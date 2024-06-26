# External modules
from mpi4py import MPI
import numpy as np
import openmdao.api as om
from openmdao.api import AnalysisError

# Local modules
from .. import DVConstraints, DVGeometry, DVGeometryESP, DVGeometryVSP


# class that actually calls the DVGeometry methods
class OM_DVGEOCOMP(om.ExplicitComponent):
    def initialize(self):
        r"""
        Set up a geometry component with either 1 DVGeo or multiple DVGeos.

        A single DVGeo are initialized by specifying its file and type and, optionally, additional options.
        Available options can be found in the specific DVGeometry class.

        Multiple DVGeos are initialized in a dictionary of these values and must have a unique name. The format is:

        >>> DVGeoInfo = {
        >>>     "name1": {"file": file1, "type": type1, "options": options1}
        >>>     "name2": {"file": file2, "type": type2, "options": options2}
        >>> }

        The two setup methods cannot currently be used together.
        """

        self.options.declare("file", default=None)
        self.options.declare("type", default=None)
        self.options.declare("options", default=None)
        self.options.declare("DVGeoInfo", default=None)

    def setup(self):
        # create a constraints object to go with this DVGeo(s)
        self.DVCon = DVConstraints()
        # hold the DVGeo(s) in a dictionary
        self.DVGeos = {}

        # conventional setup with one DVGeo. maintains old interface
        if self.options["DVGeoInfo"] is None:
            self.multDVGeo = False

            # set up DVGeoInfo so a single DVGeo can be initialized with the multi-DVGeo case
            DVGeoInfo = {
                "defaultDVGeo": {
                    "file": self.options["file"],
                    "type": self.options["type"],
                    "options": self.options["options"],
                }
            }

            # DVGeo and DVCon expect different defaults
            DVConName = "default"
            DVGeoName = None

        # we need to add multiple DVGeos to this geometry component
        # the actual initialization is handled in the same way regardless
        else:
            self.multDVGeo = True
            DVGeoInfo = self.options["DVGeoInfo"]

        # create the DVGeo object that does the computations (or multiple DVGeo objects)
        for name, info in DVGeoInfo.items():
            if self.multDVGeo:
                DVGeoName = DVConName = name

            if info.get("options") is None:
                options = {}
            else:
                options = info["options"]

            # this DVGeo uses FFD
            if info["type"] == "ffd":
                self.DVGeos.update({name: DVGeometry(info["file"], name=DVGeoName, **options)})

            # this DVGeo uses VSP
            elif info["type"] == "vsp":
                self.DVGeos.update({name: DVGeometryVSP(info["file"], comm=self.comm, name=DVGeoName, **options)})

            # this DVGeo uses ESP
            elif info["type"] == "esp":
                self.DVGeos.update({name: DVGeometryESP(info["file"], comm=self.comm, name=DVGeoName, **options)})

            # add each geometry to the constraints object
            for _, DVGeo in self.DVGeos.items():
                self.DVCon.setDVGeo(DVGeo, name=DVConName)

        self.omPtSetList = []

    def compute(self, inputs, outputs):
        # check for inputs that have been added but the points have not been added to dvgeo
        for var in inputs.keys():
            # check that the input name matches the convention for points
            if var[:2] == "x_":
                # trim the _in and add a "0" to signify that these are initial conditions initial
                var_out = var[:-3] + "0"
                if var_out not in self.omPtSetList:
                    self.nom_addPointSet(inputs[var], var_out, add_output=False)

        # handle DV update and pointset changes for all of our DVGeos
        for _, DVGeo in self.DVGeos.items():
            # inputs are the geometric design variables
            DVGeo.setDesignVars(inputs)

            # ouputs are the coordinates of the pointsets we have
            for ptName in DVGeo.points:
                if ptName in self.omPtSetList:
                    # update this pointset and write it as output
                    outputs[ptName] = DVGeo.update(ptName).flatten()

        # compute the DVCon constraint values
        constraintfunc = dict()
        self.DVCon.evalFunctions(constraintfunc, includeLinear=True)

        for constraintname in constraintfunc:
            # if any constraint returned a fail flag throw an error to OpenMDAO
            # all constraints need the same fail flag, no <name_> prefix
            if constraintname == "fail":
                raise AnalysisError("Analysis error in geometric constraints")
            outputs[constraintname] = constraintfunc[constraintname]

        # we ran a compute so the inputs changed. update the dvcon jac
        # next time the jacvec product routine is called
        self.update_jac = True

    def nom_addChild(self, ffd_file, DVGeoName=None, childName=None):
        # if we have multiple DVGeos use the one specified by name
        DVGeo = self.nom_getDVGeo(DVGeoName=DVGeoName)

        # can only add a child to a FFD DVGeo
        if not isinstance(DVGeo, DVGeometry):
            raise RuntimeError(
                f"Only FFD-based DVGeo objects can have children added to them, not type: {type(DVGeo).__name__}"
            )

        # Add child FFD
        child_ffd = DVGeometry(ffd_file, child=True, name=childName)
        DVGeo.addChild(child_ffd)

        # Embed points from parent if not already done
        for pointSet in DVGeo.points:
            if pointSet not in child_ffd.points:
                child_ffd.addPointSet(DVGeo.points[pointSet], pointSet)

    def nom_add_discipline_coords(self, discipline, points=None, DVGeoName=None):
        # TODO remove one of these methods to keep only one method to add pointsets

        if points is None:
            # no pointset info is provided, just do a generic i/o. We will add these points during the first compute
            self.add_input("x_%s_in" % discipline, distributed=True, shape_by_conn=True)
            self.add_output("x_%s0" % discipline, distributed=True, copy_shape="x_%s_in" % discipline)

        else:
            # we are provided with points. we can do the full initialization now
            self.nom_addPointSet(points, "x_%s0" % discipline, add_output=False, DVGeoName=DVGeoName)
            self.add_input("x_%s_in" % discipline, distributed=True, val=points.flatten())
            self.add_output("x_%s0" % discipline, distributed=True, val=points.flatten())

    def nom_addPointSet(self, points, ptName, add_output=True, DVGeoName=None, **kwargs):
        # if we have multiple DVGeos use the one specified by name
        DVGeo = self.nom_getDVGeo(DVGeoName=DVGeoName)

        # add the points to the dvgeo object
        DVGeo.addPointSet(points.reshape(len(points) // 3, 3), ptName, **kwargs)
        self.omPtSetList.append(ptName)

        if isinstance(DVGeo, DVGeometry):
            for child in DVGeo.children.values():
                # Embed points from parent if not already done
                for pointSet in DVGeo.points:
                    if pointSet not in child.points:
                        child.addPointSet(DVGeo.points[pointSet], pointSet)

        if add_output:
            # add an output to the om component
            self.add_output(ptName, distributed=True, val=points.flatten())

    def nom_add_point_dict(self, point_dict):
        # add every pointset in the dict, and set the ptset name as the key
        for k, v in point_dict.items():
            self.nom_addPointSet(v, k)

    def nom_getDVGeo(self, childName=None, DVGeoName=None):
        """
        Gets the DVGeometry object held in the geometry component so DVGeo methods can be called directly on it

        Parameters
        ----------
        DVGeoName : string, optional
            The name of the DVGeo to return, necessary if there are multiple DVGeo objects

        childName : str, optional
            Name of the child FFD, if you want a child DVGeo returned

        Returns
        -------
        DVGeometry object
            DVGeometry object held by this geometry component
        """
        # if we have multiple DVGeos use the one specified by name
        if self.multDVGeo:
            DVGeo = self.DVGeos[DVGeoName]
        else:
            DVGeo = self.DVGeos["defaultDVGeo"]

        # return the top level DVGeo
        if childName is None:
            return DVGeo

        # return a child DVGeo
        else:
            return DVGeo.children[childName]

    def nom_getDVCon(self):
        """
        Gets the DVConstraints object held in the geometry component so DVCon methods can be called directly on it

        Returns
        -------
        self.DVCon, DVConstraints object
            DVConstraints object held by this geometry component
        """
        return self.DVCon

    """
    Wrapper for DVGeo functions
    """

    def nom_addGlobalDV(
        self, dvName, value, func, childName=None, isComposite=False, DVGeoName=None, prependName=False, config=None
    ):
        """
        Add a global design variable to the DVGeo object. This is a wrapper for the DVGeo.addGlobalDV method.

        Parameters
        ----------
        dvName : str
            See :meth:`addGlobalDV <.DVGeometry.addGlobalDV>`

        value : float, or iterable list of floats
            See :meth:`addGlobalDV <.DVGeometry.addGlobalDV>`

        func : python function
            See :meth:`addGlobalDV <.DVGeometry.addGlobalDV>`

        childName : str, optional
            Name of the child FFD, if this DV is for a child FFD.

        isComposite : bool, optional
            Whether this DV is to be included in the composite DVs, by default False

        DVGeoName : string, optional
            The name of the DVGeo to return, necessary if there are multiple DVGeo objects

        Raises
        ------
        RuntimeError
            Raised if the underlying DVGeo object is not an FFD
        """

        # if we have multiple DVGeos use the one specified by name
        DVGeo = self.nom_getDVGeo(childName=childName, DVGeoName=DVGeoName)

        # global DVs are only added to FFD-based DVGeo objects
        if not isinstance(DVGeo, DVGeometry):
            raise RuntimeError(f"Only FFD-based DVGeo objects can use global DVs, not type: {type(DVGeo).__name__}")

        # if this DVGeo object has a name attribute, prepend it to match up with what DVGeo is expecting
        # this keeps track of DVs between multiple DVGeo objects
        if DVGeoName is not None and prependName:
            dvName = DVGeoName + "_" + dvName

        # call the dvgeo object and add this dv
        DVGeo.addGlobalDV(dvName, value, func, prependName=False, config=config)

        # define the input
        # When composite DVs are used, input is not required for the default DVs. Now the composite DVs are
        # the actual DVs. So OpenMDAO don't need the default DVs as inputs.
        if not isComposite:
            self.add_input(dvName, distributed=False, shape=len(np.atleast_1d(value)))

    def nom_addLocalDV(
        self,
        dvName,
        axis="y",
        pointSelect=None,
        childName=None,
        isComposite=False,
        DVGeoName=None,
        prependName=False,
        volList=None,
        config=None,
    ):
        # if we have multiple DVGeos use the one specified by name
        DVGeo = self.nom_getDVGeo(childName=childName, DVGeoName=DVGeoName)

        # local DVs are only added to FFD-based DVGeo objects
        if not isinstance(DVGeo, DVGeometry):
            raise RuntimeError(f"Only FFD-based DVGeo objects can use local DVs, not type: {type(DVGeo).__name__}")

        # if this DVGeo object has a name attribute, prepend it to match up with what DVGeo is expecting
        # this keeps track of DVs between multiple DVGeo objects
        if DVGeoName is not None and prependName:
            dvName = DVGeoName + "_" + dvName

        # add the DV to DVGeo
        nVal = DVGeo.addLocalDV(
            dvName, axis=axis, pointSelect=pointSelect, prependName=False, config=config, volList=volList
        )

        # define the input
        # When composite DVs are used, input is not required for the default DVs. Now the composite DVs are
        # the actual DVs. So OpenMDAO don't need the default DVs as inputs.
        if not isComposite:
            self.add_input(dvName, distributed=False, shape=nVal)
        return nVal

    def nom_addLocalSectionDV(
        self,
        dvName,
        secIndex,
        childName=None,
        axis=1,
        pointSelect=None,
        volList=None,
        orient0=None,
        orient2="svd",
        config=None,
        DVGeoName=None,
        prependName=False,
    ):
        """
        Add one or more section local design variables to the DVGeometry object
        Wrapper for :meth:`addLocalSectionDV <.DVGeometry.addLocalSectionDV>`
        Input parameters are identical to those in wrapped function unless otherwise specified

        Parameters
        ----------
        dvName : str
            Name to give this design variable

        secIndex : char or list of chars
            See wrapped

        childName : str, optional
            Name of the child FFD, if this DV is for a child FFD.

        axis : int, optional
            See wrapped

        pointSelect : pointSelect object, optional
            See wrapped

        volList : list, optional
            See wrapped

        orient0 : orientation, optional
            See wrapped

        orient2 : str, optional
            See wrapped

        config : str or list, optional
            See wrapped

        DVGeoName : string, optional
            The name of the DVGeo to return, necessary if there are multiple DVGeo objects

        Returns
        -------
        nVal, int
            number of local section DVs

        Raises
        ------
        RuntimeError
            Raised if the underlying DVGeo parameterization is not FFD-based
        """

        # if we have multiple DVGeos use the one specified by name
        DVGeo = self.nom_getDVGeo(childName=childName, DVGeoName=DVGeoName)

        # local DVs are only added to FFD-based DVGeo objects
        if not isinstance(DVGeo, DVGeometry):
            raise RuntimeError(
                f"Only FFD-based DVGeo objects can use local section DVs, not type: {type(DVGeo).__name__}"
            )

        # if this DVGeo object has a name attribute, prepend it to match up with what DVGeo is expecting
        # this keeps track of DVs between multiple DVGeo objects
        if DVGeoName is not None and prependName:
            dvName = DVGeoName + "_" + dvName

        # add the DV to DVGeo
        nVal = DVGeo.addLocalSectionDV(
            dvName, secIndex, axis, pointSelect, volList, orient0, orient2, config, prependName=False
        )

        # define the input
        self.add_input(dvName, distributed=False, shape=nVal)
        return nVal

    def nom_addShapeFunctionDV(self, dvName, shapes, childName=None, config=None, DVGeoName=None, prependName=False):
        """
        Add one or more local shape function design variables to the DVGeometry object
        Wrapper for :meth:`addShapeFunctionDV <.DVGeometry.addShapeFunctionDV>`
        Input parameters are identical to those in wrapped function unless otherwise specified

        Parameters
        ----------
        dvName : str
            Name to give this design variable

        shapes : list of dictionaries, or a single dictionary
            See wrapped

        childName : str, optional
            Name of the child FFD, if this DV is for a child FFD.

        config : str or list, optional
            See wrapped

        DVGeoName : string, optional
            The name of the DVGeo to return, necessary if there are multiple DVGeo objects

        Returns
        -------
        N : int
            The number of design variables added.

        Raises
        ------
        RuntimeError
            Raised if the underlying DVGeo parameterization is not FFD-based
        """

        # if we have multiple DVGeos use the one specified by name
        DVGeo = self.nom_getDVGeo(childName=childName, DVGeoName=DVGeoName)

        # shape function DVs are only added to FFD-based DVGeo objects
        if not isinstance(DVGeo, DVGeometry):
            raise RuntimeError(
                f"Only FFD-based DVGeo objects can use shape function DVs, not type: {type(DVGeo).__name__}"
            )

        # if this DVGeo object has a name attribute, prepend it to match up with what DVGeo is expecting
        # this keeps track of DVs between multiple DVGeo objects
        if DVGeoName is not None and prependName:
            dvName = DVGeoName + "_" + dvName

        # add the DV to DVGeo
        nVal = DVGeo.addShapeFunctionDV(dvName, shapes, config, prependName=False)

        # define the input
        self.add_input(dvName, distributed=False, shape=nVal)
        return nVal

    def nom_addGeoCompositeDV(
        self, dvName, ptSetName=None, u=None, scale=None, DVGeoName=None, prependName=False, **kwargs
    ):
        # if we have multiple DVGeos use the one specified by name
        DVGeo = self.nom_getDVGeo(DVGeoName=DVGeoName)

        # if this DVGeo object has a name attribute, prepend it to match up with what DVGeo is expecting
        # this keeps track of DVs between multiple DVGeo objects
        if DVGeoName is not None and prependName:
            dvName = DVGeoName + "_" + dvName

        # call the dvgeo object and add this dv
        DVGeo.addCompositeDV(dvName, ptSetName=ptSetName, u=u, scale=scale, prependName=False, **kwargs)
        val = DVGeo.getValues()

        # define the input
        self.add_input(dvName, distributed=False, shape=DVGeo.getNDV(), val=val[dvName][0])

    def nom_addVSPVariable(self, component, group, parm, isComposite=False, DVGeoName=None, **kwargs):
        # if we have multiple DVGeos use the one specified by name
        DVGeo = self.nom_getDVGeo(DVGeoName=DVGeoName)

        # VSP DVs are only added to VSP-based DVGeo objects
        if not isinstance(DVGeo, DVGeometryVSP):
            raise RuntimeError(f"Only VSP-based DVGeo objects can use VSP DVs, not type: {type(DVGeo).__name__}")

        # actually add the DV to VSP
        DVGeo.addVariable(component, group, parm, **kwargs)

        # full name of this DV
        dvName = "%s:%s:%s" % (component, group, parm)

        # get the value
        val = DVGeo.DVs[dvName].value.copy()

        # define the input
        # When composite DVs are used, input is not required for the default DVs. Now the composite DVs are
        # the actual DVs. So OpenMDAO don't need the default DVs as inputs.
        if not isComposite:
            self.add_input(dvName, distributed=False, shape=1, val=val)

    def nom_addESPVariable(self, desmptr_name, isComposite=False, DVGeoName=None, **kwargs):
        # if we have multiple DVGeos use the one specified by name
        DVGeo = self.nom_getDVGeo(DVGeoName=DVGeoName)

        # ESP DVs are only added to VSP-based DVGeo objects
        if not isinstance(DVGeo, DVGeometryESP):
            raise RuntimeError(f"Only ESP-based DVGeo objects can use ESP DVs, not type: {type(DVGeo).__name__}")

        # actually add the DV to ESP
        DVGeo.addVariable(desmptr_name, **kwargs)

        # get the value
        val = DVGeo.DVs[desmptr_name].value.copy()

        # add the input with the correct value, VSP DVs always have a size of 1
        # When composite DVs are used, input is not required for the default DVs. Now the composite DVs are
        # the actual DVs. So OpenMDAO don't need the default DVs as inputs.
        if not isComposite:
            self.add_input(desmptr_name, distributed=False, shape=val.shape, val=val)

    def nom_addRefAxis(
        self,
        name,
        childName=None,
        DVGeoName=None,
        curve=None,
        xFraction=None,
        yFraction=None,
        zFraction=None,
        volumes=None,
        rotType=5,
        axis="x",
        alignIndex=None,
        rotAxisVar=None,
        rot0ang=None,
        rot0axis=[1, 0, 0],
        includeVols=[],
        ignoreInd=[],
        raySize=1.5,
    ):
        # TODO: we should change `volume` to `volList`, to be consistent with other APIs.
        # But doing this may create backward incompatibility. So we will use `volumes` for now

        # if we have multiple DVGeos use the one specified by name
        DVGeo = self.nom_getDVGeo(childName=childName, DVGeoName=DVGeoName)

        # references axes are only needed in FFD-based DVGeo objects
        if not isinstance(DVGeo, DVGeometry):
            raise RuntimeError(f"Only FFD-based DVGeo objects can use reference axes, not type: {type(DVGeo).__name__}")

        # add ref axis to this DVGeo
        return DVGeo.addRefAxis(
            name=name,
            curve=curve,
            xFraction=xFraction,
            yFraction=yFraction,
            zFraction=zFraction,
            volumes=volumes,
            rotType=rotType,
            axis=axis,
            alignIndex=alignIndex,
            rotAxisVar=rotAxisVar,
            rot0ang=rot0ang,
            rot0axis=rot0axis,
            includeVols=includeVols,
            ignoreInd=ignoreInd,
            raySize=raySize,
        )
        # add ref axis to the specified child

    """
    Wrapper for DVCon functions
    """

    def nom_addThicknessConstraints2D(
        self,
        name,
        leList,
        teList,
        nSpan,
        nChord,
        scaled=True,
        addToPyOpt=True,
        surfaceName="default",
        DVGeoName="default",
        compNames=None,
        projected=False,
    ):
        self.DVCon.addThicknessConstraints2D(
            leList,
            teList,
            nSpan,
            nChord,
            name=name,
            scaled=scaled,
            addToPyOpt=addToPyOpt,
            surfaceName=surfaceName,
            DVGeoName=DVGeoName,
            compNames=compNames,
            projected=projected,
        )
        self.add_output(name, distributed=False, val=np.ones((nSpan * nChord,)), shape=nSpan * nChord)

    def nom_addThicknessConstraints1D(
        self,
        name,
        ptList,
        nCon,
        axis,
        scaled=True,
        addToPyOpt=True,
        surfaceName="default",
        DVGeoName="default",
        compNames=None,
        projected=False,
    ):
        self.DVCon.addThicknessConstraints1D(
            ptList,
            nCon,
            axis,
            name=name,
            scaled=scaled,
            addToPyOpt=addToPyOpt,
            surfaceName=surfaceName,
            DVGeoName=DVGeoName,
            compNames=compNames,
            projected=projected,
        )
        self.add_output(name, distributed=False, val=np.ones(nCon), shape=nCon)

    def nom_addVolumeConstraint(
        self,
        name,
        leList,
        teList,
        nSpan=10,
        nChord=10,
        scaled=True,
        addToPyOpt=True,
        surfaceName="default",
        DVGeoName="default",
        compNames=None,
    ):
        """
        Add a DVCon volume constraint to the problem
        Wrapper for :meth:`addVolumeConstraint <.DVConstraints.addVolumeConstraint>`
        Input parameters are identical to those in wrapped function unless otherwise specified

        Parameters
        ----------
        name :
            See wrapped
        leList :
            See wrapped
        teList :
            See wrapped
        nSpan : int, optional
            See wrapped
        nChord : int, optional
            See wrapped
        scaled : bool, optional
            See wrapped
        surfaceName : str, optional
            See wrapped
        DVGeoName : str, optional
            See wrapped
        compNames : list, optional
            See wrapped
        """

        self.DVCon.addVolumeConstraint(
            leList,
            teList,
            nSpan=nSpan,
            nChord=nChord,
            scaled=scaled,
            name=name,
            addToPyOpt=addToPyOpt,
            surfaceName=surfaceName,
            DVGeoName=DVGeoName,
            compNames=compNames,
        )
        self.add_output(name, distributed=False, val=1.0)

    def nom_addSurfaceAreaConstraint(
        self, name, scaled=True, addToPyOpt=True, surfaceName="default", DVGeoName="default", compNames=None
    ):
        """
        Add a DVCon surface area constraint to the problem
        Wrapper for :meth:`addSurfaceAreaConstraint <.DVConstraints.addSurfaceAreaConstraint>`
        Input parameters are identical to those in wrapped function unless otherwise specified

        Parameters
        ----------
        name :
            See wrapped
        scaled : bool, optional
            See wrapped
        surfaceName : str, optional
            See wrapped
        DVGeoName : str, optional
            See wrapped
        compNames : list, optional
            See wrapped
        """

        self.DVCon.addSurfaceAreaConstraint(
            name=name,
            scaled=scaled,
            addToPyOpt=addToPyOpt,
            surfaceName=surfaceName,
            DVGeoName=DVGeoName,
            compNames=compNames,
        )
        self.add_output(name, distributed=False, val=1.0)

    def nom_addProjectedAreaConstraint(
        self, name, axis, scaled=True, addToPyOpt=True, surface_name="default", DVGeoName="default", compNames=None
    ):
        """
        Add a DVCon projected area constraint to the problem
        Wrapper for :meth:`addProjectedAreaConstraint <.DVConstraints.addProjectedAreaConstraint>`
        Input parameters are identical to those in wrapped function unless otherwise specified

        Parameters
        ----------
        name :
            See wrapped
        axis :
            See wrapped
        scaled : bool, optional
            See wrapped
        surface_name : str, optional
            See wrapped
        DVGeoName : str, optional
            See wrapped
        compNames : list, optional
            See wrapped
        """

        self.DVCon.addProjectedAreaConstraint(
            axis,
            name=name,
            scaled=scaled,
            addToPyOpt=addToPyOpt,
            surfaceName=surface_name,
            DVGeoName=DVGeoName,
            compNames=compNames,
        )
        self.add_output(name, distributed=False, val=1.0)

    def nom_add_LETEConstraint(
        self,
        name,
        volID,
        faceID,
        topID=None,
        indSetA=None,
        indSetB=None,
        config=None,
        childName=None,
        comp=None,
        DVGeoName="default",
    ):
        self.DVCon.addLeTeConstraints(
            volID=volID,
            faceID=faceID,
            topID=topID,
            indSetA=indSetA,
            indSetB=indSetB,
            name=name,
            config=config,
            childName=childName,
            comp=comp,
            DVGeoName=DVGeoName,
        )
        # how many are there?
        conobj = self.DVCon.linearCon[name]
        nCon = len(conobj.indSetA)
        self.add_output(name, distributed=False, val=np.zeros((nCon,)), shape=nCon)
        return nCon

    def nom_addLERadiusConstraints(
        self,
        name,
        leList,
        nSpan,
        axis,
        chordDir,
        scaled=True,
        addToPyOpt=True,
        surfaceName="default",
        DVGeoName="default",
        compNames=None,
    ):
        self.DVCon.addLERadiusConstraints(
            leList=leList,
            nSpan=nSpan,
            axis=axis,
            chordDir=chordDir,
            name=name,
            scaled=scaled,
            addToPyOpt=addToPyOpt,
            surfaceName=surfaceName,
            DVGeoName=DVGeoName,
            compNames=compNames,
        )
        self.add_output(name, distributed=False, val=np.ones(nSpan), shape=nSpan)

    def nom_addCurvatureConstraint1D(
        self,
        name,
        start,
        end,
        nPts,
        axis,
        curvatureType="mean",
        scaled=True,
        KSCoeff=1.0,
        addToPyOpt=True,
        surfaceName="default",
        DVGeoName="default",
        compNames=None,
    ):
        self.DVCon.addCurvatureConstraint1D(
            start=start,
            end=end,
            nPts=nPts,
            axis=axis,
            name=name,
            curvatureType=curvatureType,
            scaled=scaled,
            KSCoeff=KSCoeff,
            addToPyOpt=addToPyOpt,
            surfaceName=surfaceName,
            DVGeoName=DVGeoName,
            compNames=compNames,
        )
        self.add_output(name, distributed=False, val=1.0)

    def nom_addLinearConstraintsShape(
        self, name, indSetA, indSetB, factorA, factorB, config=None, childName=None, comp=None, DVGeoName="default"
    ):
        self.DVCon.addLinearConstraintsShape(
            indSetA=indSetA,
            indSetB=indSetB,
            factorA=factorA,
            factorB=factorB,
            name=name,
            config=config,
            childName=childName,
            comp=comp,
            DVGeoName=DVGeoName,
        )
        lSize = len(indSetA)
        self.add_output(name, distributed=False, val=np.zeros(lSize), shape=lSize)

    def nom_addTriangulatedSurfaceConstraint(
        self,
        name,
        surface_1_name=None,
        DVGeo_1_name="default",
        surface_2_name="default",
        DVGeo_2_name="default",
        rho=50.0,
        heuristic_dist=None,
        max_perim=3.0,
        addToPyOpt=True,
    ):
        self.DVCon.addTriangulatedSurfaceConstraint(
            comm=self.comm,
            surface_1_name=surface_1_name,
            DVGeo_1_name=DVGeo_1_name,
            surface_2_name=surface_2_name,
            DVGeo_2_name=DVGeo_2_name,
            rho=rho,
            heuristic_dist=heuristic_dist,
            max_perim=max_perim,
            name=name,
            addToPyOpt=addToPyOpt,
        )

        self.add_output(f"{name}_KS", distributed=False, val=0)
        self.add_output(f"{name}_perim", distributed=False, val=0)

    def nom_setConstraintSurface(
        self, surface, name="default", addToDVGeo=False, DVGeoName="default", surfFormat="point-vector"
    ):
        # constraint needs a triangulated reference surface at initialization
        self.DVCon.setSurface(surface, name=name, addToDVGeo=addToDVGeo, DVGeoName=DVGeoName, surfFormat=surfFormat)

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        # only do the computations when we have more than zero entries in d_inputs in the reverse mode
        ni = len(list(d_inputs.keys()))

        if mode == "rev" and ni > 0:
            # this flag will be set to True after every compute call.
            # if it is true, we assume the design has changed so we re-run the sensitivity update
            # there can be hundreds of calls to this routine due to thickness constraints,
            # as a result, we only run the actual sensitivity comp once and save the jacobians
            # this might be better suited with the matrix-based API
            if self.update_jac:
                self.constraintfuncsens = dict()
                self.DVCon.evalFunctionsSens(self.constraintfuncsens, includeLinear=True)
                # set the flag to False so we dont run the update again if this is called w/o a compute in between
                self.update_jac = False

            for constraintname in self.constraintfuncsens:
                for dvname in self.constraintfuncsens[constraintname]:
                    if dvname in d_inputs:
                        dcdx = self.constraintfuncsens[constraintname][dvname]
                        dout = d_outputs[constraintname]
                        jvtmp = np.dot(np.transpose(dcdx), dout)
                        d_inputs[dvname] += jvtmp

            for _, DVGeo in self.DVGeos.items():
                for ptSetName in DVGeo.ptSetNames:
                    if ptSetName in self.omPtSetList:
                        dout = d_outputs[ptSetName].reshape(len(d_outputs[ptSetName]) // 3, 3)

                        # only do the calc. if d_output is not zero on ANY proc
                        local_all_zeros = np.all(dout == 0)
                        global_all_zeros = np.zeros(1, dtype=bool)
                        # we need to communicate for this check otherwise we may hang
                        self.comm.Allreduce([local_all_zeros, MPI.BOOL], [global_all_zeros, MPI.BOOL], MPI.LAND)

                        # global_all_zeros is a numpy array of size 1
                        if not global_all_zeros[0]:
                            # TODO totalSensitivityTransProd is broken. does not work with zero surface nodes on a proc
                            # xdot = DVGeo.totalSensitivityTransProd(dout, ptSetName)
                            xdot = DVGeo.totalSensitivity(dout, ptSetName)

                            # loop over dvs and accumulate
                            xdotg = {}
                            for k in xdot:
                                # check if this dv is present
                                if k in d_inputs:
                                    # do the allreduce
                                    # TODO reove the allreduce when this is fixed in openmdao
                                    # reduce the result ourselves for now. ideally, openmdao will do the reduction itself when this is fixed. this is because the bcast is also done by openmdao (pyoptsparse, but regardless, it is not done here, so reduce should also not be done here)
                                    xdotg[k] = self.comm.allreduce(xdot[k], op=MPI.SUM)

                                    # accumulate in the dict
                                    # TODO
                                    # because we only do one point set at a time, we always want the 0th
                                    # entry of this array since dvgeo always behaves like we are passing
                                    # in multiple objective seeds with totalSensitivity. we can remove the [0]
                                    # once we move back to totalSensitivityTransProd
                                    d_inputs[k] += xdotg[k][0]
