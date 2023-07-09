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
        # set up a geometry component with only 1 DVGeo (stadard)
        self.options.declare("file", default=None)
        self.options.declare("type", default=None)
        self.options.declare("options", default=None)

        # use additional options for a geometry component that needs multiple DVGeos
        self.options.declare("DVGeoInfo", default=None)

    def setup(self):
        self.DVCon = DVConstraints()

        # create the DVGeo object that does the computations (or multiple DVGeo objects)
        # conventional setup with one DVGeo. maintains old interface
        if self.options["DVGeoInfo"] is None:
            self.multDVGeo = False

            geoType = self.options["type"]
            file = self.optionsp["file"]

            if self.options["options"] is None:
                options = {}
            else:
                options = self.options["options"]
                
            # we are doing an FFD-based DVGeo
            if geoType == "ffd":
                self.DVGeo = DVGeometry(file, **options)

            # we are doing a VSP-based DVGeo
            elif geoType == "vsp":
                self.DVGeo = DVGeometryVSP(file, comm=self.comm, **options)

            # we are doing an ESP-based DVGeo
            elif geoType == "esp":
                self.DVGeo = DVGeometryESP(file, comm=self.comm, **options)

            # add the geometry to the constraints object
            self.DVCon.setDVGeo(self.DVGeo)
        
        # we need to add multiple DVGeos to this geometry component
        else:
            self.multDVGeo = True

            DVGeoInfo = self.options["DVGeoInfo"]
            self.DVGeos = {}

            # create the DVGeo object that does the computations
            for name, info in DVGeoInfo.items():
                if info.get("options") is None:
                    options = {}
                else:
                    options = info["options"]

                if info.get("name") is None:
                    name = None
                else:
                    name = info["name"]

                # this DVGeo uses FFD
                if info["type"] == "ffd":
                    self.DVGeos.update({name: DVGeometry(info["file"], name=name, **options)})

                # this DVGeo uses VSP
                elif info["type"] == "vsp":
                    self.DVGeo.update({name: DVGeometryVSP(info["file"], comm=self.comm, name=name, **options)})

                # this DVGeo uses ESP
                elif info["type"] == "esp":
                    self.DVGeos.update({name: DVGeometryESP(info["file"], comm=self.comm, name=name, **options)})

            # add each geometry to the constraints object
            for _, DVGeo in self.DVGeos.items():
                self.DVCon.setDVGeo(DVGeo, name=DVGeo.name)

        self.omPtSetList = []

    def nom_updateDVGeo(self, inputs, outputs, DVGeo):
        # inputs are the geometric design variables
        DVGeo.setDesignVars(inputs)

        # ouputs are the coordinates of the pointsets we have
        for ptName in DVGeo.points:
            if ptName in self.omPtSetList:
                # update this pointset and write it as output
                outputs[ptName] = DVGeo.update(ptName).flatten()

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
        if self.multDVGeo:
            for _, DVGeo in self.DVGeos.items():
               self.updateDVGeo(inputs, outputs, DVGeo)
        else:
            self.nom_updateDVGeo(inputs, outputs, self.DVGeo)

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

    def nom_addChild(self, ffd_file, DVGeoName=None):
        # if we have multiple DVGeos use the one specified by name
        if self.multDVGeo:
            DVGeo = self.DVGeos[DVGeoName]
        else:
            DVGeo = self.DVGeo

        # can only add a child to a FFD DVGeo
        if DVGeo.geoType != "ffd":
            raise RuntimeError(
                f"Only FFD-based DVGeo objects can have children added to them, not type:{DVGeo.geoType}"
            )

        # Add child FFD
        child_ffd = DVGeometry(ffd_file, child=True)
        DVGeo.addChild(child_ffd)

        # Embed points from parent if not already done
        for pointSet in DVGeo.points:
            if pointSet not in DVGeo.children[-1].points:
                DVGeo.children[-1].addPointSet(DVGeo.points[pointSet], pointSet)

    def nom_add_discipline_coords(self, discipline, DVGeoName=None, points=None):
        # TODO remove one of these methods to keep only one method to add pointsets

        if points is None:
            # no pointset info is provided, just do a generic i/o. We will add these points during the first compute
            self.add_input("x_%s_in" % discipline, distributed=True, shape_by_conn=True)
            self.add_output("x_%s0" % discipline, distributed=True, copy_shape="x_%s_in" % discipline)

        else:
            # if we have multiple DVGeos use the one specified by name
            if self.multDVGeo:
                DVGeo = self.DVGeos[DVGeoName]
            else:
                DVGeo = self.DVGeo

            # we are provided with points. we can do the full initialization now
            self.nom_addPointSet(DVGeo, points, "x_%s0" % discipline, add_output=False)
            self.add_input("x_%s_in" % discipline, distributed=True, val=points.flatten())
            self.add_output("x_%s0" % discipline, distributed=True, val=points.flatten())

    def nom_addPointSet(self, points, ptName, add_output=True, DVGeoName=None, **kwargs):
        # if we have multiple DVGeos use the one specified by name
        if self.multDVGeo:
            DVGeo = self.DVGeos[DVGeoName]
        else:
            DVGeo = self.DVGeo

        # add the points to the dvgeo object
        DVGeo.addPointSet(points.reshape(len(points) // 3, 3), ptName, **kwargs)
        self.omPtSetList.append(ptName)

        if DVGeo.geoType == "ffd":
            for i in range(len(DVGeo.children)):
                # Embed points from parent if not already done
                for pointSet in DVGeo.points:
                    if pointSet not in DVGeo.children[i].points:
                        DVGeo.children[i].addPointSet(DVGeo.points[pointSet], pointSet)

        if add_output:
            # add an output to the om component
            self.add_output(ptName, distributed=True, val=points.flatten())

    def nom_add_point_dict(self, point_dict):
        # add every pointset in the dict, and set the ptset name as the key
        for k, v in point_dict.items():
            self.nom_addPointSet(v, k)

    def nom_getDVGeo(self, childIdx=None, DVGeoName=None):
        """
         Gets the DVGeometry object held in the geometry component so DVGeo methods can be called directly on it

        Parameters
        ----------
        DVGeoName : string, optional
            The name of the DVGeo to return, necessary if there are multiple DVGeo objects
        childIdx : int, optional
            The zero-based index of the child FFD, if you want a child DVGeo returned

        Returns
        -------
        DVGeo object
        """
        y = 0


        # if we have multiple DVGeos use the one specified by name
        if self.multDVGeo:
            DVGeo = self.DVGeos[DVGeoName]
        else:
            DVGeo = self.DVGeo

        # return the top level DVGeo
        if childIdx is None:
            return DVGeo

        # return a child DVGeo
        else:
            return DVGeo.children[childIdx]

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

    def nom_addGlobalDV(self, dvName, value, func, childIdx=None, isComposite=False, DVGeoName=None):
        # if we have multiple DVGeos use the one specified by name
        if self.multDVGeo:
            DVGeo = self.DVGeos[DVGeoName]
        else:
            DVGeo = self.DVGeo

        # global DVs are only added to FFD-based DVGeo objects
        if DVGeo.geoType != "ffd":
            raise RuntimeError(f"Only FFD-based DVGeo objects can use global DVs, not type:{DVGeo.geoType}")

        # define the input
        # When composite DVs are used, input is not required for the default DVs. Now the composite DVs are
        # the actual DVs. So OpenMDAO don't need the default DVs as inputs.
        if not isComposite:
            print(f"add dv {dvName} with shape {value}")
            self.add_input(dvName, distributed=False, shape=len(value))

        # call the dvgeo object and add this dv
        if childIdx is None:
            DVGeo.addGlobalDV(dvName, value, func)
        else:
            DVGeo.children[childIdx].addGlobalDV(dvName, value, func)

    def nom_addLocalDV(self, dvName, axis="y", pointSelect=None, childIdx=None, isComposite=False, DVGeoName=None):
        # if we have multiple DVGeos use the one specified by name
        if self.multDVGeo:
            DVGeo = self.DVGeos[DVGeoName]
        else:
            DVGeo = self.DVGeo

        # local DVs are only added to FFD-based DVGeo objects
        if DVGeo.geoType != "ffd":
            raise RuntimeError(f"Only FFD-based DVGeo objects can use local DVs, not type:{DVGeo.geoType}")

        if childIdx is None:
            nVal = DVGeo.addLocalDV(dvName, axis=axis, pointSelect=pointSelect)
        else:
            nVal = DVGeo.children[childIdx].addLocalDV(dvName, axis=axis, pointSelect=pointSelect)

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
        childIdx=None,
        axis=1,
        pointSelect=None,
        volList=None,
        orient0=None,
        orient2="svd",
        config=None,
        DVGeoName=None,
    ):
        # if we have multiple DVGeos use the one specified by name
        if self.multDVGeo:
            DVGeo = self.DVGeos[DVGeoName]
        else:
            DVGeo = self.DVGeo

        # local DVs are only added to FFD-based DVGeo objects
        if DVGeo.geoType != "ffd":
            raise RuntimeError(f"Only FFD-based DVGeo objects can use local DVs, not type:{DVGeo.geoType}")

        # add the DV to a normal DVGeo
        if childIdx is None:
            nVal = DVGeo.addLocalSectionDV(dvName, secIndex, axis, pointSelect, volList, orient0, orient2, config)
        # add the DV to a child DVGeo
        else:
            nVal = DVGeo.children[childIdx].addLocalSectionDV(
                dvName,
                secIndex,
                axis,
                pointSelect,
                volList,
                orient0,
                orient2,
                config,
            )

        # define the input
        self.add_input(dvName, distributed=False, shape=nVal)
        return nVal

    def nom_addShapeFunctionDV(self, dvName, shapes, childIdx=None, config=None, DVGeoName=None):
        # if we have multiple DVGeos use the one specified by name
        if self.multDVGeo:
            DVGeo = self.DVGeos[DVGeoName]
        else:
            DVGeo = self.DVGeo

        # shape function DVs are only added to FFD-based DVGeo objects
        if DVGeo.geoType != "ffd":
            raise RuntimeError(f"Only FFD-based DVGeo objects can use local DVs, not type:{DVGeo.geoType}")

        # add the DV to a normal DVGeo
        if childIdx is None:
            nVal = DVGeo.addShapeFunctionDV(dvName, shapes, config)
        # add the DV to a child DVGeo
        else:
            nVal = DVGeo.children[childIdx].addShapeFunctionDV(dvName, shapes, config)

        # define the input
        self.add_input(dvName, distributed=False, shape=nVal)
        return nVal

    def nom_addGeoCompositeDV(self, dvName, ptSetName=None, u=None, scale=None, DVGeoName=None, **kwargs):
        # if we have multiple DVGeos use the one specified by name
        if self.multDVGeo:
            DVGeo = self.DVGeos[DVGeoName]
        else:
            DVGeo = self.DVGeo

        # call the dvgeo object and add this dv
        DVGeo.addCompositeDV(dvName, ptSetName=ptSetName, u=u, scale=scale, **kwargs)
        val = DVGeo.getValues()

        # define the input
        self.add_input(dvName, distributed=False, shape=DVGeo.getNDV(), val=val[dvName][0])

    def nom_addVSPVariable(self, component, group, parm, isComposite=False, DVGeoName=None, **kwargs):
        # if we have multiple DVGeos use the one specified by name
        if self.multDVGeo:
            DVGeo = self.DVGeos[DVGeoName]
        else:
            DVGeo = self.DVGeo

        # VSP DVs are only added to VSP-based DVGeo objects
        if DVGeo.geoType != "vsp":
            raise RuntimeError(f"Only VSP-based DVGeo objects can use VSP DVs, not type:{DVGeo.geoType}")

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
        if self.multDVGeo:
            DVGeo = self.DVGeos[DVGeoName]
        else:
            DVGeo = self.DVGeo

        # ESP DVs are only added to VSP-based DVGeo objects
        if DVGeo.geoType != "esp":
            raise RuntimeError(f"Only ESP-based DVGeo objects can use ESP DVs, not type:{DVGeo.geoType}")

        # actually add the DV to ESP
        DVGeo.addVariable(desmptr_name, **kwargs)

        # get the value
        val = DVGeo.DVs[desmptr_name].value.copy()

        # add the input with the correct value, VSP DVs always have a size of 1
        # When composite DVs are used, input is not required for the default DVs. Now the composite DVs are
        # the actual DVs. So OpenMDAO don't need the default DVs as inputs.
        if not isComposite:
            self.add_input(desmptr_name, distributed=False, shape=val.shape, val=val)

    def nom_addRefAxis(self, childIdx=None, DVGeoName=None, **kwargs):
        # if we have multiple DVGeos use the one specified by name
        if self.multDVGeo:
            DVGeo = self.DVGeos[DVGeoName]
        else:
            DVGeo = self.DVGeo

        # references axes are only needed in FFD-based DVGeo objects
        if DVGeo.geoType != "ffd":
            raise RuntimeError(f"Only FFD-based DVGeo objects can use reference axes, not type:{DVGeo.geoType}")
        
        # add ref axis to this DVGeo
        if childIdx is None:
            return DVGeo.addRefAxis(**kwargs)
        # add ref axis to the specified child
        else:
            return DVGeo.children[childIdx].addRefAxis(**kwargs)

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
        surfaceName="default",
        DVGeoName="default",
        compNames=None,
    ):
        self.DVCon.addThicknessConstraints2D(
            leList,
            teList,
            nSpan,
            nChord,
            name=name,
            scaled=scaled,
            surfaceName=surfaceName,
            DVGeoName=DVGeoName,
            compNames=compNames,
        )
        self.add_output(name, distributed=False, val=np.ones((nSpan * nChord,)), shape=nSpan * nChord)

    def nom_addThicknessConstraints1D(
        self, name, ptList, nCon, axis, scaled=True, surfaceName="default", DVGeoName="default", compNames=None
    ):
        self.DVCon.addThicknessConstraints1D(
            ptList,
            nCon,
            axis,
            name=name,
            scaled=scaled,
            surfaceName=surfaceName,
            DVGeoName=DVGeoName,
            compNames=compNames,
        )
        self.add_output(name, distributed=False, val=np.ones(nCon), shape=nCon)

    def nom_addVolumeConstraint(self, name, leList, teList, nSpan=10, nChord=10, surfaceName="default"):
        self.DVCon.addVolumeConstraint(leList, teList, nSpan=nSpan, nChord=nChord, name=name, surfaceName=surfaceName)
        self.add_output(name, distributed=False, val=1.0)

    def nom_add_LETEConstraint(self, name, volID, faceID, topID=None, childIdx=None):
        self.DVCon.addLeTeConstraints(volID, faceID, name=name, topID=topID, childIdx=childIdx)
        # how many are there?
        conobj = self.DVCon.linearCon[name]
        nCon = len(conobj.indSetA)
        self.add_output(name, distributed=False, val=np.zeros((nCon,)), shape=nCon)
        return nCon

    def nom_addLERadiusConstraints(self, name, leList, nSpan, axis, chordDir):
        self.DVCon.addLERadiusConstraints(leList=leList, nSpan=nSpan, axis=axis, chordDir=chordDir, name=name)
        self.add_output(name, distributed=False, val=np.ones(nSpan), shape=nSpan)

    def nom_addCurvatureConstraint1D(self, name, start, end, nPts, axis, **kwargs):
        self.DVCon.addCurvatureConstraint1D(start=start, end=end, nPts=nPts, axis=axis, name=name, **kwargs)
        self.add_output(name, distributed=False, val=1.0)

    def nom_addLinearConstraintsShape(self, name, indSetA, indSetB, factorA, factorB, childIdx=None):
        self.DVCon.addLinearConstraintsShape(
            indSetA=indSetA, indSetB=indSetB, factorA=factorA, factorB=factorB, name=name, childIdx=childIdx
        )
        lSize = len(indSetA)
        self.add_output(name, distributed=False, val=np.zeros(lSize), shape=lSize)

    def nom_addTriangulatedSurfaceConstraint(
        self,
        name,
        surface_1_name=None,
        DVGeo1=None,
        surface_2_name="default",
        DVGeo2=None,
        rho=50.0,
        heuristic_dist=None,
        max_perim=3.0,
    ):
        self.DVCon.addTriangulatedSurfaceConstraint(
            comm=self.comm,
            surface_1_name=surface_1_name,
            DVGeo1=DVGeo1,
            surface_2_name=surface_2_name,
            DVGeo2=DVGeo2,
            rho=rho,
            heuristic_dist=heuristic_dist,
            max_perim=max_perim,
            name=name,
        )

        self.add_output(f"{name}_KS", distributed=False, val=0)
        self.add_output(f"{name}_perim", distributed=False, val=0)

    def nom_setConstraintSurface(
        self, surface, name="default", addToDVGeo=False, DVGeoName="default", surfFormat="point-vector"
    ):
        # constraint needs a triangulated reference surface at initialization
        self.DVCon.setSurface(surface, name=name, addToDVGeo=addToDVGeo, DVGeoName=DVGeoName, surfFormat=surfFormat)

    def nom_computeOnDVGeo(self, DVGeo, d_inputs, d_outputs):
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
        return d_inputs

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

            if self.multDVGeo:
                for _, DVGeo in self.DVGeos.items():
                    d_inputs = self.nom_computeOnDVGeo(DVGeo, d_inputs, d_outputs)
            else:
                d_inputs = self.nom_computeOnDVGeo(DVGeo, d_inputs, d_outputs)
