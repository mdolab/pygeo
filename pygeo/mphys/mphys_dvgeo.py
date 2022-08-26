import openmdao.api as om
from .. import DVGeometry, DVConstraints

try:
    from .. import DVGeometryVSP
except ImportError:
    # not everyone might have openvsp installed, and thats okay
    pass
from mpi4py import MPI
import numpy as np

# class that actually calls the dvgeometry methods
class OM_DVGEOCOMP(om.ExplicitComponent):
    def initialize(self):

        self.options.declare("ffd_file", default=None)
        self.options.declare("vsp_file", default=None)
        self.options.declare("vsp_options", default=None)

    def setup(self):

        # create the DVGeo object that does the computations
        if self.options["ffd_file"] is not None:
            # we are doing an FFD-based DVGeo
            ffd_file = self.options["ffd_file"]
            self.DVGeo = DVGeometry(ffd_file)
        if self.options["vsp_file"] is not None:
            # we are doing a VSP based DVGeo
            vsp_file = self.options["vsp_file"]
            if self.options["vsp_options"] is None:
                vsp_options = {}
            else:
                vsp_options = self.options["vsp_options"]
            self.DVGeo = DVGeometryVSP(vsp_file, comm=self.comm, **vsp_options)

        self.DVCon = DVConstraints()
        self.DVCon.setDVGeo(self.DVGeo)
        self.omPtSetList = []

    def compute(self, inputs, outputs):
        # check for inputs thathave been added but the points have not been added to dvgeo
        for var in inputs.keys():
            # check that the input name matches the convention for points
            if var[:2] == "x_":
                # trim the _in and add a "0" to signify that these are initial conditions initial
                var_out = var[:-3] + "0"
                if var_out not in self.omPtSetList:
                    self.nom_addPointSet(inputs[var], var_out, add_output=False)

        # inputs are the geometric design variables
        self.DVGeo.setDesignVars(inputs)

        # ouputs are the coordinates of the pointsets we have
        for ptName in self.DVGeo.points:
            if ptName in self.omPtSetList:
                # update this pointset and write it as output
                outputs[ptName] = self.DVGeo.update(ptName).flatten()

        # compute the DVCon constraint values
        constraintfunc = dict()
        self.DVCon.evalFunctions(constraintfunc, includeLinear=True)
        comm = self.comm
        if comm.rank == 0:
            for constraintname in constraintfunc:
                outputs[constraintname] = constraintfunc[constraintname]

        # we ran a compute so the inputs changed. update the dvcon jac
        # next time the jacvec product routine is called
        self.update_jac = True

    def nom_addChild(self, ffd_file):
        # Add child FFD
        child_ffd = DVGeometry(ffd_file, child=True)
        self.DVGeo.addChild(child_ffd)

        # Embed points from parent if not already done
        for pointSet in self.DVGeo.points:
            if pointSet not in self.DVGeo.children[-1].points:
                self.DVGeo.children[-1].addPointSet(self.DVGeo.points[pointSet], pointSet)

    def nom_add_discipline_coords(self, discipline, points=None):
        # TODO remove one of these methods to keep only one method to add pointsets

        if points is None:
            # no pointset info is provided, just do a generic i/o. We will add these points during the first compute
            self.add_input("x_%s_in" % discipline, distributed=True, shape_by_conn=True)
            self.add_output("x_%s0" % discipline, distributed=True, copy_shape="x_%s_in" % discipline)

        else:
            # we are provided with points. we can do the full initialization now
            self.nom_addPointSet(points, "x_%s0" % discipline, add_output=False)
            self.add_input("x_%s_in" % discipline, distributed=True, val=points.flatten())
            self.add_output("x_%s0" % discipline, distributed=True, val=points.flatten())

    def nom_addPointSet(self, points, ptName, add_output=True, **kwargs):
        # add the points to the dvgeo object
        self.DVGeo.addPointSet(points.reshape(len(points) // 3, 3), ptName, **kwargs)
        self.omPtSetList.append(ptName)

        for i in range(len(self.DVGeo.children)):
            # Embed points from parent if not already done
            for pointSet in self.DVGeo.points:
                if pointSet not in self.DVGeo.children[i].points:
                    self.DVGeo.children[i].addPointSet(self.DVGeo.points[pointSet], pointSet)

        if add_output:
            # add an output to the om component
            self.add_output(ptName, distributed=True, val=points.flatten())

    def nom_add_point_dict(self, point_dict):
        # add every pointset in the dict, and set the ptset name as the key
        for k, v in point_dict.items():
            self.nom_addPointSet(v, k)

    def nom_addGlobalDV(self, dvName, value, func, childIdx=None):
        # define the input
        self.add_input(dvName, distributed=False, shape=len(value))

        # call the dvgeo object and add this dv
        if childIdx is None:
            self.DVGeo.addGlobalDV(dvName, value, func)
        else:
            self.DVGeo.children[childIdx].addGlobalDV(dvName, value, func)

    def nom_addLocalDV(self, dvName, axis="y", pointSelect=None, childIdx=None):
        if childIdx is None:
            nVal = self.DVGeo.addLocalDV(dvName, axis=axis, pointSelect=pointSelect)
        else:
            nVal = self.DVGeo.children[childIdx].addLocalDV(dvName, axis=axis, pointSelect=pointSelect)
        self.add_input(dvName, distributed=False, shape=nVal)
        return nVal

    def nom_addVSPVariable(self, component, group, parm, **kwargs):

        # actually add the DV to VSP
        self.DVGeo.addVariable(component, group, parm, **kwargs)

        # full name of this DV
        dvName = "%s:%s:%s" % (component, group, parm)

        # get the value
        val = self.DVGeo.DVs[dvName].value.copy()

        # add the input with the correct value, VSP DVs always have a size of 1
        self.add_input(dvName, distributed=False, shape=1, val=val)

    def nom_addThicknessConstraints2D(self, name, leList, teList, nSpan=10, nChord=10):
        self.DVCon.addThicknessConstraints2D(leList, teList, nSpan, nChord, lower=1.0, name=name)
        comm = self.comm
        if comm.rank == 0:
            self.add_output(name, distributed=True, val=np.ones((nSpan * nChord,)), shape=nSpan * nChord)
        else:
            self.add_output(name, distributed=True, shape=(0,))

    def nom_addThicknessConstraints1D(self, name, ptList, nCon, axis):
        self.DVCon.addThicknessConstraints1D(ptList, nCon, axis, name=name)
        comm = self.comm
        if comm.rank == 0:
            self.add_output(name, distributed=True, val=np.ones(nCon), shape=nCon)
        else:
            self.add_output(name, distributed=True, shape=(0))

    def nom_addVolumeConstraint(self, name, leList, teList, nSpan=10, nChord=10):
        self.DVCon.addVolumeConstraint(leList, teList, nSpan=nSpan, nChord=nChord, name=name)
        comm = self.comm
        if comm.rank == 0:
            self.add_output(name, distributed=True, val=1.0)
        else:
            self.add_output(name, distributed=True, shape=0)

    def nom_add_LETEConstraint(self, name, volID, faceID, topID=None, childIdx=None):
        self.DVCon.addLeTeConstraints(volID, faceID, name=name, topID=topID, childIdx=childIdx)
        # how many are there?
        conobj = self.DVCon.linearCon[name]
        nCon = len(conobj.indSetA)
        comm = self.comm
        if comm.rank == 0:
            self.add_output(name, distributed=True, val=np.zeros((nCon,)), shape=nCon)
        else:
            self.add_output(name, distributed=True, shape=0)
        return nCon

    def nom_addLERadiusConstraints(self, name, leList, nSpan, axis, chordDir):
        self.DVCon.addLERadiusConstraints(leList=leList, nSpan=nSpan, axis=axis, chordDir=chordDir, name=name)
        comm = self.comm
        if comm.rank == 0:
            self.add_output(name, distributed=True, val=np.ones(nSpan), shape=nSpan)
        else:
            self.add_output(name, distributed=True, shape=0)

    def nom_addCurvatureConstraint1D(self, name, start, end, nPts, axis, **kwargs):
        self.DVCon.addCurvatureConstraint1D(start=start, end=end, nPts=nPts, axis=axis, name=name, **kwargs)
        comm = self.comm
        if comm.rank == 0:
            self.add_output(name, distributed=True, val=1.0)
        else:
            self.add_output(name, distributed=True, shape=0)

    def nom_addLinearConstraintsShape(self, name, indSetA, indSetB, factorA, factorB, childIdx=None):
        self.DVCon.addLinearConstraintsShape(
            indSetA=indSetA, indSetB=indSetB, factorA=factorA, factorB=factorB, name=name, childIdx=childIdx
        )
        lSize = len(indSetA)
        comm = self.comm
        if comm.rank == 0:
            self.add_output(name, distributed=True, val=np.zeros(lSize), shape=lSize)
        else:
            self.add_output(name, distributed=True, shape=0)

    def nom_addRefAxis(self, childIdx=None, **kwargs):
        # we just pass this through
        if childIdx is None:
            return self.DVGeo.addRefAxis(**kwargs)
        else:
            return self.DVGeo.children[childIdx].addRefAxis(**kwargs)

    def nom_setConstraintSurface(self, surface):
        # constraint needs a triangulated reference surface at initialization
        self.DVCon.setSurface(surface)

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
                        if self.comm.rank == 0:
                            dout = d_outputs[constraintname]
                            jvtmp = np.dot(np.transpose(dcdx), dout)
                        else:
                            jvtmp = 0.0
                        d_inputs[dvname] += jvtmp
                        # OM does the reduction itself
                        # d_inputs[dvname] += self.comm.reduce(jvtmp, op=MPI.SUM, root=0)

            for ptSetName in self.DVGeo.ptSetNames:
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
                        # xdot = self.DVGeo.totalSensitivityTransProd(dout, ptSetName)
                        xdot = self.DVGeo.totalSensitivity(dout, ptSetName)

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
