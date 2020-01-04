import openmdao.api as om
from .DVGeometry import DVGeometry
from mpi4py import MPI

# class that actually calls the dvgeometry methods
class OM_DVGEOCOMP(om.ExplicitComponent):

    def initialize(self):

        self.options.declare('DVGeo', allow_none=False)
        self.options['distributed'] = True

    def setup(self):

        # set the DVGeo object that does the computations
        self.DVGeo = self.options['DVGeo']

    def compute(self, inputs, outputs):

        # inputs are the geometric design variables
        self.DVGeo.setDesignVars(inputs)

        # ouputs are the coordinates of the pointsets we have
        for ptSet in self.DVGeo.points:
            # update this pointset and write it as output
            outputs[ptSet] = self.DVGeo.update(ptSet).flatten()

    def addPointSet(self, points, ptName, **kwargs):
        # add the points to the dvgeo object
        self.DVGeo.addPointSet(points.reshape(len(points)//3, 3), ptName, **kwargs)

        # add an output to the om component
        self.add_output(ptName, val=points.flatten())

    def addGeoDVGlobal(self, dvName, value, func):
        # define the input
        self.add_input(dvName, shape=value.shape)

        # call the dvgeo object and add this dv
        self.DVGeo.addGeoDVGlobal(dvName, value, func)

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):

        # only do the computations when we have more than zero entries in d_inputs in the reverse mode
        ni = 0
        for k in d_inputs:
            ni+=1

        if mode == 'rev' and ni > 0:
            for ptSetName in self.DVGeo.ptSetNames:
                dout = d_outputs[ptSetName].reshape(len(d_outputs[ptSetName])//3, 3)
                xdot = self.DVGeo.totalSensitivityTransProd(dout, ptSetName)

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
                        d_inputs[k] += xdotg[k]

class OM_DVGEO(om.Group):
    # a group to contain the des_vars and dvgeocomp

    def initialize(self):
        # define the geoOpts to take in all the options we need
        # to initialize the DVGeometry type objects
        self.options.declare('geoOpts', allow_none=False)

    def setup(self):

        # add an indepvarcomp to contain the design variables
        des_vars = self.add_subsystem('des_vars', om.IndepVarComp())

        # get options and make all lowercase
        geoOpts = self.options['geoOpts']
        opts = {}
        for k,v in geoOpts.items():
            opts[k.lower()] = v

        # check what version of dvgeo we are working with
        if 'ffdfile' in opts:
            # this is a regular dvgeo using a single ffd
            self.DVGeo = DVGeometry(opts['ffdfile'])

        # finally add the dvgeo component
        self.add_subsystem('dvgeocomp', OM_DVGEOCOMP(DVGeo=self.DVGeo))

    def addPointSet(self, points, ptName, **kwargs):
        # just pass this through
        self.dvgeocomp.addPointSet(points, ptName, **kwargs)

    def addRefAxis(self, **kwargs):
        # we just pass this through
        return self.DVGeo.addRefAxis(**kwargs)

    def addGeoDVGlobal(self, dvName, value, func, active_dv=True, **kwargs):

        # first add the output to the indepvarcomp
        self.des_vars.add_output(dvName, shape=value.shape, val=value)

        # call the dvgeocomp to add this dv
        self.dvgeocomp.addGeoDVGlobal(dvName, value, func)

        # now connect the two
        self.connect('des_vars.%s'%dvName, 'dvgeocomp.%s'%dvName)

        if active_dv:
            self.add_design_var('des_vars.%s'%dvName, **kwargs)

    def addRefAxis(self, name, **kwargs):
        return self.DVGeo.addRefAxis(name, **kwargs)




