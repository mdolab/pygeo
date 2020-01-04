import openmdao.api as om
from .DVGeometry import DVGeometry
from mpi4py import MPI

# class that actually calls the dvgeometry methods
class OM_DVGEOCOMP(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('DVGeo', allow_none=False)
        self.options['distributed'] = True
    def setup(self):
        self.DVGeo = self.options['DVGeo']

    def compute(self, inputs, outputs):
        DVGeo = self.DVGeo

        # inputs are the geometric design variables
        DVGeo.setDesignVars(inputs)
        # print(inputs['twist'])

        # ouputs are the coordinates of the pointsets we have
        for ptSet in DVGeo.points:
            # update this pointset and write it as output
            outputs[ptSet] = DVGeo.update(ptSet).flatten()

    def addPointSet(self, points, ptName, **kwargs):
        # add the points to the dvgeo object
        self.DVGeo.addPointSet(points.reshape(len(points)//3, 3), ptName, **kwargs)

        # add an output to the om component
        self.add_output(ptName, val=points.flatten())
        # print(points.shape)

        # TODO check if we need to flatten the points

    def addGeoDVGlobal(self, dvName, value, **kwargs):
        # define the input
        self.add_input(dvName, shape=value.shape)

        # call the dvgeo object and add this dv
        self.DVGeo.addGeoDVGlobal(dvName, value, **kwargs)


    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        DVGeo = self.DVGeo

        if mode == 'rev':
            print(DVGeo.ptSetNames)
            for ptSetName in DVGeo.ptSetNames:
                dout = d_outputs[ptSetName].reshape(len(d_outputs[ptSetName])//3, 3)
                xdot = DVGeo.totalSensitivityTransProd(dout, ptSetName)

                # TODO
                # reduce the result. ideally we want to do this in the dvgeocomp
                xdotg = {}
                for k in xdot:
                    xdotg[k] = self.comm.allreduce(xdot[k], op=MPI.SUM)

                print("[%d] called jacbec product"%self.comm.rank)
                if 'twist' in d_inputs:
                    print("[%d] twist in d_inputs"%self.comm.rank)
                    d_inputs['twist'] += xdotg['twist']

# a group to contain the des_vars and dvgeocomp
class OM_DVGEO(om.Group):

    def initialize(self):
        # define the geoOpts to take in all the options we need
        # to initialize the DVGeometry type objects
        self.options.declare('geoOpts', allow_none=False)

    def setup(self):
        # add an indepvarcomp to contain the design variables
        des_vars = self.add_subsystem('des_vars', om.IndepVarComp())
        # just add a dummy output
        des_vars.add_output('foo', val=1)

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

    def addGeoDVGlobal(self, dvName, value, **kwargs):
        # we need to add the dv as output to the indepvar comp,
        # input to the dvgeocomp
        # and actually call the dvgeocomp routine to add the design variable

        # first add the output to the indepvarcomp
        self.des_vars.add_output(dvName, shape=value.shape, val=value)

        # call the dvgeocomp to add this dv
        self.dvgeocomp.addGeoDVGlobal(dvName, value, **kwargs)

        # now connect the two
        self.connect('des_vars.%s'%dvName, 'dvgeocomp.%s'%dvName)




