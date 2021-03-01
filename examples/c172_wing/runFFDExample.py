# EXCERPT 1 #
from pygeo import DVGeometry
import numpy as np
import matplotlib.pyplot as plt
from stl import mesh

def create_fresh_dvgeo():
    # The Plot3D file ffdbox.xyz contains the coordinates of the free-form deformation (FFD) volume
    # The "i" direction of the cube consists of 10 points along the x (streamwise) axis
    # The "j" direction of the cube is 2 points up and down (y axis direction)
    # The "k" direction of the cube is 8 along the span (z axis direction)
    FFDfile = "ffdbox.xyz"

    # initialize the DVGeometry object with the FFD file
    DVGeo = DVGeometry(FFDfile)


    stlmesh = mesh.Mesh.from_file('baseline_wing.stl')
    # create a pointset. pointsets are of shape npts by 3 (the second dim is xyz coordinate)
    # already have the wing mesh as a triangulated surface (stl file)
    DVGeo.addPointSet(stlmesh.v0, 'mesh_v0')
    DVGeo.addPointSet(stlmesh.v1, 'mesh_v1')
    DVGeo.addPointSet(stlmesh.v2, 'mesh_v2')
    return DVGeo, stlmesh
    # EXCERPT 1 # 

# EXCERPT 3 #
# Now that we have pointsets added, we should parameterize the geometry.

# Adding local geometric design to make local modifications to FFD box
# This option will perturb all the control points but only the y (up-down) direction
DVGeo, stlmesh = create_fresh_dvgeo()
DVGeo.addGeoDVLocal("shape", lower=-0.5, upper=0.5, axis="y", scale=1.0 )
# EXCERPT 3 #

dvdict = DVGeo.getValues()
dvdict['shape'][DVGeo.getLocalIndex(0)[:,1,5]] += 0.15
dvdict['shape'][DVGeo.getLocalIndex(0)[3,1,1]] += 0.15
DVGeo.setDesignVars(dvdict)

stlmesh.vectors[:,0,:] = DVGeo.update('mesh_v0')
stlmesh.vectors[:,1,:] = DVGeo.update('mesh_v1')
stlmesh.vectors[:,2,:] = DVGeo.update('mesh_v2')
stlmesh.save('local_wing.stl')
DVGeo.writeTecplot('local_ffd.dat')

DVGeo, stlmesh = create_fresh_dvgeo()
# add a reference axis to the FFD volume 
# it will go in the spanwise (k) direction and be located at the quarter chord line
nrefaxpts = DVGeo.addRefAxis('c4', xFraction=0.25, alignIndex='k')
nspanwise = 8
print('Num ref axis pts: ', str(nrefaxpts), ' Num spanwise FFD: ', str(nspanwise))

def twist(val, geo):
    for i in range(nrefaxpts):
        geo.rot_z['c4'].coef[i] = val[i]

def sweep(val, geo):
    C = geo.extractCoef('c4')
    C_orig = C.copy()
    sweep_ref_pt = C_orig[0,:]

    theta = -val[0]*np.pi/180
    rot_mtx = np.array([[np.cos(theta), 0., -np.sin(theta)],
                        [0.,            1., 0.            ],
                        [np.sin(theta), 0., np.cos(theta) ]])
    for i in range(nrefaxpts):
        vec = C[i,:] - sweep_ref_pt
        # need to now rotate this by the sweep angle and add it back
        C[i,:] = sweep_ref_pt + rot_mtx @ vec
    geo.restoreCoef(C, 'c4')


DVGeo.addGeoDVGlobal('twist', func=twist, value=np.zeros(nrefaxpts),
            lower=-10, upper=10, scale=0.05)

DVGeo.addGeoDVGlobal('sweep', func=sweep, value=0.,
            lower=0, upper=45, scale=0.05)

DVGeo.writeRefAxes('local')

dvdict = DVGeo.getValues()
dvdict['twist'] = np.linspace(-10., 20., nrefaxpts)
DVGeo.setDesignVars(dvdict)
stlmesh.vectors[:,0,:] = DVGeo.update('mesh_v0')
stlmesh.vectors[:,1,:] = DVGeo.update('mesh_v1')
stlmesh.vectors[:,2,:] = DVGeo.update('mesh_v2')
stlmesh.save('twist_wing.stl')
DVGeo.writeTecplot('twist_ffd.dat')

dvdict = DVGeo.getValues()
dvdict['sweep'] = 30.
dvdict['twist'] = np.linspace(0., 20., nrefaxpts)
DVGeo.setDesignVars(dvdict)

stlmesh.vectors[:,0,:] = DVGeo.update('mesh_v0')
stlmesh.vectors[:,1,:] = DVGeo.update('mesh_v1')
stlmesh.vectors[:,2,:] = DVGeo.update('mesh_v2')
stlmesh.save('sweep_wing.stl')
DVGeo.writeTecplot('sweep_ffd.dat')
DVGeo.writeRefAxes('sweep')

def chord(val, geo):
    for i in range(nrefaxpts):
        geo.scale_x['c4'].coef[i] = val[i]

DVGeo, stlmesh = create_fresh_dvgeo()
nrefaxpts = DVGeo.addRefAxis('c4', xFraction=0.25, alignIndex='k')
DVGeo.addGeoDVGlobal('twist', func=twist, value=np.zeros(nrefaxpts),
            lower=-10, upper=10, scale=0.05)
DVGeo.addGeoDVGlobal('chord', func=chord, value=np.ones(nrefaxpts),
            lower=0.01, upper=2.0, scale=0.05)
DVGeo.addGeoDVGlobal('sweep', func=sweep, value=0.,
            lower=0, upper=45, scale=0.05)
DVGeo.addGeoDVLocal('thickness', axis='y', lower=-0.5, upper=0.5)

dvdict = DVGeo.getValues()
dvdict['twist'] = np.linspace(0., 20., nrefaxpts)
dvdict['chord'] = np.linspace(1.2, 0.2, nrefaxpts)
dvdict['thickness'] = np.random.uniform(-0.1, 0.1, 160)
dvdict['sweep'] = 30.
DVGeo.setDesignVars(dvdict)

stlmesh.vectors[:,0,:] = DVGeo.update('mesh_v0')
stlmesh.vectors[:,1,:] = DVGeo.update('mesh_v1')
stlmesh.vectors[:,2,:] = DVGeo.update('mesh_v2')
stlmesh.save('all_wing.stl')
DVGeo.writeTecplot('all_ffd.dat')
DVGeo.writeRefAxes('all')






# # EXCERPT 6 #
# # Now let's deform the geometry.
# # We want to set the front and rear control points the same so we preserve symmetry along the z axis
# # and we ues the getLocalIndex function to accomplish this
# lower_front_idx = DVGeo.getLocalIndex(0)[:,0,0]
# lower_rear_idx = DVGeo.getLocalIndex(0)[:,0,1]
# upper_front_idx = DVGeo.getLocalIndex(0)[:,1,0]
# upper_rear_idx = DVGeo.getLocalIndex(0)[:,1,1]

# currentDV = DVGeo.getValues()['shape']
# newDV = currentDV.copy()

# # add a constant offset (upward) to the lower points, plus a linear ramp and a trigonometric local change
# # this will shrink the cylinder height-wise and make it wavy
# # set the front and back points the same to keep the cylindrical sections square along that axis
# for idx in [lower_front_idx, lower_rear_idx]:
#     const_offset = 0.3 * np.ones(10)
#     local_perturb = np.cos(np.linspace(0,4*np.pi,10))/10 + np.linspace(-0.05,0.05,10)
#     newDV[idx] = const_offset + local_perturb

# # add a constant offset (downward) to the upper points, plus a linear ramp and a trigonometric local change
# # this will shrink the cylinder height-wise and make it wavy
# for idx in [upper_front_idx, upper_rear_idx]:
#     const_offset = -0.3 * np.ones(10)
#     local_perturb = np.sin(np.linspace(0,4*np.pi,10))/20 + np.linspace(0.05,-0.10,10)
#     newDV[idx] = const_offset + local_perturb

# # we've created an array with design variable perturbations. Now set the FFD control points with them
# # and update the point sets so we can see how they changed
# DVGeo.setDesignVars({'shape': newDV.copy()})

# Xmod = DVGeo.update('cylinder')
# FFDmod = DVGeo.update('ffd')
# # EXCERPT 6 #

# # EXCERPT 7 #
# # cast the 3D pointsets to 2D for plotting (ignoring depth)
# FFDplt = FFDptset[:,:2]
# FFDmodplt = FFDmod[:,:2]
# Xptplt = Xpt[:, :2]
# Xmodplt = Xmod[:, :2]

# # plot the new and deformed pointsets and control points
# plt.figure()
# plt.title('Applying FFD deformations to a cylinder')

# plt.plot(Xptplt[:,0], Xptplt[:,1], color='#293bff')
# plt.plot(FFDplt[:,0], FFDplt[:,1], color='#d6daff', marker='o')

# plt.plot(Xmodplt[:,0], Xmodplt[:,1], color='#ff0000')
# plt.plot(FFDmodplt[:,0], FFDmodplt[:,1], color='#ffabab', marker = 'o')

# plt.xlabel('x')
# plt.ylabel('y')
# # plt.xlim([-0.7,1.2])
# plt.axis('equal')
# legend = plt.legend(['original shape','original FFD ctl pts','deformed shape','deformed FFD ctl pts'], loc='lower right', framealpha=0.0)
# legend.get_frame().set_facecolor('none')
# plt.tight_layout()
# plt.savefig('deformed_cylinder.png')
# # EXCERPT 7 #
