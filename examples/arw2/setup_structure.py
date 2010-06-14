# This is just a scrip that is execfile'd in the main script to setup the structure. 

geo_surface = pyGeo.pyGeo('plot3d',file_name='arw2_surface.xyz',file_type='ascii',order='f')
geo_surface.doConnectivity('arw2.con')
geo_surface.fitGlobal()
# Now setup the structure
mpiPrint('---------------------------',comm=comm)
mpiPrint('      pyLayout Setup' ,comm=comm)
mpiPrint('---------------------------',comm=comm)
    
wing_box = pyLayout2.Layout(geo_surface,[67,26],scale=.0254,comm=comm,
                            surf_list=[4,6,8,12,10,17,28,30,31,26,29,27])

le_list = array([[15.794,59,12.5],[65.7,59,113.5]])
te_list = array([[24.7,59,12.5],[70.4,59,113.5]])

nrib = 18
nspar = 2
domain = pyLayout2.domain(le_list,te_list,k=2)

rib_space = [4,3,3]
span_space = 3*ones(nrib-1,'intc')
v_space = 3

struct_def1 = pyLayout2.struct_def(
    nrib,nspar,domain=domain,t=1.0,
    rib_space=rib_space,span_space=span_space,v_space=v_space)
    
wing_box.addSection(struct_def1)
structure = wing_box.finalize2()

