# This is just a scrip that is execfile'd in the main script to setup the structure. 

geo_surface = pyGeo.pyGeo('plot3d',file_name='arw2_surface.xyz',file_type='ascii',order='f')
geo_surface.doConnectivity('arw2.con')
geo_surface.fitGlobal()
# Now setup the structure
mpiPrint('---------------------------',comm=comm)
mpiPrint('      pyLayout Setup' ,comm=comm)
mpiPrint('---------------------------',comm=comm)
    
wing_box = pyLayout2.Layout(geo_surface,[67,26],scale=.0254,comm=comm,
                            surf_list=[4,6,8,12,10,17,28,30,31,26,29,27],
                            complex=complex)

le_list = array([[15.794,59,12.5],[65.7,59,113.5]])
te_list = array([[24.7,59,12.5],[70.4,59,113.5]])

le_list = array([[10.794,59,12.5],[62.7,57,113.5]])
te_list = array([[31.7,59,12.5],[73.4,57,113.5]])


nrib = 4
nspar = 2
domain = pyLayout2.domain(le_list,te_list,k=2)

# rib_space = [2,2,2,2,2]
# span_space = 3*ones(nrib-1,'intc')
# v_space = 3

rib_space = [1,1,1]#,1,1]
span_space = 1*ones(nrib-1,'intc')
v_space = 1

spar_blank = ones([nspar,nrib-1],'intc')
spar_blank[0] = 0
spar_blank[-1] = 0

struct_def1 = pyLayout2.struct_def(
    nrib,nspar,domain=domain,t=1.0,spar_blank=spar_blank,
    rib_space=rib_space,span_space=span_space,v_space=v_space)
    
wing_box.addSection(struct_def1)
structure = wing_box.finalize2()

# temp = structure.createVec()
# temp.set(1.0)
# temp.applyBCs()
# structure.writeTecplotFile(temp,'bc_check')

