# This is just a scrip that is execfile'd in the main script to setup the structure. 

geo_surface = pyGeo.pyGeo('plot3d',file_name='arw2_surface.xyz',file_type='ascii',order='f')
geo_surface.doConnectivity('arw2.con')
geo_surface.fitGlobal()

# Now setup the structure
mpiPrint('---------------------------',comm=comm)
mpiPrint('      pyLayout Setup' ,comm=comm)
mpiPrint('---------------------------',comm=comm)

# Create Structural Object    
wing_box = pyLayout2.Layout(geo_surface,[67,26],scale=.0254,comm=comm,
                            surf_list=[4,6,8,12,10,17,28,30,31,26,29,27],
                            complex=complex)

# List of points defining the Front and Rear of Spar Box

le_list = array([[10.794,59,12.5],[62.7,57,113.5]])
te_list = array([[31.7,59,12.5],[73.4,57,113.5]])

# Number of Ribs and Spars
nrib = 18
nspar = 4

# Domain for pyLayout
domain = pyLayout2.domain(le_list,te_list,k=2)

# Number of elements on each section
rib_space = [3,3,3,3,3] # nspar + 1
span_space = 3*ones(nrib-1,'intc') # nribs-1
v_space = 3 # scalar

# Blanking for ribs
spar_blank = ones([nspar,nrib-1],'intc')
spar_blank[0] = 0
spar_blank[-1] = 0

# Blanking for spars
#None

# Get the constitutive classes we will use -> This is equal to the
# number of design variables we will have for thickness


# Rib constitutive classes
rib  = wing_box.getConstitutive(0.001,.001,1,0)

# Spar constitutive classe
spar = wing_box.getConstitutive(0.01,.001,1,1)

# Skin constitutvie classes
skin = wing_box.getConstitutive(0.001,.001,1,2)

struct_def1 = pyLayout2.struct_def(
    nrib,nspar,
    domain=domain,
    spar_blank=spar_blank,
    rib_space=rib_space,
    span_space=span_space,
    v_space=v_space,
    rib_stiffness = rib,
    spar_stiffness = spar,
    top_stiffness = skin,
    bot_stiffness = skin

    )
    
wing_box.addSection(struct_def1)
structure = wing_box.finalize2()
mass_func = functions.StructuralMass(structure)
ks_func   = functions.SimpleKSFailure(structure,40,0)
