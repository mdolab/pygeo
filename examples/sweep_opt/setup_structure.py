# This is just a scrip that is execfile'd in the main script to setup the structure. 
geo_surface = pyGeo.pyGeo('iges',file_name='sweep.igs')
geo_surface.doConnectivity('sweep.con')
#geo_surface.writeTecplot('geo.dat',edge_labels=True)

# This surface needs to be scaled by 2.33
geo_surface.coef*=2.33
geo_surface._updateSurfaceCoef()

# Now setup the structure
mpiPrint('---------------------------',comm=comm)
mpiPrint('      pyLayout Setup' ,comm=comm)
mpiPrint('---------------------------',comm=comm)

# Create Structural Object    
wing_box = pyLayout2.Layout(geo_surface,[2,7],comm=comm,complex=complex)

# List of points defining the Front and Rear of Spar Box

le_list = array([[.10,0,0],[.10,0,4.99]])*2.33
te_list = array([[.85,0,0],[.85,0,4.99]])*2.33

# Number of Ribs and Spars
nrib = 25
nspar = 4

# Domain for pyLayou

domain = pyLayout2.domain(le_list,te_list,k=2)

# Number of elements on each section
rib_space = [3,3,3,3,3] # nspar + 1
span_space = 3*ones(nrib-1,'intc') # nribs-1
v_space = 3 # scalar

# Blanking for ribs
spar_blank = ones([nspar,nrib-1],'intc')
spar_blank[1] = 0
spar_blank[-1] = 0

# Blanking for spars
#None

# Get the constitutive classes we will use -> This is equal to the
# number of design variables we will have for thickness

# Rib constitutive classes
rib  = wing_box.getConstitutive(0.002,yieldstress=87.6e6)

# Spar constitutive classe
spar = wing_box.getConstitutive(0.1,dv=0,t_min=.001,t_max=1,yieldstress=87.6e6)

# Skin constitutvie classes
skin = wing_box.getConstitutive(0.02,dv=1,t_min=.001,t_max=1,yieldstress=87.6e6)

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
ks_func   = functions.SimpleKSFailure(structure,40.0,int(0))
