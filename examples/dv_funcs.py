# --------------------------------------
# Define Design Variable functions here:
# --------------------------------------


from numpy import sin, cos, linspace, pi, zeros, where, hstack, mat, array, \
    transpose, vstack, max, dot, sqrt, append, mod, ones, interp, meshgrid, \
    real, imag, dstack, floor, size, reshape,tan

def span_extension(val,ref_axis):
    '''Single design variable for span extension'''
    ref_axis[0].x[:,2] = ref_axis[0].x0[:,2] * val
    return ref_axis

def twist(val,ref_axis):
    '''Twist'''
    ref_axis[0].rot[:,2] = ref_axis[0].rot0[:,2] + ref_axis[0].s*val
    ref_axis[1].rot[:,2] = ref_axis[0].rot[-1,2]
    return ref_axis

def sweep(val,ref_axis):
    '''Sweep the wing'''
    # Interpret the val as an ANGLE
    angle = val*pi/180
    dz = ref_axis[0].x[-1,2] - ref_axis[0].x[0,2]
    dx = dz*tan(angle)
    ref_axis[0].x[:,0] =  ref_axis[0].x0[:,0] +  dx * ref_axis[0].s

    dz = ref_axis[1].x[-1,2] - ref_axis[1].x[0,2]
    dx = dz*tan(angle)
    ref_axis[1].x[:,0] =  ref_axis[1].x0[:,0] +  dx * ref_axis[1].s

    return ref_axis

def flap(val,ref_axis):
#    print 'flap:',val
    ref_axis[2].rot[:,2] = val

    return ref_axis

