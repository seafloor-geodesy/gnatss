import numba
import math

@numba.njit
def _compute_hm(dd, sv, start_depth, end_depth, start_index):
    """
    Computes harmonic mean.
    It's a direct translation from the original Fortran code found in
    src/cal_sv_harmonic_mean/get_sv_harmonic_mean.F called
    subroutine `sv_harmon_mean`
    """
    # TODO: Find a way to vectorize this computation
    zs = start_depth
    ze = end_depth
    
    z1=dd[start_index]
    z2=dd[start_index+1]

    c_z1 = sv[start_index]
    c_z2 = sv[start_index+1]

    zi = zs
    if(z2 >= ze):
        zf = ze
    else:
        zf = z2
        
    cumsum = 0.0
    
    for i in range(start_index+1, len(dd)):
        b =  ( c_z2 - c_z1) / ( z2 - z1 )
        wi = zi - z1 + c_z1/b
        wf = zf - z1 + c_z1/b

        wi = math.log( (zi - z1)*b + c_z1)/b
        wf = math.log( (zf - z1)*b + c_z1)/b

        delta = (wf-wi)
        cumsum = cumsum + delta
        z1=zf          
        z2=dd[i+1]
        c_z1 = c_z2
        c_z2 = sv[i+1]
        zi=zf

        if ze > zi and ze < z2:
            zf = ze
        else:
            zf = z2

        if z1 >= ze: break
    
    return (ze-zs)/cumsum

def sv_harmonic_mean(svdf, start_depth, end_depth):
    """
    Computes harmonic mean from a sound profile
    containing depth (dd) and sound speed (sv)
    
    Parameters
    ----------
    svdf : pd.DataFrame
        Sound speed profile data as dataframe
    start_depth : int or float
        The start depth for harmonic mean to be computed
    end_depth : int or float
        The end depth for harmonic mean to be computed
        
    Returns
    -------
    float
        The sound speed harmonic mean value
    """
    abs_start = abs(start_depth)
    abs_end = abs(end_depth)
    abs_sv = abs(svdf)
    # Get the index for the start of depth closest to specified start depth
    start_index = abs_sv[(abs_sv['dd'].round() >= start_depth)].index[0]

    return _compute_hm(abs_sv['dd'].values, abs_sv['sv'].values, abs_start, abs_end, start_index)
