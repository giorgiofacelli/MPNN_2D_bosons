import jax
from jax import numpy as jnp
from distances import dist_min_image
from functools import partial


"""
//////////////////////////////////////////////////////////////////////
Function for computation of radial correlation function.
//////////////////////////////////////////////////////////////////////
"""

def corr_2d(x, L, rMax, dr):

    ''' Computes the radial correlation function given a set of positions.

        Args:
        - x:     positions of particles. Should be (nparticles, sdim)
        - L:     length of each side of the square region of the plane
        - rMax:  outer diameter of largest annulus
        - dr:    increment for increasing radius of annulus

        Returns:
        - g(r):  an array containing the correlation function g(r)
        - radii: an array containing the radii of the annuli used to compute g(r)
    '''

    N, sdim = x.shape

    Rs = jnp.arange(0., rMax + dr, dr)
    #num_increments = len(Rs) - 1
    #g = jnp.zeros(num_increments)
    rho = x.shape[0] / L**2

    Ds = dist_min_image(x.reshape(-1), L, sdim, norm=True)

    #need to recount distance betwen i and j when considering i and when considering j    
    #total_Ds = jnp.concatenate(Ds, Ds)

    #count distances in intervals given by Rs
    result,_ = jnp.histogram(Ds, Rs, density = False)
    g = result / (rho*len(Ds)/N) 

    #Compute radii
    
    r_inner = jnp.insert(Rs, 0, 0)
    r_outer = jnp.append(Rs, 0)
    radii = 0.5*(r_outer+r_inner)[1:-1]

    g_norm = g / (jnp.pi * (2*dr*radii + dr**2))

    return (g_norm, radii)


corr_2d_mapped = jax.vmap(corr_2d, in_axes=(0, None, None, None), out_axes = (0,None))


"""
//////////////////////////////////////////////////////////////////////
Function for computation of radial correlation function.
//////////////////////////////////////////////////////////////////////
"""


def corr_2d_xy(x, L, xedges, yedges):
    
    N, sdim = x.shape
    rho = x.shape[0] / (L[0]*L[1])

    Ds = dist_min_image(x, L, 2, False) 
    
    result,_,_ = jnp.histogram2d(Ds[:,0], Ds[:,1], bins = [xedges, yedges], density = False)

    g = result / (rho*Ds.shape[0]/N)

    inner = jnp.insert(xedges, 0, 0)
    outer = jnp.append(xedges, 0)
    xs = 0.5*(outer+inner)[1:-1]

    inner = jnp.insert(yedges, 0, 0)
    outer = jnp.append(yedges, 0)
    ys = 0.5*(outer+inner)[1:-1]

    return g, xs, ys
    
corr_2d_xy_mapped = jax.vmap(corr_2d_xy, in_axes=(0, None, None, None), out_axes = (0,None, None))


"""
//////////////////////////////////////////////////////////////////////
Function for computation of moving average & std.
//////////////////////////////////////////////////////////////////////
"""

def moving_stats(vec, window_size):

    window_size = window_size
    mov_avg = []
    mov_std = []
    for i in range(len(vec) - window_size + 1):
        
        # Store elements from i to i+window_size
        # in list to get the current window
        window = vec[i : i + window_size]
    
        # Calculate the average of current window
        window_average = jnp.mean(window)
        window_std = jnp.std(window)
        
        # Store the average of current
        # window in moving average list
        mov_avg.append(window_average)
        mov_std.append(window_std)

    mov_avg = jnp.array(mov_avg)
    mov_std = jnp.array(mov_std)

    return mov_avg, mov_std



"""
//////////////////////////////////////////////////////////////////////
Function for computation of structure factor.
//////////////////////////////////////////////////////////////////////
"""

@partial(jax.jit)
def kernel_sfactor(vec, coord):

    s = jnp.array([jnp.exp(-1j*jnp.dot(v,coord)) for v in vec])
    print(s.shape)
    return jnp.abs(jnp.sum(s))**2


def structure_factor(vec, L, n_max):
        
    ''' Computes the structure factor given a set of positions.

    Args:
    - vec:     2D positions of particles. Should be (nparticles, 2)
    - L:     length of each side of the square region of the plane. Should be (L_x, L_y)
    - n_max: Maximal integer for the wavevector

    Returns:
    - s_factors:  structure factor for each point in the wavevector meshgrid
    - coords: coords of the wavevector meshgrid
    '''

    vec = jnp.array(vec)
    ns = jnp.arange(-n_max,n_max+1, 1)
    nx, ny = jnp.meshgrid(ns,ns)
    kx, ky = 2.*jnp.pi*nx/L[0], 2.*jnp.pi*ny/L[1]

    coords = jnp.append(kx.reshape(1,-1), ky.reshape(1,-1), axis=0)
    s_factors = jnp.zeros(shape=(coords.shape[1],))

    def body_scan(carry, i):

        vec, coords, s_factors = carry
        snew = kernel_sfactor(vec, coords[:,i])
        s_factors = s_factors.at[i].set(snew)

        return (vec, coords, s_factors), i 
    
    (vec, coords, s_factors), _ = jax.lax.scan(body_scan, (vec, coords, s_factors), jnp.arange(coords.shape[1]))


    return s_factors, coords

structure_factor_mapped = jax.vmap(structure_factor, in_axes = (0, None, None, None), out_axes = (0,None))
            
