import jax
from jax import numpy as jnp



def dist_min_image(x, L, sdim, norm = False):

    ''' computes distances following minimum image convention
        
        Args:
        - x: coords on which to compute distance
        - L: size of the system
        - sdim: spatial dimension
        - norm: boolean to output norm or not
        Returns:
        distances / norm of the distances
    '''
    
    n_particles = x.shape[0]//sdim
    x = x.reshape(-1, sdim)

    distances = (-x[jnp.newaxis, :, :] + x[:, jnp.newaxis, :])[
    jnp.triu_indices(n_particles, 1)]

    distances = jnp.remainder(distances + L / 2.0, L) - L / 2.0
    if norm:
        return  jnp.linalg.norm(distances, axis=-1)
    else:
        return distances



def make_vec_periodic(vec, L):
    
    ''' makes a vector periodic

        Args:
        - vec: vector to be made periodic
        - L: size of the system
    '''
    return jnp.concatenate((jnp.sin(vec*2.0*jnp.pi/L), jnp.cos(vec*2.0*jnp.pi/L)), axis=-1)


def distance_matrix(x, L, periodic = True):

    ''' computes distances, optionally returning the trigonometric version
        
        Args:
        - x: coords on which to compute distance
        - L: size of the system
        - periodic: boolean to output periodic distances or not
    '''

    rij = x[..., :, jnp.newaxis, :] - x[..., jnp.newaxis, :, :]

    if periodic:
        return make_vec_periodic(rij, L)
    else:
        return rij