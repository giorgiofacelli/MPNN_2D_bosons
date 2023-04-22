import jax
import flax.linen as nn
import jax.numpy as jnp
from typing import Tuple, Callable
import numpy as np
from netket.utils.types import NNInitFunc, DType
from jax.nn.initializers import (
    zeros,
    ones,
    lecun_normal,
    normal
)
from distances import distance_matrix, dist_min_image




class Phi(nn.Module):

    """
        Message-passing layer. A single feed-forward neural network
    """

    output_dim: int
    widths: Tuple #= (16,)
    hidden_lyrs: int #= 1
    initializer: NNInitFunc = lecun_normal()
    activation: Callable = nn.activation.gelu
    out_lyr_activation: Callable = None

    @nn.compact
    def __call__(self, x):
        #(n_samples, N, d)
        #in_dim = x.shape[-1]

        #APPLY HIDDEN LAYERS

        for i in range(self.hidden_lyrs):

            x = nn.Dense(features = self.widths[i], kernel_init = self.initializer, param_dtype=np.float64)(x)
            x = self.activation(x)

        #APPLY LAST LAYER WITH OUTPUT DIMENSION REQUIRED

        x = nn.Dense(features = self.output_dim, kernel_init = self.initializer, param_dtype=np.float64)(x)
        
        #APPLY ACTIVATION CONDITIONALLY ON out_lyr_activation
        if self.out_lyr_activation is not None:
            x = self.out_activation(x)

        return x
    




class MPNN(nn.Module):

    """
        Class for coordinate transformations with Message-Passing Neural Network
    """

    L: float
    graph_number: int
    phi_out_dim: int
    initializer: NNInitFunc = lecun_normal()
    activation: Callable = nn.activation.gelu
    phi_hidden_lyrs: int = 1
    phi_widths: Tuple = (5,)

    @nn.compact
    def __call__(self, ri):
        
        assert len(ri.shape) == 3 
        
        N_samples, N, sdim = ri.shape

        #creation of hidden nodes and edges
        hi = self.param("hidden_state_nodes", self.initializer, (1, 1, self.phi_widths[0]), np.float64)
        hij = self.param("hidden_state_edges", self.initializer, (1, 1, 1, self.phi_widths[0]), np.float64)
        hi = jnp.tile(hi, (N_samples,N,1))
        hij = jnp.tile(hij, (N_samples,N,N,1))

        #Euclidean distance between vectors
        dist = distance_matrix(ri, self.L, periodic = False)

        #Periodic distance between vectors
        rij = distance_matrix(ri, self.L,  periodic = True)

        ri = jnp.concatenate((jnp.sin(ri*2.0*jnp.pi/self.L), jnp.cos(ri*2.0*jnp.pi/self.L)), axis=-1)

        normij = jnp.linalg.norm(jnp.sin(jnp.pi*dist/self.L) + jnp.eye(N)[..., None], axis=-1, keepdims=True)**2 * (
                    1. - jnp.eye(N)[..., None]) #NORM  OF THE TRANSFORMED DISTANCE VECTORS
         
        xi = jnp.concatenate((ri, hi), axis = -1)
        xij = jnp.concatenate((rij, normij, hij), axis = -1)

        
        for i in range(self.graph_number):
            
            phi = Phi(output_dim = self.phi_out_dim, widths = self.phi_widths, hidden_lyrs = self.phi_hidden_lyrs)
            f = Phi(output_dim = self.phi_out_dim, widths = self.phi_widths, hidden_lyrs = self.phi_hidden_lyrs)
            g = Phi(output_dim = self.phi_out_dim, widths = self.phi_widths, hidden_lyrs = self.phi_hidden_lyrs)
            nuij = phi(xij)

            if i != self.graph_number-1:
                xij = jnp.concatenate((rij, normij, f(jnp.concatenate((xij, nuij), axis=-1))), axis=-1)
  
            xi = jnp.concatenate((ri, g(jnp.concatenate((xi, jnp.sum(nuij, axis=-2)), axis=-1))), axis=-1)


        return xi
    





class logpsi(nn.Module):

    """
        Brings together MPNN and a simple feed-forward NN to model \ln(\psi)
    """

    L: float
    sdim: int
    graph_number: int
    phi_out_dim: int
    initializer: NNInitFunc = lecun_normal()
    activation: Callable = nn.activation.gelu
    phi_hidden_lyrs: int = 1
    phi_widths: Tuple = (5,)

    rho_hidden_lyrs: int = 1
    rho_widths: int = (5,)

    @nn.compact
    def __call__(self, x):

        N = x.shape[-1] // self.sdim

        x = x.reshape(-1, N, self.sdim)

        mpnn = MPNN(self.L, self.graph_number, self.phi_out_dim, self.initializer, self.activation, self.phi_hidden_lyrs, self.phi_widths)
        
        #transform coords x into MPNN-kind coords
        x = mpnn(x)

        for i in range(self.rho_hidden_lyrs):
            
            x = nn.Dense(features = self.rho_widths[i], kernel_init = self.initializer, param_dtype = np.float64)(x)
            x = self.activation(x)

        x = nn.Dense(features = 1, kernel_init = self.initializer, param_dtype = np.float64)(x)
        x = jnp.sum(x, axis = -2)

        return x.reshape(-1)