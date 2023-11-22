import jax
from jax import numpy as jnp
from flax import linen as nn
from typing import Tuple, Callable
import numpy as np
from netket.utils.types import NNInitFunc, DType
from jax.nn.initializers import (
    zeros,
    ones,
    lecun_normal,
    normal
)

from distances import dist_min_image


def transf(x, sdim, L):
    '''
    function to transform coords in the naturally periodic coords of sin and cos
    '''
    x = x.reshape(x.shape[0],-1,sdim)
    new = jnp.concatenate([jnp.sin(2*jnp.pi*x/L), jnp.cos(2*jnp.pi*x/L)], axis=-1)

    return new



class deepset(nn.Module):
    
    layers_phi: int = 2                         #layers of 1st NN
    layers_rho: int  = 2                        #layers of 2nd NN
    width_phi: Tuple = (16,16)                  #widths of 1st NN
    width_rho: Tuple  = (16,1)                  #widths of 2nd NN
    sdim: int = 2                               #spatial dimension
    L: int = 10                                 #length of the box
    initfunc: NNInitFunc = lecun_normal()       #intialization func
    activation: Callable = nn.activation.gelu   #activation

    @nn.compact
    def __call__(self, x):
        M = x.shape[0]
        #apply transformation of coords
        x = transf(x, self.sdim, self.L)
        
        #1st NN
        for i in range(self.layers_phi):
            x = nn.Dense(features=self.width_phi[i], kernel_init=self.initfunc, param_dtype=np.float64)(x)
            if i == self.layers_phi-1:
                break
            x = self.activation(x)
        
        #sum over the particles
        x = jnp.log(jnp.sum(jnp.exp(x), axis=-2))
        
        #2nd layer
        for i in range(self.layers_rho):
            x = nn.Dense(features=self.width_rho[i], kernel_init=self.initfunc, param_dtype=np.float64)(x)
            if i == self.layers_rho-1:
                break
            x = self.activation(x)
        
        #if x.shape[0] == 1:
        #    return x.reshape(-1)[0]
        #
        #return x.reshape(-1)
        return x.reshape(M)
