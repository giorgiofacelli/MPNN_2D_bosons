import jax
import flax.linen as nn
import jax.numpy as jnp
import numpy as np
import netket as nk
import numpy as np
from distances import distance_matrix, dist_min_image
from MPNN_model import logpsi

#define potential energy
def V(x, sdim, L, eps, sigma):
    #vector of distances

    norm = dist_min_image(x, L, sdim, norm = True)
    #distances = jnp.array([[jnp.linalg.norm(jnp.sin(PI*(x_i-x_j)/L)) for x_i in x] for x_j in x])
 
    #terms to be summed in the potential
    arg = jnp.exp(-norm**2/(2*sigma**2))
                                                                        
    pot = eps*jnp.sum(arg)

    return pot

#track acceptance rate of MCMC
def mycb(step, logged_data, driver):
    logged_data["acceptance"] = float(driver.state.sampler_state.acceptance)
    return True


L = 20            #system size
nparticles = 20   #number of particles
sigma = 2**-.5    #metrpolis hastings sampling std
sdim = 1          #spatial dimension
eps = 2.          #potential coefficient 

v = lambda x: V(x,sdim,L,eps,sigma)

hilb = nk.hilbert.Particle(N=nparticles, L=(L,), pbc=True)

sab = nk.sampler.MetropolisGaussian(hilb, sigma=0.15, n_chains=16, n_sweeps=32)

ekin = nk.operator.KineticEnergy(hilb, mass=1.0)
pot = nk.operator.PotentialEnergy(hilb, v)
ha = ekin + pot

model = logpsi(L= L, sdim = sdim, graph_number = 1, phi_out_dim = 10, phi_hidden_lyrs = 1, phi_widths=(10,), rho_widths=(10,), rho_hidden_lyrs=1)

vs = nk.vqs.MCState(sab, model, n_samples=10**4, n_discard_per_chain=32)
op = nk.optimizer.Sgd(0.01)
sr = nk.optimizer.SR(diag_shift=0.005)

gs = nk.VMC(ha, op, sab, variational_state=vs, preconditioner=sr)
gs.run(n_iter=500, callback=mycb, out="int_bosons_1d_N=20_L=20")
