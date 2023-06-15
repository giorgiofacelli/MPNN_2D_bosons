import jax
import flax.linen as nn
import jax.numpy as jnp
import numpy as np
import netket as nk
import numpy as np
from distances import distance_matrix, dist_min_image
from MPNN_model import logpsi
from netket.utils import mpi

def V(x, sdim, L, eps, sigma):
    #vector of distances

    norm = dist_min_image(x, L, sdim, norm = True)
    #distances = jnp.array([[jnp.linalg.norm(jnp.sin(PI*(x_i-x_j)/L)) for x_i in x] for x_j in x])
 
    #terms to be summed in the potential
    arg = jnp.exp(-norm**2/(2*sigma**2))
                                                                        
    pot = eps*jnp.sum(arg)

    return pot


def mycb(step, logged_data, driver):
    logged_data["acceptance"] = float(driver.state.sampler_state.acceptance)
    return True

l = 12
A = [1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4]
nparticles = 16
sigma = 1.
sdim = 2
eps = 1.
mass = 30

for a in A:    

    L = (l * a**-0.5, l * a**0.5)
    v = lambda x: V(x,sdim,jnp.array(L),eps,sigma)

    hilb = nk.hilbert.Particle(N=nparticles, L=L, pbc=True)

    sab = nk.sampler.MetropolisGaussian(hilb, sigma=0.1, n_chains=16, n_sweeps=32)

    ekin = nk.operator.KineticEnergy(hilb, mass=mass)
    pot = nk.operator.PotentialEnergy(hilb, v)
    ha = ekin + pot

    model = logpsi(L= L, sdim = sdim, graph_number = 1, phi_out_dim = 10, phi_hidden_lyrs = 1, phi_widths=(10,), rho_widths=(10,), rho_hidden_lyrs=1)

    vs = nk.vqs.MCState(sab, model, n_samples=5*10**3, n_discard_per_chain=32)
    op = nk.optimizer.Sgd(0.05)
    sr = nk.optimizer.SR(diag_shift=0.005)

    gs = nk.VMC(ha, op, sab, variational_state=vs, preconditioner=sr)
    gs.run(n_iter=10**3, callback=mycb, out=f"2int_bosons_2d_N=16_L={l}_{a}")

    x,_ = mpi.mpi_allgather_jax(vs.samples)
    jnp.save(f"2samples_N=16_L={l}_{a}.npy", x)
