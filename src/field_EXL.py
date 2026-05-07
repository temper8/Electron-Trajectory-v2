from src.parameters import ccc_R0, a, R0
from src.physical_constants import tau_norm
import src.env as env

u_loop, B0_tau, q0_tau, qa_tau = env.create_env_function(tau_norm*ccc_R0, R0, a)

def get_field_environment(tau):
    sf0=q0_tau(tau)
    sfb=qa_tau(tau)
    Uloop=u_loop(tau)
    B0 = B0_tau(tau)
    return sf0, sfb, Uloop, B0
