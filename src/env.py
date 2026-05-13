import numpy as np
from numba import njit
from numpy import pi, sin, cos

def create_env_function(tau_factor, R0, a):
    """
    Создает и возвращает функцию Uloop(t), привязанную к заданному интервалу.

    """
    t_start = 0
    t_end   = 1 
    # Пересчитываем границы в безразмерные единицы сразу
    t0 = t_start * tau_factor
    t1 = t_end * tau_factor
    delta_t = t1 - t0

    @njit
    def uloop_t(t:float) -> float:
        # t ожидается в нормализованных единицах
        x = (t - t0) / delta_t
        # Полином
        return (-7.887*x**6 + 80.237*x**5 - 144.31*x**4 + 
                90.855*x**3 - 21.72*x**2 + 2.0764*x + 0.6478)
    
    @njit
    def B0_t(t:float) -> float:
        """Магнитное поле от времени"""
        x = (t-t0)/delta_t
        #B0 = 0.0002*x**3 - 0.0443*x**2 + 1.9616*x - 3.0964
        B0 = 0.71
        return B0
    
    @njit
    def cur_t(t:float):
        """Полный ток от времени"""
        x = (t-t0)/delta_t
        return 45249*x**6 - 134815*x**5 + 148661*x**4 - 73887*x**3 + 13833*x**2 + 905.74*x + 44.505
    
    @njit
    def q_a(t):
        """ q-фактор на границе"""
        Bpl_curnp=2.e-4*cur_t(t)/a
        return np.abs(a*B0_t(t)/(R0*Bpl_curnp))

    ctq0=1
    t_q0=t0  
    tau_q0=0.03
    tau_q10=2.5
    t_q0=t_q0  
    tau_q0=tau_q0/tau_factor
    tau_q10=tau_q10/tau_factor
    q_0_ini=(q_a(t_q0)-ctq0)/2

    @njit
    def E0_field(r, theat, phi, tau):
        E0tor=uloop_t(tau)/(2*pi*R0)
        return E0tor

    @njit    
    def E_field(r, theta, phi, tau):
        #E0tor = E0_field(r, theta, phi, tau)
        E0tor=uloop_t(tau)/(2*pi*R0)
        Etor=E0tor*R0/(R0+r*cos(theta))
        Erad=0.
        Epol=0.
        Etot=np.sqrt(Etor**2+Erad**2+Epol**2)
        if abs(Etot) >0.:
            etor=Etor/Etot
            erad=Erad/Etot
            epol=Epol/Etot
        else:
            etor=0.
            erad=0.
            epol=0.
        return Etot,Etor,etor,Erad,erad,Epol,epol

    @njit
    def q0_t(t):
        """q-фактор в центре"""
        q_0=q_0_ini*(0.85*np.exp(-(t-t_q0)/tau_q0)+0.0*np.exp(-(t-t_q0)/tau_q10)) + ctq0
        return q_0
    
    return uloop_t, B0_t, q0_t, q_a, E_field


u_loop = lambda x: 0
B0_tau = lambda x: 0
q0_tau = lambda x: 0
qa_tau = lambda x: 0
E_field_tau = lambda r, theta, phi, tau: 0

def init_env(tau_factor, R0, a):
    global u_loop, B0_tau, q0_tau, qa_tau, E_field_tau
    u_loop, B0_tau, q0_tau, qa_tau, E_field_tau = create_env_function(tau_factor, R0, a)# (tau_norm*ccc_R0, R0, a)

def get_field_environment(tau):
    sf0=q0_tau(tau)
    sfb=qa_tau(tau)
    Uloop=u_loop(tau)
    B0 = B0_tau(tau)
    return sf0, sfb, Uloop, B0


@njit
def saf_fact(sf0,sfb,r,a):
    sf=sf0+(sfb-sf0)*(r/a)**2
    return sf

@njit
def safety_factor(tau, ro):
    sf0=q0_tau(tau)
    sfa=qa_tau(tau)
    sf=sf0+(sfa-sf0)*ro*ro
    return sf0, sfa, sf