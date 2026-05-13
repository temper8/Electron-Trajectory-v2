from numpy import pi, sqrt, log
from numba import njit
from src.physical_constants import *
from src.eqations import Mag_field, get_Btot
from src.env import B0_tau, get_field_environment
import src.config as config 

def initialize_particle_state(tau, params: config.RunParams):
    """
    Инициализирует начальное состояние частицы и параметры магнитного поля.
    """
    
    # Получение параметров окружения и расчет профиля safety factor
    sf0, sfb, Uloop, B0 = get_field_environment(tau)
    
    # Расчет компонентов магнитного поля
    Btot, *_ = Mag_field(params.r, params.theta, params.phi, tau, params)
    
    # Расчет инвариантов и начальной энергии
    pperp2 = params.pperp**2    
    mu = pperp2 / Btot
    p2 = params.ppar**2 + pperp2
    
    # Расчет полоидального потока
    psipol = pi * B0 * params.a**2 / (sfb - sf0) * log((sf0 + (sfb - sf0) * (params.r / params.a)**2) / sf0)
    
    # Энергия в эВ (или МэВ, в зависимости от констант)
    energy = m01 * ccc1**2 * (sqrt(1 + p2) - 1) / 1.6022e-12

    # Возвращаем словарь со всеми рассчитанными значениями
    return  Btot, mu, psipol, energy

@njit
def get_particle_state(tau, y, mu, params: config.RunParams):
    """
    вычисляет параметры магнитного поля и состояние частицы
    """
    ppar, r, theta, phi = y
    # Получение параметров окружения и расчет профиля safety factor
    B0 = B0_tau(tau)
    # Расчет компонентов магнитного поля
    Btot = get_Btot(r, theta, phi, tau, params)
    
    # Расчет перпендикулярного импульса и начальной энергии
    
    pperp2 = mu* Btot
    pperp  = sqrt(pperp2)
    p_tot  = ppar**2 + pperp2
    
    # Энергия в эВ (или МэВ, в зависимости от констант)
    energy = m01 * ccc1**2 * (sqrt(1 + p_tot) - 1) / 1.6022e-12
    return energy, p_tot