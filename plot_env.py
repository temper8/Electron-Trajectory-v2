import sys

import numpy as np
import pandas as pd
from numpy import cos, sin, pi
import matplotlib.pyplot as plt    

import src.config as config
from src.config import RunParams, SolverParams, load_configs, read_config_hdf5
from src.physical_constants import *


if len(sys.argv) > 1:
    shot_file=sys.argv[1]
else:
    shot_file = 'short.toml'

   
run_cfg, solver, params = config.load_configs(f'shots/{shot_file}')

print(solver)
print(params)

ccc_R0= ccc/params.R0
a = params.a
R0 = params.R0
n = params.n

from src.plot_environment import plot_field_environment
from src.env import init_env

init_env(tau_norm*ccc_R0, R0, a)

plt.ion() # Включаем интерактивный режим

tau_start = run_cfg.time_start*ccc_R0/tau_norm
tau_end   = tau_start + run_cfg.delta_tau*run_cfg.num_it
tau = np.linspace(tau_start, tau_end, 1000)
plot_field_environment(tau, ccc_R0*tau_norm, shot_file)
plt.draw() # Принудительная отрисовка
plt.pause(0.1)

tau_start = 0.0*ccc_R0/tau_norm
tau_end   = 1.0*ccc_R0/tau_norm

tau = np.linspace(tau_start, tau_end, 1000)
plot_field_environment(tau, ccc_R0*tau_norm, 'shot_file')

plt.draw() 
plt.pause(0.1)

plt.ioff() # Выключаем интерактивный режим
plt.show() # Блокируем выход, пока вы сами не закроете окна



