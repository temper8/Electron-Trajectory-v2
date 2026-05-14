import sys

import numpy as np
import pandas as pd
from numpy import cos, sin, pi
import matplotlib.pyplot as plt    


from src.config import RunParams, SolverParams, load_configs, read_config_hdf5
from src.physical_constants import *
from src.utils import get_dataset_sizes, select_h5_file

#race_file = 'race/EXL-50U_13976/2026_05_04_22_20_06.h5'
race_file = select_h5_file() 
if race_file is None:
    sys.exit()

solver, params, cfg = read_config_hdf5(race_file)
sizes = get_dataset_sizes(race_file, ['trajectory', 'poincare_points'])
print(sizes)
print(solver)
print(params)

ccc_R0= ccc/params.R0
a = params.a
R0 = params.R0
n = params.n

from src.env import init_env
init_env(tau_norm*ccc_R0, R0, a)

from src.plot_environment import plot_field_environment
from src.envelope_fit import get_extremums, plot_envelope_fit
from src.plot_r_phi import plot_r_phi_segments
from src.plotting import plot_partice_state, plot_poincare, plot_timeline_r_phi_poincare, plot_traj, polar_plot_traj

downsample_step = 1
#df = pd.read_hdf(race_file, 'trajectory', mode='r', start=0, stop=100000)

df = pd.read_hdf(race_file, 'trajectory', mode='r')
df['R'] = R0 + df['r']*cos(df['theta'])
df['Z'] = df['r']*sin(df['theta'])
df['time']=df['tau']/ccc_R0*tau_norm
df['floor_phi'] =  np.floor(df['phi']/(2*pi)).astype(int)

#if downsample_step>0:
#    indices = np.arange(0, sizes['poincare_points'], downsample_step)
#    pp_df = pd.read_hdf(race_file, 'poincare_points', mode='r', where= pd.Index(indices))    

pp_df = pd.read_hdf(race_file, 'poincare_points', mode='r')
pp_df['time']=pp_df['tau']/ccc_R0*tau_norm
pp_df['R'] = R0 + pp_df['r']*cos(pp_df['theta'])
pp_df['Z'] = pp_df['r']*sin(pp_df['theta'])

#print(df.head(5).to_string())
print(f"trajectory size = {len(df)}")
print(f"poincare size   = {len(pp_df)}")

ion = True
def show():
    if ion:
        plt.draw() # Принудительная отрисовка
        plt.pause(1.5)
    else:
        plt.show()


if ion:
    plt.ion() # Включаем интерактивный режим
time_start = df['time'].iloc[0] 
time_end = df['time'].iloc[-1]
tau_start = df['tau'].iloc[0] 
tau_end = df['tau'].iloc[-1]

title = f"r = {params.r}, delta_time = {time_end-time_start:0.7f} delta_tau = {tau_end-tau_start:0.1f}\n"
title += f" solver={solver.method}, rtol={solver.rtol}, atol={solver.atol}"
plot_envelope_fit(df, a, title, race_file.stem)
show()

df_thin = df.iloc[::downsample_step]
pp_df_thin = pp_df.iloc[::downsample_step]
plot_field_environment(pp_df_thin['tau'], ccc_R0*tau_norm, title, race_file.stem)
show()

#plot_r_phi_segments(df, 43)
#show()

plot_traj(df_thin, pp_df_thin, title, race_file.stem)
show()

polar_plot_traj(df_thin, pp_df_thin, a, race_file.stem)
show()

plot_timeline_r_phi_poincare(df_thin, pp_df_thin, a, race_file.stem)
show()

plot_poincare(df_thin, pp_df_thin, race_file.stem)
show()

plot_partice_state(df_thin, a, title, race_file.stem)
show()



#ex_df = get_extremums(df)
#title = f"extremums solver={solver.method}, rtol={solver.rtol}, atol={solver.atol}"
#plot_envelope_fit(ex_df, a, title, race_file.stem + "_ext")
#show()

if ion:
    plt.ioff() # Выключаем интерактивный режим
    plt.show() # Блокируем выход, пока вы сами не закроете окна

sys.exit(0)

from scipy import stats
slope, intercept, r_value, p_value, std_err = stats.linregress(df['theta'] , df['phi'])
coeffs = np.polyfit(df['theta'] , df['phi'], 1)
trend_line = np.poly1d(coeffs)

# Создаем колонку с предсказанными значениями
df['trend_phi'] = trend_line(df['theta'])
plt.figure()

plt.scatter(df['theta']%(2*pi), 
            df['phi']-df['trend_phi'], 
            c= df['time'], cmap='plasma', 
            alpha=0.05, edgecolors='none', s=12)
#plt.plot(df['theta'], df['phi']- df['trend_phi'], color='red')

plt.title("theta-phi plot")
plt.xlabel('theta')
#plt.ylim(0.,1.0)
#plt.savefig('pictures/FT2_r_0.01_t_15_p_m0.025_segment_4_rto_a.svg')
plt.grid()
plt.draw() 
plt.pause(0.1)

plt.ioff() # Выключаем интерактивный режим
plt.show() # Блокируем выход, пока вы сами не закроете окна

