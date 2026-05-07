import sys

import numpy as np
import pandas as pd
from numpy import cos, sin, pi
import matplotlib.pyplot as plt    
from src.envelope_fit import plot_envelope_fit
from src.config import RunParams, SolverParams, load_configs, read_config_hdf5
from src.physical_constants import *
from src.plot_r_phi import plot_r_phi_segments
from src.plot_trajectory import plot_12, plot_123, plot_hilbert, plot_poincare, plot_traj, polar_plot_traj
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

df = pd.read_hdf(race_file, 'trajectory', mode='r')
df['R'] = R0 + df['r']*cos(df['theta'])
df['Z'] = df['r']*sin(df['theta'])
df['time']=df['tau']/ccc_R0*tau_norm
df['floor_phi'] =  np.floor(df['phi']/(2*pi)).astype(int)

pp_df = pd.read_hdf(race_file, 'poincare_points', mode='r')
pp_df['time']=pp_df['tau']/ccc_R0*tau_norm
pp_df['R'] = R0 + pp_df['r']*cos(pp_df['theta'])
pp_df['Z'] = pp_df['r']*sin(pp_df['theta'])

#print(df.head(5).to_string())
print(f"trajectory size = {len(df)}")
print(f"poincare size   = {len(pp_df)}")


plt.ion() # Включаем интерактивный режим

plot_r_phi_segments(df, 43)
plt.draw() # Принудительная отрисовка
plt.pause(0.1)

plot_traj(df, pp_df, race_file.stem)
plt.draw() # Принудительная отрисовка
plt.pause(0.1)

polar_plot_traj(df, a, race_file.stem)
plt.draw() 
plt.pause(0.1)

plot_123(df, pp_df, a)
plt.draw() 
plt.pause(0.1)

plot_poincare(df, pp_df, race_file.stem)
plt.draw() 
plt.pause(0.1)

title = f"solver={solver.method}, rtol={solver.rtol}, atol={solver.atol}"
plot_envelope_fit(df,a, title, race_file.stem)
plt.draw() 
plt.pause(0.1)


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


sys.exit(0)

rpr=df['r']/a
thetpr=df['theta']

mmn=0
mmx= 10000
mmn1=100000
mmx1=109999
mmn2=200000
mmx2=209999
mmn3=300000
mmx3=309990


rpr0=rpr[mmn:mmx]
thetpr0=thetpr[mmn:mmx]
rpr1=rpr[mmn1:mmx1]
thetpr1=thetpr[mmn1:mmx1]
rpr2=rpr[mmn2:mmx2]
thetpr2=thetpr[mmn2:mmx2]
rpr3=rpr[mmn3:mmx3]
thetpr3=thetpr[mmn3:mmx3]
#rpr=rpr[0:629200]
#thetpr=thetpr[0:629200]
#tpr=df['t_ini']
#tpr=tpr[0:45]




#print('rini=',sol[nrange-1,1])
#plt.plot(sol.t, sol.y[1]/a, 'g', label='r(t)/a')
#rpr=df['rini']/a
tinipr=df['time']

#rpr=rpr[mmn:mmx]
tinipr0=tinipr[mmn:mmx]
tinipr1=tinipr[mmn1:mmx1]
tinipr2=tinipr[mmn2:mmx2]
tinipr3=tinipr[mmn3:mmx3]
#plt.plot(df['t_ini'], df['rini']/a, 'g', label='r(t)/a')
plt.figure()
plt.plot(tinipr,rpr, 'm', label='r(t)/a')
plt.plot(tinipr0,rpr0, 'r', label='r(t)/a')
plt.plot(tinipr1,rpr1, 'g', label='r(t)/a')
plt.plot(tinipr2,rpr2, 'y', label='r(t)/a')
plt.plot(tinipr3,rpr3, 'b', label='r(t)/a')
plt.legend(loc='best')
plt.xlabel('t(ms)')
#plt.xlim(0.31,0.32)
#plt.xlim(0.346,0.350)
plt.ylim(0.,1.0)
#plt.xlim(0.348,0.349)
#plt.xlim(0.328,0.331)
plt.savefig('pictures/FT2_r_0.01_t_15_p_m0.025_segment_4_rto_a.svg')
plt.grid()
plt.draw() # Принудительная отрисовка
plt.pause(0.1)


plt.figure()

plt.plot(df['time'], df['r'], 'g', label='r(t)')
plt.legend(loc='best')
plt.xlabel('t')
plt.xlim(15,47)

plt.ylim(0.0,0.601)
plt.grid()
plt.figure()
plt.draw() # Принудительная отрисовка
plt.pause(0.1)

plt.ioff() # Выключаем интерактивный режим
plt.show() # Блокируем выход, пока вы сами не закроете окна

#plt.plot(sol.t, m01*ccc1**2*(sqrt(1+(sol.y[4])+(sol.y[0])**2)-1.)/1.6022e-12, 'r', label='energy(eV)')
#df['energy']= m01*ccc1**2*(np.sqrt(1+df['pperp2']+(df['ppar'])**2)-1.)/1.6022e-12
enrg=df['energy']
enrg0=enrg[mmn:mmx]
enrg1=enrg[mmn1:mmx1]
enrg2=enrg[mmn2:mmx2]
#plt.plot(df['t_ini'], df['energy'], 'r', label='energy(eV)')
plt.plot(df['time'], df['energy'], 'g', label='Energy(eV)')
plt.plot(tinipr0,enrg0, 'y', label='Energy(eV)')
plt.plot(tinipr1,enrg1, 'r', label='Energy(eV)')
plt.plot(tinipr2,enrg2, 'b', label='Energy(eV)')

#plt.plot(df['t_ini'], df['energyini'], 'g', label='energy(eV)')
plt.legend(loc='best')
plt.xlabel('t(s)')

plt.ylim(0.e7,1.e7)
plt.savefig('pictures/FT2_r_0.01_t_15_p_m0.025_segment_4_Wkin.svg')
plt.grid()
plt.show()

df['gamnp']=  np.sqrt(1 + df['pperp2'].astype(float) + df['ppar'].astype(float)**2)
#print(len(pperp2np),len(pparini))
plt.plot(df['time'], df['gamnp'], 'r', label='relativistic gamma factor')
plt.legend(loc='best')
plt.xlabel('t')
plt.grid()
plt.show()

figure, axes = plt.subplots( 1 )
x= (df['r'].astype(float))*cos(df['thet'].astype(float))/a
y= (df['r'].astype(float))*sin(df['thet'].astype(float))/a
x=x[300000:668000]
y=y[300000:668000]
axes.plot(x, y, 'g', label='r(t)/a')
axes.set_aspect( 1 )

plt.title( 'trajectory' )
plt.show()