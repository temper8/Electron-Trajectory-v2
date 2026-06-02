import gc
import sys
import time
from math import pi, sqrt, log
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp  

from src.analyze_harmonics import compute_guiding_center_harmonics, plot_guiding_center_harmonics, save_harmonics
import src.config as config 
from src.envelope_fit import get_extremums
from src.logger_config import get_memory_usage, logger
from src.physical_constants import *
from src.poincare import find_poincare_points
from src.config import save_namedtuple


if len(sys.argv) > 1:
    shot_file=sys.argv[1]
else:
    shot_file = 'short.toml'

logger.info(f"shot file: {shot_file}")
    
run_cfg, solver, params = config.load_configs(f'shots/{shot_file}')

logger.info(f"Tokamak: {run_cfg.tokamak_name} Shot number: {run_cfg.shot_number}")
logger.info(config.param_string(params))

race_folder = Path(f"race/{run_cfg.tokamak_name}_{run_cfg.shot_number}")
race_folder.mkdir(parents=True, exist_ok=True)
race_file = Path(f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.h5")

ccc_R0 = ccc/params.R0
# инициализация функций окружения
from src.env import init_env
init_env(tau_norm*ccc_R0, params.R0, params.a)

from src.eqations import guiding_center_dynamics, hit_wall
from src.particle_state import get_particle_state, initialize_particle_state

# Расчет нормированного начального времени
tau = run_cfg.time_start * ccc_R0 / tau_norm

Btot, muini, psipol, energy = initialize_particle_state(tau, params)

logger.info(f'r= {params.r}, theta={params.theta}, phi={params.phi}, ppar= {params.ppar}, energy= {energy}')
logger.info(f"---------------------- start -------------------------------")

calculation_start_time = time.time()
file = race_folder/race_file
logger.debug(file)

with pd.HDFStore(race_folder/race_file, mode='w') as store:
    save_namedtuple(store, 'params', params)
    save_namedtuple(store, 'solver', solver)
    save_namedtuple(store, 'config', run_cfg)
    logger.info(f"Open the HDF5 file :  {file.name}")
    tau_start = tau
    r = params.r
    theta = params.theta
    phi = params.phi
    ppar = params.ppar
    logger.info(f"num_it= {run_cfg.num_it}, max_tau_step= {run_cfg.max_tau_step}, delta_tau= {run_cfg.delta_tau}")
    
    max_step = run_cfg.max_tau_step if run_cfg.max_tau_step>0 else np.inf
    
    for it in range(run_cfg.num_it):
        logger.info(f"Iteration {it}. Start")
        iteration_start_time = time.time()
        
        y0= [ppar, r, theta, phi] #, pperp2ini, Bpolini, Btotini, Bradini, Btorini, psipolini, psitorini, energyini]
        tau_end= tau_start + run_cfg.delta_tau  #t1UL

        logger.info(f'r= {r}, theta= {theta}, phi= {phi}, ppar= {ppar}')
        logger.info(f't_start(s)= {tau_start*params.R0/ccc*tau_norm}, del_t_calculation(s)= {(tau_end-tau_start)*params.R0/ccc*tau_norm}, time(s)={tau_end*params.R0/ccc*tau_norm}')
        logger.info(f'solve_ivp: method= {solver.method}, dense_output=True')
        logger.info(f'solve_ivp: rtol= {solver.rtol}, atol= {solver.atol}')
        sol= solve_ivp(guiding_center_dynamics,
                    [tau_start, tau_end], 
                    y0, 
                    method= solver.method, 
                    dense_output=True, 
                    args=(params, muini),
                    events=hit_wall,
                    rtol= solver.rtol,
                    atol= solver.atol,
                    max_step= max_step) 
        logger.info(f"Number of function evaluations {sol.nfev}")
        iteration_time = time.time() - iteration_start_time
        logger.info(f"Number of function evaluations per sec {(sol.nfev/iteration_time):0.2f}")
        
        delta_tau= sol.t[-1] - sol.t[0]
        f_polo = abs(sol.y[3, -1] - sol.y[3, 0])/(2*pi*delta_tau)
        f_toro = abs(sol.y[2, -1] - sol.y[2, 0])/(2*pi*delta_tau)
        logger.info(f'f_polo= {f_polo:0.2f}, f_toro= {f_toro:0.2f}, delta_tau= {delta_tau}')
        # Шаг 1: Только считаем (график не выводится, можно крутить в цикле по разным координатам)
        freqs, times, spec = compute_guiding_center_harmonics(
            sol=sol, 
            coordinate_idx=3,   # Например, полоидальный угол
            f_polo=f_polo, # Ожидаемая полоидальная частота (Гц или у.е.)
            f_toro=f_toro,  # Ожидаемая тороидальная частота
            is_angle=True       # Сглаживаем угол синусом
        )
        save_harmonics(store, it, 
                       freqs, times, spec, 
                       f_toro, f_polo, 3, True, title_suffix="полоидальный угол")
        # Шаг 2: Передаем результаты в визуализатор
        #plot_guiding_center_harmonics(
        #    frequencies=freqs, 
        #    times=times, 
        #    amplitude_spectrogram=spec, 
        #    f_toro=f_toro,
        #    title_suffix="(полоидальный угол)"
        #)        
        tau_start= sol.t[-1]
        y_last = sol.y[:, -1]
        #pparini, rini, thetini, fiini , pperp2ini, Bpolini, Btotini, Bradini, Btorini, psipolini, psitorini, energyini = y_last
        ppar, r, theta, phi = y_last

        logger.info(f'theta revolutions= {theta/(2*pi):0.2f}, phi revolutions= {phi/(2*pi):0.2f}')
        theta = theta%(2*pi)
        phi   = phi%(2*pi)
        #exit(0)
        #energy, p_tot = get_particle_state(sol.t, sol.y, muini, params)
        energy = np.empty(len(sol.t))
        p_tot = np.empty(len(sol.t))
        for i in range(len(sol.t)):
            energy[i], p_tot[i] = get_particle_state(sol.t[i], sol.y[:, i], muini, params)

        df = pd.DataFrame(sol.y.T, columns=['ppar','r','theta','phi'])
        df['tau']    =  sol.t
        df['energy'] =  energy
        df['p_tot']  =  p_tot

        logger.debug("\n" + df.head().to_string())
        logger.info(f"df size= {len(df)}, {get_memory_usage()}.")

        # Инкрементная запись в HDF5 
        store.append('trajectory', get_extremums(df), index=False)
        #store.append('trajectory', df, index=False)
        store.append('poincare_points', find_poincare_points(sol), index=False)

        logger.info(f"Iteration {it}. calculation time: {iteration_time:0.2f} sec")
        logger.info(f"------------------------------------------------------------")
        
        if sol.status == 1:
            logger.info(f"The event was recorded: the particle touched the wall.")
            tau_collision = sol.t_events[0][0]
            logger.info(f"tau collision = {tau_collision} ")
            
            # Можно также узнать координаты точки столкновения
            #R_collision = sol.y_events[0][0][0]
            #Z_collision = sol.y_events[0][0][2]
            #print(f"Координаты столкновения: R={R_collision:.3f}, Z={Z_collision:.3f}")
            break
        gc.collect()


logger.info(f"Full calculationtime: {time.time() - calculation_start_time:0.2f} sec")        
