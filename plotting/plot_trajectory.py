from matplotlib import pyplot as plt
import numpy as np
from numpy import sin



def plot_traj(df, pp_df, race_name):
    fig = plt.figure(figsize=(10,5), num=f"{race_name}_trajctory")
    ax0, ax1 = fig.subplots(1, 2)
    ax0.scatter(df['R'], df['Z'], c= df['time'], cmap='plasma', alpha=0.05, edgecolors='none', s=8)
    ax0.set_title(f'Trayectory {len(df)} points')
    ax0.set_xlabel('R')
    ax0.set_ylabel('Z')
    ax0.grid(True)
    ax0.axis('equal')

    ax1.scatter(pp_df['R'], pp_df['Z'], c= pp_df['time'], cmap='plasma', alpha=0.05, edgecolors='none', s=8)
    ax1.set_title(f'Poincare ({len(pp_df)}) points')
    ax1.set_xlabel('R')
    #ax1.set_ylabel('Z')
    ax1.grid(True)
    ax1.axis('equal')


def polar_plot_traj(df, a, race_name):
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, num=f"{race_name}_polar_traj")
    ax.scatter( df['theta'], df['r']/a, alpha=0.05, color='blue', edgecolors='none', s=10)
    ax.set_rmax(1)
    #ax.set_rticks([0.2, 0.4, 0.6, 0.8])  # Less radial ticks
    ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
    ax.grid(True)
    #ax.set_title("Electron trajectory in poloidal crossection", va='bottom')
    #plt.savefig('pictures/FT2_r_0.01_t_15_p_m0.1_segment_4_cross_sect.png')

def plot_12(df, pp_df, a):
    fig = plt.figure(figsize=(10,8), layout='constrained')
    ax0, ax1 = fig.subplots(2, 1, sharex=True)
    ax0.plot(df['time'], df['r']/a)
    ax0.scatter(pp_df['time'], pp_df['r']/a, c= 'r', s=8)
    ax0.set_title(f'r(t)/a and sin(phi(t))')
    #ax0.set_xlabel('R')
    ax0.set_ylabel('r(t)/a')
    ax0.set_ylim(0.0, 1.0)
    ax0.grid(True)

    ax1.plot(df['time'], sin(df['phi']))
    ax1.scatter(pp_df['time'],  sin(pp_df['phi']), c= 'r', s=8)
    #ax1.set_title(f'Trayectory {len(df)} points')
    ax1.set_xlabel('time (ms)')
    ax1.set_ylabel('sin(phi(t))')
    ax1.grid(True)
    #plt.savefig('pictures/FT2_r_0.01_t_15_p_m0.025_segment_4_rto_a.svg')


def plot_123(df, pp_df, a):
    fig = plt.figure(figsize=(10,8), layout='constrained')
    ax0, ax1, ax2 = fig.subplots(3, 1, sharex=True)
    ax0.plot(df['time'], df['r']/a)
    ax0.scatter(pp_df['time'], pp_df['r']/a, c= 'r', s=8)
    ax0.set_title(f'r(t)/a and sin(phi(t))')
    ax0.set_ylabel('r(t)/a')
    ax0.set_ylim(0.0, 1.0)
    ax0.grid(True)

    ax1.plot(df['time'], sin(df['phi']))
    ax1.scatter(pp_df['time'],  sin(pp_df['phi']), c= 'r', s=8)
    ax1.set_ylabel('sin(phi(t))')
    ax1.grid(True)

    ax2.plot(pp_df['time'], sin(pp_df['phi']))
    ax2.scatter(pp_df['time'],  sin(pp_df['phi']), c= 'r', s=8)
    #ax1.set_title(f'Trayectory {len(df)} points')
    ax2.set_xlabel('time (ms)')
    ax2.set_ylabel('phi(t) poincare points')
    ax2.grid(True)
    #plt.savefig('pictures/FT2_r_0.01_t_15_p_m0.025_segment_4_rto_a.svg')
   
def plot_poincare(df, pp_df, race_name):
    plt.figure(num=f"{race_name}_poincare")
    #plt.plot(df['time'], sin(df['phi']), marker='o', linestyle='-', color='b')
    plt.plot(pp_df['time'], sin(pp_df['phi']), marker='o', linestyle='-', color='r')
    plt.title("phi(t) poincare points")
    plt.xlabel('t(ms)')
    #plt.ylim(0.,1.0)
    #plt.savefig('pictures/FT2_r_0.01_t_15_p_m0.025_segment_4_rto_a.svg')
    plt.grid()

#from scipy.fftpack import hilbert
from scipy.signal import hilbert, chirp
def plot_hilbert(df, a):
 
    duration, fs = 1, 400  # 1 s signal with sampling frequency of 400 Hz
    t = np.arange(int(fs*duration)) / fs  # timestamps of samples
    signal = chirp(t, 20.0, t[-1], 100.0)
    signal *= (1.0 + 0.5 * np.sin(2.0*np.pi*3.0*t) )

    analytic_signal = hilbert(signal)
    amplitude_envelope = np.abs(analytic_signal)
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    instantaneous_frequency = np.diff(instantaneous_phase) / (2.0*np.pi) * fs
    fig, (ax0, ax1) = plt.subplots(nrows=2, sharex='all', tight_layout=True)
    ax0.set_title("Amplitude-modulated Chirp Signal")
    ax0.set_ylabel("Amplitude")
    ax0.plot(t, signal, label='Signal')
    ax0.plot(t, amplitude_envelope, label='Envelope')
    ax0.legend()
    ax1.set(xlabel="Time in seconds", ylabel="Frequency in Hz", ylim=(0, 120))
    ax1.plot(t[1:], instantaneous_frequency, 'C2-',
            label='Instantaneous Frequency')
    ax1.legend()

from scipy.signal import find_peaks
from scipy.interpolate import interp1d

def plot_envelope_fit(df, a, title, race_name):

    # t_raw, x_raw — ваши данные ОДУ (допустим, с неравномерным шагом)

    # 1. Поиск экстремумов
    # Максимумы
    x_raw = np.array(df['r']/a)
    t_raw = np.array(df['time'])
    peaks_idx, _ = find_peaks(x_raw)
    # Минимумы (ищем максимумы инвертированного сигнала)
    troughs_idx, _ = find_peaks(-x_raw)

    t_p, x_p = t_raw[peaks_idx], x_raw[peaks_idx]
    t_t, x_t = t_raw[troughs_idx], x_raw[troughs_idx]

    # 2. Интерполяция огибающих (сплайнами)
    # Кубический сплайн дает плавность, но на краях может "гулять"
    upper_env_func = interp1d(t_p, x_p, kind='cubic', fill_value="extrapolate")
    lower_env_func = interp1d(t_t, x_t, kind='cubic', fill_value="extrapolate")

    upper_env = upper_env_func(t_raw)
    lower_env = lower_env_func(t_raw)

    # 3. Расчет характеристик
    # Амплитуда A(t) — это полурасстояние между огибающими
    A_t = (upper_env - lower_env) / 2

    # Средняя линия (offset) — если центр колебаний смещен
    offset_t = (upper_env + lower_env) / 2

    # Частота w(t) — используем и пики, и впадины для лучшего разрешения
    t_all = np.sort(np.concatenate([t_p, t_t]))
    half_periods = np.diff(t_all)
    # Частота через полупериоды
    w_vals = np.pi / half_periods 
    t_w = t_all[:-1] + half_periods / 2
    w_interp = interp1d(t_w, w_vals, kind='linear', fill_value="extrapolate")
    w_t = w_interp(t_raw)

    # 3. Визуализация
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, num=f"{race_name}_envelope_fit")

    # Верхний график: Сигнал и Огибающие
    ax1.plot(t_raw, x_raw, color='gray', alpha=0.3, label='r(t)/a')
    # Для огибающих используем линейную интерполяцию по найденным точкам
    ax1.plot(t_p, x_p, 'r--', label='Верхняя огибающая')
    ax1.plot(t_t, x_t, 'b--', label='Нижняя огибающая')
    ax1.plot(t_raw, offset_t, 'k', label='Средняя линия (offset)', alpha=0.5)
    #ax1.fill_between(t_raw, lower_env, upper_env, color='yellow', alpha=0.1)
    ax1.set_ylabel('r(t)/a')
    ax1.legend()
    ax1.set_title(title)

    # Нижний график: Две частоты
    ax2.plot(t_raw, w_t, 'r.-', label='w(t) по максимумам', alpha=0.7)
    #ax2.plot(t_w_upper, w_upper, 'r.-', label='w(t) по максимумам', alpha=0.7)
    #ax2.plot(t_w_lower, w_lower, 'b.-', label='w(t) по минимумам', alpha=0.7)
    ax2.set_ylabel('Частота w')
    ax2.set_xlabel('time (s)')
    ax2.grid(True, which='both', alpha=0.2)
    ax2.legend()

    fig.tight_layout()

    #plt.show()

