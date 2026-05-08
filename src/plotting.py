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
    fig.tight_layout()

def polar_plot_traj(df, pp_df, a, race_name):
    fig = plt.figure(figsize=(10,5), num=f"{race_name}_polar_projection_trajctory")
    #fig.set_title("Electron trajectory in poloidal crossection", va='bottom')
    #fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, num=f"{race_name}_polar_traj")
    ax0, ax1 = fig.subplots(1, 2, subplot_kw={'projection': 'polar'})
    ax0.scatter( df['theta'], df['r']/a, c= df['time'], cmap='plasma', alpha=0.05, edgecolors='none', s=8)
    ax0.set_rmax(1)
    #ax.set_rticks([0.2, 0.4, 0.6, 0.8])  # Less radial ticks
    ax0.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
    ax0.grid(True)
    #ax.set_title("Electron trajectory in poloidal crossection", va='bottom')
    
    ax1.scatter( pp_df['theta'], pp_df['r']/a, c= pp_df['time'], cmap='plasma', alpha=0.05, edgecolors='none', s=8)
    ax1.set_rmax(1)
    #ax.set_rticks([0.2, 0.4, 0.6, 0.8])  # Less radial ticks
    ax1.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
    ax1.grid(True)    
    fig.tight_layout()

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
    fig.tight_layout()

def plot_timeline_r_phi_poincare(df, pp_df, a, race_name):
    fig = plt.figure(figsize=(10,8), layout='constrained', num=f"{race_name}_timeline")
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
    fig.tight_layout()

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



