from matplotlib import pyplot as plt
from numpy import sin


def plot_traj(df, pp_df):
    fig = plt.figure(figsize=(10,5))
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


def polar_plot_traj(df, a):
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
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
   
def plot_poincare(df, pp_df):
    plt.figure()
    #plt.plot(df['time'], sin(df['phi']), marker='o', linestyle='-', color='b')
    plt.plot(pp_df['time'], sin(pp_df['phi']), marker='o', linestyle='-', color='r')
    plt.title("phi(t) poincare points")
    plt.xlabel('t(ms)')
    #plt.ylim(0.,1.0)
    #plt.savefig('pictures/FT2_r_0.01_t_15_p_m0.025_segment_4_rto_a.svg')
    plt.grid()