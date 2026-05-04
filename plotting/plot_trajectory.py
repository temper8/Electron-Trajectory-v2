from matplotlib import pyplot as plt


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
