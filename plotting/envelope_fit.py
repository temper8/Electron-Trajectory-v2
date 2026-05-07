from matplotlib import pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from scipy.interpolate import interp1d

def get_envelope_fit(t_raw, x_raw):
    # t_raw, x_raw — ваши данные ОДУ (допустим, с неравномерным шагом)

    # 1. Поиск экстремумов
    # Максимумы

    peaks_idx, _ = find_peaks(x_raw)
    # Минимумы (ищем максимумы инвертированного сигнала)
    troughs_idx, _ = find_peaks(-x_raw)

    t_p, x_p = t_raw[peaks_idx], x_raw[peaks_idx]
    t_t, x_t = t_raw[troughs_idx], x_raw[troughs_idx]

    # 2. Интерполяция огибающих (сплайнами)
    # Кубический сплайн дает плавность, но на краях может "гулять"
    upper_env_func = interp1d(t_p, x_p, kind='linear', fill_value="extrapolate")
    lower_env_func = interp1d(t_t, x_t, kind='linear', fill_value="extrapolate")



    # 3. Расчет характеристик
    # Амплитуда A(t) — это полурасстояние между огибающими
    #A_t = (upper_env - lower_env) / 2
 

    # Частота w(t) — используем и пики, и впадины для лучшего разрешения
    t_all = np.sort(np.concatenate([t_p, t_t]))
    half_periods = np.diff(t_all)
    # Частота через полупериоды
    w_vals = np.pi / half_periods 
    t_w = t_all[:-1] + half_periods / 2
    w_interp = interp1d(t_w, w_vals, kind='linear', fill_value="extrapolate")
    
    w_t = w_interp(t_all)
    upper_env = upper_env_func(t_all)
    lower_env = lower_env_func(t_all)
    # Средняя линия (offset) — если центр колебаний смещен
    offset_t = (upper_env + lower_env) / 2   
    return lower_env, upper_env, offset_t, w_t, t_all

from scipy import stats

def trend_line(x ,y):
    #slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
    coeffs = np.polyfit(x, y, 1)
    trend_line = np.poly1d(coeffs)
    xy = (x[0], trend_line(x[0]))
    return xy, coeffs[0]

def plot_envelope_fit(df, a, title, race_name, show_trend_line= True):
    x_raw = np.array(df['r'])/a
    t_raw = np.array(df['time'])

    lower_env, upper_env, offset_t, w_t, t_all = get_envelope_fit(t_raw, x_raw)
 
    # 3. Визуализация
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True, num=f"{race_name}_envelope_fit")

    # Верхний график: Сигнал и Огибающие
    ax1.plot(t_raw, x_raw, color='gray', alpha=0.3, label='r(t)/a')
    # Для огибающих используем линейную интерполяцию по найденным точкам
    ax1.plot(t_all, upper_env, 'r:', label='Верхняя огибающая')
    if show_trend_line:
        xy, slope = trend_line(t_all ,upper_env)
        ax1.axline(xy, slope= slope, color='r', linestyle='--', label= f'slope = {slope:.3f}', alpha=0.5)

    ax1.plot(t_all, lower_env, 'b--', label='Нижняя огибающая')
    ax1.plot(t_all, offset_t, 'k', label='Средняя линия (offset)', alpha=0.5)
    #ax1.fill_between(t_raw, lower_env, upper_env, color='yellow', alpha=0.1)
    ax1.set_ylabel('r(t)/a')
    ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    ax1.set_title(title)

    # Нижний график: Две частоты

    ax2.plot(t_all, w_t, 'r.-', label='w(t) по максимумам', alpha=0.7)
    #ax2.plot(t_w_upper, w_upper, 'r.-', label='w(t) по максимумам', alpha=0.7)
    #ax2.plot(t_w_lower, w_lower, 'b.-', label='w(t) по минимумам', alpha=0.7)
    if show_trend_line:
        xy, slope = trend_line(t_all ,w_t)
        ax2.axline(xy, slope= slope, color='blue', linestyle='--', label= f'slope = {slope:.3e}')

    ax2.set_ylabel('Частота w')
    ax2.set_xlabel('time (s)')
    ax2.grid(True, which='both', alpha=0.2)
    ax2.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)

    fig.tight_layout()

    #plt.show()