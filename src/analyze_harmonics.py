import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
from scipy.integrate._ivp.ivp import OdeResult

def compute_guiding_center_harmonics(sol:OdeResult, coordinate_idx:int, f_polo, f_toro, is_angle=False):
    """
    Вычисляет гармонические моменты (STFT) из плотного решения solve_ivp.
    Параметры:
    ----------
    sol : OdeResult
        Результат solve_ivp с включенным флагом dense_output=True.
    coordinate_idx : int
        Индекс анализируемой координаты в векторе состояния sol.y (например, 0, 1).
    f_polo : float
        Ожидаемая полоидальная частота (для настройки ширины окна).
    f_toro : float
        Ожидаемая тороидальная частота (для настройки масштаба графика).
    is_angle : bool
        Если True, координата считается угловой (применяется sin() для гладкости).    
    Возвращает:
    -----------
    frequencies : ndarray (массив частот)
    times : ndarray (массив временных меток окон)
    amplitude_spectrogram : ndarray (матрица амплитуд гармоник)
    """
    # 1. Генерация равномерной сетки времени через dense_output
    fs = f_toro * 15  
    delta_tau= sol.t[-1] - sol.t[0]
    dt = 1 / fs
    print(f'delta_tau= {delta_tau}, dt = {dt}')
    t_uniform = np.arange(sol.t[0], sol.t[-1], dt)
    print(len(t_uniform))
    # 2. Извлечение данных и обработка цикличности углов
    raw_data = sol.sol(t_uniform)[coordinate_idx, :]
    signal = np.sin(raw_data) if is_angle else raw_data
    
    # 3. Настройка параметров окна Ханна
    points_per_period = int(fs / f_polo)
    nperseg = points_per_period * 50       # Окно в 10 полоидальных периодов
    noverlap = int(nperseg * 0.75)         # Перекрытие 75%
    # 4. Расчет оконного преобразования Фурье
    frequencies, times, Zxx = stft(
        signal, 
        fs=fs, 
        window='hann', 
        nperseg=nperseg, 
        noverlap=noverlap
    )
    amplitude_spectrogram = np.abs(Zxx)
    
    return frequencies, times, amplitude_spectrogram


def plot_guiding_center_harmonics(frequencies, times, amplitude_spectrogram, f_toro, title_suffix=""):
    """
    Визуализирует рассчитанные гармонические моменты в виде спектрограммы.
    """
    plt.figure(figsize=(12, 6))
    
    # Ограничиваем график первыми четырьмя тороидальными гармониками
    max_freq_to_show = f_toro * 8
    freq_mask = frequencies <= max_freq_to_show
    
    # Построение тепловой карты
    mesh = plt.pcolormesh(
        times, 
        frequencies[freq_mask], 
        amplitude_spectrogram[freq_mask, :], 
        shading='gouraud', 
        cmap='inferno'
    )
    
    plt.title(f'Эволюция гармонических моментов траектории {title_suffix}')
    plt.xlabel('Время симуляции (t)')
    plt.ylabel('Частота (f)')
    plt.colorbar(mesh, label='Амплитуда гармоники')
    plt.grid(alpha=0.2, linestyle='--')
    plt.tight_layout()
    plt.show()
