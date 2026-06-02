import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
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
    fs = f_toro * 50  
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
    nperseg = points_per_period * 1000       # Окно в 10 полоидальных периодов
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
    max_freq_to_show = f_toro * 16
    freq_mask = frequencies <= max_freq_to_show
    log_amplitude = np.log10(amplitude_spectrogram[freq_mask, :] + 1e-4)
    # Построение тепловой карты
    mesh = plt.pcolormesh(
        times, 
        frequencies[freq_mask], 
        log_amplitude,
        #amplitude_spectrogram[freq_mask, :], 
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

def plot_harmonics(ax, frequencies, times, amplitude_spectrogram, f_toro, title_suffix=""):
    """
    Визуализирует рассчитанные гармонические моменты в виде спектрограммы.
    """
    
    # Ограничиваем график первыми четырьмя тороидальными гармониками
    max_freq_to_show = f_toro * 16
    freq_mask = frequencies <= max_freq_to_show
    log_amplitude = np.log10(amplitude_spectrogram[freq_mask, :] + 1e-4)
    # Построение тепловой карты
    mesh = ax.pcolormesh(
        times, 
        frequencies[freq_mask], 
        log_amplitude,
        #amplitude_spectrogram[freq_mask, :], 
        shading='gouraud', 
        cmap='inferno'
    )
    
    #plt.title(f'Эволюция гармонических моментов траектории {title_suffix}')
    #plt.xlabel('Время симуляции (t)')
    #plt.ylabel('Частота (f)')
    #plt.colorbar(mesh, label='Амплитуда гармоники')
    ax.grid(alpha=0.2, linestyle='--')



def save_harmonics(store: pd.HDFStore, it_num, frequencies, times, amplitude_spectrogram, f_toro, f_polo, coordinate_idx, is_angle, title_suffix=""):
    """
    Сохранение гармонических моменты в hdf.
    """

    # 4. Упаковка матрицы спектра в pandas.DataFrame
    # Строки — частоты, Столбцы — временные метки окон
    df_spec = pd.DataFrame(
        data=    amplitude_spectrogram,
        index=   frequencies,
        columns= times
    )
    key=f'harmonics/it_{it_num}'

    # Используем формат 'fixed', так как мы записываем матрицу целиком и не планируем append по строкам
    store.put(key, df_spec, format='fixed')
        
        # Сохраняем физические метаданные в атрибуты хранилища (storer)
    storer = store.get_storer(key) 
    storer.attrs.metadata = {
            'f_polo': f_polo,
            'f_toro': f_toro,
            'coordinate_idx': coordinate_idx,
            'is_angle': is_angle,
            'title_suffix': title_suffix
        }

    print(f"Данные сохранены в HDFStore под ключом '{key}'")

def load_from_hdf(filepath, it_num):
    """
    Загружает DataFrame спектра из pd.HDFStore и строит спектрограмму.
    """
    key=f'harmonics/it_{it_num}'
    # 1. Извлечение DataFrame и метаданных
    with pd.HDFStore(filepath, mode='r') as store:
        df_spec = store.get(key)
        # Считываем сохраненные атрибуты
        storer = store.get_storer(key)
        metadata = getattr(storer.attrs, 'metadata', {})
        coordinate_idx = metadata.get('coordinate_idx', 0)
        f_polo = metadata.get('f_polo', 0)
        f_toro = metadata.get('f_toro', 0)
        is_angle = metadata.get('is_angle', 0)
        title_suffix= metadata.get('title_suffix', 0)
    return df_spec, f_toro, f_polo, coordinate_idx, is_angle, title_suffix    