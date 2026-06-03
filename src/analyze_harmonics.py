import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import stft, find_peaks
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
    #signal = np.exp(-1j * raw_data) if is_angle else raw_data
    # 3. Настройка параметров окна Ханна
    # 1. Считаем базовый физический размер окна (строго 8 периодов)
    points_per_period = int(fs / f_polo)
    raw_nperseg = points_per_period * 64      
    
    # 2. Округляем до ближайшей честной степени 2 (например: 256, 512, 1024, 2048)
    nperseg = 1 << int(np.round(np.log2(raw_nperseg)))
    print(f"nperseg={nperseg}")
    #nperseg = points_per_period * 100       # Окно в 10 полоидальных периодов
    noverlap = int(nperseg * 0.75)         # Перекрытие 75%
    # 4. Расчет оконного преобразования Фурье
    frequencies, times, Zxx = stft(
        signal, 
        fs=fs, 
        window='hann', 
        #window='blackman',
        nperseg=nperseg, 
        noverlap=noverlap,
        boundary=None,   # <--- Убирает искусственное дополнение краев
        padded=False     # <--- Отключает падинг до ближайшей степени двойки
    )
    amplitude_spectrogram = np.abs(Zxx)
    # --- МАГИЯ ДЛЯ КОМПЛЕКСНОГО СИГНАЛА ---
    # Сдвигаем частоты и строки матрицы спектра, чтобы они шли от минус бесконечности к плюс бесконечности
    #frequencies = np.fft.fftshift(frequencies)
    #amplitude_spectrogram = np.fft.fftshift(amplitude_spectrogram, axes=0)
    return frequencies, times, amplitude_spectrogram


def plot_guiding_center_harmonics(frequencies, times, amplitude_spectrogram, f_toro, title_suffix=""):
    """
    Визуализирует рассчитанные гармонические моменты в виде спектрограммы.
    """
    plt.figure(figsize=(12, 6))
    
    # Ограничиваем график первыми четырьмя тороидальными гармониками
    #max_freq_to_show = f_toro * 16
    #freq_mask = frequencies <= max_freq_to_show
    #amp = np.log10(amplitude_spectrogram[freq_mask, :] + 1e-4)
    #amp = amplitude_spectrogram[freq_mask, :]
    # Построение тепловой карты
    mesh = plt.pcolormesh(
        times, 
        frequencies, 
        #amplitude_spectrogram,
        np.log10(amplitude_spectrogram + 1e-4),
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
    #max_freq_to_show = f_toro * 16
    #freq_mask = frequencies <= max_freq_to_show
    #amp = np.log10(amplitude_spectrogram[freq_mask, :] + 1e-4)
    #amp = amplitude_spectrogram[freq_mask, :]
    amp = np.log10(amplitude_spectrogram + 1e-4)
    #amp = amplitude_spectrogram
    # Построение тепловой карты
    mesh = ax.pcolormesh(
        times, 
        frequencies, 
        amp,
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

def approximate_peak_parabolic(spectrum_slice, frequencies, idx_max):
    """
    Выполняет параболическую аппроксимацию спектрального пика по 3 точкам.
    Находит суб-пиксельное (непрерывное) положение частоты и истинную амплитуду.
    
    Параметры:
    ----------
    spectrum_slice : ndarray
        Одномерный массив амплитуд спектра в текущий момент времени.
    frequencies : ndarray
        Массив дискретных частот БПФ.
    idx_max : int
        Индекс дискретного максимума в массивах.
        
    Возвращает:
    -----------
    f_true : float
        Аппроксимированная истинная частота пика.
    amp_true : float
        Аппроксимированная истинная амплитуда (высота купола).
    """
    # Защита от выхода за границы массива (если пик на самом краю спектра)
    if idx_max <= 0 or idx_max >= len(frequencies) - 1:
        return frequencies[idx_max], spectrum_slice[idx_max]
        
    # Извлекаем три точки: сам пик (beta) и его соседей слева (alpha) и справа (gamma)
    alpha = spectrum_slice[idx_max - 1]
    beta  = spectrum_slice[idx_max]
    gamma = spectrum_slice[idx_max + 1]
    
    # Знаменатель формулы (определяет кривизну параболы)
    denom = 2.0 * beta - alpha - gamma
    
    # Если знаменатель равен 0 (абсолютно плоский пик), сдвига нет
    if denom == 0:
        return frequencies[idx_max], beta
        
    # Расчет математического сдвига 'p' относительно центральной корзины (в долях шага БПФ)
    # Значение 'p' всегда лежит в строго пределах [-0.5, 0.5]
    p = 0.5 * (alpha - gamma) / denom
    
    # Шаг дискретизации сетки частот БПФ
    df = frequencies[1] - frequencies[0]
    
    # 1. Вычисляем истинную непрерывную частоту
    f_true = frequencies[idx_max] + p * df
    
    # 2. Вычисляем истинную амплитуду в вершине параболы
    #amp_true = beta - 0.25 * (alpha - gamma) * p
    amp_true = beta + 0.125 * ((alpha - gamma) ** 2) / denom
    return f_true, amp_true

def save_harmonic_peaks(store: pd.HDFStore, frequencies, times, amplitude_spectrogram, f_toro, f_polo, coordinate_idx, is_angle, title_suffix=""):
    """
    Сохранение максимумов гармонических моменты в hdf.
    """
    num_peaks = 10
    # Списки для сборки структурированных данных
    data_records = []
    
    # 2. Сканируем спектр по времени (столбец за столбцом)
    for idx, t in enumerate(times):
        spectrum_slice = amplitude_spectrogram[:, idx]
        
        # Ищем пики на спектре в этом временном окне
        # prominence помогает отсечь случайный шум
        peaks, props = find_peaks(spectrum_slice, prominence=0.001)

        if len(peaks) == 0:
            continue

        peak_freqs = frequencies[peaks]
        peak_amps = spectrum_slice[peaks]

                # Сортируем пики строго по возрастанию частоты
        sort_idx = np.argsort(peak_freqs)
        peaks = peaks[sort_idx]
        peak_freqs = peak_freqs[sort_idx]
        peak_amps = peak_amps[sort_idx]
        
        # Формируем строку таблицы для текущего момента времени
        record = {'time': t}
        for i in range(num_peaks):
            if i < len(peak_freqs):
                #record[f'f{i+1}'] = peak_freqs[i]
                #record[f'amp{i+1}'] = peak_amps[i]
                f_true, amp_true = approximate_peak_parabolic(spectrum_slice, frequencies, peaks[i])
                #print((f_true- peak_freqs[i])/(frequencies[1] - frequencies[0]), (amp_true-peak_amps[i]))
                record[f'f{i+1}'] = f_true
                record[f'amp{i+1}'] =amp_true
            else:
                # Если пиков обнаружилось меньше, чем запрошено, пишем NaN
                record[f'f{i+1}'] = np.nan
                record[f'amp{i+1}'] = np.nan
                
        data_records.append(record)
        
    key=f'harmonic_pieaks'
    # 3. Создание DataFrame и запись в HDFStore
    df_peaks = pd.DataFrame(data_records)
    #print(df_peaks.to_string())
    store.append(key, df_peaks, index=False)
    storer = store.get_storer(key) # type: ignore
    storer.attrs.metadata = {
        'num_peaks': num_peaks,
        'f_polo': f_polo,
        'f_toro': f_toro,
        'coordinate_idx': coordinate_idx,
        'is_angle': is_angle,
        'title_suffix': title_suffix            
    }
            
    


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

def load_and_plot_peaks(filepath, tau_factor, key='harmonic_pieaks'):
    """
    Загружает таблицу пиков из HDFStore и строит их частотные треки.
    """
    with pd.HDFStore(filepath, mode='r') as store:
        df_peaks = store.get(key)
        storer = store.get_storer(key)
        metadata = getattr(storer.attrs, 'metadata', {})
        num_peaks = metadata.get('num_peaks', 4)

    
    # 2. Создаем сетку из двух графиков (sharex=True связывает оси времени)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Цветовая палитра для соответствия линий на верхнем и нижнем графиках
    colors = plt.cm.tab10(range(num_peaks))

    # 3. Цикл по всем сохраненным пикам
    for i in range(1, num_peaks + 1):
        f_col = f'f{i}'
        amp_col = f'amp{i}'
        
        if f_col in df_peaks.columns and amp_col in df_peaks.columns:
            # Верхний график: Частота
            ax1.plot(
                df_peaks['time']/tau_factor, 
                df_peaks[f_col], 
                label=f'Пик {i}', 
                color=colors[i-1], 
                linewidth=1.5
            )
            
            # Нижний график: Амплитуда
            ax2.plot(
                df_peaks['time']/tau_factor, 
                df_peaks[amp_col], 
                color=colors[i-1], 
                linewidth=1.5
            )

    # 4. Оформление верхнего графика (Частота)
    ax1.set_title('Эволюция частот и амплитуд доминирующих пиков спектра', fontsize=12)
    ax1.set_ylabel('Частота (f)', fontsize=10)
    ax1.grid(alpha=0.3, linestyle='--')
    ax1.legend(loc='upper left')

    # 5. Оформление нижнего графика (Амплитуда)
    ax2.set_xlabel('Время симуляции (t)', fontsize=10)
    ax2.set_ylabel('Амплитуда (A)', fontsize=10)
    ax2.grid(alpha=0.3, linestyle='--')
    
    # Рекомендуется использовать логарифмический масштаб для амплитуды, 
    # если высшие гармоники сильно слабее первой (раскомментируйте при необходимости):
    # ax2.set_yscale('log')


