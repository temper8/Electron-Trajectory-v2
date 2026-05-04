import numpy as np
import matplotlib.pyplot as plt

def plot_r_phi_segments(df, max_segments=15, step=1, k = 1):
    """
    Рисует наложение участков траектории r(phi) друг на друга.
    
    Параметры:
    sol: объект решения solve_ivp
    R0: большой радиус токамака
    max_segments: сколько первых сегментов (оборотов) рисовать
    step: шаг отрисовки (например, рисовать каждый 10-й оборот)
    """

    # 1. Извлекаем данные
    phi_raw = df['phi']
     
    # Делаем phi непрерывным
    phi_cont = np.unwrap(phi_raw)

    # Считаем малый радиус r
    r = df['r']

    # 2. Находим границы оборотов (учитываем рост и убывание)
    # Считаем накопленное количество полных оборотов
    phi_dist = np.abs(phi_cont - phi_cont[0])
    n_turns = phi_dist // (2 * np.pi * k)
    
    # Индексы, где меняется номер оборота
    turn_indices = np.where(np.diff(n_turns) != 0)[0]

    if len(turn_indices) == 0:
        print("Предупреждение: Частица не совершила ни одного полного оборота.")
        return

    # 3. Отрисовка
    plt.figure(figsize=(10, 6))
    
    start_idx = 0
    plotted_count = 0
    
    # Проходим по индексам с заданным шагом
    for i in range(0, len(turn_indices), step):
        if plotted_count >= max_segments:
            break
            
        end_idx = turn_indices[i]
        
        # Срез данных
        phi_seg = phi_cont[start_idx:end_idx]
        r_seg = r[start_idx:end_idx]
        
        # Приводим к [0, 2pi]
        phi_wrapped = phi_seg + (2 * np.pi * k)*i
        
        # Сортируем для плавной линии
        #sort_mask = np.argsort(phi_wrapped)
        
        plt.plot(phi_wrapped, r_seg, alpha=0.7)
                 #label=f'Об. {int(n_turns[end_idx])}', alpha=0.7)
        
        start_idx = end_idx
        plotted_count += 1

    plt.xlabel('Тороидальный угол $\phi $\mod{2\pi}$ (рад)')
    plt.ylabel('Малый радиус $r$ (м)')
    plt.title(f'Наложение траекторий (первые {plotted_count} участков)')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    #plt.show()

def plot_radial_evolution(sol, R0):
    """
    Рисует развертку радиуса от количества оборотов (для анализа дрейфа).
    """
    phi_cont = np.unwrap(sol.y[1, :])
    r = np.sqrt((sol.y[0, :] - R0)**2 + sol.y[2, :]**2)
    
    plt.figure(figsize=(10, 4))
    plt.plot(phi_cont / (2 * np.pi), r)
    plt.xlabel('Номер оборота ($\phi / 2\pi$)')
    plt.ylabel('Малый радиус $r$ (м)')
    plt.title('Эволюция радиуса (дрейф)')
    plt.grid(True)
    plt.show()
