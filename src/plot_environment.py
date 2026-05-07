

from matplotlib import pyplot as plt
import numpy as np

from src.field_EXL import get_field_environment


def plot_field_environment(time_array, race_name):
    """
    Визуализация параметров окружения плазмы.
    time_array: numpy.ndarray с моментами времени.
    race_name: строка с идентификатором расчета.
    """
    
    # Собираем данные. Если get_field_environment не векторизована,
    # используем list comprehension для прохода по массиву numpy
    results = [get_field_environment(t) for t in time_array]
    
    # Превращаем список словарей в структуру, удобную для numpy/matplotlib
    # Это позволяет избежать циклов при отрисовке
    q0 = np.array([r[0] for r in results])
    qa = np.array([r[1] for r in results])
    u_loop = np.array([r[2] for r in results])
    b0 = np.array([r[3] for r in results])

    fig, (ax_q, ax_u, ax_b) = plt.subplots(3, 1, figsize=(10, 7), sharex=True, num=f"{race_name}_field_environment")
    fig.suptitle(f'{race_name}', fontsize=12)

    # 1. Safety Factor
    ax_q.plot(time_array, q0, label='$q_0$ (axis)', lw=2)
    ax_q.plot(time_array, qa, label='$q_a$ (edge)', lw=1.5, ls='--')
    ax_q.set_ylabel('Safety Factor')
    ax_q.grid(True, which='both', alpha=0.2)
    ax_q.legend(frameon=False)

    # 2. Loop Voltage
    ax_u.plot(time_array, u_loop, color='crimson', lw=2)
    ax_u.set_ylabel('$U_{loop}$ [V]')
    ax_u.grid(True, which='both', alpha=0.2)

    # 3. Magnetic Field
    ax_b.plot(time_array, b0, color='forestgreen', lw=2)
    ax_b.set_ylabel('$B_0$ [T]')
    ax_b.set_xlabel('Time [s]')
    ax_b.grid(True, which='both', alpha=0.2)

    # Убираем лишние отступы
    plt.tight_layout()