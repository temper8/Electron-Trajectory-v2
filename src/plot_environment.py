

from matplotlib import pyplot as plt
import numpy as np

from src.env import get_field_environment


def plot_field_environment(tau_array, time_scale, race_name):
    """
    Визуализация параметров окружения плазмы.
    tau_array: numpy.ndarray с моментами времени.
    race_name: строка с идентификатором расчета.
    """
    
    # Собираем данные. Если get_field_environment не векторизована,
    # используем list comprehension для прохода по массиву numpy
    results = [get_field_environment(t) for t in tau_array]
    
    # Превращаем список словарей в структуру, удобную для numpy/matplotlib
    # Это позволяет избежать циклов при отрисовке
    q0 = np.array([r[0] for r in results])
    qa = np.array([r[1] for r in results])
    u_loop = np.array([r[2] for r in results])
    b0 = np.array([r[3] for r in results])

    fig, (ax_q, ax_u, ax_b) = plt.subplots(3, 1, figsize=(10, 7), sharex=True, num=f"{race_name}_field_environment")
    fig.suptitle(f'{race_name}', fontsize=12)

    # 1. Safety Factor
    ax_q.plot(tau_array/time_scale, q0, label='$q_0$ (axis)', lw=2)
    ax_q.plot(tau_array/time_scale, qa, label='$q_a$ (edge)', lw=1.5, ls='--')
    ax_q.set_ylabel('Safety Factor')
    ax_q.grid(True, which='both', alpha=0.2)
    ax_q.legend(frameon=False)

    # 2. Loop Voltage
    ax_u.plot(tau_array/time_scale, u_loop, color='crimson', lw=2)
    ax_u.set_ylabel('$U_{loop}$ [V]')
    ax_u.grid(True, which='both', alpha=0.2)

    # Добавляем текст с min и max значениями U_loop
    u_min, u_max = u_loop.min(), u_loop.max()
    #text_info = f"min: {u_min:.7f} V\nmax: {u_max:.7f} V"
    text_info = f"min: {u_min} V\nmax: {u_max} V"
    # Размещаем текст в левом верхнем углу подзаголовка (координаты 0.02, 0.95 от осей)
    ax_u.text(0.02, 0.95, text_info, transform=ax_u.transAxes, 
              verticalalignment='top', fontsize=10, 
              bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # 3. Magnetic Field
    ax_b.plot(tau_array/time_scale, b0, color='forestgreen', lw=2)
    ax_b.set_ylabel('$B_0$ [T]')
    ax_b.set_xlabel('Time [s]')
    ax_b.grid(True, which='both', alpha=0.2)

    # Убираем лишние отступы
    plt.tight_layout()