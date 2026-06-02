import sys

import numpy as np
import pandas as pd
from numpy import cos, sin, pi
import matplotlib.pyplot as plt
from rich.console import Console    


from src.analyze_harmonics import load_from_hdf, plot_guiding_center_harmonics
from src.config import RunParams, SolverParams, load_configs, read_config_hdf5
from src.physical_constants import *
from src.utils import dict_to_str, get_dataset_sizes, nt_to_str, select_h5_file

#race_file = 'race/EXL-50U_13976/2026_05_04_22_20_06.h5'
race_file = select_h5_file() 
if race_file is None:
    sys.exit()

solver, params, cfg = read_config_hdf5(race_file)
sizes = get_dataset_sizes(race_file, ['trajectory', 'poincare_points'])
console = Console()
console.print(f"   [bold blue]Solver:[/bold blue] {nt_to_str(solver)}")
console.print(f"   [bold blue]Params:[/bold blue] {nt_to_str(params)}")
console.print(f"   [bold blue]Config:[/bold blue] {nt_to_str(cfg)}")
console.print(f"   [bold blue]Sizes:[/bold blue] {dict_to_str(sizes)}")

ccc_R0= ccc/params.R0
a = params.a
R0 = params.R0
n = params.n

from src.env import init_env
init_env(tau_norm*ccc_R0, R0, a)

for it in range(5):
    df_spec, f_toro, f_polo, coordinate_idx, is_angle, title_suffix = load_from_hdf(race_file, it)

    freqs = df_spec.index.values
    times = df_spec.columns.values
    spec = df_spec.values

    plot_guiding_center_harmonics(
        frequencies=freqs, 
        times=times, 
        amplitude_spectrogram=spec, 
        f_toro=f_toro,
        title_suffix= title_suffix
    )        