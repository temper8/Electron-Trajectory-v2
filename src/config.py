import tomllib
from typing import NamedTuple

import pandas as pd


class RunParams(NamedTuple):
    R0: float
    a: float
    delr: float
    delfi: float
    nfi: int
    n: int
    r: float
    theta: float
    phi: float
    ppar: float
    pperp: float

class RunConfig(NamedTuple):
    tokamak_name: str
    shot_number: int
    time_start: float # [sec]
    num_it: int
    max_tau_step: float
    delta_tau: float


class SolverParams(NamedTuple):
    method: str
    rtol: float
    atol: float    




def load_configs(discharge_path):
    with open(discharge_path, "rb") as f:
        cfg = tomllib.load(f)
    params = RunParams(
        R0=    cfg['tokamak']['R0'],
        a=     cfg['tokamak']['a'],
        delr=  cfg['discharge']['perturbations']['delr'],
        delfi= cfg['discharge']['perturbations']['delfi'],
        nfi=   cfg['discharge']['perturbations']['nfi'],
        n=     cfg['discharge']['perturbations']['n'],
        r=     cfg['initial_conditions']['r'],
        theta=  cfg['initial_conditions']['theta'],
        phi=    cfg['initial_conditions']['phi'],
        ppar=  cfg['initial_conditions']['ppar'],
        pperp= cfg['initial_conditions']['pperp'],
    )
    solver = SolverParams(        
        method       = cfg['solver']['method'],
        rtol         = cfg['solver']['rtol'],
        atol         = cfg['solver']['atol']
    )
    run_config = RunConfig(
        tokamak_name = cfg['tokamak']['name'],
        shot_number  = cfg['discharge']['main']['shot_number'],
        time_start   = cfg['initial_conditions']['time_start'],
        num_it       = cfg['initial_conditions']['num_it'],
        max_tau_step = cfg['initial_conditions']['max_tau_step'],
        delta_tau    = cfg['initial_conditions']['delta_tau'],
    )
    return run_config, solver, params

def param_string(p:RunParams):
    info = f"R0 = {p.R0}, a = {p.a}, "
    info += f"delr = {p.delr}, delr = {p.delr}, "
    info += f"nfi = {p.nfi}"
    return info

def read_dict_hdf5(store, name):
    df = store[name]
    meta_dict = df.set_index('param')['value'].to_dict()
    return meta_dict

def read_config_hdf5(file):
    with pd.HDFStore(file, mode='r') as store:
        meta_dict = read_dict_hdf5(store, 'params')
        params = RunParams(**meta_dict)
        df_solver = store['solver']
        meta_dict = read_dict_hdf5(store, 'solver')
        solver = SolverParams(**meta_dict)
        meta_dict = read_dict_hdf5(store, 'config')
        config = RunConfig(**meta_dict)
        return solver, params, config