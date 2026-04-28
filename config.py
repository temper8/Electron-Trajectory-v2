import tomllib
from typing import NamedTuple

class RunConfig(NamedTuple):
    R0: float
    a: float
    delr: float
    delfi: float
    nfi: int
    n: int
    r: float
    thet: float
    fi: float
    ppar: float
    pperp: float


def get_tokamak(discharge_path):
    with open(discharge_path, "rb") as f:
        cfg = tomllib.load(f)
    return f"Tokamak: {cfg['tokamak']['name']}"

def load_configs(discharge_path):
    with open(discharge_path, "rb") as f:
        cfg = tomllib.load(f)
    return RunConfig(
        R0=    cfg['tokamak']['R0'],
        a=     cfg['tokamak']['a'],
        delr=  cfg['discharge']['perturbations']['delr'],
        delfi= cfg['discharge']['perturbations']['delfi'],
        nfi=   cfg['discharge']['perturbations']['nfi'],
        n=     cfg['discharge']['perturbations']['n'],
        r=     cfg['initial_conditions']['r'],
        thet=  cfg['initial_conditions']['thet'],
        fi=    cfg['initial_conditions']['fi'],
        ppar=  cfg['initial_conditions']['ppar'],
        pperp= cfg['initial_conditions']['pperp'],
    )

def info_string(cfg):
    info = f"R0 = {cfg.R0}, a = {cfg.a}, "
    info += f"delr = {cfg.delr}, delr = {cfg.delr}, "
    info += f"nfi = {cfg.nfi}"
    return info

