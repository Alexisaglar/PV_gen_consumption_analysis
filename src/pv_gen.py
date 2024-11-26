from pvlib import pvsystem
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.constants import PV_PARAMETERS



def beta(temperature):
    beta = (PV_PARAMETERS['sd_t_c'] - PV_PARAMETERS['epv_t_c']) * temperature 
    return beta 

    
def phi(irradiance, technology):
    phi = (PV_PARAMETERS[f'pce_@0sun_{technology}'] + ((PV_PARAMETERS[f'pce_@1sun_{technology}'] - PV_PARAMETERS[f'pce_@0sun_{technology}']) / 1000 ) * irradiance)
    return phi

def delta_mat(irradiance, temperature, technology):
    delta_mat = (beta(temperature) + phi(irradiance, technology))/ PV_PARAMETERS[f'pce_@1sun_{technology}']
    print(delta_mat)
    return delta_mat

def power_generation_pv(irradiance, temperature):
    I_L, I_0, R_s, R_sh, nNsVth = pvsystem.calcparams_desoto(
        effective_irradiance = irradiance,
        temp_cell = temperature,
        alpha_sc = PV_PARAMETERS['alpha_sc'],
        a_ref = PV_PARAMETERS['a_ref'],
        I_L_ref = PV_PARAMETERS['I_L_ref'],
        I_o_ref = PV_PARAMETERS['I_o_ref'],
        R_sh_ref = PV_PARAMETERS['R_sh_ref'],
        R_s = PV_PARAMETERS['R_s'],
        EgRef = PV_PARAMETERS['EgRef'],
        dEgdT = PV_PARAMETERS['dEgdT'],
    )
    
    curve_info = pvsystem.singlediode(
        photocurrent=I_L,
        saturation_current=I_0,
        resistance_series=R_s,
        resistance_shunt=R_sh,
        nNsVth=nNsVth,
        method='lambertw'
    )
    # Calculate module power output with single diode diode model
    pv_power_sd = curve_info['v_mp'] * curve_info['i_mp'] 
    pv_power_sd = np.array(pv_power_sd * PV_PARAMETERS['series_cell'] * PV_PARAMETERS['parallel_cell']) / 10000
    pv_power_sd_decrease = pv_power_sd * delta_mat(irradiance, temperature, 'sd')

    # Calculating emerging PV possible power generation
    # P_EPV = P_SG * delta_mat
    # new_pv_power_sd = pv_power_sd * 0.70
    pv_power_epv = pv_power_sd * 0.75 * delta_mat(irradiance, temperature, 'epv')
    pv_power_epv_increased = pv_power_sd * delta_mat(irradiance, temperature, 'epv')
    # plot_power_output(pv_power_sd, pv_power_epv)
    plt.plot(pv_power_sd, label = 'sd')
    plt.plot(pv_power_sd_decrease, label='sd_decrease')
    plt.plot(pv_power_epv, label='epv')
    plt.plot(pv_power_epv_increased, label='epv_increased')
    plt.legend()
    plt.show()
    plt.plot(irradiance)
    plt.show()
    
    return pv_power_sd_decrease, pv_power_epv, pv_power_epv_increased

def plot_power_output(sd, epv):
    plt.plot(sd, label='sd')
    plt.plot(epv, label='epv')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    irradiance = np.arange(0, 1100, 100)
    temperature = np.full(11, 25)
    print(temperature)
    pv_power_sd, pv_power_epv, pv_power_epv_increased = power_generation_pv(irradiance, temperature)
    # plot_power_output(pv_power_sd, pv_power_epv)
