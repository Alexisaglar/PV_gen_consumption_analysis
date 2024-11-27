from pvlib import pvsystem
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.constants import PV_PARAMETERS
import time

total_percentage_output_technology = []

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

def pv_power_plot(data, label, season):
    for i, _ in enumerate(data):
        plt.plot(data[i], label=label[i])
        plt.title(f'power {season}')
    plt.legend()
    plt.show()

# def plot_power_output(sd, epv):
#     plt.plot(sd, label='sd')
#     plt.plot(epv, label='epv')
#     plt.legend()
#     plt.show()

def single_diode(irradiance, temperature):
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

    sd_output = curve_info['v_mp'] * curve_info['i_mp']

    return sd_output

def actual_pg(technology_output, season):
    # Calculate module power output with single diode diode model
    percentage_output_technology = []
    for i, _ in enumerate(technology_output):
        # print(f'this is the total data im giving: {technology_output}')
        pg_percentage =  technology_output[i] / ((single_diode(1000, 25)* 15) / 10000)
        # plt.plot(pg_percentage)
        # print(f'this is the single value in the array: {pg_percentage}')
        percentage_output_technology.append(pg_percentage)
        # print(f'this is the total appended in the array: {percentage_output_technology}')
        # plt.show()
    print(percentage_output_technology)
    plt.plot(np.arange(1, 25), percentage_output_technology[0], label='epv_percentage_potential')
    plt.plot(np.arange(1, 25), percentage_output_technology[1], label='epv_increased_percentage_potential')
    plt.plot(np.arange(1, 25), percentage_output_technology[2], label='sd_percentage_potential')
    plt.plot(np.arange(1, 25), percentage_output_technology[3], label='sd_percentage_increased_potential')
    plt.legend()
    plt.title(season)
    plt.show()
    return percentage_output_technology


def power_generation_pv(irradiance, temperature, season):
    # Calculate module power output with single diode diode model

    silicon_pv_power = single_diode(irradiance, temperature)
    pv_power_sd = np.array(silicon_pv_power * PV_PARAMETERS['series_cell'] * PV_PARAMETERS['parallel_cell']) / 10000
    pv_power_sd_decrease = pv_power_sd * delta_mat(irradiance, temperature, 'sd')

    # Calculating emerging PV possible power generation
    pv_power_epv = pv_power_sd * 0.75 * delta_mat(irradiance, temperature, 'epv')
    pv_power_epv_increased = pv_power_sd * delta_mat(irradiance, temperature, 'epv')

    # plotting pv output
    pv_power_plot([pv_power_epv, pv_power_epv_increased, pv_power_sd, pv_power_sd_decrease], ['epv', 'epv_increased', 'sd', 'sd_decreased'], season)

    # Calculating potential in % of energy output depending on season
    percentage_output_technology = actual_pg([pv_power_epv, pv_power_epv_increased, pv_power_sd, pv_power_sd_decrease], season)
    total_percentage_output_technology.append(percentage_output_technology)

    
    return pv_power_sd_decrease, pv_power_epv, pv_power_epv_increased, total_percentage_output_technology


if __name__ == '__main__':
    irradiance = np.arange(0, 1100, 100)
    temperature = np.full(11, 25)
    temperature_seasons = np.load('data/season_temperature.npy')
    irradiance_seasons = np.load('data/season_irradiance.npy')
    seasons = ['autumn', 'spring', 'summer', 'winter']
    for i in range(irradiance_seasons.shape[1]):
        pv_power_sd, pv_power_epv, pv_power_epv_increased, total_potential_energy = power_generation_pv(irradiance_seasons[:, i], temperature_seasons[:, i], seasons[i])
        # plot_power_output(pv_power_sd, pv_power_epv)
