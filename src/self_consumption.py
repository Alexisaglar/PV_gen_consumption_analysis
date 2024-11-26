from sre_constants import IN
from matplotlib.transforms import blended_transform_factory
from networkx import nodes
from networkx.algorithms.bipartite import eppstein_matching
import numpy as np
from pandapower import results
from pandapower.plotting import powerflow_results
from src.constants import *
from src.pv_gen import power_generation_pv
import matplotlib.pyplot as plt
import time as time

from matplotlib import cm
from matplotlib.colors import LightSource
# files directory
temperature_seasons = np.load('data/season_temperature.npy')
irradiance_seasons = np.load('data/season_irradiance.npy')

nodes_load_profile = np.zeros((33,24))
INCREASE_COEFFICIENT = np.array([0.5,1,1.5,2])

def calculate_self_consumption(pv_power, peak_load):
    # Initialize variables
    state_of_charge = np.zeros((33,24))  
    power_battery = np.zeros((33, 24))
    nodes_results = np.zeros((33,24,5)) # node_results(:, :, 0) = power required, node_results(:, :, 1) = power available, node_results(:, :, 2) = state_of_charge, 
    power_available = np.zeros((33,24))
    power_required = np.zeros((33,24))
    power_home = np.zeros((33,24))
    power_no_storage = np.zeros((33,24))
    node_bat_cap = np.zeros(33)
    node_results_coefficient = np.zeros((33, 24, 5, 4))


    for i in range(len(peak_load)):
        if NODE_TYPE[i] == 'industrial':
            nodes_load_profile[i] = INDUSTRIAL_LOAD_FACTOR * peak_load[i] 

        if NODE_TYPE[i] == 'commercial':
            nodes_load_profile[i] = COMMERCIAL_LOAD_FACTOR * peak_load[i]

        else:
            nodes_load_profile[i] = RESIDENTIAL_LOAD_FACTOR * peak_load[i] 

    # assigning storage depending on their demand
    for i in range(len(nodes_load_profile)):
        node_bat_cap[i] = nodes_load_profile[i].max()

    for j in range(len(INCREASE_COEFFICIENT)):
        pv_power = pv_power * INCREASE_COEFFICIENT[j]
        for i in range(nodes_results.shape[0]):
            for t in range(nodes_results.shape[1]):
                if t == 0: state_of_charge[i, t-1] = 0 # do not add load to SLACK bus
                power_available[i, t] = ((state_of_charge[i, t-1] - SOC_MIN) * (node_bat_cap[i] * INCREASE_COEFFICIENT[j] / DELTA_TIME)) / DISCHARGE_EFF # Power available
                power_required[i, t] = (SOC_MAX - state_of_charge[i, t-1]) * (node_bat_cap[i] * INCREASE_COEFFICIENT[j] / DELTA_TIME) / CHARGE_EFF  # Power required

                if nodes_load_profile[i, t] > pv_power[t]:
                    power_battery[i, t] = max(pv_power[t] - nodes_load_profile[i, t], -power_available[i, t], -MAX_POWER_DISCHARGE)

                else:
                    power_battery[i, t] = min(pv_power[t] - nodes_load_profile[i, t], power_required[i, t], MAX_POWER_CHARGE)
                

                Z_bat = 1 if power_battery[i, t] > 0 else 0
                if Z_bat == 1 :
                    state_of_charge[i, t] = (state_of_charge[i, t-1] + (CHARGE_EFF * ((power_battery[i, t] / DELTA_TIME) / (node_bat_cap[i] * INCREASE_COEFFICIENT[j])) ))
                else:
                    state_of_charge[i, t] = state_of_charge[i, t-1] + (((power_battery[i, t] / DELTA_TIME * DISCHARGE_EFF ) / (node_bat_cap[i] * INCREASE_COEFFICIENT[j])))

                # Calculate P_h^t , P_H2G, P_G2H
                power_home[i, t] = nodes_load_profile[i, t] - pv_power[t] + power_battery[i, t]
                if nodes_load_profile[i, t] - pv_power[t] > 0:
                    power_no_storage[i, t] = nodes_load_profile[i, t] - pv_power[t]
                else:
                    power_no_storage[i, t] = 0

            
                print(f'Node {i}, time {t}, coefficent{j}:\n power_available = {power_available[i, t]},\n power_required ={power_required[i, t]},\n power_battery = {power_battery[i, t]},\n load = {nodes_load_profile[i, t]},\n pv_power = {pv_power[t]}, \n power_home = {power_home[i, t]},')

        # addding results to nodes_results 3d matrix 
        # nodes_results = results_to_matrix(power_home[i, :], power_battery[i, :], state_of_charge[i, :], nodes_load_profile[i, :], power_no_storage[i, :])


            nodes_results[i, :, 0] = power_home[i, :]
            nodes_results[i, :, 1] = power_battery[i, :]
            nodes_results[i, :, 2] = state_of_charge[i, :]
            nodes_results[i, :, 3] = nodes_load_profile[i, :]
            nodes_results[i, :, 4] = power_no_storage[i, :]

        node_results_coefficient[:,:,:,j] = nodes_results
    
    return  node_results_coefficient


def plot_consumption(epv_data, sdpv_data, epv_increased):
    # 22, 23, 24 are the nodes that I would like to create profiles for
    fig, ax = plt.subplots(2, 3, sharex='col', sharey='row')
    fig.suptitle('PV generation and consumption per node')

    # Silicon values
    ax[0][0].plot(sdpv_data[22, :, 4, 0], label=f'Silicon node22 cap 0')
    ax[1][0].plot(sdpv_data[22, :, 0, 0], label=f'Silicon-bess node22 cap 0')
    ax[0][1].plot(sdpv_data[23, :, 4, 0], label=f'Silicon node23 cap 0')
    ax[1][1].plot(sdpv_data[23, :, 0, 0], label=f'Silicon-bess node23 cap 0')
    ax[0][2].plot(sdpv_data[24, :, 4, 0], label=f'Silicon node23 cap 0')
    ax[1][2].plot(sdpv_data[24, :, 0, 0], label=f'Silicon-bess node23 cap 0')

    # Silicon values 2 times more cap
    ax[0][0].plot(sdpv_data[22, :, 4, 3], label=f'Silicon node22 cap 2')
    ax[1][0].plot(sdpv_data[22, :, 0, 3], label=f'Silicon-bess node22 cap 2')
    ax[0][1].plot(sdpv_data[23, :, 4, 3], label=f'Silicon node23 cap 2')
    ax[1][1].plot(sdpv_data[23, :, 0, 3], label=f'Silicon-bess node23 cap 2')
    ax[0][2].plot(sdpv_data[24, :, 4, 3], label=f'Silicon node23 cap 2')
    ax[1][2].plot(sdpv_data[24, :, 0, 3], label=f'Silicon-bess node23 cap 2')

    # Epv values
    ax[0][0].plot(epv_data[22, :, 4, 0], label=f'EPV node22 cap 0')
    ax[1][0].plot(epv_data[22, :, 0, 0], label=f'EPV-bess node22 cap 0')
    ax[0][1].plot(epv_data[23, :, 4, 0], label=f'EPV node22 cap 0')
    ax[1][1].plot(epv_data[23, :, 0, 0], label=f'EPV-bess node22 cap 0')
    ax[0][2].plot(epv_data[24, :, 4, 0], label=f'EPV node22 cap 0')
    ax[1][2].plot(epv_data[24, :, 0, 0], label=f'EPV-bess node22 cap 0')

    # EPV values 2 times more cap
    ax[0][0].plot(epv_data[22, :, 4, 3], label=f'EPV node22 cap 2')
    ax[1][0].plot(epv_data[22, :, 0, 3], label=f'EPV-bess node22 cap 2')
    ax[0][1].plot(epv_data[23, :, 4, 3], label=f'EPV node22 cap 2')
    ax[1][1].plot(epv_data[23, :, 0, 3], label=f'EPV-bess node22 cap 2')
    ax[0][2].plot(epv_data[24, :, 4, 3], label=f'EPV node22 cap 2')
    ax[1][2].plot(epv_data[24, :, 0, 3], label=f'EPV-bess node22 cap 2')


    # EPV improved
    ax[0][0].plot(epv_increased[22, :, 4, 0], label=f'EPV increased node22 cap 2')
    ax[1][0].plot(epv_increased[22, :, 0, 0], label=f'EPV-bess increased node22 cap 2')
    ax[0][1].plot(epv_increased[23, :, 4, 0], label=f'EPV increased node22 cap 2')
    ax[1][1].plot(epv_increased[23, :, 0, 0], label=f'EPV-bess increased node22 cap 2')
    ax[0][2].plot(epv_increased[24, :, 4, 0], label=f'EPV increased node22 cap 2')
    ax[1][2].plot(epv_increased[24, :, 0, 0], label=f'EPV-bess increased node22 cap 2')
    
    # loads of each node
    ax[0][0].plot(nodes_load_profile[22, :], label='load')
    ax[1][0].plot(nodes_load_profile[22, :], label='load')

    ax[0][1].plot(nodes_load_profile[23, :], label='load')
    ax[1][1].plot(nodes_load_profile[23, :], label='load')

    ax[0][2].plot(nodes_load_profile[24, :], label='load')
    ax[1][2].plot(nodes_load_profile[24, :], label='load')

    for i in range(2):
        for j in range(3):
            ax[i][j].legend()
            ax[i][j].plot(np.full(24,0), color='black')

    plt.show()


if __name__ == '__main__':

    # 22, 23, 24 are the nodes that I would like to create profiles for
    for s in range(irradiance_seasons.shape[1]):
        # calculate the PV power with for different seasons
        sdpv_power, epv_power, epv_increased = power_generation_pv(irradiance_seasons[:, s], temperature_seasons[:, s]) 

        # calculate node_results depending on the PV material
        node_results_sdpv = calculate_self_consumption(sdpv_power, PEAK_LOAD)

        node_results_epv = calculate_self_consumption(epv_power, PEAK_LOAD)

        node_results_epv_increased = calculate_self_consumption(epv_increased, PEAK_LOAD)

        plot_consumption(node_results_epv, node_results_sdpv, node_results_epv_increased)
