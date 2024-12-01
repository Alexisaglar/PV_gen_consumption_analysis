import numpy as np
import time
import pandas as pd
import pandapower as pp
import pandapower.networks as nw
import networkx as nx
import matplotlib.pyplot as plt
from copy import deepcopy
import h5py
from torch.autograd import variable
from utils.constants import *
# from constants import COMMERCIAL_LOAD_FACTOR, COMMERCIAL_NODES, INDUSTRIAL_LOAD_FACTOR, INDUSTRIAL_NODES, RESIDENTIAL_LOAD_FACTOR, RESIDENTIAL_NODES 
import random

# Constants
re_configuration = False 
network = nw.case33bw()
NUM_NETWORKS_TO_SIMULATE = 100

class PowerFlowSimulator:
    def __init__(self, net):
        self.net = net
        self.load_factors = pd.DataFrame(data=[COMMERCIAL_LOAD_FACTOR, INDUSTRIAL_LOAD_FACTOR, RESIDENTIAL_LOAD_FACTOR]).T
        self.season_factor = pd.DataFrame(data=[WINTER_LOAD_FACTOR, SPRING_LOAD_FACTOR, SUMMER_LOAD_FACTOR, AUTUMN_LOAD_FACTOR]).T
        self.pv_potential = self.read_pv_potential('data/mean_power_potential.csv')
        self.successful_nets = []
        self.all_results = {}
        self.original_loads = self.net.load[['p_mw', 'q_mvar']].copy()
        self.run_simulation()

    def calculate_incidence_matrix(self):
        nodes = self.net.bus.index
        lines = self.net.line.index
        incidence_matrix = np.zeros((len(nodes), len(lines)))
        for idx, line in self.net.line.iterrows():
            incidence_matrix[line.from_bus, idx] = -1
            incidence_matrix[line.to_bus, idx] = 1
        return incidence_matrix

    def service_matrix(self):
        switch_matrix = np.zeros((len(self.net.line), len(self.net.line)))
        in_service = self.net.line['in_service']
        np.fill_diagonal(switch_matrix, in_service)
        return switch_matrix

    def is_network_radial(self, switch_matrix, incidence_matrix):
        AS_matrix = np.dot(incidence_matrix, switch_matrix)
        rank = np.linalg.matrix_rank(AS_matrix)
        if rank == len(self.net.bus.index) - 1:
            return True
        return False

    def configure_network(self):
        # Reset all lines to in-service to avoid cumulative deactivation from previous attempts
        self.net.line['in_service'] = True
        selected_lines = np.random.choice(self.net.line.index, size=5, replace=False)
        self.net.line.loc[selected_lines, 'in_service'] = False

    def run_simulation(self):
        incidence_matrix = self.calculate_incidence_matrix()
        while len(self.successful_nets) < NUM_NETWORKS_TO_SIMULATE:
            if re_configuration:
                self.configure_network()
            switch_matrix = self.service_matrix()
            if self.is_network_radial(switch_matrix, incidence_matrix):
                print("Configured a radial network. Now simulating loads...")
                seasonal_results = self.simulate_loads()
                if seasonal_results:
                    self.successful_nets.append(deepcopy(self.net))
                    self.all_results[len(self.successful_nets)] = seasonal_results
                    print(f"Successfully saved configuration {len(self.successful_nets)}.")
                    self.plot_network(self.net, len(self.successful_nets))  # Plot each successful network
                else:
                    print("Failed to converge for all seasons. Trying a new configuration...")
            else:
                print("Configured network is not radial.")

    def simulate_loads(self):
        # Drop not in service lines and not neccesary columns
        line_data = self.net.line[['from_bus', 'to_bus', 'length_km', 'r_ohm_per_km', 'x_ohm_per_km']]
        bus_data = self.net.bus[['vn_kv', 'max_vm_pu', 'min_vm_pu']]

        variable_NODE_TYPE = NODE_TYPE
        for i in range(len(NODE_TYPE)):
            if i > 0: 
                variable_NODE_TYPE[i] = random.choice(['industrial', 'residential', 'commercial'])

        vectorized_mapping = np.vectorize(CLASS_MAPPING.get)
        bus_data['node_type'] = vectorized_mapping(variable_NODE_TYPE)
        bus_data['grid_factor'] =  np.round(np.random.uniform(0.1,0.5), 2)
        
        # Save network configuration data
        seasonal_results = {'network_config': {
            # 'line': deepcopy(line_data.values.astype(float)),
            'line': deepcopy(line_data[self.net.line['in_service']].values),
            'bus': deepcopy(bus_data[self.net.bus['in_service']].values),
            'pv_potential': self.pv_potential
        }}

        for season in self.season_factor.columns:
            time_step_results = {}
            for time_step in range(self.load_factors.shape[0]):
                self.reset_and_apply_loads(time_step, season, variable_NODE_TYPE)
                try:
                    pp.runpp(self.net, verbose=True, numba=False)
                    if np.any((self.net.res_bus.vm_pu < 0.8) | (self.net.res_bus.vm_pu > 1.1)):
                        print(f"Voltage out of bounds at time step {time_step}, season {season}. Simulation aborted for this step.")
                        return None  # Skip saving this time ste
                    lfa_results = {
                        'res_bus': deepcopy(self.net.res_bus.values),
                        'res_line': deepcopy(self.net.res_line[self.net.line['in_service']].values)
                    }
                    print(lfa_results)
                    time_step_results[time_step] = lfa_results
                    # plt.plot(self.net.res_bus.vm_pu)
                    # plt.plot(self.net.res_bus.p_mw)
                    # plt.show()


                except pp.LoadflowNotConverged:
                    print(f'Load flow did not converge for time step {time_step}, season {season}.')
                    return None  # Terminate and return None to indicate failure

            seasonal_results[season] = time_step_results
            # plt.show()
            # print(f'season_{season}')
        return seasonal_results

    def reset_and_apply_loads(self, time_step, season, variable_NODE_TYPE):
        # Reset loads to original before applying scaling factors
        self.net.load['p_mw'] = self.original_loads['p_mw']
        self.net.load['q_mvar'] = self.original_loads['q_mvar']
        ######### adding a "single season" based increase factor
        self.net.load.loc[variable_NODE_TYPE[1:] == 'residential', 'p_mw'] *= RESIDENTIAL_LOAD_FACTOR[time_step]
        self.net.load.loc[variable_NODE_TYPE[1:] == 'commerical', 'p_mw'] *= COMMERCIAL_LOAD_FACTOR[time_step]
        self.net.load.loc[variable_NODE_TYPE[1:] == 'industrial', 'p_mw'] *= INDUSTRIAL_LOAD_FACTOR[time_step]

        # scaling reactive power (starting from [1:] to avoid slack bus)
        self.net.load.loc[variable_NODE_TYPE[1:] == 'residential', 'q_mvar'] *= RESIDENTIAL_LOAD_FACTOR[time_step]
        self.net.load.loc[variable_NODE_TYPE[1:] == 'commerical', 'q_mvar'] *= COMMERCIAL_LOAD_FACTOR[time_step]
        self.net.load.loc[variable_NODE_TYPE[1:] == 'industrial', 'q_mvar'] *= INDUSTRIAL_LOAD_FACTOR[time_step]
        
        print('*' *  100)
        # print(f'Total residential nodes: {np.sum(variable_NODE_TYPE == 'residential')},\nTotal commercial nodes: {np.sum(variable_NODE_TYPE == 'commercial')},\nTotal industrial nodes: {np.sum(variable_NODE_TYPE == 'industrial')}')
        #
        # print(f'Total p_mw: {self.net.load.p_mw.sum()}, Total q_mvar: {self.net.load.p_mw.sum()}')
        # self.net.load.loc[RESIDENTIAL_NODES[1:] == 1, 'q_mvar'] *= RESIDENTIAL_LOAD_FACTOR[time_step]
        # self.net.load.loc[COMMERCIAL_NODES[1:] == 1, 'q_mvar'] *= COMMERCIAL_LOAD_FACTOR[time_step]
        # self.net.load.loc[INDUSTRIAL_NODES[1:] == 1, 'q_mvar'] *= INDUSTRIAL_LOAD_FACTOR[time_step]


    def plot_network(self, net, config_number):
        graph = pp.topology.create_nxgraph(net)
        pos = nx.spring_layout(graph, k=1, iterations=1000)
        plt.figure(figsize=(10, 6))
        nx.draw_networkx(graph, pos, with_labels=True, node_color='black', node_size=300, font_color='white')
        plt.title(f'Power Network Topology - Configuration {config_number}')
        # plt.savefig(f'plots/Network_{config_number}', dpi=300)
        # plt.show()

    def save_results(self):
        with h5py.File('data/power_potential_networks.h5', 'w') as f:
            for net_id, net_data in self.all_results.items():
                net_group = f.create_group(f'network_{net_id}')
                static_group = net_group.create_group('network_config')
                static_group.create_dataset('line', data=net_data['network_config']['line'])
                static_group.create_dataset('bus', data=net_data['network_config']['bus'])
                
                for season, time_step_data in net_data.items():
                    if season == 'network_config':
                        continue
                    # time.sleep(0.1)
                    season_group = net_group.create_group(f'season_{season}')
                    for time_step, results in time_step_data.items():
                        print(f'network id: {net_id}, time step: {time_step}, season: {season}')
                        time_step_group = season_group.create_group(f'time_step_{time_step}')
                        time_step_group.create_dataset('res_bus', data=results['res_bus'])
                        time_step_group.create_dataset('res_line', data=results['res_line'])
                        # time_step_group.create_dataset('load', data=results['load'])

    def read_pv_potential(self, file):
        pv_potential = np.genfromtxt(
                    file,
                    delimiter=',',
                    dtype=None,  # Automatically determines the data type
                    encoding='utf-8',  # Handle text properly
                    names=True  # Use the first row as column names
                )# pv_potential = pd.read_csv(file)
        return pv_potential 


if __name__ == '__main__':
    simulator = PowerFlowSimulator(network)
    if simulator.successful_nets:
        simulator.save_results()

