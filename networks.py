#Author: Nand Chandravadia
#email: ndc2136@columbia.edu

import numpy as np
from math import ceil
from neuron import LIF

class LIF_NETWORK:
    def __init__(self, num_neurons, connection_probability, simulation_time):

        # ==== pre-defined parameters =========
        self.delta_t = 1e-4
        arr_len = ceil(simulation_time / self.delta_t)
        # =====================================

        self.num_neurons = num_neurons
        self.connection_probability = connection_probability
        self.simulation_time = simulation_time

        #intialize network connections
        self.network = np.array([[0, 1, 0],
                                 [0, 0, 1],
                                 [1, 0, 0]]) #self.initialize_network()      #network topology
        self.weights = np.multiply(np.random.normal(0, 1/self.num_neurons, size=(self.num_neurons, self.num_neurons)),
                                   self.network)
        self.lif_network = self.generate_lif_network()
        self.spike_times = np.zeros(shape = (self.num_neurons, arr_len))


    def initialize_network(self):

        #assumptions
        # 1. no autapse!

        n = self.num_neurons
        network = np.zeros(shape = (n, n))

        #initialize network connections
        for row_idx, row in enumerate(network):
            for col_idx, col in enumerate(row):
                if row_idx != col_idx: #no autapse!
                    isConnection = np.random.binomial(n=1, p=self.connection_probability)

                    if isConnection:
                        network[row_idx, col_idx] = 1

        return network

    def generate_lif_network(self):

        LIF_neurons = {}

        for neuron_i in range(1, self.num_neurons + 1):

            # ===== parameters ===================
            C_m = 100e-12
            V_rest = -70e-3
            V_reset = -80e-3
            G_l = 10e-9
            E_l = -70e-3
            V_th = -50e-3
            # =====================================

            nr = LIF(neuron_number = neuron_i, C_m = C_m, V_rest = V_rest, V_reset = V_reset, G_l = G_l, E_l = E_l,
                     V_th = V_th, network_simulation_time = self.simulation_time)

            LIF_neurons[neuron_i] = nr

            neuron_i += 1

        return LIF_neurons

    def simulate(self, I_app = 0.24e-9):

        # ===== we start by simulating neuron 1, alternatively we can randomly sample our network and sample a neuron
        neuron_start = 1

        nr = self.lif_network[neuron_start] #get neuron 1

        nr.simulate(t_start= 0, t_end = self.simulation_time, I_app = I_app, I_app_start = .15, I_app_end = .35) #simulate neuron 1

        self.update_spike_times(nr)


        for row_idx, row in enumerate(self.network):
            for col_idx, col in enumerate(row):
                if col_idx == 1: #we have a connection between neurons!



        test = 1




        return 0


    def update_spike_times(self, LIF_neuron):

        for spike_index, spiketime in enumerate(LIF_neuron.spike):

            if spiketime != 0: #we have a spike!
                self.spike_times[LIF_neuron.neuron_number, spike_index] = spiketime

        return None

a = LIF_NETWORK(num_neurons = 3, connection_probability=.3, simulation_time= 1)

a.simulate(I_app = 0.24e-9)

