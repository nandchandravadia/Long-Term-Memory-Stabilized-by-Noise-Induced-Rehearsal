#Author: Nand Chandravadia
#email: ndc2136@columbia.edu

import numpy as np
from math import ceil


class LIF:
    def __init__(self, neuron_number, C_m, V_rest, V_reset, G_l, E_l, V_th, network_simulation_time):

        # ==== pre-defined parameters =========
        self.delta_t = 1e-4
        arr_len = ceil(network_simulation_time/self.delta_t)
        # =====================================

        # ==== inputs =========================
        self.neuron_number = neuron_number
        self.C_m = C_m          # membrane capacitance
        self.V0 = V_rest        # resting membrane potential (V0 = -70e-3)
        self.V_reset = V_reset  # reset membrane potential
        self.G_l = G_l          # membrane conductance
        self.E_l = E_l          # leak potential
        self.V_th = V_th        # threshold potential
        self.network_simulation_time = network_simulation_time #for LIF network
        # ===========================================

        self.V_m = self.V0*np.ones(shape = (1, arr_len ))               # membrane potential (through time)
        self.spike_times = np.zeros(shape = (1, arr_len ))       # record times of spikes (in s)
        self.t = np.arange(start=0, stop=network_simulation_time, step=self.delta_t)         # total simulation time
        self.I_app_time = np.zeros(shape = (1, arr_len))        # current amplitude (through time)
        self.G = np.zeros(shape = (1, arr_len))                 # synaptic conductance
        self.I_syn = np.zeros(shape = (1, arr_len))             #synaptic current

    @staticmethod
    def lif_model(self, V_i, I_app):

        dVdt = (self.G_l*(self.E_l - V_i) + I_app) * (self.C_m ** -1)

        return dVdt

    def compute_conductance(self, incoming_spikes):

        #====== pre-defined parameters ============
        delta_G = 1e-9  # 1nS
        T_syn = 100e-3  # 100ms
        # ==========================================

        index = 1
        for time in range(1, len(self.t)):

            G_i = self.G[0, index-1] - (self.G[0, index-1] / T_syn) * self.delta_t

            if time in incoming_spikes:
                G_i += delta_G

            self.G[0, index] = G_i

            index += 1

        return


    def compute_current(self):

        # ====== pre-defined parameters ============
        Vsyn = 0  # reversal potential
        # ==========================================


        for index, time in enumerate(self.t):
            self.I_syn[0, index] = self.G[0, index]*(Vsyn - self.V_m[0, index])

            #update current
            self.I_app_time[0, index] += self.I_syn[0, index]

        return

    def simulate(self, t_start= 0, t_end = .5, I_app = 0, I_app_start = .15, I_app_end = .35, incoming_spikes = []):

        if t_start == 0: #if initial simulation
            start_insertion_index = ceil(I_app_start/self.delta_t) #index of start current
            stop_insertion_index = ceil(I_app_end/self.delta_t)    #index of end current

            self.I_app_time[0, start_insertion_index:stop_insertion_index] = I_app

        #now, let's compute synaptic conductance upon the neuron
        self.compute_conductance(incoming_spikes)

        #now, let's compute the resulting synaptic current upon neuron
        self.compute_current()


        #now, solve for V(t) using the forward Euler Method (and record spikes)
        start_index = ceil(t_start/self.delta_t)
        end_index = ceil(t_end/self.delta_t)


        if t_start == start_index:
            index = 1
        else:
            index = start_index

        while start_index <= index < end_index:

            Vprev = self.V_m[0,index-1]

            I_app_i = self.I_app_time[0,index]

            dvdt = self.lif_model(self, Vprev, I_app_i)

            Vi = dvdt * self.delta_t + Vprev

            # did we induce a spike?
            if Vi > self.V_th:
                spiketime = self.t[index] #we got a spike
                self.spike_times[0, index] = spiketime

                self.V_m[0, index] = self.V_reset  # reset membrane potential
            else:
                self.V_m[0, index] = Vi

            index += 1

        return
