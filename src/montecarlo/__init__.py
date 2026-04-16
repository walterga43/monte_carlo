"""Top-level package for montecarlo."""
import numpy as np
import math      
import copy as cp 
from . bitstring import BitString     

""" BitString 
class BitString:
    def __init__(self, N):
        self.N = N
        self.config = np.zeros(N, dtype=int) 

    def __repr__(self):
        out = ""
        for i in self.config:
            out += str(i)
        return out

    def __eq__(self, other):        
        return all(self.config == other.config)
    
    def __len__(self):
        return len(self.config)

    def on(self):
        return self.config.count(1)

    def off(self):
        return self.config.count(0)

    def flip_site(self,i):
        if self.config[i] == 0:
            self.config[i] = 1
        else:
            self.config[i] = 0

    
    def integer(self):
        place = len(self.config) - 1
        number = 0
        for i in self.config:
            if i == 1:
                number = math.pow(2, place) + number
            place-=1
        return number 
 

    def set_config(self, s:list[int]):
        self.config = s

    def set_integer_config(self, dec:int):
        self.config = np.zeros(self.N, dtype=int)
        num = dec
        place = self.N - 1
        while num != 0:
            self.config[place] = num % 2
            place-=1
            num//= 2
        return self.config
"""
class IsingHamiltonian:
    def __init__(self, G):
        self.G = G
        self.J = np.asarray([self.G.edges[e].get('weight', 1.0) for e in self.G.edges], dtype=float)
        self.mu = np.zeros(self.G.number_of_nodes(), dtype=float)
    
    def energy(self, config:BitString):
        E = 0
        for e in self.G.edges:
            i, j = e
            if config.config[i] == 1 and config.config[j] == 1:
                E += self.G.edges[e]['weight'] * 1 * 1
            elif config.config[i] == 1 and config.config[j] == 0:
                E += self.G.edges[e]['weight'] * 1 * -1
            elif config.config[i] == 0 and config.config[j] == 1:
                E += self.G.edges[e]['weight'] * -1 * 1
            else:
                E += self.G.edges[e]['weight'] * -1 * -1

        for i in self.G.nodes:
            spin_i = 1 if config.config[i] == 1 else -1
            E += self.mu[i] * spin_i
        return E
        
    def set_mu(self, mus: np.array):
        mu_values = np.asarray(mus, dtype=float)
        if len(mu_values) != self.G.number_of_nodes():
            raise ValueError("Length of mus must match the number of graph nodes")
        self.mu = mu_values

    def compute_average_values(self, T: int):
        if T <= 0:
            raise ValueError("Temperature must be positive")

        E = 0.0
        M = 0.0
        Z = 0.0
        EE = 0.0
        MM = 0.0

        beta = 1.0 / T
        nsites = self.G.number_of_nodes()
        ncfg = 2 ** nsites

        conf = BitString(nsites)
        for idx in range(ncfg):
            conf.set_integer_config(idx)

            Ei = self.energy(conf)
            Mi = np.sum(conf.config) * 2 - nsites
            w = math.exp(-beta * Ei)

            Z += w
            E += w * Ei
            M += w * Mi
            EE += w * Ei * Ei
            MM += w * Mi * Mi

        E /= Z
        M /= Z
        EE /= Z
        MM /= Z

        HC = (EE - E * E) / (T * T)
        MS = (MM - M * M) / T

        return E, M, HC, MS

class MonteCarlo:
    def __init__(self, ham):
        self.Hamiltonian = ham

    def run(self, T, n_samples, n_burn):
        nsites = self.Hamiltonian.G.number_of_nodes()
        conf = BitString(nsites)
        conf.set_config(np.random.randint(0, 2, size=nsites))

        beta = 1.0 / T
        E_current = self.Hamiltonian.energy(conf)
        M_current = np.sum(conf.config) * 2 - nsites

        energies = np.zeros(n_samples, dtype=float)
        mags = np.zeros(n_samples, dtype=float)

        total_steps = n_burn + n_samples
        sample_idx = 0

        for step in range(total_steps):
            for i in range(nsites):
                old_bit = conf.config[i]
                conf.flip_site(i)

                E_new = self.Hamiltonian.energy(conf)
                delta_E = E_new - E_current

                accept = delta_E <= 0 or np.random.random() < math.exp(-beta * delta_E)
                if accept:
                    E_current = E_new
                    if old_bit == 0:
                        M_current += 2
                    else:
                        M_current -= 2
                else:
                    conf.flip_site(i)

            if step >= n_burn and sample_idx < n_samples:
                energies[sample_idx] = E_current
                mags[sample_idx] = M_current
                sample_idx += 1

        return energies, mags
        