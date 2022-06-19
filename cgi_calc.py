from io_data import *
from masses import *
import numpy as np
from numpy import sign, log, power
import pandas as pd
import os

inp_file = "inp/msfr_mix1_benchmark_burn"
years = timesteps(inp_file)

h1 = 6.58e-2
h2 = 8.42e-2
h3 = 1.07e-1
h4 = 1.36e-1
h5 = 1.84e-1
h6 = 2.69e-1
h = [h1, h2, h3, h4, h5, h6]

def calculate_p(epsilon32, epsilon21, r32, r21, s, q, p_out):

    if p_out != 0:
        a = (power(r21, p_out) - s) / (power(r32, p_out) - s)
        q = log(a)

    p_in = (1 / log(r21)) * (log(abs(epsilon32 / epsilon21)) + q)

    if p_in <= 1: p_in = 1
    elif p_in >= 2: p_in = 2 

    error = (abs((p_in - p_out)) / p_in) * 100

    if (error <= 0.01 and error >= -0.01): return p_in

    elif np.isnan(error): return 0

    else: return calculate_p(epsilon32, epsilon21, r32, r21, s, q, p_in)


# Coarsest to Finest -> M6 < M5 < M4 < M3 < M2 < M1
# Variables 1, 2, 3: vectors with the variables
# Mesh: number of the mesh we want the GCI
def cgi(variable1, variable2, variable3, mesh):

    phi1 = np.array(variable1)
    phi2 = np.array(variable2)
    phi3 = np.array(variable3)

    if mesh == 0: return 0

    #[0 to x] logic to [1 to x+1]
    mesh1 = mesh - 1
    mesh2 = mesh 
    mesh3 = mesh + 1

    r32 = h[mesh3] / h[mesh2]
    r21 = h[mesh2] / h[mesh1]

    epsilon32 = phi3 - phi2
    epsilon21 = phi2 - phi1
    
    s = sign(epsilon32 / epsilon21)

    p = []

    for i in range(len(phi1)):
        p.append(calculate_p(epsilon32[i], epsilon21[i], r32, r21, s[i], 0, 0))

    print(p)

    e21 = abs((phi1 - phi2) / phi1)

    cgi = (1.25 * e21) / (power(r21, p) - 1)

    for i, v in enumerate(cgi):
        if np.isnan(v): cgi[i] = 0

    return cgi

class cgi_values():
    def __init__(self, file_type, variable, mix):
        self.variable = variable
        self.mix = mix
        if file_type == 'dep': self.cgi_var = self.__dep()
        else:                  self.cgi_var = self.__res()
        
    def __dep(self):
        cgi_var = []
        for i in range(1, 5):
            var = []
            for k in range(i, i+3):
                var.append(neutronic_output(f'res/m{k}_msfr_mix{self.mix}_benchmark_burn_res.m', inp_file).keff)
            cgi_var.append(cgi(var[0], var[1], var[2], i))
        cgi_var.append(cgi(var[0], var[1], var[2], i))
        ind = [f'Keff_mix{self.mix} 1', '2', '3', '4', '5']
        cgi_var = pd.DataFrame(cgi_var, ind, years)
        return cgi_var
    
    def __res(self):
        cgi_var = []
        for i in range(1, 5):
            var = []
            for k in range(i, i+3):
                if self.variable == 'FIR':
                    a = fir_values(f'dep/m{k}_msfr_mix{self.mix}_benchmark_burn_dep.m', inp_file, len(years))
                elif self.variable == 'Ing.' or self.variable == 'Inh.':
                    a = toxicity(f'dep/m{k}_msfr_mix{self.mix}_benchmark_burn_dep.m', inp_file, len(years))
                else:
                    a = fuel_mass(f'dep/m{k}_msfr_mix{self.mix}_benchmark_burn_dep.m', inp_file, len(years))
                a = a[[f'{self.variable}']]
                a = a.to_numpy()
                a = np.transpose(a)
                a = list(a[0])
                var.append(a)     
            cgi_var.append(cgi(var[0], var[1], var[2], i))
        cgi_var.append(cgi(var[0], var[1], var[2], i))
        ind = [f'{self.variable}_mix{self.mix} 1', '2', '3', '4', '5']
        cgi_var = pd.DataFrame(cgi_var, ind, years)
        return cgi_var


def main():

    variables = [
        'keff',
        'Pa',
        'U',
        'Np',
        'Pu',
        'Am',
        'Cm',
        '232U',
        '233U',
        '231Pa',
        '238Pu',
        '239Pu',
        '240Pu',
        '241Pu',
        'FIR',
        'Inh.',
        'Ing.'
    ]

    cgi = []
    for item in variables:
        if item == 'keff':
            cgi.append(cgi_values('dep', item, 1).cgi_var)
        else:
            cgi.append(cgi_values('res', item, 1).cgi_var)

    cgi = pd.concat(cgi)

    if os.path.exists('cgi.xlsx'):
        os.remove('cgi.xlsx')
    cgi.to_excel('cgi.xlsx')
    
    

if __name__ == "__main__":
    main()