from io_data import *
from masses import *
import numpy as np
from numpy import sign, log, power
import pandas as pd

inp = "inp/msfr_mix1_benchmark_burn"

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

    #print(error)

    if (error <= 1 and error >= -1): return p_in

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

    e21 = abs((phi1 - phi2) / phi1)

    cgi = (1.25 * e21) / (power(r21, p) - 1)

    return cgi

def main():

    ts = timesteps(inp)

    #Keff
    cgi_keff = []
    for i in range(1, 5):
        var = []
        for k in range(i, i+3):
            var.append(neutronic_output(f'res/m{k}_msfr_mix1_benchmark_burn_res.m', inp).keff)
        cgi_keff.append(cgi(var[0], var[1], var[2], i))
    cgi_keff.append(cgi(var[0], var[1], var[2], i))
    ind = ['Keff 1', '2', '3', '4', '5']
    cgi_keff = pd.DataFrame(cgi_keff, ind, ts)

    #Fir
    cgi_fir = []
    for i in range(1, 5):
        var = []
        for k in range(i, i+3):
            a = fir_values(f'dep/m{k}_msfr_mix1_benchmark_burn_dep.m', inp, len(ts))
            var.append(a)
        cgi_fir.append(cgi(var[0], var[1], var[2], i))
    cgi_fir.append(cgi(var[0], var[1], var[2], i))
    ind = ['Fir 1', '2', '3', '4', '5']
    cgi_fir = pd.DataFrame(cgi_fir, ind, ts)

    cgi_fir.to_excel(f'fir.xlsx')


if __name__ == "__main__":
    main()