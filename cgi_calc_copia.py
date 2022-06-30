from io_data import *
from masses import *
import numpy as np
from numpy import sign, log, power
import pandas as pd
import os
import matplotlib.pyplot as plt

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

    #print(p)

    e21 = abs((phi1 - phi2) / phi1)

    cgi = (1.25 * e21) / (power(r21, p) - 1)

    for i, v in enumerate(cgi):
        if np.isnan(v): cgi[i] = 0

    return cgi

class cgi_values():
    def __init__(self, file_type, variable, mix):
        self.variable = variable
        self.mix = mix
        if file_type == 'res': self.cgi_var = self.__res()
        else:                  self.cgi_var = self.__dep()
        
    def __res(self):
        cgi_var = []
        for i in range(1, 5):
            var = []
            for k in range(i, i+3):
                if self.variable == 'keff':
                    a = neutronic_output(f'res/m{k}_msfr_mix{self.mix}_benchmark_burn_res.m', inp_file).keff

                elif self.variable == 'feedback':
                    keff = neutronic_output(f'res/m{k}_msfr_mix{self.mix}_benchmark_burn_res.m', inp_file).keff
                    keff_tmp = neutronic_output(f'res/m{k}_msfr_mix{self.mix}_benchmark_burn_temperature_res.m', inp_file).keff
                    keff_den = neutronic_output(f'res/m{k}_msfr_mix{self.mix}_benchmark_burn_density_res.m', inp_file).keff

                    keff = np.array(keff)
                    keff_tmp = np.array(keff_tmp)
                    keff_den = np.array(keff_den)

                    doppler_coef = abs((keff - keff_tmp)/300) * -1
                    density_coef = abs((keff - keff_den)/230) * -1

                    a = doppler_coef + density_coef
                    

                elif self.variable == 'doppler':
                    keff = neutronic_output(f'res/m{k}_msfr_mix{self.mix}_benchmark_burn_res.m', inp_file).keff
                    keff_tmp = neutronic_output(f'res/m{k}_msfr_mix{self.mix}_benchmark_burn_temperature_res.m', inp_file).keff

                    keff = np.array(keff)
                    keff_tmp = np.array(keff_tmp)

                    doppler_coef = abs((keff - keff_tmp)/300) * -1

                    a = doppler_coef

                elif self.variable == 'density':
                    keff = neutronic_output(f'res/m{k}_msfr_mix{self.mix}_benchmark_burn_res.m', inp_file).keff
                    keff_den = neutronic_output(f'res/m{k}_msfr_mix{self.mix}_benchmark_burn_density_res.m', inp_file).keff

                    keff = np.array(keff)
                    keff_den = np.array(keff_den)

                    density_coef = abs((keff - keff_den)/230) * -1

                    a = density_coef

                var.append(a)
            cgi_var.append(cgi(var[0], var[1], var[2], i))
        cgi_var.append(cgi(var[0], var[1], var[2], i))
        ind = [f'{self.variable}_mix{self.mix} 1', '2', '3', '4', '5']
        cgi_var = pd.DataFrame(cgi_var, ind, years)
        return cgi_var
    
    def __dep(self):
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
        # 'keff',
        # 'feedback',
        # 'doppler',
        # 'density',
        'Pa',
        'U',
        'Np',
        'Pu',
        'Am',
        'Cm',
        # '232U',
        # '233U',
        # '231Pa',
        # '238Pu',
        # '239Pu',
        # '240Pu',
        # '241Pu',
        # 'FIR',
        # 'Inh.',
        # 'Ing.'
    ]

    cgi = []
    for item in variables:
        if item == 'keff' or item == 'feedback' or item == 'doppler' or item == 'density':
            #cgi.append(cgi_values('res', item, 1).cgi_var)
            pass
        else:
            cgi.append(cgi_values('dep', item, 1).cgi_var)
            pass

    out_files = [
        'dep/msfr_mix1_benchmark_burn_dep.m',
        'dep/m1_msfr_mix1_benchmark_burn_dep.m',
        'dep/m2_msfr_mix1_benchmark_burn_dep.m',
        'dep/m3_msfr_mix1_benchmark_burn_dep.m',
        'dep/m4_msfr_mix1_benchmark_burn_dep.m',
        'dep/m5_msfr_mix1_benchmark_burn_dep.m',
        'dep/m6_msfr_mix1_benchmark_burn_dep.m',
    ]

    print(cgi)

    #No mesh
    years = timesteps(inp_file)
    mass = fuel_mass(out_files[0], inp_file, len(years))
    mass = mass[['Pa', 'U', 'Np', 'Pu', 'Am', 'Cm']]   

    bench_time = [0.206, 0.24, 1.956, 7.153, 8.214, 8.823, 9.64, 9.714, 10.176, 
                18.552, 19.086, 19.712, 20.247, 33.144, 49.128, 49.607, 49.741, 
                50.025, 57.585, 81.198, 84.034, 99.771, 99.809, 99.811, 99.889, 
                99.934, 100.11, 199.023, 199.093, 199.242, 199.243, 199.342, 199.668]

    plt.plot(mass, label = mass.columns)
    plt.gca().set_prop_cycle(None)
    plt.plot(bench_time, 
            [124.38, 124.63, 137.98, 139.82, 140.2, 140.42, 140.03, 139.99, 
            139.77, 135.85, 135.78, 135.68, 135.61, 133.74, 131.79, 131.73, 131.71, 
            131.68, 131.43, 130.64, 130.54, 130.02, 130.02, 130.02, 130.02, 130.02, 
            130.01, 128.93, 128.93, 128.92, 128.92, 128.92, 128.92], 
            '--', label = 'Pa (Benchmark)')
    plt.plot(bench_time, 
            [4911.8, 4918, 5231.6, 6308.6, 6554.5, 6611, 6687.6, 6694.6, 6738.4, 
            7583.1, 7640.4, 7657, 7671.2, 8022, 8478.9, 8483.2, 8484.4, 8487, 
            8554.9, 8770.6, 8796.9, 8808.6, 8808.7, 8808.7, 8808.7, 8808.8, 
            8808.9, 8883, 8883, 8883.1, 8883.1, 8883.2, 8883.5], 
            '--', label = 'U (Benchmark)')
    plt.plot(bench_time, 
            [0.021861, 0.022287, 0.057784, 1.03507, 1.8664, 2.6168, 4.1183, 
            4.2926, 4.66, 20.678, 22.739, 25.419, 26.096, 49.181, 107.868, 110.438, 
            111.168, 111.396, 117.615, 139.36, 142.23, 159.26, 159.31, 159.31, 159.31, 
            159.32, 159.34, 169.02, 169.02, 169.04, 169.04, 169.05, 169.08], 
            '--', label = 'Np (Benchmark)')
    plt.plot(bench_time, 
            [0.00027271, 0.00028113, 0.0012619, 0.11915, 0.30173, 0.51397, 
            1.05038, 1.12131, 1.6795, 12.929, 14.726, 17.154, 19.541, 49.466, 156.39, 
            161.88, 162.17, 162.81, 180.52, 249.23, 259.08, 321.21, 321.38, 321.39, 321.73, 
            321.74, 321.8, 353.34, 353.36, 353.41, 353.41, 353.45, 353.56],
            '--', label = 'Pu (Benchmark)')
    plt.plot(bench_time, 
            [0.14879, 0.14897, 0.1579, 0.18836, 0.19527, 0.19935, 0.20495, 0.20547, 
            0.20872, 0.27735, 0.28243, 0.2885, 0.29378, 0.45516, 0.78306, 0.7959, 0.79953, 
            0.80729, 1.04346, 2.3258, 2.5609, 4.369, 4.3747, 4.375, 4.3866, 4.3933, 4.4196, 
            6.3997, 6.4014, 6.405, 6.405, 6.4074, 6.4152], 
            '--', label = 'Am (Benchmark)')
    plt.plot(bench_time, 
            [0.016379, 0.016408, 0.017918, 0.023394, 0.024704, 0.025488, 0.026579, 
            0.026681, 0.02732, 0.041991, 0.043157, 0.044567, 0.045807, 0.088792, 0.20165, 0.20667, 
            0.2081, 0.21116, 0.31124, 1.04555, 1.2094, 2.712, 2.7173, 2.7176, 2.7285, 2.7348, 
            2.7384, 5.6867, 5.6896, 5.6959, 5.6959, 5.7001, 5.7138], 
            '--', label = 'Cm (Benchmark)')

    x = np.array(years)

    for i, v in enumerate(variables):
        y = np.array(mass[[v]]).transpose()[0]
        err = np.array(cgi[i])
        for k in range(5):
            plt.errorbar(x, y, err[k], linestyle = 'None', 
                    color = 'black', capsize = 3)
        pass

    plt.yscale('log')
    plt.xlim(0, 200)

    if mass.to_numpy().max() > 8884:
        ymax = mass.to_numpy().max()
    else:
        ymax = 8884

    y_upper_lim = 1

    while ymax > y_upper_lim:
        y_upper_lim *= 10

    plt.ylim(1, y_upper_lim)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.xlabel('Operation Time (Years)')
    plt.ylabel('Fuel Salt Inventory (Kg)')
    plt.savefig('vazao_certa.png', bbox_inches='tight')
    plt.clf()

    ### PLOT ###

    # out_files = [
    #     'res/msfr_mix1_benchmark_burn_res.m',
    #     'res/m1_msfr_mix1_benchmark_burn_res.m',
    #     'res/m2_msfr_mix1_benchmark_burn_res.m',
    #     'res/m3_msfr_mix1_benchmark_burn_res.m',
    #     'res/m4_msfr_mix1_benchmark_burn_res.m',
    #     'res/m5_msfr_mix1_benchmark_burn_res.m',
    #     'res/m6_msfr_mix1_benchmark_burn_res.m',
    # ]

    # out_files_tmp = [
    #     'res/msfr_mix1_benchmark_burn_temperature_res.m',
    #     'res/m1_msfr_mix1_benchmark_burn_temperature_res.m',
    #     'res/m2_msfr_mix1_benchmark_burn_temperature_res.m',
    #     'res/m3_msfr_mix1_benchmark_burn_temperature_res.m',
    #     'res/m4_msfr_mix1_benchmark_burn_temperature_res.m',
    #     'res/m5_msfr_mix1_benchmark_burn_temperature_res.m',
    #     'res/m6_msfr_mix1_benchmark_burn_temperature_res.m',
    # ]

    # out_files_den = [
    #     'res/msfr_mix1_benchmark_burn_density_res.m',
    #     'res/m1_msfr_mix1_benchmark_burn_density_res.m',
    #     'res/m2_msfr_mix1_benchmark_burn_density_res.m',
    #     'res/m3_msfr_mix1_benchmark_burn_density_res.m',
    #     'res/m4_msfr_mix1_benchmark_burn_density_res.m',
    #     'res/m5_msfr_mix1_benchmark_burn_density_res.m',
    #     'res/m6_msfr_mix1_benchmark_burn_density_res.m',
    # ]

    # cgi = np.array(cgi_values('res', 'feedback', 1).cgi_var)
    # print(np.array(cgi))

    # for i in range(5):
    #     keff = np.array(neutronic_output(out_files[i], inp_file).plt_data[['keff']]).transpose()
    #     keff_tmp = np.array(neutronic_output(out_files_tmp[i], inp_file).plt_data[['keff']]).transpose()
    #     keff_den = np.array(neutronic_output(out_files_den[i], inp_file).plt_data[['keff']]).transpose()

    #     doppler = abs((keff - keff_tmp)/300) * -1
    #     density = abs((keff - keff_den)/230) * -1

    #     feedback = doppler + density

    #     feedback = pd.DataFrame(
    #             feedback,
    #             columns = years
    #     )

    #     feedback = feedback.transpose()

    #     plt.plot(feedback, '.-', label = f'Mesh {i}')

    #     print(cgi[i])

    #     cgi[i] = cgi[i]/300

    #     plt.errorbar(np.array(years), np.array(feedback).transpose()[0], cgi[i],
    #                  linestyle = 'None', color = 'black', capsize = 3)

    # plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    # plt.xscale('log')
    # plt.xlabel('Operation Time (Years)')
    # plt.ylabel('Feedback Coefficient (pcm/K)')
    # plt.savefig('keff_feedback_msh.png', bbox_inches='tight')
    # plt.clf()

    ### PLOT ###

    #cgi = pd.concat(cgi)

    #if os.path.exists('cgi.xlsx'):
    #    os.remove('cgi.xlsx')
    #cgi.to_excel('cgi.xlsx')
    
    

if __name__ == "__main__":
    main()