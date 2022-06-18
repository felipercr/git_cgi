import re
import pandas as pd

#Represents a neutronic input file
class neutronic_input():

    #If the file already exists
    def __init__(self, name): 
        self.name = name
        self.__find_U_and_Th()

    #Get the values from U and Th from an existing input file
    def __find_U_and_Th(self):                            
        path_file = f"{self.name}"
        with open(path_file, 'r') as f:
            for line in f:
                if re.search("Th-232.09c", line):
                    self.Th = float(line.split()[1])
                if re.search("U-233.09c", line):
                    self.U  = float(line.split()[1])

#Represents the neutronic output file
class neutronic_output():
    def __init__(self, out_file, inp_file = None):
        self.out_file = out_file
        self.inp_file = inp_file
        self.__find_variables()

    #Get some variables from the output file
    def __find_variables(self):
        path_file = f"{self.out_file}"

        keff         = []
        keff_sd      = []
        beta_zero    = []
        beta_zero_sd = []
        gen_time     = []
        gen_time_sd  = []
        beta_eff     = []
        beta_eff_sd  = []

        with open(path_file, 'r') as file:

            for line in file:
                if re.search("ANA_KEFF", line):
                    keff.append(float(line.split()[6]))
                    keff_sd.append(float(line.split()[7]))
                    #self.keff = keff[0]
                    #self.keff_sd = keff_sd[0]

                if re.search("FWD_ANA_BETA_ZERO", line):
                    beta_zero.append(float(line.split()[6]))
                    beta_zero_sd.append(float(line.split()[7]))

                if re.search("ADJ_IFP_ANA_BETA_EFF", line):
                    beta_eff.append(float(line.split()[6]))
                    beta_eff_sd.append(float(line.split()[7]))

                if re.search("ADJ_IFP_GEN_TIME", line):
                    gen_time.append(float(line.split()[6]))
                    gen_time_sd.append(float(line.split()[7]))

        self.keff = keff
        self.keff_sd = keff_sd

        values = [
                    f"ANA_KEFF = {keff[0]} {keff_sd[0]}\n", 
                    f"FWD_ANA_BETA_ZERO = {beta_zero[0]} {beta_zero_sd[0]}\n", 
                    f"ADJ_IFP_ANA_BETA_EFF = {beta_eff[0]} {beta_eff_sd[0]}\n", 
                    f"ADJ_IFP_GEN_TIME = {gen_time[0]} {gen_time_sd[0]}\n",
                 ]

        data = [keff, beta_zero, gen_time, beta_eff, 
                keff_sd, beta_zero_sd, gen_time_sd, beta_eff_sd]

        if self.inp_file:
            self.plt_data = pd.DataFrame(
                    data, 
                    index=['keff', 'beta_zero', 'gen_time', 'beta_eff', 
                        'keff_sd', 'beta_zero_sd', 'gen_time_sd', 'beta_eff_sd'],
                    columns=timesteps(self.inp_file)
                )
            self.plt_data = self.plt_data.transpose()
            values = [
                        f"ANA_KEFF = {keff} {keff_sd}\n", 
                        f"FWD_ANA_BETA_ZERO = {beta_zero} {beta_zero_sd}\n", 
                        f"ADJ_IFP_ANA_BETA_EFF = {beta_eff} {beta_eff_sd}\n", 
                        f"ADJ_IFP_GEN_TIME = {gen_time} {gen_time_sd}\n",
                     ]
        #print(keff)
        self.values = values

def timesteps(file):
    with open(file, 'r') as f:
        for line in f:
            if re.search("\Adaystep", line):
                daystep = line.split()
    
    daystep = daystep[1:len(daystep)]

    years = []
    for x in range(len(daystep)+1):
        sum = 0
        for y in range(x):
            sum = sum + int(daystep[y])
        sum = sum / 365
        years.append(sum)

    return years