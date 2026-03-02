import numpy as np 
import plot_tools
import matplotlib.pyplot as plt
import math
import ast
import re
import SLSQP_zfit

# Open the file in read mode
#file = open("CIs.txt", "r")

# Read the first line
#line = file.readline()

#preCIs = []

#text_file = open("CIs.txt", "r")
#lines = text_file.readlines()
#print lines
#print len(lines)
#text_file.close()

CIs = []

with open("CIs.txt", "r") as f:
    for line in f:
        # Remove "np.float(...)" using regex
        cleaned_line = re.sub(r'np\.float\d*\s*\(\s*([^\)]+)\s*\)', r'\1', line.strip())
        
        #print(cleaned_line)
        # Evaluate the cleaned string into a Python object
        parsed = ast.literal_eval(cleaned_line)  # This gives something like [[1.0, 2.0]]
        
        # Flatten one level if needed
        if len(parsed) == 1 and isinstance(parsed[0], list):
            parsed = parsed[0]
        
        CIs.append(parsed)


space = 0.5
mu = 1.5
N_ints = len(CIs)
num = 0
height = N_ints*space

plt.figure()
for o in range(N_ints):

    u_bound = CIs[o][1]
    l_bound = CIs[o][0]

    if l_bound <= mu <= u_bound:
        num+=1
    else:
        pass

    plt.scatter(x=CIs[o], y=[space, space], s=4, color='blue')
    plt.hlines(y=space, xmin=CIs[o][0], xmax=CIs[o][1], color='blue')
    space+=0.5

beta = np.round(num/N_ints, 3)
plt.vlines(mu, 0, height, color='purple', label='$\\mu_v$', linestyle='--')
plt.title('CIs at 90% CL', loc='left')
plt.title(f'$\\beta=${beta}', loc='right')
plt.xlim(1.35, 1.65)
plt.xlabel('$\\mu$')
plt.legend()
plt.yticks([])
plt.savefig('Test_results/ExtendedKFC/Condor_Confidence_intervals_CL_90.png')
plt.close()

sorted_CIs = sorted(CIs, key=lambda CI : CI[0])

#print(sorted_CIs)

space = 0.5
mu = 1.5
N_ints = len(CIs)
#num = 0
height = N_ints*space

plt.figure()
for o in range(N_ints):

    plt.scatter(x=sorted_CIs[o], y=[space, space], s=4, color='blue')
    plt.hlines(y=space, xmin=sorted_CIs[o][0], xmax=sorted_CIs[o][1], color='blue')
    space+=0.5

#beta = np.round(num/N_ints, 3)
plt.vlines(mu, 0, height, color='purple', label='$\\mu_v$', linestyle='--')
plt.title('CIs at 90% CL', loc='left')
plt.title(f'$\\beta=${beta}', loc='right')
plt.xlim(1.35, 1.65)
plt.xlabel('$\\mu$')
plt.legend()
plt.yticks([])
plt.savefig('Test_results/ExtendedKFC/Sorted_CIs_CL_90.png')
plt.close()


