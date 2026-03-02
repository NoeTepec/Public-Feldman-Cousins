import numpy as np 
import plot_tools
import matplotlib.pyplot as plt
import ast
import re

CIs = []

with open('Preliminary_data/CIs_60.txt', 'r') as file:
    for line in file:
        stripped_line = line.strip()  # Remove newline and extra whitespace
        if stripped_line:  # Skip empty lines
            CIs.append(ast.literal_eval(stripped_line))

print(len(CIs))


space = 0.5
mu = 0.0539
N_ints = len(CIs)
num = 0
height = N_ints*space + 2

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
plt.vlines(mu, 0, height, color='orange', label='True $F_H$', linestyle='--')
plt.title(f'{len(CIs)} CIs at 68% CL', loc='left')
plt.title(f'$\\beta=${beta}', loc='right')
plt.xlim(-0.2, 1.0)
plt.xlabel('$F_H$')
plt.legend()
plt.yticks([])
plt.savefig('Preliminary_data/Condor_CIs_CL_68.png')
plt.close()


sorted_CIs = sorted(CIs, key=lambda CI : CI[0])

#print(sorted_CIs)

space = 0.5
mu = 0.0539
N_ints = len(CIs)
#num = 0
height = N_ints*space +2

plt.figure()
for o in range(N_ints):

    plt.scatter(x=sorted_CIs[o], y=[space, space], s=4, color='blue')
    plt.hlines(y=space, xmin=sorted_CIs[o][0], xmax=sorted_CIs[o][1], color='blue')
    space+=0.5

#beta = np.round(num/N_ints, 3)
plt.vlines(mu, 0, height, color='orange', label='True $F_H$', linestyle='--')
plt.title(f'{len(CIs)} CIs at 68% CL', loc='left')
plt.title(f'$\\beta=${beta}', loc='right')
plt.xlim(-0.2, 1)
plt.xlabel('$F_H$')
plt.legend()
plt.yticks([])
plt.savefig('Preliminary_data/Condor_Sorted_CIs_CL_68.png')
plt.close()