""" Model parameters of the coupled tank 

(c) Martin Doff-Sotta, University of Oxford (martin.doff-sotta@eng.ox.ac.uk)

"""
import numpy as np
g = 981                                     # Gravity acceleration (cm/s^2)
k_p = 3.3                                   # Pump gain (cm3 s-1 V-1)
d = 4.4                                     # Tank diameter (cm3)
A = d**2*np.pi/4                            # Tank area (cm2)
#v_0 = 7                                    # Estimated nominal pump voltage (V)
A1 = np.sqrt((k_p*7.3)**2/(2*g*16))         # sigma_1*a_1 (obtained experimentally)
A2 = np.sqrt((k_p*7.3)**2/(2*g*15))         # sigma_2*a_2 (obtained experimentally)
h2_0 = .1                                   # Initial height tank 2
h1_0 = .2                                   # Initial height tank 1 (or (A2/A1)**2*h2_0))
x_init = np.array([h1_0, h2_0])             # Initial state
h2_r = 15                                   # Reference height tank 2
h1_r = (A2/A1)**2*h2_r                      # Reference height tank 1
h_r = np.array([h1_r, h2_r])                # Reference state
u_r = np.array([A1/k_p*np.sqrt(2*g*h1_r)])  # Reference input (6.1 - 9.3)
u_init = u_r                                # Initial input
u_term = 1                                  # Terminal set bound on input 
x_term = 1                                  # Terminal set bound on state   
u_min = np.array([0])                       # min input
u_max = np.array([24])                      # max input
x_min = np.array([0.1, 0.1])                # min state
x_max = np.array([30, 30])                  # max state