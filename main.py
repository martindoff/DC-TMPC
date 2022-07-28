""" DC-TMPC: A tube-based MPC algorithm for systems that can be expressed as a difference 
of convex functions. 

Application: coupled water tank

dh1/dt  = -A1/A sqrt(2g h1) + k/A u
dh2/dt  = A1/A sqrt(2g h1) - A2/A sqrt(2g h2)

This program computes a control law to robustly stabilise a water tank system according to
the DC-TMPC algorithm in the paper: 'Difference of convex functions in robust tube
nonlinear MPC' by Martin Doff-Sotta and Mark Cannon. 

(c) Martin Doff-Sotta, University of Oxford (martin.doff-sotta@eng.ox.ac.uk)

"""

import numpy as np
from scipy.linalg import block_diag
import matplotlib.pyplot as plt 
import matplotlib.patches as patches
import cvxpy as cp
import mosek
import time
import os
import param_init as param
from tank_model import linearise, f, f1, f2, terminal
from control_custom import eul, dlqr, dp  

##########################################################################################
#################################### Initialisation ######################################
##########################################################################################

# Solver parameters
N = 50                                         # horizon 
T = 50                                         # terminal time
delta = T/N                                    # time step
tol1 = 10e-3                                   # tolerance 
alpha, beta = 1, .1                            # objective penalty parameters  
maxIter = 5                                    # max number of iterations

# Variables initialisation
N_state = param.x_init.size                    # number of states
N_input = param.u_init.size                    # number of inputs
x = np.zeros((N_state, N+1))                   # state
x[:, 0] =  param.x_init
u = np.zeros((N_input, N))                     # control input
u_0 = param.u_init*np.ones((N_input, N))       # (feasible) guess control input                     
x_0 = np.zeros((N_state, N+1))                 # (feasible) guess trajectory
x_r = np.ones_like(x)*param.h_r[:, None]       # reference trajectory 
t = np.zeros(N+1)                              # time vector 
K = np.zeros((N+1, N_input, N_state))          # gain matrix 
Phi1 = np.zeros((N+1, N_state, N_state))       # closed-loop state transition matrix of f1
Phi2 = np.zeros((N+1, N_state, N_state))       # closed-loop state transition matrix of f2
real_obj = np.zeros((N, maxIter+1))            # objective value
X_0 = np.zeros((N, maxIter+1, N_state, N+1))   # store guess trajectories 
S_low = np.zeros((N, maxIter+1, N_state, N+1)) # store perturbed state (lower bound)
S_up = np.zeros((N, maxIter+1, N_state, N+1))  # store perturbed state (upper bound)
S = np.zeros((N, maxIter+1, N_state, N+1))     # store perturbed state    

# Terminal set computation
C = np.array([[0, alpha]])
Q = C.T @ C
R = beta*np.eye(N_input)
Q_N, gamma_N, K_hat = terminal(param, Q, R, delta)

##########################################################################################
####################################### TMPC loop ########################################
##########################################################################################

for i in range(N):

    print("Computation at time step {}/{}...".format(i+1, N)) 
    
    # Guess trajectory update 
    x_0[:, :-1] = eul(f, u_0[:, :-1], x[:, i], delta, param)
    u_0[:, -1] = K_hat @ ( x_0[:,-2, None]  - x_r[:, -2, None]) + param.u_r  # terminal u
    x_0[:, -1]  = x_0[:, -2] + delta*(f(x_0[:, -2], u_0[:, -1] , param))     # terminal x 
    
    # Iteration
    k = 0 
    real_obj[i, 0] = 5000 
    delta_obj = 5000
    print('{0: <6}'.format('iter'), '{0: <5}'.format('status'), 
          '{0: <18}'.format('time'), '{}'.format('cost'))
    while real_obj[i, k] > tol1 and k < maxIter and delta_obj > 0.1:
        
        # Linearise system at x_0, u_0 
        A1, B1, A2, B2 = linearise(x_0, u_0, delta, param)   
        A = A1 - A2
        B = B1 - B2
        
        # Compute K matrix (using dynamic programming)
        P = Q_N
        for l in reversed(range(N)): 
            K[l, :, :], P = dp(A[l, :, :], B[l, :, :], Q, R, P)
            Phi1[l, :, :] = A1[l, :, :] +  B1[l, :, :] @ K[l, :, :]
            Phi2[l, :, :] = A2[l, :, :] +  B2[l, :, :] @ K[l, :, :]   
        
        # State transition of the closed loop
        Phi = Phi1 - Phi2
        
        ##################################################################################
        ############################ Optimisation problem ################################
        ##################################################################################
        
        N_ver = 2**N_state                     # number of vertices 
        
        # Optimisation variables
        theta = cp.Variable(N+1)               # cost 
        v = cp.Variable((N_input, N))          # input perturbation 
        s_low = cp.Variable((N_state, N+1))    # state perturbation (lower bound)
        s_up = cp.Variable((N_state, N+1))     # state perturbation (upper bound)
        s_ = {}                                # create dictionary for 3D variable 
        for l in range(N_ver):
            s_[l] = cp.Expression

        # Define blockdiag matrices for page-wise matrix multiplication
        K_ = block_diag(*K[:-1,:,:])
        Phi1_ = block_diag(*Phi1[:-1,:,:])
        Phi2_ = block_diag(*Phi2[:-1,:,:])
        B1_ = block_diag(*B1[:-1,:,:])
        B2_ = block_diag(*B2[:-1,:,:])
        
        # Objective
        objective = cp.Minimize(cp.sum(theta))
        
        # Constraints
        constr = []
        
        # Assemble vertices
        s_[0] = s_low
        s_[1] = s_up
        s_[2] = cp.vstack([s_low[0, :], s_up[1, :]])
        s_[3] = cp.vstack([s_up[0, :], s_low[1, :]])
    
        for l in range(N_ver):
            # Define some useful variables
            s_r = cp.reshape(s_[l][:, :-1], (N_state*N,1))
            v_r = cp.reshape(v, (N_input*N,1))
            K_s = (K_ @ s_r).T 
            Phi1_s = cp.reshape(Phi1_ @ s_r, ((N_state, N)))
            Phi2_s = cp.reshape(Phi2_ @ s_r, ((N_state, N)))
            B1_v = cp.reshape(B1_ @ v_r, (N_state, N))
            B2_v = cp.reshape(B2_ @ v_r, (N_state, N))
            
            # SOC objective constraints 
            constr += [theta[:-1] >= alpha*cp.square(s_[l][1,:-1]+x_0[1,:-1]-x_r[1,:-1])\
                                    + beta*cp.square(v + u_0 + K_s - param.u_r)[0, :]]
                                    
            constr += [theta[-1] >= cp.quad_form(s_[l][:,-1] + x_0[:,-1] - x_r[:,-1],Q_N)]
            
            # Input constraints  
            constr += [v + u_0 + K_s >= param.u_min,
                       v + u_0 + K_s  <= param.u_max]
            
            # Tube          
            constr += [s_low[:, 1:] <= Phi1_s + B1_v\
                       - f2(x_0[:, :-1] + s_[l][:, :-1], delta, param)\
                       + f2(x_0[:, :-1], delta, param)]
            
            constr += [s_up[:, 1:] >= -Phi2_s - B2_v\
                       + f1(x_0[:, :-1] + s_[l][:, :-1], v + u_0 + K_s, delta, param)\
                       - f1(x_0[:, :-1], u_0, delta, param)]
                  
        # State constraints
        constr += [s_low[:, :-1] + x_0[:, :-1] >= param.x_min[:, None],
                  s_up[:, :-1] + x_0[:, :-1] >= param.x_min[:, None],
                  s_up[:, :-1] + x_0[:, :-1]  <= param.x_max[:, None], 
                  s_low[:, :-1] + x_0[:, :-1]  <= param.x_max[:, None],
                  s_low[:, 0] == x[:, i] - x_0[:, 0], 
                  s_up[:, 0]  == x[:, i] - x_0[:, 0]] 
                                 
        # Terminal set constraint 
        constr += [gamma_N >= theta[-1]]
        
        # Solve problem
        problem = cp.Problem(objective, constr)
        t_start = time.time()
        problem.solve(solver = cp.MOSEK, verbose=False)
        
        print('{0: <5}'.format(k+1), '{0: <5}'.format(problem.status), 
              '{0: <5.2f}'.format(time.time()-t_start), '{0: <5}'.format(problem.value))
        if problem.status not in ["optimal"] and k > 0:
            print("Problem status {} at iteration k={}".format(problem.status, k))
            break
        
        ##################################################################################
        ############################### Iteration update #################################
        ##################################################################################
        
        # Save variables 
        S_low[i, k, :, :] = s_low.value.copy()
        S_up[i, k, :, :] = s_up.value.copy()
        X_0[i, k, :, :] = x_0.copy()
        x_0_old = x_0.copy()
        
        # Input and state update
        s = np.zeros((N_state, N+1))
        s[:, 0] = x[:, i] - x_0[:, 0]  # implies s_0 = 0
        Ks = np.zeros_like(v.value)
        for l in range(N):
            Ks[:, l] =   K[l, :, :] @ s[:, l, None]
            u_0[:, l] += v.value[:, l] + Ks[:, l]  
            x_0[:, l+1] =  eul(f, u_0[:, l], x_0[:, l], delta, param)
            s[:, l+1] = x_0[:, l+1]-x_0_old[:, l+1]
  
        S[i, k, :, :] = s.copy()  
        
        
        # Step update 
        k += 1
        real_obj[i, k] = problem.value
        delta_obj = real_obj[i, k-1]-real_obj[i, k]
    
    ######################################################################################
    #################################### System update ###################################
    ######################################################################################
    
    # Uncomment to exit at first iteration 
    """x = x_0
    u = u_0
    t = np.cumsum(np.ones(x.shape[1])*delta)-delta
    x_r_0 = x_r
    break"""
    
    u[:, i] = u_0[:, 0]                                 # apply first input
    u_0[:, :-1] = u_0[:, 1:]                            # extract tail of the input
    x[:, i+1] = eul(f, u[:, i], x[:, i], delta, param)  # update nonlinear dynamics 
    t[i+1] = t[i] + delta
    print('Height:', x[:, i], 'Voltage:', u[:,i])


##########################################################################################
##################################### Plot results #######################################
##########################################################################################

if not os.path.isdir('plot'):
    os.mkdir('plot')
    
# Trajectories 
fig, axs = plt.subplots(2, 1)
axs[0].plot(t, x[0,:], label=r'$x_1$')
axs[0].plot(t, x[1,:], label=r'$x_2$')
axs[0].plot(t, x_r[0,:], '--', label=r'$x^r_1$')
axs[0].plot(t, x_r[1,:], '--', label=r'$x^r_2$')
axs[0].legend(loc='upper right', prop={'size': 6.5})
axs[0].set(xlabel='Time (s)', ylabel='State x (cm)')
axs[1].plot(t[:-1], u[0,:])
axs[1].set(xlabel='Time (s)', ylabel='Input u (V)')
#fig.savefig('plot/tmpc1.eps', format='eps')

# Convergence of trajectory at first time step
plt.figure()
for j in range(maxIter): 
    plt.plot(t[:-1], X_0[0, j, 1, :-1], '-b')
    #plt.plot(t[:-1], X_0[0, j, 0, :-1], '-b')
    plt.ylabel('Convergence of $x_2$ (first time step)')
    
if maxIter <=4: fig, axs = plt.subplots(2, 2)
else: fig, axs = plt.subplots(2, 3)
k =0
l = 0
for j in range(maxIter):
    if j< 2: k, l = j, 0
    elif j>=2 and j < 4: k, l = j%2, 1
    else: k,l = 0, 2 
    axs[k, l].plot(X_0[0, j, 1, :] - np.abs(S[0, j, 1, :]/2), '--', 
                   label=r'$x_0 + \underbar{s} $')
    axs[k, l].plot(X_0[0, j, 1, :] + np.abs(S[0, j, 1, :]/2), '--', 
                   label=r'$x_0 + \overline{s}$')
    axs[k, l].plot(X_0[0, j, 1, :], label='$x_0$')
    axs[k, l].set_ylim([0, 20])
    if j == maxIter-1: axs[k, l].legend()
    axs[k, l].set_title('Iteration {}'.format(j+1))

for ax in axs.flat:
    ax.set(xlabel='Step (-)', ylabel='$x_2$ (cm)')
    ax.label_outer()
    
#fig.savefig('plot/tmpc2.eps', format='eps')   

# Objective value
plt.figure()
plt.semilogy(range(0, N), real_obj[:, 1])
plt.ylabel('Objective value $J$ at first iteration (-)')
plt.xlabel('Time step n (-)')
#plt.savefig('plot/tmpc3.eps', format='eps') 

# Final state perturbation 
plt.figure()
plt.semilogy(range(1, maxIter+1), np.linalg.norm(S[0, :-1, :, -1], axis=1), '-b')
plt.ylabel('Final state perturbation $s_N$, first time step (cm)')
plt.xlabel('Iteration (-)')

# Phase plot
plt.figure()
for j in range(maxIter):
    plt.plot(X_0[0, j, 0, :], X_0[0, j, 1, :], label='Iter {}'.format(j+1))
    plt.xlabel('$h_1$')
    plt.ylabel('$h_2$')
    for l in range(0, N, 1): 
        width = S_up[0, j, 0, l] - S_low[0, j, 0, l]
        height = S_up[0, j, 1, l] - S_low[0, j, 1, l]
        rect = patches.Rectangle((X_0[0, j, 0, l]+S_low[0, j, 0, l], 
                                  X_0[0, j, 1, l]+S_low[0, j, 1, l]), 
                                  width, height, fill=False, color="black")
        plt.gca().add_patch(rect)
    rect_term = patches.Rectangle((x_r[0, -1]-param.x_term, x_r[1, -1] - param.x_term), 
                                   param.x_term*2, param.x_term*2, fill=False,color="red")
    plt.gca().add_patch(rect_term)
plt.legend()
#plt.savefig('plot/tmpc4.eps', format='eps') 

plt.show()
