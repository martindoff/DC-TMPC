""" Control systems -related functions

(c) Martin Doff-Sotta, University of Oxford (martin.doff-sotta@eng.ox.ac.uk)

"""
import numpy as np
import scipy
import matplotlib.pyplot as plt

def eul(f, u, x_0, d, param):
    
    
    """Integrate dynamics using forward Euler method
    
    x[k+1] = x[k] + delta*(f(x[k], u[k]))
    
    Input: dynamics f, control input u, initial condition x_0, step d
    Output: trajectory x 
    """  
    if u.ndim == 1:
        return  x_0 + d*(f(x_0, u, param))

    N = u.shape[1]
    N_state = x_0.shape[0]
    x = np.zeros((N_state, N+1))
    x[:, 0] = x_0  
    for i in range(N):
        x[:, i+1] = x[:, i] + d*(f(x[:, i], u[:, i], param))
        
    return x

    
def ode4(f, u, x_0, d, param):
    
    
    """Integrate dynamics using Runge Kutta 4 method
    
    Input: dynamics f, control input u, initial condition x_0, step d
    Output: trajectory x 
    """  
    if u.ndim == 1:
        k1 = f(x_0, u, param)
        k2 = f(x_0+k1*d/2, u, param)
        k3 = f(x_0+k2*d/2, u, param)
        k4 = f(x_0+k3*d, u, param)
        return  x_0 + (k1+2*k2+2*k3+k4)*d/6

    N = u.shape[1]
    N_state = x_0.shape[0]
    x = np.zeros((N_state, N+1))
    x[:, 0] = x_0  
    for i in range(N):
        k1 = f(x[:, i], u[:, i], param)
        k2 = f(x[:, i]+k1*d/2, u[:, i], param)
        k3 = f(x[:, i]+k2*d/2, u[:, i], param)
        k4 = f(x[:, i]+k3*d, u[:, i], param)
        x[:, i+1] = x[:, i] + (k1+2*k2+2*k3+k4)*d/6
        
    return x

        
def dlqr(A,B,Q,R):
    
    
    """Solve the discrete time LQR controller for dynamic system
     
    x[k+1] = A x[k] + B u[k]
    
    with u[k] = K x[k] and cost = sum x[k].T Q x[k] + u[k].T R u[k]
    
    Input: state space matrices A and B, state penalty Q, input penalty R
    Output: gain matrix K, Riccati equation solution X, eigenvalues eig_vals
    """

    # Ricatti equation
    X = np.matrix(scipy.linalg.solve_discrete_are(A, B, Q, R))
     
    # Compute the LQR gain
    if hasattr(B, "T"):
        K = -np.matrix(scipy.linalg.inv(B.T*X*B+R)*(B.T*X*A))
    else:
        K = -np.matrix(scipy.linalg.inv(B*X*B+R)*(B*X*A))  
     
    eig_vals, eig_vecs = scipy.linalg.eig(A-B*K)


def dp(A, B, Q, R, P):


    """ Implement one iteration of the DP recursion to compute K 
    Input: state space matrices A and B, state penalty Q, input penalty R, 
           Riccati equation solution P
    Output: gain K, P
    """
    
    # Compute gain 
    S = np.linalg.inv(B.T @ P @ B + R)
    K = - S @ B.T @ P @ A
    
    # Update P
    P = Q + A.T @ P @ A - A.T @ P @ B @ S @ B.T @ P @ A
    
    return K, P    
    return K, X, eig_vals