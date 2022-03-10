""" Tank model 

(c) Martin Doff-Sotta, University of Oxford (martin.doff-sotta@eng.ox.ac.uk)

"""
import numpy as np
import cvxpy as cp


def linearise(h_0, v_0, delta, param):


    """ Form the linearised discrete-time model around x_0, u_0 
    
    h_0[k+1] = (A1_d - A2_d) h_0[k] + (B1_d - B2_d) u_0[k]
    
    where A_d = A1_d - A2_d and B = B1_d - B2_d
    
    Input: guess state trajectory h_0, guess input trajectory u_0, time step delta, 
           parameter structure param
    Output: Discrete-time matrices A1_d, B1_d, A2_d, B2_d
    
    """
    assert np.all(h_0 != 0), 'Function sqrt(x) is not continuously differentiable in 0'
    
    # Dimensions 
    N = v_0.shape[1]
    N_input = v_0.shape[0]
    N_state = h_0.shape[0]
      
    # Linearised discrete-time model
    A1 = np.zeros((N+1, N_state, N_state))
    A1[:, 0, 0] = -param.A1*param.g/(param.A*np.sqrt(2*param.g*h_0[0, :]))
    A1[:, 1, 1] = -param.A2*param.g/(param.A*np.sqrt(2*param.g*h_0[1, :]))
    B1 = np.ones((N+1, N_state, N_input))
    B1 = B1*np.array([[param.k_p/param.A], [0]])
    A2 = np.zeros((N+1, N_state, N_state))
    A2[:, 1, 0] = -param.A1*param.g/(param.A*np.sqrt(2*param.g*h_0[0, :]))
    B2 = np.zeros((N+1, N_state, N_input))
    
    # Linearised discrete-time model
    A1_d = np.eye(N_state) + delta*A1
    A2_d = delta*A2
    B1_d = delta*B1
    B2_d = delta*B2
    
    return A1_d, B1_d, A2_d, B2_d


def f(h, u, param):


    """ Continuous-time system dynamics function f such that
    
    h[k+1] = f(h[k], u[k]) 
    
    Input: state h, input u, parameter structure param
    Output: f dynamics function
    """
    f1 = (param.k_p*u[0] - param.A1*np.sqrt(2*param.g*h[0]))/param.A
    f2 = (param.A1*np.sqrt(2*param.g*h[0]) - param.A2*np.sqrt(2*param.g*h[1]))/param.A
    return np.array([f1, f2], dtype=object)


def f1(h, u, delta, param):


    """ Return the discrete-time f1 convex dynamics from the DC decomposition
    
    f = f1 - f2
    
    where f is the system dynamics and f1, f2 are convex functions of the state / inputs
    
    Input: state h, input u, time step delta, parameter structure param
    Output: f1 convex dynamics function
    """
    f_1 = h[0,:] -delta*param.A1/param.A*cp.sqrt(2*param.g*h[0,:])\
                 + delta*param.k_p/param.A*u[0,:]
    f_2 = h[1,:] -delta*param.A2/param.A*cp.sqrt(2*param.g*h[1,:])
    
    return cp.vstack([f_1, f_2]) 


def f2(h, delta, param):


    """ Return the discrete-time f2 convex dynamics from the DC decomposition
    
    f = f1 - f2
    
    where f is the system dynamics and f1, f2 are convex functions of the state / inputs
    
    Input: state h, time step delta, parameter structure param
    Output: f2 convex dynamics function
    """
    f_1 = h[0,:]*0
    f_2 = -delta*param.A1/param.A*cp.sqrt(2*param.g*h[0,:])
    
    return cp.vstack([f_1, f_2])

    
def terminal(param, Q, R, delta):
    
    
    """ Compute terminal cost, terminal constraint bound and terminal matrix
    Input: parameter structure param, penalty matrices Q and R, time step delta
    Ouput: terminal matrix Q_N, terminal constraint bound gamma_N and terminal gain K_N 
    """
    
    # Initialisation 
    alpha = 10                                                  # objective weight
    n = len(param.x_init)                                       # number of states
    m = len(param.u_init)                                       # number of inputs
    I = np.eye(n)
    C = Q[-1, None, :] 
    eps = np.finfo(float).eps
    
    
    # Variables definition (SDP)
    Q_N = cp.Variable((n,n), symmetric=True)
    S = cp.Variable((n,n), symmetric=True)
    Y = cp.Variable((m,n))
    gamma_inv = cp.Variable((1,1))
    
    # Terminal set definition 
    dx, du = param.x_term, param.u_term                         # terminal set size
    Ver = np.array([[dx, dx, -dx, -dx],
                    [dx, -dx, dx, -dx]]) + param.h_r[:, None]   # vertices of terminal set
    
    # Objective 
    objective = cp.Minimize(cp.trace(Q_N) + alpha*gamma_inv)
    
    # Initialise constraints
    constr = []
    
    # Constraint S = Q_N^-1
    constr += [cp.vstack([cp.hstack([S, I]), cp.hstack([I, Q_N])]) >> eps*np.eye(n*2)]
    
    # Terminal cost constraint
    Y_ = cp.vstack([np.zeros((n-m, n)), Y])
    R_ = cp.diag(np.array([*(0,)*(n-m), np.linalg.inv(R)], dtype=object))
    CS = cp.vstack([C @ S, np.zeros_like(C)])
    O = np.zeros((n, n))
    for i in range(4):
        A1, B1, A2, B2 = linearise(Ver[:, i, None], np.zeros((1, 0)), delta, param)
        A = A1[0]-A2[0]
        B = B1[0]-B2[0]
        M = (A @ S + B @ Y)
        constr += [cp.vstack([cp.hstack([S, M.T, CS.T, Y_.T]), cp.hstack([M, S, O, O]),
        cp.hstack([CS, O, I, O]), cp.hstack([Y_, O, O, R_])]) >> eps*np.eye(n*4) ]
    
    # Terminal constraint F x + G u <= h
    G = cp.vstack([np.zeros((2,1)), np.zeros((2,1)), np.array([[1]]), -np.array([[1]])])
    F = cp.vstack([I, -I, I*0])
    h = cp.vstack([param.x_max[0], param.x_max[1], -param.x_min[0], 
                  -param.x_min[1], param.u_max, -param.u_min])
    h_loc = cp.vstack([h- (F @ param.h_r[:, None] + G @ np.array([[param.u_r[0]]])), 
                       np.array([[du]]), np.array([[dx]]), np.array([[dx]])])
    G_loc = cp.vstack([G, np.array([[1]]), np.zeros((2,1))])
    F_loc = cp.vstack([F, np.zeros((1, 2)), I])
    for o in range(h_loc.shape[0]):
        block1 = cp.hstack([gamma_inv @ (h_loc[o,:,None])**2, 
                            F_loc[o, None, :] @ S + G_loc[o, :, None] @ Y])
        block2 = cp.hstack([(F_loc[o, None, :] @ S + G_loc[o, :, None] @ Y).T, S]) 
        constr += [cp.vstack([block1, block2]) >> eps*np.eye(n+m)] 
    
    # Solve SDP problem    
    problem = cp.Problem(objective, constr)
    problem.solve(verbose=False)
    
    # Post-processing 
    gamma_N = 1/(gamma_inv.value[0, 0])
    K_N = Y.value @ Q_N.value
    
    return Q_N.value, gamma_N, K_N


def seed_cost(x0, u0, Q, R, Q_N, param):

    """ Compute cost 
    Input: trajectories x0 and u0, penalty matrices Q, R, Q_N, parameter structure param
    Output: cost J 
    """
    J = 0
    N = len(u0)
    for k in range(N):
        J = J + (x0[:, k]-param.h_r).T @ Q @ (x0[:, k]-param.h_r)\
         + (u0[:, k]- param.u_r[0]) * R * (u0[:, k]-param.u_r[0])
    
    # Terminal cost term
    J = J + (x0[:, N+1]-param.h_r).T @ Q_N @ (x0[:, N+1]-param.h_r)
    
    return J     