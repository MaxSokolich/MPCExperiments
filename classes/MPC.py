import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import math


class MPC:
    def __init__(self ,A,B, ref, N, Q, R) -> None:
        """
        MPC controller using Gurobi optimization.

        Parameters:
        - A, B: System matrices.
        - x0: Initial state.
        - 
        - N: Prediction horizon.
        - Q, R: Weight matrices for the state and input.
        - umin, umax: Minimum and maximum control inputs.
        
        """

        self.A = A
        self.B = B
        self.ref = ref
        self.N =  N
        self.Q = Q
        self.R = R

        control_bound = 15
        self.umax =  control_bound
        self.umin = -control_bound


        self.nx = len(Q) # Number of states
        self.nu = len(R) # Number of inputs
           
        
    def control_cvx(self, x0, Dist, ref):
        # Variables
        x = cp.Variable((self.N+1, self.nx))
        u = cp.Variable((self.N, self.nu))

        # Constraints
        constraints = [x[0, :] == x0]  # Initial state constraint

        # Dynamics constraints
        for t in range(self.N):
            constraints.append(x[t+1, :] == x[t, :] + self.B @ u[t, :] + Dist[t, :])

        # Input constraints
        constraints += [
            u >= self.umin,
            u <= self.umax
        ]

        # Define the cost function
        cost = 0
        gamma = 1
        for t in range(self.N):
            cost += cp.quad_form(x[t, :] - ref[t, :], self.Q) + cp.quad_form(u[t, :], self.R)

        # Define and solve the problem
        problem = cp.Problem(cp.Minimize(cost), constraints)
        problem.solve()

        # Check if the problem was successfully solved
        if problem.status not in ["infeasible", "unbounded"]:
            # Assuming the problem is feasible and bounded
            u_opt = u.value[0, :]  # Optimal control input for the current time step
            predict_traj = x.value
            return u_opt, predict_traj
        else:
            raise Exception("The problem is infeasible or unbounded.")
    def convert_control(self, u_mpc):

        f_t = np.linalg.norm(u_mpc)
        alpha_t = math.atan2(u_mpc[0], u_mpc[1])
        return f_t, alpha_t
                
            
 