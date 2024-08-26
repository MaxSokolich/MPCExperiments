import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB
import math
import scipy.io
import random
import torch
import numpy as np
from torch import nn
import copy
import matplotlib.pyplot as plt
import time
import matplotlib.animation as animation


class MLP(nn.Module):

    def __init__(self, nInput, nOutput):
        super(MLP, self).__init__()

        # two hidden layers
        self.layers = nn.Sequential(
            nn.Linear(nInput, 40),
            nn.ReLU(),
            nn.Linear(40, 60),
            nn.ReLU(),
            nn.Linear(60, nOutput)
        )

    def forward(self, x):
        x = self.layers(x)
        return x


class Interp1d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, xnew, out=None):
        """
        Linear 1D interpolation on the GPU for Pytorch.
        This function returns interpolated values of a set of 1-D functions at
        the desired query points `xnew`.
        This function is working similarly to Matlab™ or scipy functions with
        the `linear` interpolation mode on, except that it parallelises over
        any number of desired interpolation problems.
        The code will run on GPU if all the tensors provided are on a cuda
        device.

        Parameters
        ----------
        x : (N, ) or (D, N) Pytorch Tensor
            A 1-D or 2-D tensor of real values.
        y : (N,) or (D, N) Pytorch Tensor
            A 1-D or 2-D tensor of real values. The length of `y` along its
            last dimension must be the same as that of `x`
        xnew : (P,) or (D, P) Pytorch Tensor
            A 1-D or 2-D tensor of real values. `xnew` can only be 1-D if
            _both_ `x` and `y` are 1-D. Otherwise, its length along the first
            dimension must be the same as that of whichever `x` and `y` is 2-D.
        out : Pytorch Tensor, same shape as `xnew`
            Tensor for the output. If None: allocated automatically.

        """
        # making the vectors at least 2D
        is_flat = {}
        require_grad = {}
        v = {}
        device = []
        eps = torch.finfo(y.dtype).eps
        for name, vec in {'x': x, 'y': y, 'xnew': xnew}.items():
            assert len(vec.shape) <= 2, 'interp1d: all inputs must be '\
                                        'at most 2-D.'
            if len(vec.shape) == 1:
                v[name] = vec[None, :]
            else:
                v[name] = vec
            is_flat[name] = v[name].shape[0] == 1
            require_grad[name] = vec.requires_grad
            device = list(set(device + [str(vec.device)]))
        assert len(device) == 1, 'All parameters must be on the same device.'
        device = device[0]

        # Checking for the dimensions
        assert (v['x'].shape[1] == v['y'].shape[1]
                and (
                     v['x'].shape[0] == v['y'].shape[0]
                     or v['x'].shape[0] == 1
                     or v['y'].shape[0] == 1
                    )
                ), ("x and y must have the same number of columns, and either "
                    "the same number of row or one of them having only one "
                    "row.")

        reshaped_xnew = False
        if ((v['x'].shape[0] == 1) and (v['y'].shape[0] == 1)
           and (v['xnew'].shape[0] > 1)):
            # if there is only one row for both x and y, there is no need to
            # loop over the rows of xnew because they will all have to face the
            # same interpolation problem. We should just stack them together to
            # call interp1d and put them back in place afterwards.
            original_xnew_shape = v['xnew'].shape
            v['xnew'] = v['xnew'].contiguous().view(1, -1)
            reshaped_xnew = True

        # identify the dimensions of output and check if the one provided is ok
        D = max(v['x'].shape[0], v['xnew'].shape[0])
        shape_ynew = (D, v['xnew'].shape[-1])
        if out is not None:
            if out.numel() != shape_ynew[0]*shape_ynew[1]:
                # The output provided is of incorrect shape.
                # Going for a new one
                out = None
            else:
                ynew = out.reshape(shape_ynew)
        if out is None:
            ynew = torch.zeros(*shape_ynew, device=device)

        # moving everything to the desired device in case it was not there
        # already (not handling the case things do not fit entirely, user will
        # do it if required.)
        for name in v:
            v[name] = v[name].to(device)

        # calling searchsorted on the x values.
        ind = ynew.long()

        # expanding xnew to match the number of rows of x in case only one xnew is
        # provided
        if v['xnew'].shape[0] == 1:
            v['xnew'] = v['xnew'].expand(v['x'].shape[0], -1)

        # the squeeze is because torch.searchsorted does accept either a nd with
        # matching shapes for x and xnew or a 1d vector for x. Here we would
        # have (1,len) for x sometimes
        torch.searchsorted(v['x'].contiguous().squeeze(),
                           v['xnew'].contiguous(), out=ind)

        # the `-1` is because searchsorted looks for the index where the values
        # must be inserted to preserve order. And we want the index of the
        # preceeding value.
        ind -= 1
        # we clamp the index, because the number of intervals is x.shape-1,
        # and the left neighbour should hence be at most number of intervals
        # -1, i.e. number of columns in x -2
        ind = torch.clamp(ind, 0, v['x'].shape[1] - 1 - 1)

        # helper function to select stuff according to the found indices.
        def sel(name):
            if is_flat[name]:
                return v[name].contiguous().view(-1)[ind]
            return torch.gather(v[name], 1, ind)

        # activating gradient storing for everything now
        enable_grad = False
        saved_inputs = []
        for name in ['x', 'y', 'xnew']:
            if require_grad[name]:
                enable_grad = True
                saved_inputs += [v[name]]
            else:
                saved_inputs += [None, ]
        # assuming x are sorted in the dimension 1, computing the slopes for
        # the segments
        is_flat['slopes'] = is_flat['x']
        # now we have found the indices of the neighbors, we start building the
        # output. Hence, we start also activating gradient tracking
        with torch.enable_grad() if enable_grad else contextlib.suppress():
            v['slopes'] = (
                    (v['y'][:, 1:]-v['y'][:, :-1])
                    /
                    (eps + (v['x'][:, 1:]-v['x'][:, :-1]))
                )

            # now build the linear interpolation
            ynew = sel('y') + sel('slopes')*(
                                    v['xnew'] - sel('x'))

            if reshaped_xnew:
                ynew = ynew.view(original_xnew_shape)

        ctx.save_for_backward(ynew, *saved_inputs)
        return ynew

    @staticmethod
    def backward(ctx, grad_out):
        inputs = ctx.saved_tensors[1:]
        gradients = torch.autograd.grad(
                        ctx.saved_tensors[0],
                        [i for i in inputs if i is not None],
                        grad_out, retain_graph=True)
        result = [None, ] * 5
        pos = 0
        for index in range(len(inputs)):
            if inputs[index] is not None:
                result[index] = gradients[pos]
                pos += 1
        return (*result,)



class MLP_controller(object):

    def __init__(self,mlp_model_a = None,mlp_model_b = None,a_means=None,b_means = None,dt = 0.1) -> None:
        if mlp_model_a is None or mlp_model_b is None:
            nInput = 2
            nOutput = 2+2+2
            model_a = MLP(nInput, nOutput)
            model_b = MLP(nInput, nOutput)
            model_a.load_state_dict(torch.load('/home/ahmad/Desktop/uBots_MARSS/multi-ubots-learning/genDataMatlab/model_a.pt'))
            model_b.load_state_dict(torch.load('/home/ahmad/Desktop/uBots_MARSS/multi-ubots-learning/genDataMatlab/model_b.pt'))
            self.model_a = model_a
            self.model_b = model_b
        if a_means is None or b_means is None:
            train_data_a = scipy.io.loadmat('/home/ahmad/Desktop/uBots_MARSS/multi-ubots-learning/genDataMatlab/data_a_2.mat')
            train_data_b = scipy.io.loadmat('/home/ahmad/Desktop/uBots_MARSS/multi-ubots-learning/genDataMatlab/data_b_2.mat')
            self.a_means = torch.tensor(train_data_a['means'])
            self.b_means = torch.tensor(train_data_b['means'])
        pass
        self.dt = dt
    def getControl(self,Dx):#da, db):
        da = Dx[0,:]
        db = Dx[1,:]
        mag_a = torch.sqrt(torch.dot(da,da))
        mag_b = torch.sqrt(torch.dot(db,db))
        if mag_a > mag_b:
            angle = torch.arctan2(da[1], da[0]) - torch.pi/2;
            #print(angle)
            A = torch.tensor([[torch.cos(-angle), -torch.sin(-angle)], [torch.sin(-angle), torch.cos(-angle)]]) / mag_a
            db = A @ db
            #print(db)

            reflect = False
            if db[0] < 0:
                db[0] = -db[0]
                reflect = True

            #print(db)
            ctrl = self.model_a(db.float()).double() + self.a_means
            #print(ctrl)
            if reflect:
                #print(ctrl[:,2:4])
                #print(torch.sin(ctrl[:,2:4]))
                #print(torch.cos(ctrl[:,2:4]))
                ctrl[:,2:4] = torch.arctan2(torch.sin(ctrl[:,2:4]), -torch.cos(ctrl[:,2:4]))
                #print(ctrl)

            ctrl[:,2:4] += angle
            ctrl[:,4:6] *= mag_a

            return ctrl
        else:
            angle = torch.arctan2(db[1], db[0]) - torch.pi/2;
            A = torch.tensor([[torch.cos(-angle), -torch.sin(-angle)], [torch.sin(-angle), torch.cos(-angle)]]) / mag_b
            da = A @ da

            reflect = False
            if da[0] < 0:
                da[0] = -da[0]
                reflect = True


            ctrl = self.model_b(da.float()).double() + self.b_means

            if reflect:
                ctrl[:,2:4] = torch.arctan2(torch.sin(ctrl[:,2:4]), -torch.cos(ctrl[:,2:4]))

            ctrl[:,2:4] += angle
            ctrl[:,4:6] *= mag_b

            return ctrl
    def discretization(self,u):
        #TODO
        DTs= u[4:]/self.dt
        freqs = np.zeros(int(DTs[0]+DTs[1]))
        alphas = np.zeros(int(DTs[0]+DTs[1]))
        # frequincy sequence:
        freqs[0:int(DTs[0])] = u[0]
        freqs[int(DTs[0]+1):int(DTs[0]+DTs[1])] = u[1]
        # heading sequence:  
        alphas[0:int(DTs[0])] = u[2]
        alphas[int(DTs[0]+1):int(DTs[0]+DTs[1])] = u[3]
        return freqs, alphas 

# ---------------------------


class MPC:
    def __init__(self ,A,B, N, Q, R) -> None:
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
        self.N =  N
        self.Q = Q
        self.R = R

        control_bound = 3.5
   
        self.umax =  control_bound
        self.umin = -control_bound


        self.nx = len(Q) # Number of states
        self.nu = len(R) # Number of inputs
           
    def control_gurobi(self, x0, ref, Dist):
        print('x_0 = ', x0)
        print('ref= ', ref)
        # print('ref_shape =', ref.shape)
        # print('R = ', self.R)
        # print('Q=', self.Q)
        # print('A=', self.A)
        # print('B= ', self.B)
    
        """
        calculate the control signal.
        - x0: Initial state.
        - Dist: Estimation of the Disturbance
        - ref: Reference trajectory (Nx1 vector).
        Returns:
        - u_opt: Optimal control input for the current time step.
        - predict_traj: prediction of the trajectory
        """

        # env = gp.Env(empty=True)

       

        # Turn off Gurobi output to the console
        # env.setParam('OutputFlag', 0)

        # Now you can create your model and solve it without seeing output in the console
        # model = gp.Model("example_model", env=env)

        
        # Create a new model
        m = gp.Model("mpc")
       

        x0 = np.reshape(x0, 2)
        # Decision variables for states and inputs
        x = m.addMVar((self.N+1, self.nx), lb=-GRB.INFINITY, name="x")
        u = m.addMVar((self.N, self.nu), lb= self.umin, ub=self.umax, name="u")
  
        # Initial state constraint
        m.addConstr(x[0, :] == x0, name="init")
    


        # Dynamics constraints
        for t in range(self.N):
            m.addConstr(x[t+1, :] ==  x[t, :] + self.B @ u[t, :]+Dist, name=f"dyn_{t}")
            # m.addConstr(u[t, 0]**2+u[t,1]**2 ==  1, name=f"sincos_{t}")
      
        # State constraints
        # for t in range(N+1):
        #     m.addConstr(x[t, :] >= xmin, name=f"xmin_{t}")
        #     m.addConstr(x[t, :] <= xmax, name=f"xmax_{t}")

        # Objective: Minimize cost function
        # abs_diff = m.addVars(self.N-1, name="abs_diff")

        # Define the cost function
        cost = gp.QuadExpr()

        gamma = 0.85
        for t in range(self.N):
            cost += gamma**t*(x[t, :] - ref[t, :]) @ self.Q @ (x[t, :] - ref[t, :]) + u[t, :] @ self.R @ u[t, :]
        
        # cost+=1000* (x[t, :] - ref[t, :]) @ Q @ (x[t, :] - ref[t, :])
        # for t in range(self.N-1):
        #     u_sq_diff = (u[t, 0]**2 + u[t, 1]**2) - (u[t+1, 0]**2 + u[t+1, 1]**2)
        #     m.addConstr(abs_diff[t] >= u_sq_diff, name=f"abs_diff_pos_{t}")
        #     m.addConstr(abs_diff[t] >= -u_sq_diff, name=f"abs_diff_neg_{t}")
        #     cost += abs_diff[t]
        # for t in range(self.N-1):
        #     cost += (u[t, :]-u[t+1,:]) @ self.R @ (u[t, :]-u[t+1,:])
           
   
        m.setObjective(cost, GRB.MINIMIZE)

        
        # Optimize model
        # m.params.NonConvex = 2
        m.optimize()
     
        
        u_opt = u.X[0, :]  # Get optimal control input for the current time step
        predict_traj = x.X
        return u_opt, predict_traj
    

    """def control_cvx(self, x0, Dist, ref):
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
            raise Exception("The problem is infeasible or unbounded.")"""
    

    def convert_control(self, u_mpc):

        f_t = np.linalg.norm(u_mpc)
        #alpha_t = math.atan2(-u_mpc[1], u_mpc[0]) - np.pi/2
        alpha_t = math.atan2(u_mpc[1], u_mpc[0]) - np.pi/2
        
        #alpha_t = np.pi 
        #f_t = 5
        return f_t, alpha_t
                
def main(): 
    mlpC = MLP_controller()
    mlpC.discretization(u  = np.array([1,2,.1,.2,5,6]))  
if __name__ == '__main__':
    main()