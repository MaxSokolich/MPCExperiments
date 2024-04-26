
import numpy as np
import sys
import classes.Learning_module_2d as GP # type: ignore


import math 
from classes.MPC import  MPC


class algorithm:
    def __init__(self):
        
        
        self.gp_sim = GP.LearningModule()

        self.gp_sim.load_GP()
        self.a0_sim = np.load('classes/a0_est.npy')

        # freq = 4
        # a0_def = 1.5
        self.dt = 0.040 #assume a timestep of 30 ms

        x0 = 2200
        y0 = 1800

        r = 100
        center_x = x0 -r
        center_y = y0
        time_steps = 500

        theta_ls = np.linspace(0, 2*np.pi,time_steps)
        x_ls = center_x + r*(np.cos(theta_ls))
        y_ls = center_y+ r*np.sin(theta_ls)
        ref = np.ones((time_steps,2))
        ref[:,0]= x_ls
        ref[:,1]= y_ls
        
        self.ref = ref
        print(self.ref)

        time_steps = len(self.ref)


        x0 = [self.ref[0,0],self.ref[0,1]]


        ########MPC parameters
        B  = self.a0_sim*self.dt*np.array([[1,0],[0,1]])
        A = np.eye(2)
        # Weight matrices for state and input
        Q = np.array([[1,0],[0,1]])
        self.R = 0.01*np.array([[1,0],[0,1]])
        self.N = 1
        self.mpc = MPC(A= A, B=B, ref=self.ref, N=self.N, Q=Q, R=self.R)


        x_traj = np.zeros((time_steps+1, 2))  # +1 to include initial state

        u_traj = np.zeros((time_steps, 2))
        x_traj[0, :] = x0

        self.alpha_t = 0
        self.freq_t =0
        self.counter = 0
        self.time_range = 200 #frames

        #### ref Trjactory


        #########Ref Trajectory

        """node_ls = np.load('classes/node_path.npy')
        node_ls[3] = np.array([1500, 1600])
        node_ls = np.delete(node_ls, 1, 0)
        gpath_planner_traj = self.generate_in_between_points(node_ls)
        self.ref = gpath_planner_traj
        np.save('ref.npy', self.ref)"""
        # ref = np.load('ref.npy')
        

        
    
    
    def generate_traj(self, robot_list):
        microrobot_latest_position_x = robot_list[-1].position_list[-1][0]
        microrobot_latest_position_y = robot_list[-1].position_list[-1][1]

        x0 = microrobot_latest_position_x
        y0 = microrobot_latest_position_y

        r = 100
        center_x = x0 -r
        center_y = y0
        time_steps = 500

        theta_ls = np.linspace(0, 2*np.pi,time_steps)
        x_ls = center_x + r*(np.cos(theta_ls))
        y_ls = center_y+ r*np.sin(theta_ls)
        ref = np.ones((time_steps,2))
        ref[:,0]= x_ls
        ref[:,1]= y_ls
        
        self.ref = ref
        print(self.ref)

        time_steps = len(self.ref)


        x0 = [self.ref[0,0],self.ref[0,1]]


        ########MPC parameters
        B  = self.a0_sim*self.dt*np.array([[1,0],[0,1]])
        A = np.eye(2)
        # Weight matrices for state and input
        Q = np.array([[1,0],[0,1]])
        self.R = 0.01*np.array([[1,0],[0,1]])
        self.N = 1
        self.mpc = MPC(A= A, B=B, ref=self.ref, N=self.N, Q=Q, R=self.R)


        x_traj = np.zeros((time_steps+1, 2))  # +1 to include initial state

        u_traj = np.zeros((time_steps, 2))
        x_traj[0, :] = x0

        self.alpha_t = 0
        self.freq_t =0
        self.counter = 0
        self.time_range = 200 #frames



    def generate_in_between_points(self, node_ls):
        """
        Generates in-between points for a given list of segment endpoints.

        Parameters:
        - node_ls: Array of shape (number_of_segments, 2, 2), where each entry represents
                a segment with [start_point, end_point] and each point is [x, y].
        - num_points_per_segment: Number of in-between points to generate per segment.

        Returns:
        - full_trajectory: Array of points representing the full trajectory, including
                        the original endpoints and the newly generated in-between points.
        """
        full_trajectory = []

        for i in range(len(node_ls)-1):
            start_point, end_point = node_ls[i], node_ls[i+1]
            length = np.linalg.norm(end_point-start_point)
            num_points_per_segment= int(2*length/2)
            # Generate a sequence of numbers between 0 and 1, which will serve as interpolation factors.
            interpolation_factors = np.linspace(0, 1, num_points_per_segment + 2)
            
            # Interpolate x and y separately
            x_points = (1 - interpolation_factors) * start_point[0] + interpolation_factors * end_point[0]
            y_points = (1 - interpolation_factors) * start_point[1] + interpolation_factors * end_point[1]
            
            # Combine the x and y coordinates
            segment_points = np.vstack((x_points, y_points)).T
            full_trajectory.extend(segment_points[:-1].tolist())

        # Ensure the last point of the last segment is included
        full_trajectory.append(node_ls[-1].tolist())

        return np.array(full_trajectory)



    def run(self, robot_list): #this executes at every frame
        

        
        self.counter += 1

        current_ref = self.ref[self.counter:min(self.counter+self.N, self.time_range), :]

        if current_ref.shape[0] < self.N:
            # Pad the reference if it's shorter than the prediction horizon
            current_ref = np.vstack((current_ref, np.ones((self.N-current_ref.shape[0], 1)) * self.ref[-1, :]))

        ### Disturbance Compensator 
        muX,sigX = self.gp_sim.gprX.predict(np.array([[self.alpha_t, self.freq_t]]), return_std=True)
        muY,sigY = self.gp_sim.gprY.predict(np.array([[self.alpha_t, self.freq_t]]), return_std=True)
        v_e = np.array([muX[0], muY[0]])
        
        Qz = 0*self.R
        
        #microrobot_latest_position = x_traj[t, :]
        
        
        #define robot position
        microrobot_latest_position_x = robot_list[-1].position_list[-1][0]
        microrobot_latest_position_y = robot_list[-1].position_list[-1][1]
        microrobot_latest_position = np.array([microrobot_latest_position_x, 
                                               microrobot_latest_position_y]).reshape([2,1])
        
        #print(microrobot_latest_position)
        print("X0", microrobot_latest_position)
        u_mpc , pred_traj = self.mpc.control_gurobi(microrobot_latest_position, current_ref, (v_e)*self.dt)
        #x0 = x_traj[t,:] - current_ref[0,:]



        #u_current = u_mpc
        #u_traj[t, :] = u_current # Assuming u_opt is the control input for the next step
        
        self.f_t, self.alpha_t = self.mpc.convert_control(u_mpc)
        ### f_t and alpha_t must be passed to the system as the control inputs


        #x_traj[t+1, :] =  sim.last_state# Update state based on non_linear dynamics
            #output: actions which is the magetnic field commands applied to the arduino

        Bx = 0 
        By = 0 
        Bz = 0
        alpha = self.alpha_t
        gamma = np.pi/2
        freq = self.f_t
        psi = np.pi/2
        gradient = 0 # gradient has to be 1 for the gradient thing to work
        acoustic_freq = 0
        
        
        return Bx, By, Bz, alpha, gamma, freq, psi, gradient, acoustic_freq