
import numpy as np
import sys
import os

# Add the directory containing the 'classes' folder to the Python path


from classes.Learning_module_2d import LearningModule # type: ignore
from classes.MR_simulator import Simulator
import math 
from classes.MPC import  MPC
import matplotlib.pyplot as plt
import cv2


class algorithm:
    def __init__(self):
        self.alpha_ls = [0]
        self.freq_ls = [0]

        self.gp = LearningModule(3)
        self.reset()
        
        # self.gp_sim = GP.LearningModule()

        self.a0_sim = np.load('a0_est.npy')

        # freq = 4
        # a0_def = 1.5
        self.dt = 1/10 #assume a timestep of 30 ms

        x0 = 1200
        y0 = 1300

        #######creating circle
        # center_x = x0 
        # center_y = y0
        # tx = 100
        # ty = 100
        # time_steps = 300
        # t =np.linspace(0, 2*np.pi,time_steps)
        # x_ls = center_x +tx*t
        # y_ls = center_y+ ty*t
        # ref = np.ones((time_steps,2))
        # ref[:,0]= x_ls
        # ref[:,1]= y_ls
        # r = 400
        # theta_ls = np.linspace(0, 2*np.pi,time_steps)
        # x_ls = center_x + r*(np.cos(theta_ls))
        # y_ls = center_y+ r*np.sin(theta_ls)
        # ref = np.ones((time_steps,2))
        # ref[:,0]= x_ls
        # ref[:,1]= y_ls
        # self.ref = ref


       
        node_ls = np.load('classes/node_path.npy')
        # node_ls[3] = np.array([1500, 1600])
        # node_ls = np.delete(node_ls, 1, 0)
        gpath_planner_traj = self.generate_in_between_points(node_ls)
        width = 1600
        height = 2000
        center = [1200, 1000]
        self.ref = self.generate_infinity_path(width, height, center)
        print(self.ref)
        # ref = np.load('classes')


         
        self.init_point_x = self.ref[0,0]
        self.init_point_y = self.ref[0, 1]
        
        print('whole ref_shape =', self.ref.shape)
        #print(self.ref)

        time_steps = len(self.ref)

        self.goal = np.array([1500,1500])
        x0 = [self.ref[0,0],self.ref[0,1]]


        ########MPC parameters
        # B  = self.a0_sim*self.dt*np.array([[1,0],[0,1]])
        B  = self.a0_sim*np.array([[1,0],[0,-1]])
        A = np.eye(2)
        # Weight matrices for state and input
        Q = np.array([[1,0],[0,1]])
        R = 0.01*np.array([[1,0],[0,1]])
        self.N = 4
        self.mpc = MPC(A= A, B=B, N=self.N, Q=Q, R=R)


        x_traj = np.zeros((time_steps+1, 2))  # +1 to include initial state

        u_traj = np.zeros((time_steps, 2))
        x_traj[0, :] = x0

        self.alpha_t = 0
        self.freq_t =0
        self.counter = 0
        self.time_range = time_steps #frames

        #### ref Trjactory
        self.umpc_history = []

        self.robot_list = None

        #########Ref Trajectory

        """node_ls = np.load('classes/node_path.npy')
        node_ls[3] = np.array([1500, 1600])
        node_ls = np.delete(node_ls, 1, 0)
        gpath_planner_traj = self.generate_in_between_points(node_ls)
        self.ref = gpath_planner_traj
        np.save('ref.npy', self.ref)"""
        # ref = np.load('ref.npy')

        # self.sim()
        
    def reset(self):
        self.counter = 0
        self.current_index = 0
    
    def load_GP(self):
        
        self.gp.load_GP()

    def generate_infinity_path(self, width, height, center, num_points=1600):
        """
        Generate points for an infinity-shaped path with constant velocity.

        Args:
        width (float): The width of the infinity path.
        height (float): The height of the infinity path.
        center (tuple): The (x, y) center of the infinity path.
        num_points (int): Number of points to generate.

        Returns:
        np.array: Array of points (x, y) that form the infinity path.
        """
        # Unpack the center coordinates
        cx, cy = center[0], center[1]
        
        # Generate t values from 0 to 2*pi
        t = np.linspace(0, 2 * np.pi, num_points)
        
        # Parametric equations for the lemniscate
        x = cx + (width / 2) * np.sin(t) / (1 + np.cos(t)**2)
        y = cy + (height / 2) * np.sin(t) * np.cos(t) / (1 + np.cos(t)**2)
        
        # Calculate cumulative distances
        distances = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
        velocity_track = 2
        cumulative_distances = 1/velocity_track*np.insert(np.cumsum(distances), 0, 0)
        
        # Generate uniform distances
        uniform_distances = np.linspace(0, cumulative_distances[-1], num_points)
        
        # Interpolate to get points at uniform distances
        x_uniform = np.interp(uniform_distances, cumulative_distances, x)
        y_uniform = np.interp(uniform_distances, cumulative_distances, y)
        
        return np.column_stack((x_uniform, y_uniform))


    def correct_position(self, robot_list):
        microrobot_latest_position_x = robot_list[-1].position_list[-1][0]
        microrobot_latest_position_y = 2048 - robot_list[-1].position_list[-1][1] 
        y_max = 2048
        y = y_max - microrobot_latest_position_y
        out = np.array([microrobot_latest_position_x, y])
        out = out.reshape([2])
        return out

        
    def run_sim(self, actions,init_pos=None,noise_var = 1,a0 =1.5, is_mismatched = False):
        # state_prime = np.empty((0,2))
        # states      = np.empty((0,2))
        sim = Simulator()
        sim.reset_start_pos(init_pos)
        sim.noise_var = noise_var
        sim.a0 = a0
        sim.is_mismatched = is_mismatched
        time_steps = len(actions)
        X = np.zeros(time_steps)
        Y = np.zeros(time_steps)
        # X[0] = init_pos[0]
        # Y[0] = init_pos[1]
        # state       = env.reset(init = init_pos,noise_var = noise_var,a0=a0, is_mismatched = is_mismatched)
        # init
        # states      = np.append(states, env.last_pos, axis=0)
        # state_prime = np.append(state_prime, np.array([0,0]), axis=0)
        counter = 0
        for action in actions:
            sim.step(f_t=action[0], alpha_t=action[1])
            X[counter] = sim.last_state[0]
            Y[counter] = sim.last_state[1]
            counter += 1
        alpha   = actions[:,1]
        freq    = actions[:,0]
        time    = np.linspace(0, (len(X) - 1)/30.0, len(X)) # (np.arange(len(X))) / 30.0 #timestep is 1/30
        
        return X,Y,alpha,time,freq
    
    
    def sim(self):
        

        gp_sim = self.gp.LearningModule()

    
        # gp_sim.load_GP()
        a0_sim = np.load('classes/a0_est.npy')

        # freq = 4
        # a0_def = 1.5
        dt = self.dt#assume a timestep of 30 ms
        noise_var = 0.0
        sim = Simulator()
        time_steps = len(self.ref)

        self.live_path = False
        #### ref Trjactory

        
        # r0=500
        # xi_0 = 1000
        # yi_0 =1000
        # theta_ls = np.linspace(0, 2*np.pi,time_steps)
        # x_ls = r0*(1-np.cos(theta_ls))+xi_0
        # y_ls = r0*np.sin(theta_ls)+yi_0
        # ref = np.ones((time_steps,2))
        # ref[:,0]= x_ls
        # ref[:,1]=y_ls



        x0 = self.ref[0]

        sim.reset_start_pos(x0)
        sim.noise_var = noise_var
        sim.a0 = a0_sim
        sim.is_mismatched = True
        #########Ref Trajectory
        # r0 = 2
    


                
        ########MPC parameters
        
        N = self.N


        # Simulates the dynamic system over T time steps using MPC control.
        # 
        #     Parameters:
            # - All parameters are as previously described.
            # - T: Total number of time steps to simulate.

            # Returns:
            # - x_traj: Trajectory of states.
            # - u_traj: Trajectory of control inputs.
            # """
        x_traj = np.zeros((time_steps+1, 2))  # +1 to include initial state

        u_traj = np.zeros((time_steps, 2))
        x_traj[0, :] = x0

        alpha_t = 0
        freq_t =0



        

        for t in range(time_steps):
            goal = self.goal
            current_ref = self.find_stright_path( x_traj[t, :], goal)
            # Update reference for the current time step
            # current_ref = self.ref[t:min(t+N, time_steps), :]
            # if current_ref.shape[0] < N:
            #     # Pad the reference if it's shorter than the prediction horizon
            #     current_ref = np.vstack((current_ref, np.ones((N-current_ref.shape[0], 1)) * self.ref[-1, :]))

            ### Disturbance Compensator 
            muX,sigX = gp_sim.gprX.predict(np.array([[alpha_t, freq_t]]), return_std=True)
            muY,sigY = gp_sim.gprY.predict(np.array([[alpha_t, freq_t]]), return_std=True)
            # D = np.array([gp_sim.Dx, gp_sim.Dy])
            v_e = np.array([muX[0], muY[0]])
            
            
            # u_mpc,pred_traj = self.mpc.mpc_control(B, x_traj[t, :], current_ref, N, Q, R,(v_e)*dt)
            u_mpc,pred_traj = self.mpc.control_gurobi(x0 =  x_traj[t, :], ref =current_ref, Dist=0)
            z0 = x_traj[t,:]-current_ref[0,:]
            # u_mpc = mpc_with_integral_action(A, B, Q, R, Qz, x0, z0, ref, N,np.hstack((v_e*dt, [0,0])) )


            u_current = u_mpc
            u_traj[t, :] = u_current # Assuming u_opt is the control input for the next step
            f_t = np.linalg.norm(u_current)
            alpha_t = math.atan2(u_current[1], u_current[0])


            action = np.array([ [f_t], [alpha_t], [t*dt]])


            sim.step(f_t=f_t, alpha_t= alpha_t)
            x_traj[t+1, :] =  sim.last_state# Update state based on non_linear dynamics
            # x_traj_lin[t+1, :] =x_traj_lin[t, :] + B @ u_opt


        # Plotting
        time_span = np.arange(time_steps+1)



        plt.figure(figsize=(12, 6))

        plt.subplot(1, 3, 1)
        plt.plot(time_span, x_traj[:, 0], label='Actual Trajectory (x1)')
        plt.plot(time_span[1:], self.ref[:, 0], 'r--', label='Desired Trajectory (x1)')
        plt.xlabel('Time step')
        plt.ylabel('State x1')
        plt.title('Trajectory of State x1')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 3, 2)
        plt.plot(time_span, x_traj[:, 1], label='Actual Trajectory (x2)')
        plt.plot(time_span[1:], self.ref[:, 1], 'r--', label='Desired Trajectory (x2)')
        plt.xlabel('Time step')
        plt.ylabel('State x2')
        plt.title('Trajectory of State x2')
        plt.legend()
        plt.grid(True)


        plt.subplot(1, 3, 3)
        plt.plot(x_traj[:,0], x_traj[:, 1], label='Actual Trajectory (x1)')
        plt.plot(self.ref[:,0], self.ref[:, 1], 'r--', label='Desired Trajectory (x1)')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.title('Trajectory')
        plt.legend()
        plt.axis('equal')
        plt.grid(True)
        plt.show()

            
        
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
        # print(self.ref)

        time_steps = len(self.ref)


        x0 = [self.ref[0,0],self.ref[0,1]]


        
    
        


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
            num_points_per_segment= int(2*length/5)
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

    def find_stright_path(self, start, goal, length = 10, size = 3 ):
        #### find a line between the start and goal
        direction = (goal -start)/np.linalg.norm((goal -start))
        t = np.linspace(0, length, size)
        path = []
        for ti in t:
            path.append( start + ti*direction)
        path = np.array(path)

        return path

    def find_closest_index(self, current_state, reference_trajectory, current_index, window_size=10):
        # Define the search window
        print("curent", current_index, window_size)
        start_index = max(0, current_index - window_size)
        end_index = min(len(reference_trajectory), current_index + window_size)
        
        # Extract the relevant part of the reference trajectory
        search_window = reference_trajectory[start_index:end_index]
        
        # Calculate the Euclidean distance within the window
        distances = np.linalg.norm(search_window - current_state, axis=1)
        
        # Find the index of the minimum distance within the window
        closest_index_within_window = np.argmin(distances)
        
        # Convert back to the original reference trajectory index
        closest_index = start_index + closest_index_within_window
        return closest_index



    def run(self, robot_list, frame): #this executes at every frame

        self.counter += 1

        #cv2.circle(frame,(self.goal[0], self.goal[1]),20,(0,0,0), -1)


        #determines what type of path were following from trajecotry list
        if len(robot_list[-1].trajectory) > 0:
            ref = robot_list[-1].trajectory
            ref = np.reshape(np.array(ref), [len(ref),2])
            self.ref = ref
        

        ### Disturbance Compensator 
        # muX,sigX = self.gp_sim.gprX.predict(np.array([[self.alpha_t, self.freq_t]]), return_std=True)
        # muY,sigY = self.gp_sim.gprY.predict(np.array([[self.alpha_t, self.freq_t]]), return_std=True)
        # v_e = np.array([muX[0], muY[0]])
        
        
        
        #microrobot_latest_position = x_traj[t, :]
        
        #define robot position
        microrobot_latest_position_x = robot_list[-1].position_list[-1][0]
        microrobot_latest_position_y = robot_list[-1].position_list[-1][1] 
        microrobot_latest_position = np.array([microrobot_latest_position_x, 
                                               microrobot_latest_position_y]).reshape(2)
        vx = robot_list[-1].velocity_list[-1][0]
        vy = robot_list[-1].velocity_list[-1][1]
        print("velocity",(vx,vy))

        

    

        x_current = np.array([microrobot_latest_position_x, microrobot_latest_position_y])
        closest_index = self.find_closest_index(x_current, self.ref, self.current_index)
         # Update the current reference index
        self.current_index = closest_index
        current_ref = self.ref[self.counter:min(self.counter+self.N, self.time_range), :]
        #current_ref = self.ref[self.counter:min(closest_index+self.N, self.time_range), :]
        if current_ref.shape[0] < self.N:
            # Pad the reference if it's shorter than the prediction horizon
            current_ref = np.vstack((current_ref, np.ones((self.N-current_ref.shape[0], 1)) * self.ref[-1, :]))

        #print(microrobot_latest_position)
        #print("X0", microrobot_latest_position)
        # u_mpc , pred_traj = self.mpc.control_gurobi(microrobot_latest_position, current_ref, (v_e)*self.dt)
        goal_trsh = 10
        if np.linalg.norm(microrobot_latest_position-self.ref)<goal_trsh:
            print('reached to the goal')
            self.f_t, self.alpha_t = 0,0
            path = []
        else:



            alpha_GP = self.alpha_ls[-1]
            freq_GP = self.freq_ls [-1]
            muX,sigX = self.gp.gprX.predict(np.array([[alpha_GP, freq_GP]]), return_std=True)
            muY,sigY = self.gp.gprY.predict(np.array([[alpha_GP, freq_GP]]), return_std=True)
            # D = np.array([gp_sim.Dx, gp_sim.Dy])
            try:
                print("using GP",[muX, muY])
                v_e = np.array([muX[0], muY[0]])
            except Exception:
                print("not using GP")
                v_e = np.array([0,0])
           
            #v_e = np.array([muX[0], muY[0]])
            #v_e = np.array([0,0])
            u_mpc , pred_traj = self.mpc.control_gurobi(microrobot_latest_position, current_ref, v_e)
            
            #u_mpc = np.array([0,1])
            path = current_ref
            print('u_mpc = ', u_mpc)
            self.f_t, self.alpha_t = self.mpc.convert_control(u_mpc)
            self.freq_ls.append(self.f_t)
            self.alpha_ls.append(self.alpha_t)
            

        
        
        print('current_ref =', current_ref)

        cv2.putText(frame, "counter: {}".format(self.counter),
                        (int(2448 * (6/10)),
                        int(2048 * (9.9/10))),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1, 
                        thickness=4,
                        color = (255, 255, 255))
        # print('self_N = ', self.N)
        goal = self.goal#np.array([1500,1500])
        # path_len = 150
        # if np.linalg.norm(goal-self.correct_position(robot_list))> path_len:
        #     path = self.find_stright_path(microrobot_latest_position, goal, length=path_len)
        # else:
        #     path = self.find_stright_path(microrobot_latest_position, goal, length= np.linalg.norm(goal-self.correct_position(robot_list)))
        # u_mpc , pred_traj = self.mpc.control_gurobi(x0 = self.correct_position(robot_list), ref =path, Dist=0)
        # u_mpc , pred_traj = self.mpc.control_gurobi(x0 = microrobot_latest_position, ref =path, Dist=0)
    
        self.umpc_history.append(u_mpc)

        #plot direction vec
        color = (255, 255, 255)  # Blue color in BGR
        thickness = 10
        line_type = 4
        shift = 0
        start_point = microrobot_latest_position
        sc = 10
        end_point = start_point.reshape(2) + sc*u_mpc.reshape(2)
     
        
        #plot ref
        ref_pts = np.array(self.ref, np.int32)
        cv2.polylines(frame, [ref_pts], False, (100, 100, 100), 4)
        # print('ref = ', path)
        # print('pred_trj = ', pred_traj)

        #u vec
        cv2.arrowedLine(frame, start_point.astype(np.int32), end_point.astype(np.int32), color, thickness, line_type, shift)
        
        #plot path
        path_pts = np.array(path, np.int32)
        cv2.polylines(frame, [path_pts], False, (0, 255, 0), 8)


        

        #plot pred traj pts pts
        pred_traj_pts = np.array(pred_traj, np.int32)
        cv2.polylines(frame, [pred_traj_pts], False, (0, 0, 255), 6)

        
        
        

        

        
        
        #x0 = x_traj[t,:] - current_ref[0,:]



        #u_current = u_mpc
        #u_traj[t, :] = u_current # Assuming u_opt is the control input for the next step
        

        rotatingfield = "alpha: {:.2f}, freq: {:.2f}".format(np.degrees(self.alpha_t),  self.f_t) #adding 90 to alpha for display purposes only
                
        cv2.putText(frame, rotatingfield,
            (int(2448 / 1.8),int(2048 / 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1.5, 
            thickness=3,
            color = (255, 255, 255),
        )
        
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
        
        

        #     else:
        #         Bx = 0 
        #         By = 0 
        #         Bz = 0
        #         alpha = self.alpha_t
        #         gamma = np.pi/2
        #         freq = self.f_t
        #         psi = np.pi/2
        #         gradient = 0 # gradient has to be 1 for the gradient thing to work
        #         acoustic_freq = 0
        # else:
        #     Bx = 0 
        #     By = 0 
        #     Bz = 0
        #     alpha = self.alpha_t
        #     gamma = np.pi/2
        #     freq = self.f_t
        #     psi = np.pi/2
        #     gradient = 0 # gradient has to be 1 for the gradient thing to work
        #     acoustic_freq = 0
        # 
        
        return frame, Bx, By, Bz, alpha, gamma, freq, psi, gradient, acoustic_freq