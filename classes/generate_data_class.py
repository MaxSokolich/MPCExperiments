
import numpy as np
import sys
import classes.Learning_module_2d as GP # type: ignore

from classes.MR_simulator import Simulator
import math 
from classes.MPC import  MPC
import matplotlib.pyplot as plt
import cv2


class gen_data:
    def __init__(self):
        self.reset()
        

        
    def reset(self):
        self.counter = 0
        self.run_calibration_status = True
        self.robot_list = None
        
        self.reading_completed = False

        self.reading_actions = False
        self.dataset_GP = []  #data from generate data function 
        freq_ls = np.linspace(0, 7, 2)
        #actions = np.array([[1, 0.3*np.pi*((t/time_steps)-1)*(-1)**(t//300)] 
        #                        for t in range(1,time_steps)]) # [T,action_dim]
        actions_learn = np.array([])

        for freq in freq_ls:
            time_steps = 36 #train for 10s at 30 hz
            cycles = 1 #train my moving in 3 circles

            steps = (int)(time_steps / cycles)

            #generate actions to move in a circle at a constant frequency
            actions_circle = np.zeros( (steps, 2))
            actions_circle[:,0] = freq
            actions_circle[:,1] = np.linspace(0, 2*np.pi, steps)

            #stack the circle actions to get our learning set
            actions_circle_combined = np.vstack([actions_circle]*cycles)
            if len(actions_learn) == 0 :
                actions_learn = actions_circle_combined
            else:
                actions_learn = np.vstack((actions_learn, actions_circle_combined))
            


        self.actions_learn = actions_learn
        print('action data size = ',len(self.actions_learn))

    def run_calibration(self, robot_list):
        
        curernt_pos = robot_list[-1].position_list[-1] #the most recent position at the time of clicking run algo
        
        #print(curernt_pos)

        direction_vec = [1800 - curernt_pos[0], 1400 - curernt_pos[1]]
        error = np.sqrt(direction_vec[0] ** 2 + direction_vec[1] ** 2)
        start_alpha = np.arctan2(-direction_vec[1], direction_vec[0]) - np.pi/2
        
        if error < 5:
            Bx = 0
            By = 0
            Bz = 0
            alpha = 0
            gamma = 0
            freq = 0
            psi = 0
            gradient = 0
            acoustic_freq = 0
            self.run_calibration_status = False
            self.reading_actions = True
            
        else:
            self.run_calibration_status = True
            Bx = 0
            By = 0
            Bz = 0
            alpha = start_alpha
            gamma = np.pi/2
            freq = 10
            psi = np.pi/2
            gradient = 0
            acoustic_freq = 0

        return Bx, By, Bz, alpha, gamma, freq, psi, gradient, acoustic_freq

    def run(self): #this executes at every frame

        
       
        
    
            
        if self.counter < len(self.actions_learn):
                
            self.reading_actions = True
            Bx = 0
            By = 0
            Bz = 0
            alpha = self.actions_learn[self.counter][1]
            gamma = np.pi/2
            freq = self.actions_learn[self.counter][0]
            psi = np.pi/2
            gradient = 0
            acoustic_freq = 0
            if self.counter == len(self.actions_learn)-1:
                self.reading_completed = True
                print('reading_completed')
         
                
            
        else:
            self.reading_completed = False
            self.reading_actions = False
            Bx = 0
            By = 0
            Bz = 0
            alpha = 0
            gamma = np.pi/2
            freq = 0
            psi = np.pi/2
            gradient = 0
            acoustic_freq = 0

        self.counter +=1
           
    
            

        return Bx, By, Bz, alpha, gamma, freq, psi, gradient, acoustic_freq