
import numpy as np
import sys
import classes.Learning_module_2d as GP # type: ignore

from classes.MR_simulator import Simulator
import math 
from classes.MPC import  MPC
import matplotlib.pyplot as plt
import cv2
from classes.algorithm_class import algorithm


class gen_data2:
    def __init__(self):

        self.reset()
        
        

        
    def reset(self):
 
        self.max_steps = 1000
        
        ### freq range for gen data
        self.f_min = 0
        self.f_max =5
   
        self.frange_size = self.f_max-self.f_min +1 
        self.run_calibration_status = True
        self.robot_list = None
        
        self.reading_completed = False

        self.reading_actions = False
        self.dataset_GP2 = []  #data from generate data function 
      
        self.time_steps = 480 ##number of data points for each freq
        #actions = np.array([[1, 0.3*np.pi*((t/time_steps)-1)*(-1)**(t//300)] 
        #                        for t in range(1,time_steps)]) # [T,action_dim]

        self.algorithm  = algorithm()
        self.calibration_coord = [self.algorithm.init_point_x, self.algorithm.init_point_y]

 

    def run_infinity(self, robot_list, frame): #this executes at every frame

        
        if self.algorithm.counter == self.max_steps:
            self.reading_completed = True
            print('reading_completed')

        else:
            self.reading_completed = False
            self.reading_actions = False
            frame, Bx, By, Bz, alpha, gamma, freq, psi, gradient, acoustic_freq = self.algorithm.generate_data(robot_list, frame)

        

        return frame, Bx, By, Bz, alpha, gamma, freq, psi, gradient, acoustic_freq
    
    def run_calibration_infinity(self, robot_list):
        
        curernt_pos = robot_list[-1].position_list[-1] #the most recent position at the time of clicking run algo

        direction_vec = [self.calibration_coord[0] - curernt_pos[0], self.calibration_coord[1] - curernt_pos[1]]
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