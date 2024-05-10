

import pandas as pd

import numpy as np
import sys
import Learning_module_2d as GP # type: ignore
from utils import readfile,test_gp,find_alpha_corrected
from utils import plot_xy,plot_traj,plot_vel,plot_bounded_curves
from scipy.ndimage import uniform_filter1d
from MR_simulator import Simulator
import math 
# from MPC import mpc_control
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Path to your Excel file
# file_path = 'expriment_data/2024.04.10-16.04.35.xlsx'
def num_velocity(time, px, py, freq, alpha):
    time -= time[0]
    
    # apply smoothing to the position signals before calculating velocity 
    # dt is ~ 35 ms, so filter time ~= 0.035*N (this gives N = 38)
    N = (int)(1 / 0.035 / 2) #filter position data due to noisy sensing

    px = uniform_filter1d(px, N, mode="nearest")
    py = uniform_filter1d(py, N, mode="nearest")

    #calculate velocity via position derivative
    vx = np.gradient(px, time)
    vy = np.gradient(py, time)

    # apply smoothing to the velocity signal
    vx = uniform_filter1d(vx, (int)(N/2), mode="nearest")
    vy = uniform_filter1d(vy, (int)(N/2), mode="nearest")

    
    vx = vx[N:-N]
    vy = vy[N:-N]
    time = time[N:-N]
    alpha = alpha[N:-N]
    freq = freq[N:-N]
    return vx, vy, freq, alpha


def linear_reg(alpha, freq, vx, vy):
    ux = freq*np.sin(alpha)
    uy = freq*np.cos(alpha)



    model_x = LinearRegression()
    model_y = LinearRegression()

    # Train the model
    model_x.fit(np.reshape(ux,(-1,1)), np.reshape(vx,(-1,1)))

    model_y.fit(np.reshape(uy,(-1,1)), np.reshape(vy,(-1,1)))
    # Display the coefficients
    print("a0_x:", model_x.coef_)
    print("D_x:", model_x.intercept_)

    X_new = np.array([[ux.min()], [ux.max()]])  # X values for prediction
    y_predict = model_x.predict(X_new)

    # Plotting the data points
    # plt.scatter(ux, vx, color='blue', label='Data points')

    # # Plotting the regression line
    # plt.plot(X_new, y_predict, color='red', label='Regression line')
    print("a0_y:", model_y.coef_)
    print("D_y:", model_y.intercept_)

    # Adding labels and title
    # plt.xlabel('Independent variable X')
    # plt.ylabel('Dependent variable y')
    # plt.title('Linear Regression')
    # plt.legend()
    # plt.show()


    X_new = np.array([[uy.min()], [uy.max()]])  # X values for prediction
    y_predict = model_y.predict(X_new)

    # Plotting the data points
    # plt.scatter(uy, vy, color='blue', label='Data points')

    # # Plotting the regression line
    # plt.plot(X_new, y_predict, color='red', label='Regression line')

    # # Adding labels and title
    # plt.xlabel('Independent variable X')
    # plt.ylabel('Dependent variable y')
    # plt.title('Linear Regression')
    # plt.legend()
    # plt.show()
    a0 =0.5*model_x.coef_[0][0]+0.5*model_y.coef_[0][0]


    ####Visualize the error
    # ex = vx-a0*ux
    # ey = vy-a0*uy
    # fig, ax = plt.subplots()
    # ax.plot(alpha[600:697], ex[600:697])
    # ax.plot(alpha[700:797], ex[700:797])
    # ax.plot(alpha[800:897], ex[800:897])


    # freq_grid, alpha_grid = np.meshgrid(freq, alpha)

    # # Create a new figure for plotting
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    # # Plotting the surface
    # surface = ax.plot_surface(freq_grid, alpha_grid, np.reshape(ex,(-1,1)), cmap='viridis')

    # # Add a color bar which maps values to colors
    # fig.colorbar(surface, shrink=0.5, aspect=5)

    # # Labels and title
    # ax.set_xlabel('Frequency (freq)')
    # ax.set_ylabel('Alpha')
    # ax.set_zlabel('Ex Value')
    # ax.set_title('Surface Plot of Ex')

    # Show plot
    # plt.show()
    return a0


def esitmate_a0(alpha, freq, vx, vy):
   



    model = LinearRegression()
    
    v_square = vx**2+vy**2
    # Train the model
    model.fit(np.reshape(freq**2,(-1,1)), np.reshape(v_square,(-1,1)))

    
    # Display the coefficients
    print("a0:", np.sqrt(model.coef_))
    print("D**2:", model.intercept_)

    X_new = np.array([[(freq**2).min()], [(freq**2).max()]]) # X values for prediction
    y_predict = model.predict(X_new)

    # Plotting the data points
    plt.scatter(freq**2, v_square, color='blue', label='Data points')
    plt.plot(X_new, y_predict, color='red', label='Regression line')
    # Plotting the regression line
    
    # Adding labels and title
    plt.xlabel('Independent variable X')
    plt.ylabel('Dependent variable y')
    plt.title('Linear Regression')
    plt.legend()
    plt.show()


    return np.sqrt(model.coef_)




        



def read_data_action(file_path):
    # If you are not sure about the sheet names, you can list all sheet names like this
    xls = pd.ExcelFile(file_path)
    sheet_names = xls.sheet_names  # This will list all sheet names



    # To read a specific sheet by name
    sheet1_df = pd.read_excel(file_path, sheet_name=sheet_names[0])
    sheet2_df = pd.read_excel(file_path, sheet_name=sheet_names[1])
   
    first_non_zero_alpha = sheet1_df[sheet1_df['Alpha'] != 0].index[0]
    ###Combing important rows and save it to df
    start_frame_1 =sheet1_df['Frame'][0]
    start_frame_2 =sheet2_df['Frame'][0]

    end_frame_1 =sheet1_df['Frame'][len(sheet1_df['Frame'])-1]
    end_frame_2 =sheet2_df['Frame'][len(sheet2_df['Frame'])-1]
    end_frame = min(end_frame_1, end_frame_2)
    df = pd.DataFrame([])

    freq_ls = np.unique(sheet1_df['Rolling Frequency'][first_non_zero_alpha:].to_numpy())
    freq_ls = freq_ls[:-1]
    alpha_ls = np.unique(sheet1_df['Alpha'][first_non_zero_alpha:].to_numpy())


    last_full_cycle = np.where(sheet1_df['Rolling Frequency'].to_numpy() == freq_ls[-1])[0][-1]+1

    dt_ls = sheet2_df['Times'][first_non_zero_alpha:last_full_cycle].to_list()
    time = np.zeros_like(dt_ls)
    for i in range(len(time)-1):
        time[i+1]= time[i]+dt_ls[i]
    df['Frame'] = sheet1_df['Frame'][first_non_zero_alpha:last_full_cycle]
    df['Rolling Frequency'] = sheet1_df['Rolling Frequency'][first_non_zero_alpha:]
    df['Alpha'] = sheet1_df['Alpha'][first_non_zero_alpha:last_full_cycle]
    df['Times'] = time
    df['Pos X'] = sheet2_df['Pos X'][first_non_zero_alpha:last_full_cycle]
    df['Pos Y'] = sheet2_df['Pos Y'][first_non_zero_alpha:last_full_cycle]
    df['Stuck?'] = sheet2_df['Stuck?'][first_non_zero_alpha:last_full_cycle]
    df['Vel X'] = sheet2_df['Vel X'][first_non_zero_alpha:last_full_cycle]
    df['Vel Y'] = sheet2_df['Vel Y'][first_non_zero_alpha:last_full_cycle]

    
   
    
    lf  = len(freq_ls)
    la = len(alpha_ls)
    vx_grid = np.zeros([3,len(freq_ls),len(alpha_ls)])
    vy_grid = np.zeros([3,len(freq_ls),len(alpha_ls)])
    
    for i in range(lf):
        ind = int(3*i)
        vx_grid[0,i,:] = df['Vel X'][int(ind*la):int((ind+1)*la)].to_numpy()
        vx_grid[1,i,:] = df['Vel X'][int((ind+1)*la):int((ind+2)*la)].to_numpy()
        vx_grid[2,i,:] = df['Vel X'][int((ind+2)*la):int((ind+3)*la)].to_numpy()
        vy_grid[0,i,:] = df['Vel Y'][int(ind*la):int((ind+1)*la)].to_numpy()
        vy_grid[1,i,:] = df['Vel Y'][int((ind+1)*la):int((ind+2)*la)].to_numpy()
        vy_grid[2,i,:] = df['Vel Y'][int((ind+2)*la):int((ind+3)*la)].to_numpy()

        


    
    alpha_grid, freq_grid  = np.meshgrid(alpha_ls, freq_ls)
    return  alpha_grid, freq_grid ,np.mean(vx_grid, axis=0), np.mean(vy_grid, axis=0)


def plot_surface(alpha_grid, freq_grid ,vx_grid, vy_grid):

    
    # Create a new figure for plotting
    fig = plt.figure()
    

    # Plotting the surface
    ax1 = fig.add_subplot(121, projection='3d')  # Changed from 111 to 121 for 1 row, 2 cols, 1st subplot
    surface1 = ax1.plot_surface(alpha_grid, freq_grid, np.mean(vx_grid, axis=0), cmap='viridis')
    fig.colorbar(surface1, shrink=0.5, aspect=5)

    # Second subplot
    ax2 = fig.add_subplot(122, projection='3d')  # Changed to 122 for 1 row, 2 cols, 2nd subplot
    surface2 = ax2.plot_surface(alpha_grid, freq_grid, np.mean(vy_grid, axis=0), cmap='viridis')
    fig.colorbar(surface2, shrink=0.5, aspect=5)

    plt.show()




gp_sim = GP.LearningModule()

#first we will do absolutely nothing to try and calculate the drift term
# file_path_idle = 'expriment_data/idle_action_data.xlsx'

# px_idle,py_idle,alpha_idle,time_idle,freq_idle,vx_idle, vy_idle = read_data(file_path_idle,0)
# gp_sim.estimateDisturbance(px_idle, py_idle, time_idle)

file_path_action = 'expriment_data/control_action_data.xlsx'

alpha_grid, freq_grid ,vx_grid, vy_grid= read_data_action(file_path_action)
X = np.vstack( [alpha_grid.flatten(), freq_grid.flatten()] ).transpose()
Y = np.vstack([vx_grid.flatten(), vy_grid.flatten()]).transpose()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15, random_state=42)
# plt.scatter(freq_sim*np.cos(alpha_sim),vy_d)
# plt.scatter(freq_sim*np.sin(alpha_sim),vx_d)
# plt.show()
# vx, vy, freq, alpha= num_velocity(time_sim, px_sim, py_sim, freq_sim, alpha_sim)

# a0_est = esitmate_a0(alpha[0:4000], freq[0:4000], vx[0:4000],vy[0:4000])

a0_est = linear_reg(alpha_grid.flatten(), freq_grid.flatten(), vx_grid.flatten(),vy_grid.flatten())
print(a0_est)
# a0_est = linear_reg(alpha, freq, vx, vy)
# learn noise and a0 -- note px_desired and py_desired need to be at the same time
# mean_vx = (vx_d[600:697]+vx_d[700:797]+vx_d[800:897])/3
# mean_vy =(vy_d[600:697]+vy_d[700:797]+vy_d[800:897])/3
gp_sim.a0 = a0_est
# gp_sim.learn(vx_grid.flatten(), vy_grid.flatten(), alpha_grid.flatten(), freq_grid.flatten())
# gp_sim.learn( Y_train[:,0], Y_train[:,1], X_train[:,0], X_train[:,1])

gp_sim.load_GP()
# gp_sim.visualize_mu(alpha_grid, freq_grid ,vx_grid, vy_grid)

# a0_sim = gp_sim.learn(mean_vx, mean_vy, alpha_sim[600:697], freq_sim[600:697], a0_est)
print("Estimated a0 value is " + str(a0_est))




######################Checking if it is an overfit


from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error



muX,sigX = gp_sim.gprX.predict(X_test, return_std=True)
muY,sigY = gp_sim.gprY.predict(X_test, return_std=True)
# mse = mean_squared_error(Y_test, Y_pred)

maeX = mean_absolute_error(Y_test[:,0]-a0_est*X_test[:,1]*np.sin(X_test[:,0]), muX)
maeY = mean_absolute_error(Y_test[:,1]-a0_est*X_test[:,1]*np.cos(X_test[:,0]), muY)
print('maeX = ', maeX, 'maeY=', maeY)

# gp_sim.visualize()
# ux = freq_sim*np.sin(alpha_sim)
# uy = freq_sim*np.cos(alpha_sim)
# ex = vx_d-a0_est*ux
# ey = vy_d-a0_est*uy



# mean_ux = (ux[600:697]+ux[700:797]+ux[800:897])/3
# mean_uy =(uy[600:697]+uy[700:797]+uy[800:897])/3
# fig, ax = plt.subplots(2)



# ax[0].plot(alpha_sim[600:697], ex[600:697])
# ax[0].plot(alpha_sim[700:797], ex[700:797])
# ax[0].plot(alpha_sim[800:897], ex[800:897])
# ax[0].plot(alpha_sim[600:697],mean_vx-a0_est*mean_ux , color = 'black')

# ax[1].plot(alpha_sim[600:697], ey[600:697])
# ax[1].plot(alpha_sim[700:797], ey[700:797])
# ax[1].plot(alpha_sim[800:897], ey[800:897])
# ax[1].plot(alpha_sim[600:697],mean_vy-a0_est*mean_uy ,  color = 'black')



# X = np.vstack( [alpha_sim[600:697], freq_sim[600:697]] ).transpose()
# muX,sigX = gp_sim.gprX.predict(X, return_std=True)
# muY,sigY = gp_sim.gprY.predict(X, return_std=True)
# ax[0].plot(alpha_sim[600:697], muX)
# ax[1].plot(alpha_sim[600:697], muY)
# # ax[2].plot(alpha_sim[600:697], sigX)
# # ax[3].plot(alpha_sim[600:697], sigY)
# plt.show()