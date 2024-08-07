from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import *
from scipy.ndimage import uniform_filter1d
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

from mpl_toolkits.mplot3d import Axes3D

from scipy.optimize import minimize, minimize_scalar

from sklearn.model_selection import train_test_split
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.linear_model import LinearRegression
import os

# Function to find the closest grid point by Euclidean distance
def find_closest_grid_point(alpha, freq, alpha_grid, freq_grid):
    distances = np.sqrt((alpha_grid - alpha) ** 2 + (freq_grid - freq) ** 2)
    closest_idx = np.unravel_index(np.argmin(distances), distances.shape)
    return closest_idx

def objective(X, a0, v_d, GPx, GPy):

    
    alpha = X[0]
    freq  = X[1]
    
    X = np.array([alpha, freq]).transpose()
    
    mux = GPx.predict(X.reshape(1, -1))
    muy = GPy.predict(X.reshape(1, -1))


    #return (a0*freq*np.cos(alpha) + mux - v_d[0])**2 + (a0*freq*np.sin(alpha) + mux - v_d[1])**2

    return (a0*freq)**2 + (mux - v_d[0])**2 + 2*a0*freq*np.cos(alpha)*(mux - v_d[0]) + (muy - v_d[1])**2 + 2*a0*freq*np.sin(alpha)*(muy - v_d[1])


class LearningModule:
    def __init__(self,cycle):
        # kernel list is the kernel cookbook from scikit-learn
        
        kernel = ConstantKernel(1.0, (1e-2, 500.0))* RBF(length_scale=1.0, length_scale_bounds=(1e-2, 500.0)) + ConstantKernel(1.0, (1e-2, 500.0))*WhiteKernel()

        # kernel = ConstantKernel(1.1, (1e-2, 1e2))* RBF(length_scale=1.0, length_scale_bounds=(1e-2, 100.0)) + WhiteKernel()
        #create the X and Y GP regression objects
        self.gprX = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
        self.gprY = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)

        self.X = []
        self.Yx = []
        self.Yy = []

        self.a0 = 0
        self.f = 0
        self.Dx = 0
        self.Dy = 0
        self.plot = False
        self.cycle = cycle
        
    def normalize_angle(self, angles):
        """
        Normalize a single angle, list of angles, or numpy array of angles to the range [0, 2π].
        Returns:
        float, list, or numpy array: Normalized angle(s) in the range [0, 2π].
        """
        if isinstance(angles, (list, np.ndarray)):
            return np.mod(angles, 2 * np.pi)
        else:
            return angles % (2 * np.pi)

        
    def load_GP(self, directory=None):
        """
        Load Gaussian Process models and other necessary files.

        Args:
        directory (str, optional): Directory to load files from. Defaults to None.
        """
        # Set the default directory if none is provided
        if directory is None:
            directory = 'classes'
        
        # Construct file paths
        gpY_path = os.path.join(directory, 'gpY_2d.pkl')
        gpX_path = os.path.join(directory, 'gpX_2d.pkl')
        a0_path = os.path.join(directory, 'a0_est.npy')
        
        # Load files
        self.gprY = joblib.load(gpY_path)
        self.gprX = joblib.load(gpX_path)
        self.a0 = np.load(a0_path)
        
        print('GP is loaded')

    def visualize_mu(self, alpha_grid, freq_grid ,vx_grid, vy_grid):
        error_x = vx_grid - self.a0 * freq_grid * np.sin(alpha_grid)
        error_y = vy_grid - self.a0 * freq_grid * np.cos(alpha_grid)
        

        X = np.vstack((alpha_grid.ravel(), freq_grid.ravel())).T
        # for ix in range(alpha_grid.shape[0]):
        #     for iy in range(alpha_grid.shape[1]):
        #         X = np.vstack([[alpha_grid[ix][iy]], [freq_grid[ix][iy]]] ).transpose()
              

        #         #evaluate the GPs
        muX,sigX = self.gprX.predict(X, return_std=True)
        muY,sigY = self.gprY.predict(X, return_std=True)
        gp_est_x = muX.reshape(alpha_grid.shape)
        gp_est_y = muY.reshape(alpha_grid.shape)
        # Create a new figure for plotting
        fig = plt.figure()
        

        # Plotting the surface
        ax1 = fig.add_subplot(121, projection='3d')  # Changed from 111 to 121 for 1 row, 2 cols, 1st subplot
        # surface1 = ax1.plot_surface(alpha_grid, freq_grid, error_x, cmap='viridis')
        surface11 = ax1.plot_surface(alpha_grid, freq_grid, gp_est_x, cmap='viridis')
        # fig.colorbar(surface1, shrink=0.5, aspect=5)

        # Second subplot
        ax2 = fig.add_subplot(122, projection='3d')  # Changed to 122 for 1 row, 2 cols, 2nd subplot
        # surface2 = ax2.plot_surface(alpha_grid, freq_grid, error_y, cmap='viridis')
        surface2 = ax2.plot_surface(alpha_grid, freq_grid, gp_est_y, cmap='viridis')
        # fig.colorbar(surface2, shrink=0.5, aspect=5)

        plt.show()


    def read_data_action(self, data, obj):
        # If you are not sure about the sheet names, you can list all sheet names like this
        # cycle = 1
        # sc_factor = 0.28985 * obj
        sc_factor = 0.28985 * obj
        # To read a specific sheet by name
        data = np.array(data)
        freq_read = data[:,-1]
        alpha_read = data[:,-2]
        vx_read = data[:,3]
        vy_read = data[:,4]
        freq_ls = np.unique(freq_read)
        alpha_ls = np.unique(alpha_read)


        lf  = len(freq_ls)
        la = len(alpha_ls)
        vx_grid = np.zeros([self.cycle,len(freq_ls),len(alpha_ls)])
        vy_grid = np.zeros([self.cycle,len(freq_ls),len(alpha_ls)])
        
        for i in range(lf):
            
            for ci in range(self.cycle):
                start_ind= int(la*self.cycle*i+ci*la)
                end_ind = int(la*self.cycle*i+(ci+1)*la)
                vx_grid[ci,i,:] = vx_read[start_ind:end_ind]
                # vx_grid[1,i,:] = vx_read[int((ind+1)*la):int((ind+2)*la)]
                # vx_grid[2,i,:] = vx_read[int((ind+2)*la):int((ind+3)*la)]
                vy_grid[ci,i,:] = vy_read[start_ind:end_ind]
                # vy_grid[1,i,:] = vy_read[int((ind+1)*la):int((ind+2)*la)]
                # vy_grid[2,i,:] = vy_read[int((ind+2)*la):int((ind+3)*la)]

                


        
        alpha_grid, freq_grid  = np.meshgrid(alpha_ls, freq_ls)
        self.alpha_grid, self.freq_grid ,self.vx_grid, self.vy_grid = self.normalize_angle(np.pi/2+alpha_grid), freq_grid ,np.mean(vx_grid, axis=0), np.mean(vy_grid, axis=0)
        # return  alpha_grid, freq_grid ,np.mean(vx_grid, axis=0), np.mean(vy_grid, axis=0)



     
    def dyn_model(self, alpha, freq):
      

        #evaluate the GPs
        X = np.array([[alpha, freq]])
  
        muX,sigX = self.gprX.predict(X, return_std=True)
        muY,sigY = self.gprY.predict(X, return_std=True)

        v_x = self.a0*freq*np.cos(alpha)+muX
        v_y = -self.a0*freq*np.sin(alpha)+muY
        return [v_x, v_y], [self.a0*freq*np.cos(alpha), -self.a0*freq*np.sin(alpha)],[sigX, sigY]
        
    def read_data_action2(self, data, obj):
        # If you are not sure about the sheet names, you can list all sheet names like this
        # cycle = 1
        # sc_factor = 0.28985 * obj
        sc_factor = 0.28985 * obj
        # To read a specific sheet by name
        data = np.array(data)
        freq_read = data[:,-1]
        alpha_read = np.pi/2+data[:,-2]
        vx_read = data[:,3]
        vy_read = data[:,4]
        # freq_ls = np.unique(freq_read)
        # alpha_ls = np.unique(alpha_read)
  
    
        self.alpha_infinity, self.freq_infinity ,self.vx_infinity, self.vy_infinity = self.normalize_angle(alpha_read), freq_read ,vx_read, vy_read
        self.alpha_all, self.freq_all = np.hstack([self.alpha_grid.flatten(), self.alpha_infinity.flatten()]), np.hstack([self.freq_grid.flatten(), self.freq_infinity.flatten()])
        self.vx_all, self.vy_all = np.hstack([self.vx_grid.flatten(), self.vx_infinity.flatten()]), np.hstack([self.vy_grid.flatten(), self.vy_infinity.flatten()])
        
        
        data = {
            'alpha': self.alpha_all,
            'freq': self.freq_all,
            'vx': self.vx_all,
            'vy': self.vy_all
        }
        df = pd.DataFrame(data)
        # Define the bin sizes for alpha and freq
        

        # # Create bins for alpha and freq based on bin sizes
        # alpha_bins = np.arange(df['alpha'].min(), df['alpha'].max() + bin_size_alpha, bin_size_alpha)
        # freq_bins = np.arange(df['freq'].min(), df['freq'].max() + bin_size_freq, bin_size_freq)

        # # Assign each point to a bin
        # df['alpha_bin'] = pd.cut(df['alpha'], bins=alpha_bins, labels=False)
        # df['freq_bin'] = pd.cut(df['freq'], bins=freq_bins, labels=False)

        # # Group by the bins and calculate the mean vx and vy
        # grouped = df.groupby(['alpha_bin', 'freq_bin']).agg({'vx': 'mean', 'vy': 'mean'}).reset_index()
        #         # return  alpha_grid, freq_grid ,np.mean(vx_grid, axis=0), np.mean(vy_grid, axis=0)
        # print('all_dataset_shape=', self.vx_all.shape)
        # self.alpha_grouped, self.freq_grouped, self.vx_grouped, self.vy_grouped = np.array(grouped['alpha_bin']),  np.array(grouped['freq_bin']),  np.array(grouped['vx']),  np.array(grouped['vy'])
        # print('grouped_dataset_shape=', self.alpha_grouped.shape)






        # Define the range and resolution of alpha and freq
        alpha_min, alpha_max, alpha_steps = 0, 2*np.pi, 200  # Replace with your values
        freq_min, freq_max, freq_steps = 0, 5, 15  # Replace with your values

        alpha_range = np.linspace(alpha_min, alpha_max, alpha_steps)
        freq_range = np.linspace(freq_min, freq_max, freq_steps)

        # Create the mesh grid
        alpha_grid, freq_grid = np.meshgrid(alpha_range, freq_range)

        # Initialize vx and vy grids with None and count grids with zeros
        vx_grid = np.full(alpha_grid.shape, None)
        vy_grid = np.full(freq_grid.shape, None)
        count_grid = np.zeros(alpha_grid.shape)


      

        # Process each data point
        for index, point in df.iterrows():
            alpha_idx, freq_idx = find_closest_grid_point(point['alpha'], point['freq'], alpha_grid, freq_grid)

            if vx_grid[alpha_idx, freq_idx] is None:
                vx_grid[alpha_idx, freq_idx] = point['vx']
                vy_grid[alpha_idx, freq_idx] = point['vy']
                count_grid[alpha_idx, freq_idx] = 1
            else:
                count_grid[alpha_idx, freq_idx] += 1
                vx_grid[alpha_idx, freq_idx] = ((vx_grid[alpha_idx, freq_idx] * (count_grid[alpha_idx, freq_idx] - 1)) + point['vx']) / count_grid[alpha_idx, freq_idx]
                vy_grid[alpha_idx, freq_idx] = ((vy_grid[alpha_idx, freq_idx] * (count_grid[alpha_idx, freq_idx] - 1)) + point['vy']) / count_grid[alpha_idx, freq_idx]

        # Extract non-None values into lists for the dataset
        self.alpha_grouped = []
        self.freq_grouped = []
        self.vx_grouped = []
        self.vy_grouped = []

        for i in range(vx_grid.shape[0]):
            for j in range(vx_grid.shape[1]):
                if vx_grid[i, j] is not None and vy_grid[i, j] is not None:
                    self.alpha_grouped.append(alpha_grid[i, j])
                    self.freq_grouped.append(freq_grid[i, j])
                    self.vx_grouped.append(vx_grid[i, j])
                    self.vy_grouped.append(vy_grid[i, j])

        # Create the dataset and convert to a DataFrame
        # data = {
        #     'alpha': alpha_all,
        #     'freq': freq_all,
        #     'vx': vx_all,
        #     'vy': vy_all
        # }

        
        self.alpha_grouped, self.freq_grouped, self.vx_grouped, self.vy_grouped = np.array( self.alpha_grouped),  np.array(self.freq_grouped),  np.array(self.vx_grouped),  np.array(self.vy_grouped)
        print('all_dataset_shape=',self.vx_grouped.shape)
      

















    
    




    
    def estimate_a0(self, mod):
        if mod == 0:
        #### Estimate a0 and train GP
            X = np.vstack( [self.alpha_grid.flatten(), self.freq_grid.flatten()] ).transpose()
            Y = np.vstack([self.vx_grid.flatten(), self.vy_grid.flatten()]).transpose()

            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15, random_state=42)
            # plt.scatter(self.freq_sim*np.cos(self.alpha_sim),self.vx_grid.flatten())
            # plt.scatter(self.freq_sim*np.sin(self.alpha_sim),self.vy_grid.flatten())
            # plt.show()
            
            a0_est = self.linear_reg(self.alpha_grid.flatten(), self.freq_grid.flatten(), self.vx_grid.flatten(),self.vy_grid.flatten())
            print('a0_est=',a0_est)
            np.save('a0_est.npy', a0_est)
        
    
            self.a0 = a0_est
            self.learn(self.vx_grid.flatten(), self.vy_grid.flatten(), self.alpha_grid.flatten(), self.freq_grid.flatten())
            # self.learn( Y_train[:,0], Y_train[:,1], X_train[:,0], X_train[:,1])
            print('Trainig completed')
            if self.plot:
                self.visualize_mu(self.alpha_grid, self.freq_grid ,self.vx_grid, self.vy_grid)


        if mod == 1:
            print('reacehd here 2')
        #### Estimate a0 and train GP
            # X = np.vstack( [self.alpha_all.flatten(), self.freq_all.flatten()] ).transpose()
            # Y = np.vstack([self.vx_all.flatten(), self.vy_all.flatten()]).transpose()
            # print('datasize===', len(self.vx_all.flatten()))

            # X = np.vstack( [self.alpha_grouped.flatten(), self.freq_grouped.flatten()] ).transpose()
            # Y = np.vstack([self.vx_grouped.flatten(), self.vy_grouped.flatten()]).transpose()
            # print('datasize===', len(self.vx_grouped.flatten()))

            # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15, random_state=42)
            # plt.scatter(self.freq_sim*np.cos(self.alpha_sim),self.vx_grid.flatten())
            # plt.scatter(self.freq_sim*np.sin(self.alpha_sim),self.vy_grid.flatten())
            # plt.show()
            
            self.a0 = self.linear_reg(self.alpha_all.flatten(), self.freq_all.flatten(), self.vx_all.flatten(),self.vy_all.flatten())
            print('a0=',self.a0)
            np.save('a0.npy', self.a0)
        
    
            # self.a0 = a0_est
            self.learn(self.vx_all.flatten(), self.vy_all.flatten(), self.alpha_all.flatten(), self.freq_all.flatten())
            # self.learn( Y_train[:,0], Y_train[:,1], X_train[:,0], X_train[:,1])
            print('Trainig completed')
            if self.plot:
                self.visualize_mu(self.alpha_grid, self.freq_grid ,self.vx_grid, self.vy_grid)
            
        if mod == 2:
            n = int(len(self.alpha_all.flatten() * 70 / 100))
            
            # Generate random indices
            indices = np.random.choice(len(self.alpha_all.flatten()), n, replace=False)
            
            # Select and return the elements at the generated indices
            self.learn(self.vx_all.flatten()[indices], self.vy_all.flatten()[indices], self.alpha_all.flatten()[indices], self.freq_all.flatten()[indices])
        if mod == 3:
            # Select and return the elements at the generated indices
            print("mod 3")
            self.a0 = self.linear_reg(self.alpha_grid.flatten(), self.freq_grid.flatten(), self.vx_grid.flatten(),self.vy_grid.flatten())
            self.learn(self.vx_grouped.flatten(), self.vy_grouped.flatten(), self.alpha_grouped.flatten(), self.freq_grouped.flatten())



        

    

    def linear_reg(self, alpha, freq, vx, vy):
        
        ux = freq*np.cos(alpha)
        uy = freq*np.sin(alpha)



        model_x = LinearRegression()
        model_y = LinearRegression()

        # Train the model
        model_x.fit(np.reshape(ux,(-1,1)), np.reshape(vx,(-1,1)))

        model_y.fit(np.reshape(uy,(-1,1)), np.reshape(vy,(-1,1)))
        # Display the coefficients
        print("a0_x:", model_x.coef_)
        print("D_x:", model_x.intercept_)



        print("a0_y:", -model_y.coef_)
        print("D_y:", -model_y.intercept_)

       

        # Adding labels and title
        # plt.xlabel('Independent variable X')
        # plt.ylabel('Dependent variable y')
        # plt.title('Linear Regression')
        # plt.legend()
        # plt.show()


        
        if self.plot:
            X_new_x = np.array([[ux.min()], [ux.max()]])  # X values for prediction
            y_predict_x = model_x.predict(X_new_x)

            # Plotting the data points
            plt.scatter(ux, vx, color='blue', label='Data points')

            # Plotting the regression line
            plt.plot(X_new_x, y_predict_x, color='red', label='Regression line')
            print("a0_y:", model_y.coef_)
            print("D_y:", model_y.intercept_)
             ### Adding labels and title
            plt.xlabel('Independent variable X ')
            plt.ylabel('Dependent variable y')
            plt.title('Linear Regression_ x axis')
            plt.legend()
            plt.show()


            

            # Plotting the regression line
            X_new_y = np.array([[uy.min()], [uy.max()]])  # X values for prediction
            y_predict_y = model_y.predict(X_new_y)
            plt.plot(X_new_y, y_predict_y, color='red', label='Regression line')
            print("a0_y:", model_y.coef_)
            print("D_y:", model_y.intercept_)

           
            plt.scatter(uy, vy, color='blue', label='Data points')# Plotting the data points

            

            # Adding labels and title
            plt.xlabel('Independent variable X')
            plt.ylabel('Dependent variable y')
            plt.title('Linear Regression y Axis')
            plt.legend()
            plt.show()
        a0 =0.5*model_x.coef_[0][0]-0.5*model_y.coef_[0][0]
        print('a0=================', a0)
        self.a0 = a0


        ####Visualize the error
       

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









    def estimateDisturbance(self, px, py, time):
        N = (int)(1 / 0.035 / 2) #filter position data due to noisy sensing
     
        px = uniform_filter1d(px, N, mode="nearest")
        py = uniform_filter1d(py, N, mode="nearest")
        #calculate velocity via position derivative
        vx = np.gradient(px, time)
        vy = np.gradient(py, time)
        # apply smoothing to the velocity signal
        vx = uniform_filter1d(vx, (int)(N/2), mode="nearest")
        vy = uniform_filter1d(vy, (int)(N/2), mode="nearest")

        self.Dx = np.mean(vx)
        self.Dy = np.mean(vy)
        print("Estimated a D value of [" + str(self.Dx) + ", " + str(self.Dy) + "].")

    # px, py, alpha, time are numpy arrays, freq is constant
    # returns an estimate of a_0
    def learn(self, vx, vy, alpha, freq):
        #set time to start at 0
        # time -= time[0]
        # trsh = 900
        # px = px[0:trsh]
        # py = py[0:trsh]
        # time = time[0:trsh]
        # alpha= alpha[0:trsh]
        # freq = freq[0:trsh]

        # # apply smoothing to the position signals before calculating velocity 
        # # dt is ~ 35 ms, so filter time ~= 0.035*N (this gives N = 38)
        # N = (int)(1 / 0.035 / 2) #filter position data due to noisy sensing

        # px = uniform_filter1d(px, N, mode="nearest")
        # py = uniform_filter1d(py, N, mode="nearest")

        # #calculate velocity via position derivative
        # vx = np.gradient(px, time)
        # vy = np.gradient(py, time)

        # # apply smoothing to the velocity signal
        # vx = uniform_filter1d(vx, (int)(N/2), mode="nearest")
        # vy = uniform_filter1d(vy, (int)(N/2), mode="nearest")
       

        #calculate speed to fit a_0
        # speed = np.sqrt( (vx - self.Dx )**2 + (vy - self.Dy)**2 )

        #alpha  = 1k means the controller is off, delete those frames
        # todel = np.argwhere(alpha >= 500)
        # if len(todel) > 0:
        #     todel = int(todel[0])
        #     alpha = alpha[0:todel-1]
        #     freq  = freq[0:todel-1]
        #     px = px[0:todel-1]
        #     py = py[0:todel-1]
        #     vx = vx[0:todel-1]
        #     vy = vy[0:todel-1]
        #     time = time[0:todel-1]
        #     speed = speed[0:todel-1]

        #smoothing creates a boundary effect -- let's remove it
        # alpha = alpha[N:-N]
        # freq = freq[N:-N]
        # px = px[N:-N]
        # py = py[N:-N]
        # vx = vx[N:-N]
        # vy = vy[N:-N]
        # time = time[N:-N]
        # speed = speed[N:-N]

     

    # Tr
        #generate empty NP arrays for X (data) and Y (outputs)
        #X = alpha.reshape(-1,1)
        
        X = np.vstack( [alpha, freq] ).transpose()
                
        #v_e = v_actual - v_desired = v - a0*f*[ cos alpha; sin alpha]
        Yx = vx - self.a0 * freq * np.cos(alpha)
        Yy = vy + self.a0 * freq * np.sin(alpha)

        self.gprX.fit(X, Yx)
        self.gprY.fit(X, Yy)

        print("GP Learning Complete!")
        print("r^2 are " + str(self.gprX.score(X, Yx)) + " and " + str(self.gprY.score(X, Yy)) )

        
        a = np.linspace( np.min(X), np.max(X))
        f = np.zeros(a.shape) + freq[0]
        
        Xe = np.vstack( [a, f] ).transpose()
        
        e = self.gprX.predict(Xe)
        
        #plt.figure()
        #plt.plot(X, Yx, 'kx')
        #plt.plot(a, e, '-r')
        #plt.show()

        #plot the velocity error versus time
        #plt.figure()
        #plt.plot(time, vx, time, a0*freq*np.cos(alpha))
        #plt.show()


        self.X = X; self.Yx = Yx; self.Yy = Yy
        
        self.freq = freq
        joblib.dump(self.gprX, 'classes/gpX_2d.pkl')
        joblib.dump(self.gprY, 'classes/gpY_2d.pkl')

        # return a0



    def learn_sim(self, px, py, alpha, freq, time):
        #set time to start at 0
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

        #calculate speed to fit a_0
        speed = np.sqrt( (vx - self.Dx )**2 + (vy - self.Dy)**2 )

        #alpha  = 1k means the controller is off, delete those frames
        todel = np.argwhere(alpha >= 500)
        if len(todel) > 0:
            todel = int(todel[0])
            alpha = alpha[0:todel-1]
            freq  = freq[0:todel-1]
            px = px[0:todel-1]
            py = py[0:todel-1]
            vx = vx[0:todel-1]
            vy = vy[0:todel-1]
            time = time[0:todel-1]
            speed = speed[0:todel-1]

        #smoothing creates a boundary effect -- let's remove it
        alpha = alpha[N:-N]
        freq = freq[N:-N]
        px = px[N:-N]
        py = py[N:-N]
        vx = vx[N:-N]
        vy = vy[N:-N]
        time = time[N:-N]
        speed = speed[N:-N]

        a0 = np.median(speed / freq)

        #generate empty NP arrays for X (data) and Y (outputs)
        #X = alpha.reshape(-1,1)
        
        X = np.vstack( [alpha, freq] ).transpose()
                
        #v_e = v_actual - v_desired = v - a0*f*[ cos alpha; sin alpha]
        Yx = vx - a0 * freq * np.cos(alpha)
        Yy = vy - a0 * freq * np.sin(alpha)

        self.gprX.fit(X, Yx)
        self.gprY.fit(X, Yy)

        print("GP Learning Complete!")
        print("r^2 are " + str(self.gprX.score(X, Yx)) + " and " + str(self.gprY.score(X, Yy)) )

        
        a = np.linspace( np.min(X), np.max(X))
        f = np.zeros(a.shape) + freq[0]
        
        Xe = np.vstack( [a, f] ).transpose()
        
        e = self.gprX.predict(Xe)
        
        #plt.figure()
        #plt.plot(X, Yx, 'kx')
        #plt.plot(a, e, '-r')
        #plt.show()

        #plot the velocity error versus time
        #plt.figure()
        #plt.plot(time, vx, time, a0*freq*np.cos(alpha))
        #plt.show()


        self.X = X; self.Yx = Yx; self.Yy = Yy
        self.a0 = a0
        self.freq = freq
        joblib.dump(self.gprX, 'gpX_2d.pkl')
        joblib.dump(self.gprY, 'gpY_2d.pkl')

        return a0



    def visualize_1d(self):

        alpha_range = np.linspace( np.min(self.X[:,0]), np.max(self.X[:,0]), 200 )
        freq_range  = np.linspace( np.min(self.X[:,1]), np.max(self.X[:,1]), 200 )

        X = alpha_range.reshape(-1, 1)

        #evaluate the GPs
        muX,sigX = self.gprX.predict(X, return_std=True)
        muY,sigY = self.gprY.predict(X, return_std=True)

        #plot what the GP looks like for x velocity
        plt.figure()
        #plot mean and stdv
        plt.fill_between(alpha_range, muX + 2*sigX, muX - 2*sigX)
        plt.fill_between(alpha_range, muX + sigX, muX - sigX)
        plt.plot(alpha_range, muX, 'r')
        #plot data points
        plt.plot(self.X, self.Yx, 'kx')
        
        plt.xlabel('alpha')
        plt.ylabel('v_e^x')
        
        plt.show()
        
        #plot what the GP looks like for y velocity
        plt.figure()
        #plot mean and stdv
        plt.fill_between(alpha_range, muY + 2*sigY, muY - 2*sigY)
        plt.fill_between(alpha_range, muY + sigY, muY - sigY)
        plt.plot(alpha_range, muY, 'r')
        #plot data points
        plt.plot(self.X, self.Yy, 'kx')
        
        plt.xlabel('alpha')
        plt.ylabel('v_e^y')
        
        plt.show()
      

    def visualize(self):

        alpha_range = np.linspace( -np.pi, np.pi, 200 )
        freq_range  = np.linspace( 1, 10, 200 )
        
        
        alpha,freq = np.meshgrid(alpha_range, freq_range)
        
        print(alpha.shape)
        print(freq.shape)

        alpha_flat = np.ndarray.flatten(alpha)
        freq_flat = np.ndarray.flatten(freq)
        
        print(alpha_flat.shape)
        print(freq_flat.shape)


        X = np.vstack( [alpha_flat, freq_flat] ).transpose()

        #evaluate the GPs
        muX,sigX = self.gprX.predict(X, return_std=True)
        muY,sigY = self.gprY.predict(X, return_std=True)

        #plot what the GP looks like for x velocity
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(alpha, freq, np.reshape(sigX, alpha.shape ))
        ax.set_xlabel('alpha')
        ax.set_ylabel('f')
        ax.set_title('X Velocity Uncertainty')
        # ax.colorbar()
        
        # plt.plot(self.X[:,0], self.X[:,1], 'kx')
        
        plt.show()
        
        #plot what the GP looks like for y velocity
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(alpha, freq, np.reshape(sigY, alpha.shape ))
        ax.set_xlabel('alpha')
        ax.set_ylabel('f')
        ax.set_title('Y Velocity Uncertainty')
        # ax.colorbar()
        
        # plt.plot(self.X[:,0], self.X[:,1], 'kx')

        plt.show()
        
        #plot pm 2 stdev
        #plt.fill_between(alpha_range,  muX - 2*sigX,  muX + 2*sigX)
        #plt.fill_between(alpha_range,  muX - sigX,  muX + sigX)
        #plot the data
        #plt.plot(self.X[:,0], self.Yx, 'xk')
        #plot the approximate function
        #plt.plot(alpha_range, muX, 'g')
        #plt.title('X Axis Learning')
        #plt.xlabel("alpha")
        #plt.ylabel("V_e^x")


        #plot what the GP looks like for y velocity
        #plt.figure()
        #plot pm 2 stdev
        #plt.fill_between(alpha_range,  muY - 2*sigY,  muY + 2*sigY)
        #plt.fill_between(alpha_range,  muY - sigY,  muY + sigY)
        #plot the data
        #plt.plot(self.X[:,0], self.Yy, 'xk')
        #plot the approximate function
        #plt.plot(alpha_range, muY, 'g')
        #plt.title('Y Axis Learning')
        #plt.xlabel("alpha")
        #plt.ylabel("V_e^x")
    def visualize_mu(self, alpha_grid, freq_grid ,vx_grid, vy_grid):
        error_x = vx_grid - self.a0 * freq_grid * np.sin(alpha_grid)
        error_y = vy_grid - self.a0 * freq_grid * np.cos(alpha_grid)
        

        X = np.vstack((alpha_grid.ravel(), freq_grid.ravel())).T
        # for ix in range(alpha_grid.shape[0]):
        #     for iy in range(alpha_grid.shape[1]):
        #         X = np.vstack([[alpha_grid[ix][iy]], [freq_grid[ix][iy]]] ).transpose()
              

        #         #evaluate the GPs
        muX,sigX = self.gprX.predict(X, return_std=True)
        muY,sigY = self.gprY.predict(X, return_std=True)
        gp_est_x = muX.reshape(alpha_grid.shape)
        gp_est_y = muY.reshape(alpha_grid.shape)
        # Create a new figure for plotting
        fig = plt.figure()
        

        # Plotting the surface
        ax1 = fig.add_subplot(121, projection='3d')  # Changed from 111 to 121 for 1 row, 2 cols, 1st subplot
        # surface1 = ax1.plot_surface(alpha_grid, freq_grid, error_x, cmap='viridis')
        surface11 = ax1.plot_surface(alpha_grid, freq_grid, np.abs(gp_est_x-error_x)/np.abs(error_x), cmap='viridis')
        fig.colorbar(surface11, shrink=0.5, aspect=5)
        ax1.set_title('gpX')
        ax1.set_zlim(0,10)

        # Second subplot
        ax2 = fig.add_subplot(122, projection='3d')  # Changed to 122 for 1 row, 2 cols, 2nd subplot
        # surface2 = ax2.plot_surface(alpha_grid, freq_grid, error_y, cmap='viridis')
        surface2 = ax2.plot_surface(alpha_grid, freq_grid, np.abs(error_y-gp_est_y)/np.abs(error_y), cmap='viridis')
        fig.colorbar(surface2, shrink=0.5, aspect=5)
        ax2.set_zlim(0,10)
        plt.show()

        

    def error(self, vd):
        #alpha desired comes from arctan of desired velocity
        alpha_d = np.array(math.atan2(vd[1], vd[0]))
        f_d = np.linalg.norm(vd) / self.a0
        
        X = np.array([alpha_d, f_d])
         
        #estimate the uncertainty for the desired alpha
        muX,sigX = self.gprX.predict(X.reshape(1,-1), return_std=True)
        muY,sigY = self.gprY.predict(X.reshape(1,-1), return_std=True)

        return muX, muY, sigX, sigY

    def predict(self, vd):
        #alpha desired comes from arctan of desired velocity
        alpha_d = np.array(math.atan2(vd[1], vd[0]))
        f_d = np.linalg.norm(vd) / self.a0

        X = np.array([alpha_d, f_d])
        

        #estimate the uncertainty for the desired alpha
        muX = self.gprX.predict(X.reshape(1,-1))
        muY = self.gprY.predict(X.reshape(1,-1))


        #select the initial alpha guess as atan2 of v_d - v_error
        x0 = np.hstack( [alpha_d, f_d] )
        
        
        result = minimize(objective, x0, args=(self.a0, vd, self.gprX, self.gprY), bounds=[(-np.pi, np.pi), (0, 5)])

        #result = minimize_scalar(objective, method='Bounded', args=(self.a0, self.freq, vd, self.gprX, self.gprY), bounds=[-np.pi, np.pi] )


        X = np.array(result.x)

        #generate the uncertainty for the new alpha we're sending
        muX,sigX = self.gprX.predict(X.reshape(1,-1), return_std=True)
        muY,sigY = self.gprY.predict(X.reshape(1,-1), return_std=True)

        return X, muX, muY, sigX, sigY

    '''
    #get bounds on learning - 2stdv ~= 95% of data
    plt.figure()
    plt.fill_between(time,   v_learned[:,0] - 2*gpX[:,1],   v_learned[:,0] + 2*gpX[:,1])
    plt.fill_between(time,   v_learned[:,0] -   gpX[:,1],   v_learned[:,0] +   gpX[:,1])

    plt.plot(time, v_learned[:,0], '-r', label="learned")
    plt.plot(time, v_desired[:,0], '-b', label="desired")
    plt.plot(time, vx, '-k', label="data")

    plt.legend()

    plt.show()

    '''






