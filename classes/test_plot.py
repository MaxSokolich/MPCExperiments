import numpy as np
import matplotlib.pyplot as plt


x0 = 1000
y0 = 1000

x0 = 1000
y0 = 1000
time_steps = 900

center_x = x0 
center_y = y0
tx = 100
ty = 100
# time_steps = 9000
t =np.linspace(0, 2*np.pi,time_steps)
x_ls = center_x +tx*t
y_ls = center_y+ ty*t
ref = np.ones((time_steps,2))
ref[:,0]= x_ls
ref[:,1]= y_ls
plt.plot(x_ls, y_ls)
plt.show()