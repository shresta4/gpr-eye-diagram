import time
import pandas as pd 
df = pd.read_csv('EyeProbeSummary_ustrip.csv', warn_bad_lines = True); 
all_inputs = df[['w', 'sp', 'ep_r']].values;
all_outputs = df['RX_Probe.Height'].values

from sklearn.model_selection import train_test_split
X_train_original, X_test_original, y_train_original, y_test_original = train_test_split(all_inputs, all_outputs, test_size=0.2, random_state=42)

# Reduce data count 
import sklearn
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
max_n = len(y_train_original)
x_axis_array = (range(1, max_n + 1))[::5]
total_increments = len(x_axis_array)
time_array = [0] * total_increments #max_n 
num_data_points = [0] * total_increments #max_n  # deal with this later 
max_sigma = [0] * total_increments #* max_n 
max_error = [0] * total_increments #* max_n 

for index in range(1, len(x_axis_array)): # max_n + 1): # i represents the current value of n
    i = x_axis_array[index]
    print("Finding every " + str(i) + "th element"); 

    start_time = time.time(); 
    X_train = X_train_original[::i]
    X_test = X_test_original[::i]
    y_train = y_train_original[::i] 
    y_test = y_test_original[::i] 
    
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF([5] * X_train.shape[1], (1e-2, 1e2)) 
    gp = GaussianProcessRegressor(kernel = kernel, n_restarts_optimizer = 15)
    gp.fit(X_train, y_train); 
    y_pred, stdev = gp.predict(X_test, return_std = True) 
    end_time = time.time() 
    print("done training")
    time_array[index] = end_time - start_time; 
    num_data_points[index] = len(y_train); 
    
    #maximum sigma 
    max_sigma[index] = max(stdev) 
    
    #maximum relative error 
    relative_error = (y_pred - y_test) / (y_test)
    max_error[index] = max(relative_error)  


import matplotlib.pyplot as plt
import math
# graph 1 
# max sigma vs n 
plt.figure() 
plt.plot(x_axis_array, max_sigma)
plt.title("Max sigma vs. n"); 

#graph 2 
plt.figure()
plt.plot(x_axis_array, max_error) 
plt.title("Max error vs. n"); 

#graph 3
import numpy as np
plt.figure()
print(str(x_axis_array))
print(str(num_data_points))
plt.plot(np.log(x_axis_array), np.log(num_data_points)) 
plt.title("Log Time vs. log num data points"); 
 
plt.show()  

for i in range (0, len(x_axis_array)): 
    print(str(x_axis_array[i]) + " " + str(max_error[i]) + " " + str(max_sigma[i]) + " " + str(time_array[i]))
print("time")
for i in range (0, len(x_axis_array)): 
    print(str(num_data_points[i]) + " " + str(time_array[i]))
