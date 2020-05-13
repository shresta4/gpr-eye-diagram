# https://stackoverflow.com/questions/41572058/how-to-correctly-use-scikit-learns-gaussian-process-for-a-2d-inputs-1d-output
# https://archive.ics.uci.edu/ml/datasets/Airfoil+Self-Noise
import time
start_time = time.time(); 

import numpy as np
def normalize(nparray): 
    return nparray / nparray.max(axis = 0); 

import pandas as pd 
df = pd.read_csv('EyeProbeSummary_ustrip.csv', warn_bad_lines = True); 
all_inputs = df[['w', 'sp', 'ep_r']].values;
all_outputs = df['RX_Probe.Height'].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(all_inputs, all_outputs, test_size=0.2, random_state=42)

# Reduce data count 
import sklearn
n = 10
X_train = X_train[::n]
X_train = normalize(X_train)
X_train = X_train * 10; 

X_test = X_test[::n]
X_test = normalize(X_test)
X_test = X_test * 10; 


y_train = y_train[::n] 
y_train = normalize(y_train)

y_test = y_test[::n] 
y_test = normalize(y_test)



from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF([5] * X_train.shape[1], (1e-2, 1e2)) 
gp = GaussianProcessRegressor(kernel = kernel, n_restarts_optimizer = 15)  
gp.fit(X_train, y_train); 
y_pred, stdev = gp.predict(X_test, return_std = True) 
#y_pred, stdev = gp.predict(X_train, return_std = True) 
end_time = time.time() 
#y_test = y_train #

import matplotlib.pyplot as plt
import math
# graph 1 
# sigma vs. sample number -- confidence interval
plt.figure() 
sample_number = range(0, len(y_test))
plt.scatter(sample_number, stdev)
plt.title("Sigma vs. sample number"); 

#graph 2 
#error = y_pred - y_test # absolute error
relative_error = (y_pred - y_test) / (y_test)
plt.figure()
plt.plot(relative_error) 
plt.title("Error"); 

# test with training data to see if model works 

print("Execution time: " + str(end_time - start_time)); 
plt.show()


# print out data 
for i in range(0, len(y_test)): 
	print(str(i) + " " + str(relative_error[i]) + " " + str(stdev[i]))