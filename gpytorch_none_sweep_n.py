import torch
import time
import pandas as pd 
import gpytorch
torch.set_default_dtype(torch.float64)
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

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

for index in range(1, len(x_axis_array)): # max_n + 1): # i represents the current value of n
    i = x_axis_array[index]
    print("Finding every " + str(i) + "th element"); 

    start_time = time.time(); 
    X_train = torch.from_numpy(X_train_original[::i])
    X_test = torch.from_numpy(X_test_original[::i])
    y_train = torch.from_numpy(y_train_original[::i]) 
    y_test = torch.from_numpy(y_test_original[::i]) 
    
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(X_train, y_train, likelihood) 
    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},  # Includes GaussianLikelihood parameters
    ], lr=0.1)

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    training_iter = 50
    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(X_train)
        loss = -mll(output, y_train)

        loss.backward()
        '''print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
            i + 1, training_iter, loss.item(),
            model.covar_module.base_kernel.lengthscale.item(),
            model.likelihood.noise.item()
        ))''' 
        optimizer.step()

    model.eval()
    likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        # test_x = torch.linspace(0, 1, 51)
        observed_pred = likelihood(model(X_test))
        print ("Hello")

    f_preds = model(X_test)
    y_preds = likelihood(model(X_test))

    f_mean = f_preds.mean
    f_var = f_preds.variance
    f_covar = f_preds.covariance_matrix

    end_time = time.time() 
    print("done training")




    time_array[index] = end_time - start_time; 
    num_data_points[index] = len(y_train.detach().numpy()); 

    #maximum sigma 
    max_sigma[index] = max(f_preds.variance.detach().numpy()) 
    
    #maximum relative error 
    relative_error = (f_preds.mean.detach().numpy() - y_test.detach().numpy()) / (y_test.detach().numpy())
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
