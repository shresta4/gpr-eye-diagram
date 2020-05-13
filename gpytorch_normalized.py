import math
import torch
import gpytorch
from matplotlib import pyplot as plt
import numpy as np
# use this program 
torch.set_default_dtype(torch.float64)

# normalize
import numpy as np
def normalize(nparray): 
    return nparray / nparray.max(axis = 0); 


# Read data from csv 
import pandas as pd 
df = pd.read_csv('EyeProbeSummary_ustrip.csv', warn_bad_lines = True); 
all_inputs = df[['w', 'sp', 'ep_r']].values;
all_outputs = df['RX_Probe.Height'].values

from sklearn.model_selection import train_test_split
train_x, X_test, train_y, y_test = train_test_split(all_inputs, all_outputs, test_size=0.2, random_state=42)
n = 10
train_x = normalize(train_x)
train_y = normalize(train_y)
X_test = normalize(X_test)
y_test = normalize(y_test)
train_x = torch.from_numpy(train_x[::n])
train_y = torch.from_numpy(train_y[::n]) 
X_test = torch.from_numpy(X_test[::n])
y_test = torch.from_numpy(y_test[::n])




# We will use the simplest form of GP model, exact inference
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# initialize likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = ExactGPModel(train_x, train_y, likelihood)

# Find optimal model hyperparameters
model.train()
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam([
    {'params': model.parameters()},  # Includes GaussianLikelihood parameters
], lr=0.1)

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

training_iter = 50
for i in range(training_iter):
    # Zero gradients from previous iteration
    optimizer.zero_grad()
    # Output from model
    output = model(train_x)
    loss = -mll(output, train_y)

    loss.backward()
    print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
        i + 1, training_iter, loss.item(),
        model.covar_module.base_kernel.lengthscale.item(),
        model.likelihood.noise.item()
    ))
    optimizer.step()


 # Get into evaluation (predictive posterior) mode
model.eval()
likelihood.eval()

# Test points are regularly spaced along [0,1]
# Make predictions by feeding model through likelihood
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    # test_x = torch.linspace(0, 1, 51)
    observed_pred = likelihood(model(X_test))
    print ("Hello")

f_preds = model(X_test)
y_preds = likelihood(model(X_test))

f_mean = f_preds.mean
f_var = f_preds.variance
f_covar = f_preds.covariance_matrix
print(y_preds.mean)

import matplotlib.pyplot as plt
# plot covariance

# plot error 
relative_error = (f_preds.mean.detach().numpy() - y_test.detach().numpy()) / (y_test.detach().numpy())
plt.figure()
plt.plot(relative_error) 
plt.title("Error"); 
plt.show()

plt.figure() 
sample_number = range(0, len(y_test))
plt.scatter(sample_number, f_preds.variance.detach().numpy())
plt.title("Sigma vs. sample number"); 
plt.show()

# print out data 
for i in range(0, len(y_test)): 
    print(str(i) + " " + str(relative_error[i]) + " " + str(f_preds.variance.detach().numpy()[i]))

# f_samples = f_preds.sample(sample_shape=torch.Size(1000,))