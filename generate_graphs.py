from matplotlib import pyplot as plt

# dataframes for all sheets
import pandas as pd 
df_sk = pd.read_excel(pd.ExcelFile('all_data.xlsx'), 'sklearn_none');
df_sk_norm = pd.read_excel(pd.ExcelFile('all_data.xlsx'), 'sklearn_normalized')
df_gp = pd.read_excel(pd.ExcelFile('all_data.xlsx'), 'gpytorch_none')
df_gp_norm = pd.read_excel(pd.ExcelFile('all_data.xlsx'), 'gpytorch_normalized')

# all error
sk_error = df_sk['rel_error'].values
sk_norm_error = df_sk_norm['rel_error'].values
gp_error = df_gp['rel_error'].values
gp_norm_error = df_gp_norm['rel_error'].values

plt.figure()
line, = plt.plot(sk_error); line.set_label('sklearn')
line2, = plt.plot(sk_norm_error); line2.set_label('sklearn normalized')
line3, = plt.plot(gp_error); line3.set_label('gpytorch')
line4, = plt.plot(gp_norm_error); line4.set_label('gpytorch normalized')

plt.legend()
plt.title("Relative Error")
plt.show()

# all sigma 
sk_sigma = df_sk['confidence'].values
sk_norm_sigma = df_sk_norm['confidence'].values
gp_sigma = df_gp['confidence'].values
gp_norm_sigma = df_gp_norm['confidence'].values

plt.figure() 
line, = plt.plot(sk_sigma); line.set_label('sklearn')
line2, = plt.plot(sk_norm_sigma); line2.set_label('sklearn normalized')
line3, = plt.plot(gp_sigma); line3.set_label('gpytorch')
line4, = plt.plot(gp_norm_sigma); line4.set_label('gpytorch normalized')

plt.legend()
plt.title("Sigma")
plt.show()

# move on to different sheets of the doc 
df_sk = pd.read_excel(pd.ExcelFile('all_data.xlsx'), 'sklearn_none_sweep_n');
df_sk_norm = pd.read_excel(pd.ExcelFile('all_data.xlsx'), 'sklearn_normalized_sweep_n')
df_gp = pd.read_excel(pd.ExcelFile('all_data.xlsx'), 'gpytorch_none_sweep_n')
df_gp_norm = pd.read_excel(pd.ExcelFile('all_data.xlsx'), 'gpytorch_normalized_sweep_n')

# max error vs n
x_axis = df_sk['n'].values 
sk_max_error = df_sk['rel_error'].values
sk_norm_max_error = df_sk_norm['rel_error'].values
gp_max_error = df_gp['rel_error'].values
gp_norm_max_error = df_gp_norm['rel_error'].values

plt.figure() 
line, = plt.plot(x_axis, sk_max_error); line.set_label('sklearn')
line2, = plt.plot(x_axis, sk_norm_max_error); line2.set_label('sklearn normalized')
line3, = plt.plot(x_axis, gp_max_error); line3.set_label('gpytorch')
line4, = plt.plot(x_axis, gp_norm_max_error); line4.set_label('gpytorch normalized')

plt.legend()
plt.title("Max Error vs. n")
plt.show()

# max sigma vs n
x_axis = df_sk['n'].values 
sk_max_sigma = df_sk['confidence'].values
sk_norm_max_sigma = df_sk_norm['confidence'].values
gp_max_sigma = df_gp['confidence'].values
gp_norm_max_sigma = df_gp_norm['confidence'].values

plt.figure() 
line, = plt.plot(x_axis, sk_max_sigma); line.set_label('sklearn')
line2, = plt.plot(x_axis, sk_norm_max_sigma); line2.set_label('sklearn normalized')
line3, = plt.plot(x_axis, gp_max_sigma); line3.set_label('gpytorch')
line4, = plt.plot(x_axis, gp_norm_max_sigma); line4.set_label('gpytorch normalized')

plt.legend()
plt.title("Max Sigma vs. n")
plt.show()

# log(time) vs log(# of samples)
import numpy as np 
x_axis = np.reciprocal(np.log(df_sk['n'].values)) # log(1/n)
sk_time = np.log(df_sk['time'].values)
sk_norm_time = np.log(df_sk_norm['time'].values)
gp_time = np.log(df_gp['time'].values)
gp_norm_time = np.log(df_gp_norm['time'].values)

plt.figure() 
line, = plt.plot(x_axis, sk_time); line.set_label('sklearn')
line2, = plt.plot(x_axis, sk_time); line2.set_label('sklearn normalized')
line3, = plt.plot(x_axis, gp_time); line3.set_label('gpytorch')
line4, = plt.plot(x_axis, gp_time); line4.set_label('gpytorch normalized')

plt.legend()
plt.title("log(time) vs log(1/n)")
plt.show()

#all_inputs = df[['frequency', 'angle of attack', 'chord length', 'free-stream velocity', 'suction side displacement thickness']].values;
#all_outputs = df['scaled sound pressure level'].values