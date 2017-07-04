import numpy as np
import matplotlib.pyplot as plt

#Data from the baseline
baseline_data = np.load("baseline_time_loss_data.npy")

#Data from the cnn
cnn_data = np.load("cnn/time_loss_data.npy")

min_time = np.min([baseline_data[-1,0],cnn_data[-1,0]])

#Limit both datasets to the 
baseline_data = baseline_data[baseline_data[:,0]<=min_time]
cnn_data = cnn_data[cnn_data[:,0]<=min_time]

#Reduce number of plot points to roughly a maximum of MAX_NUM_POINTS
MAX_NUM_POINTS = 200
b_num_points = baseline_data.shape[0]
c_num_points = cnn_data.shape[0]

if b_num_points>MAX_NUM_POINTS:
	baseline_data = baseline_data[::b_num_points/MAX_NUM_POINTS]
if c_num_points>MAX_NUM_POINTS:
	cnn_data = cnn_data[::c_num_points/MAX_NUM_POINTS]


#Finally plot the figure and save to file
fig = plt.figure()
plt.plot(baseline_data[:,0],baseline_data[:,1],'r-',figure=fig,label="Baseline")
plt.plot(cnn_data[:,0],cnn_data[:,1],'b-',figure=fig,label="CNN")
plt.legend(bbox_to_anchor=(0.95, 0.95), loc=1, borderaxespad=0.)
plt.ylabel("Cross Entropy Loss")
plt.xlabel("Time (s)")
plt.savefig('loss_comparison.png')
