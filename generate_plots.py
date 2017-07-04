import numpy as np
import matplotlib.pyplot as plt

#Data from the baseline
baseline_data = np.load("baseline_time_loss_data.npy")

#Data from the cnn
cnn_data_smoothed = np.load("cnn/jpg_time_loss_data.npy")
cnn_data_unsmoothed = np.load("cnn/png_time_loss_data.npy")

min_time = np.min([baseline_data[-1,0],cnn_data_smoothed[-1,0],cnn_data_unsmoothed[-1,0]])

#Limit both datasets to the 
baseline_data = baseline_data[baseline_data[:,0]<=min_time]
cnn_data_smoothed = cnn_data_smoothed[cnn_data_smoothed[:,0]<=min_time]
cnn_data_unsmoothed = cnn_data_unsmoothed[cnn_data_unsmoothed[:,0]<=min_time]

#Reduce number of plot points to roughly a maximum of MAX_NUM_POINTS
MAX_NUM_POINTS = 200
b_num_points = baseline_data.shape[0]
c_u_num_points = cnn_data_unsmoothed.shape[0]
c_s_num_points = cnn_data_smoothed.shape[0]

if b_num_points>MAX_NUM_POINTS:
	baseline_data = baseline_data[::b_num_points/MAX_NUM_POINTS]
if c_u_num_points>MAX_NUM_POINTS:
	cnn_data_unsmoothed = cnn_data_unsmoothed[::c_u_num_points/MAX_NUM_POINTS]
if c_s_num_points>MAX_NUM_POINTS:
	cnn_data_smoothed = cnn_data_smoothed[::c_s_num_points/MAX_NUM_POINTS]


#Finally plot the figures and save to file
fig = plt.figure()
plt.plot(baseline_data[:,0],baseline_data[:,1],'r-',figure=fig,label="Baseline")
plt.plot(cnn_data_smoothed[:,0],cnn_data_smoothed[:,1],'b-',figure=fig,label="CNN")
plt.legend(bbox_to_anchor=(0.95, 0.95), loc=1, borderaxespad=0.)
plt.ylabel("Cross Entropy Loss")
plt.xlabel("Time (s)")
plt.savefig('loss_comparison_baseline_smoothed.png')


fig = plt.figure()
plt.plot(cnn_data_unsmoothed[:,0],cnn_data_unsmoothed[:,1],'r-',figure=fig,label="Unsmoothed")
plt.plot(cnn_data_smoothed[:,0],cnn_data_smoothed[:,1],'b-',figure=fig,label="Smoothed")
plt.legend(bbox_to_anchor=(0.95, 0.95), loc=1, borderaxespad=0.)
plt.ylabel("Cross Entropy Loss")
plt.xlabel("Time (s)")
plt.savefig('loss_comparison_smoothed_unsmoothed.png')
