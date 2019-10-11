# This script will take all the separate time series data recorded by the
# Empatica E4 wristband and combine it into one dataframe. Then it will
# present the data in separate time series plots.

# Author: Brenden Eum (2019)

# Start fresh~~
import pandas as pd 
import numpy as np 
import csv
import matplotlib.pyplot as plt
import os
import datetime
import pickle

# Set directories
code_dir = str(os.getcwd()) + "\\"
data_dir = str(os.path.join(code_dir, "..\\Brenden pilot data\\"))
fig_dir = str(os.path.join(code_dir, "..\\Figures\\"))

# Generate a dataframe for each time series.
## ACC: Accelerometer sensor, units 1/64g.
ACC = pd.read_csv(data_dir + 'ACC.csv', names=['ACC_x','ACC_y','ACC_z'])
ACC['drop'] = range(0,len(ACC))
ACC['time_unix'] = ACC['drop']*(1/ACC.ACC_x[1])+ACC.ACC_x[0]
ACC = ACC.drop([0,1])
ACC = ACC.drop(['drop'],axis=1)
delta = ACC.diff()
ACC['ACC_delta'] = np.abs(delta.ACC_x) + np.abs(delta.ACC_y) + np.abs(delta.ACC_z)
## BVP: Data from photoplethysmograph.
BVP = pd.read_csv(data_dir + 'BVP.csv', names=['BVP'])
BVP['drop'] = range(0,len(BVP))
BVP['time_unix'] = BVP['drop']*(1/BVP.BVP[1])+BVP.BVP[0]
BVP = BVP.drop([0,1])
BVP = BVP.drop(['drop'],axis=1)
## EDA: Data from electrodermal activity sensor expressed as microsiemens.
EDA = pd.read_csv(data_dir + 'EDA.csv', names=['EDA'])
EDA['drop'] = range(0,len(EDA))
EDA['time_unix'] = EDA['drop']*(1/EDA.EDA[1])+EDA.EDA[0]
EDA = EDA.drop([0,1])
EDA = EDA.drop(['drop'],axis=1)
## HR: Average heart rate extracted from the BVP signal.
HR = pd.read_csv(data_dir + 'HR.csv', names=['HR'])
HR['drop'] = range(0,len(HR))
HR['time_unix'] = HR['drop']*(1/HR.HR[1])+HR.HR[0]
HR = HR.drop([0,1])
HR = HR.drop(['drop'],axis=1)
## IBI: Time between individual's heart beats extracted from BVP.
IBI = pd.read_csv(data_dir + 'IBI.csv', names=['IBI_time','IBI_duration'])
IBI['time_unix'] = IBI.IBI_time[0]
IBI = IBI.drop([0])
IBI['time_unix'] = IBI['time_unix']+IBI['IBI_time']
IBI = IBI.drop(['IBI_time'],axis=1)
## Tags: Times for a physical button press.
tags = pd.read_csv(data_dir + 'tags.csv', names=['tag'])
tag0 = tags.tag[0]
tag1 = tags.tag[1]

# Combine all the dataframes into one. Merge by time.
data = ACC
data = data.merge(BVP, how='outer', on='time_unix', sort=True)
data = data.merge(EDA, how='outer', on='time_unix', sort=True)
data = data.merge(HR, how='outer', on='time_unix', sort=True)
data = data.merge(IBI, how='outer', on='time_unix', sort=True)

# Drop all data before and after the experiment.
data = data[(data['time_unix'] > tag0) & (data['time_unix'] < tag1)]

# Store this data
data.to_pickle(data_dir + "01_merged_data.pkl")

# Plot all the time series on one subplot.
plt.style.use('seaborn-whitegrid')
## ACC
plt.subplot(4,1,1)
plt.plot(data.time_unix[np.logical_not(np.isnan(data.ACC_delta))], data.ACC_delta[np.logical_not(np.isnan(data.ACC_delta))])
plt.title('Total change in acceleration (1/64g)')
## BVD
plt.subplot(4,1,2)
plt.plot(data.time_unix[np.logical_not(np.isnan(data.BVP))], data.BVP[np.logical_not(np.isnan(data.BVP))])
plt.title('Blood volume changes in tissue')
## EDA
plt.subplot(4,1,3)
plt.plot(data.time_unix[np.logical_not(np.isnan(data.EDA))], data.EDA[np.logical_not(np.isnan(data.EDA))])
plt.title('Electrodermal activity (microsiemens)')
## HR
plt.subplot(4,1,4)
plt.plot(data.time_unix[np.logical_not(np.isnan(data.HR))], data.HR[np.logical_not(np.isnan(data.HR))])
plt.title('Heart rate')

# Show the plot and save it.
plt.tight_layout()
plt.savefig(fig_dir + 'graph.png')
plt.show()
