# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 15:11:23 2023
@author: HAN
"""

import pickle
import numpy as np
import os

import matplotlib.pyplot as plt
import scipy.stats as stats


left_right = 0       # left lane change:0  /  right lane change:1

Lc_complete = {}
cnt = 1  
n = 0     
for i in range(0,57):
    temp_name = "LC_vehcile_features_" + str(cnt) + ".pkl"
    cnt = cnt + 1
    prexfix_path = "C:/Users/HAN/.spyder-py3/data_pkl"  # Please replace the path
    pkl_path = os.path.join(prexfix_path, temp_name)

    data = {}
    f=open(pkl_path,'rb')
    data=pickle.load(f)
    
    for key in data.keys():
        Lc_vehicle = {}   
        
        if (data[key]['is_complete_start'] == True): 
            Lc_start = data[key]['LC_action_start_local_frame']
            Lc_end = data[key]['LC_action_end_local_frame'] 
            if np.array(data[key]['s_Velocity'])[Lc_start] >= 60/3.6:
               
                if np.array(data[key]['left_right_maveuver']) == left_right:
             
                    if isinstance(data[key]['LC_interaction']['action_ego_LTCpreceding_S_velocity'], np.float64):
                        n = n+1
                        Lc_vehicle['id'] = data[key]['vehicle_id'] 
                        Lc_vehicle['class'] = data[key]['class']
                        Lc_vehicle['left_right'] = data[key]['left_right_maveuver']
                        Lc_vehicle['LTC_start_forward_TTC'] = np.array(data[key]['LC_interaction']['action_LTC_forward_TTC'])     
                        Lc_vehicle['LTC_start_forward_DHW'] = np.array(data[key]['LC_interaction']['action_ego_LTCpreceding_S_distance'])                          
                        Lc_vehicle['speed_ego'] = np.array(data[key]['s_Velocity'])[Lc_start]
                        Lc_vehicle['speed_forward'] = Lc_vehicle['speed_ego'] - np.array(data[key]['LC_interaction']['action_ego_LTCpreceding_S_velocity'])
                        
                        Lc_vehicle['i'] = i+1
                        Lc_complete[n] = Lc_vehicle      
                
            
   
LTCstartForwardTTC = []
LTCstartForwardTHW = []
LTCstartForwardDHW = []

speedEgo = []
speedFor = []
safeModelFor = []
effiModelFor =[]

nnn = 0
for key in Lc_complete.keys():
  
    if Lc_complete[key]['LTC_start_forward_DHW'] != 0:
        LSFT = Lc_complete[key]['LTC_start_forward_TTC']        
        LSFD = Lc_complete[key]['LTC_start_forward_DHW']   
        SE = Lc_complete[key]['speed_ego'] 
        SF = Lc_complete[key]['speed_forward'] 
        LSFTH = LSFD/SE      
           
        SMF = -(SE*SE-277.78)/(SF*SF-277.78+8*LSFD)
        EMF = -((2*SE)/(LSFD+2*SF))*SE*SE/SF/SF


        LTCstartForwardTTC.append(LSFT)
        LTCstartForwardTHW.append(LSFTH)
        LTCstartForwardDHW.append(LSFD)

        speedEgo.append(SE)
        speedFor.append(SF)     
        safeModelFor.append(SMF)
        effiModelFor.append(EMF)

    else:
        nnn = nnn+1
print(nnn)



filter_data1 = np.array(safeModelFor)  
mean_value = np.mean(filter_data1)
variance_value = np.var(filter_data1, ddof=1) 

x = np.linspace(filter_data1.min() - 0.1, filter_data1.max() + 0.1, 1000)
y = stats.norm.pdf(x, loc=mean_value, scale=np.sqrt(variance_value))

plt.figure(figsize=(12, 8))


plt.subplot(1, 2, 1)
plt.hist(filter_data1, bins=300, density=True, alpha=0.7, color='blue', edgecolor='black', label='Data Histogram')
plt.plot(x, y, color='red', linewidth=2, label='Normal Distribution')
plt.xlabel('Value')
plt.ylabel('Frequency / Probability Density')
plt.title('Data Histogram and Fitted Normal Distribution')
plt.legend()
plt.grid(True)


plt.subplot(1, 2, 2)
stats.probplot(filter_data1, plot=plt)
plt.title('Q-Q Plot')


filter_data2 = np.array(effiModelFor)  
mean_value2 = np.mean(filter_data2)
variance_value2 = np.var(filter_data2, ddof=1) 

x = np.linspace(filter_data2.min() - 0.1, filter_data2.max() + 0.1, 1000)
y = stats.norm.pdf(x, loc=mean_value2, scale=np.sqrt(variance_value2))

plt.figure(figsize=(12, 8))


plt.subplot(1, 2, 1)
plt.hist(filter_data2, bins=300, density=True, alpha=0.7, color='blue', edgecolor='black', label='Data Histogram')
plt.plot(x, y, color='red', linewidth=2, label='Normal Distribution')
plt.xlabel('Value')
plt.ylabel('Frequency / Probability Density')
plt.title('Data Histogram and Fitted Normal Distribution')
plt.legend()
plt.grid(True)


plt.subplot(1, 2, 2)
stats.probplot(filter_data2, plot=plt)
plt.title('Q-Q Plot')



