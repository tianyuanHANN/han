# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 15:11:23 2023
@author: HAN
"""

import pickle
import numpy as np
import os
import math
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
        
        if (data[key]['is_complete_start'] == True): #
            Lc_start = data[key]['LC_action_start_local_frame']
            Lc_end = data[key]['LC_action_end_local_frame'] 
            if np.array(data[key]['s_Velocity'])[Lc_start] >= 60/3.6:
                if np.array(data[key]['left_right_maveuver']) == left_right:
    
                    if isinstance(data[key]['LC_interaction']['action_ego_LTCfollowing_S_velocity'], np.float64):
                        n = n+1
                        Lc_vehicle['id'] = data[key]['vehicle_id'] 
                        Lc_vehicle['class'] = data[key]['class']
                        Lc_vehicle['left_right'] = 0
                        Lc_vehicle['LTC_start_backward_TTC'] = np.array(data[key]['LC_interaction']['action_LTC_backward_TTC'])     
                        Lc_vehicle['LTC_start_backward_DHW'] = np.array(data[key]['LC_interaction']['action_ego_LTCfollowing_S_distance'])                          
                        Lc_vehicle['speed_ego'] = np.array(data[key]['s_Velocity'])[Lc_start]
                        Lc_vehicle['speed_backward'] = Lc_vehicle['speed_ego'] - np.array(data[key]['LC_interaction']['action_ego_LTCfollowing_S_velocity'])
                        
                        Lc_vehicle['i'] = i+1
                        Lc_complete[n] = Lc_vehicle      
                
            
   
LTCstartBackwardTTC = []
LTCstartBackwardTHW = []
LTCstartBackwardDHW = []

speedEgo = []
speedBack = []
safeModelBack = []


nnn = 0
for key in Lc_complete.keys():
    if Lc_complete[key]['LTC_start_backward_DHW'] != 0:
        LSBT = Lc_complete[key]['LTC_start_backward_TTC']        
        LSBD = Lc_complete[key]['LTC_start_backward_DHW']   
        SE = Lc_complete[key]['speed_ego'] 
        SB = Lc_complete[key]['speed_backward'] 
        LSBTH = LSBD/SB      
        
        SMB =  -(SB*2.5)/(LSBD+SE*2.5)*(SB/SE)*(SB/SE)
        LTCstartBackwardTTC.append(LSBT)
        LTCstartBackwardTHW.append(LSBTH)
        LTCstartBackwardDHW.append(LSBD)

        speedEgo.append(SE)
        speedBack.append(SB)     
        safeModelBack.append(SMB)

    else:
        nnn = nnn+1
print(nnn)



filter_data1 = np.array(safeModelBack)  
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




