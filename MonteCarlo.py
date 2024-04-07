# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 20:12:37 2023

@author: HAN
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import math
from sklearn.mixture import GaussianMixture
import pickle
import os
import scipy.stats as stats
from scipy.stats import multivariate_normal


Lc_complete1 = {}
Lc_complete2 = {}
Lc_complete3 = {}
Lc_complete4 = {}
cnt = 1   
n1 = 0     
n2 = 0
n3 = 0
n4 = 0
for i in range(0,57):
    temp_name = "LC_vehcile_features_" + str(cnt) + ".pkl"  # Please replace the path
    cnt = cnt + 1
    prexfix_path = "C:/Users/HAN/.spyder-py3/data_pkl"  
    pkl_path = os.path.join(prexfix_path, temp_name)

    data = {}
    f=open(pkl_path,'rb')
    data=pickle.load(f)
    
    for key in data.keys():
        Lc_vehicle1 = {}   
        Lc_vehicle2 = {} 
        Lc_vehicle3 = {} 
        Lc_vehicle4 = {} 
        if (data[key]['is_complete_start'] == True): #
            Lc_start = data[key]['LC_action_start_local_frame']
            if np.array(data[key]['s_Velocity'])[Lc_start] >= 60/3.6:
                if np.array(data[key]['left_right_maveuver']) == 0:
                    if isinstance(data[key]['LC_interaction']['action_ego_preceding_S_velocity'], np.float64):
                        n1 = n1+1           
                        Lc_vehicle1['start_forward_DHW'] = np.array(data[key]['LC_interaction']['action_ego_preceding_S_distance'])                          
                        Lc_vehicle1['speed_ego'] = np.array(data[key]['s_Velocity'])[Lc_start]
                        Lc_vehicle1['speed_lead'] = Lc_vehicle1['speed_ego'] - np.array(data[key]['LC_interaction']['action_ego_preceding_S_velocity'])                       
                        Lc_complete1[n1] = Lc_vehicle1 
                    if isinstance(data[key]['LC_interaction']['action_ego_LTCpreceding_S_velocity'], np.float64):
                        n2 = n2+1
                        Lc_vehicle2['LTC_start_forward_DHW'] = np.array(data[key]['LC_interaction']['action_ego_LTCpreceding_S_distance'])                          
                        Lc_vehicle2['speed_ego'] = np.array(data[key]['s_Velocity'])[Lc_start]
                        Lc_vehicle2['speed_forward'] = Lc_vehicle2['speed_ego'] - np.array(data[key]['LC_interaction']['action_ego_LTCpreceding_S_velocity'])
                        Lc_complete2[n2] = Lc_vehicle2   
                    if isinstance(data[key]['LC_interaction']['action_ego_LTCfollowing_S_velocity'], np.float64):
                        n3 = n3+1
                        Lc_vehicle3['LTC_start_backward_DHW'] = np.array(data[key]['LC_interaction']['action_ego_LTCfollowing_S_distance'])                          
                        Lc_vehicle3['speed_ego'] = np.array(data[key]['s_Velocity'])[Lc_start]
                        Lc_vehicle3['speed_backward'] = Lc_vehicle3['speed_ego'] - np.array(data[key]['LC_interaction']['action_ego_LTCfollowing_S_velocity'])
                        Lc_complete3[n3] = Lc_vehicle3
                    if isinstance(data[key]['LC_interaction']['action_ego_preceding_S_velocity'], np.float64) or isinstance(data[key]['LC_interaction']['action_ego_LTCpreceding_S_velocity'], np.float64) or isinstance(data[key]['LC_interaction']['action_ego_LTCfollowing_S_velocity'], np.float64):
                        n4 = n4+1                                     
                        Lc_vehicle4['speed_ego'] = np.array(data[key]['s_Velocity'])[Lc_start]    
                        Lc_complete4[n4] = Lc_vehicle4

startForwardDHW = []
speedEgo = []
speedLead = []
LTCstartForwardDHW = []
speedFor = []
LTCstartBackwardDHW = []
speedBack = []


for key in Lc_complete1.keys():
    if Lc_complete1[key]['start_forward_DHW'] > 5:
        SFD = Lc_complete1[key]['start_forward_DHW']        
        SL = Lc_complete1[key]['speed_lead']
        startForwardDHW.append(SFD)
        speedLead.append(SL)  

for key in Lc_complete2.keys():
    if Lc_complete2[key]['LTC_start_forward_DHW'] != 0:      
        LSFD = Lc_complete2[key]['LTC_start_forward_DHW']   
        SF = Lc_complete2[key]['speed_forward'] 
        LTCstartForwardDHW.append(LSFD)
        speedFor.append(SF)     
for key in Lc_complete3.keys():
    if Lc_complete3[key]['LTC_start_backward_DHW'] != 0:     
        LSBD = Lc_complete3[key]['LTC_start_backward_DHW']        
        SB = Lc_complete3[key]['speed_backward'] 
        LTCstartBackwardDHW.append(LSBD)
        speedBack.append(SB)     
for key in Lc_complete4.keys():
    SE = Lc_complete4[key]['speed_ego'] 
    speedEgo.append(SE)



def initialize_parameters(data, num_components):
    np.random.seed(0)
    random_indices = np.random.choice(len(data), num_components, replace=False)
    initial_means = data[random_indices]
    initial_covariances = [np.cov(data, rowvar=False)] * num_components
    initial_weights = np.ones(num_components) / num_components
    return initial_means, initial_covariances, initial_weights

def expectation_maximization(data, num_components, num_iterations=1000):
    means, covariances, weights = initialize_parameters(data, num_components)

    for _ in range(num_iterations):
        responsibilities = np.zeros((len(data), num_components))
        for k in range(num_components):
            responsibilities[:, k] = multivariate_normal.pdf(data, mean=means[k], cov=covariances[k]) * weights[k]
        responsibilities /= np.sum(responsibilities, axis=1)[:, np.newaxis]

        total_responsibilities = np.sum(responsibilities, axis=0)
        weights = total_responsibilities / len(data)
        means = np.dot(responsibilities.T, data) / total_responsibilities[:, np.newaxis]
        covariances = [np.dot(responsibilities[:, k] * (data - means[k]).T, data - means[k]) / total_responsibilities[k] for k in range(num_components)]

    return means, covariances, weights


plt.figure(figsize=(12, 8))        

filter_data1 =(np.array(speedLead)) ** 2 
filter_data1 = filter_data1.reshape(-1, 1)
num_components = 2  
means1, covariances1, weights1 = expectation_maximization(filter_data1, num_components)
plt.hist(filter_data1, bins=100, density=True, alpha=0.7, color='gray', edgecolor='black', label='Data Histogram')

x = np.linspace(min(filter_data1), max(filter_data1), 1000)
for i in range(num_components):
    pdf = multivariate_normal.pdf(x, mean=means1[i, 0], cov=covariances1[i][0, 0])
    plt.plot(x, pdf * weights1[i], label=f'Gaussian {i+1}')
plt.xlabel('Speed of CP')
plt.ylabel('Density')
plt.legend()

plt.show()




plt.figure(figsize=(12, 8))
filter_data2 = np.log(np.array(startForwardDHW))   
m_2 = np.mean(filter_data2)
de_2= np.sqrt(np.var(filter_data2))
x = np.linspace(filter_data2.min() - 0.1, filter_data2.max() + 0.1, 1000)
y = stats.norm.pdf(x, loc = m_2, scale = de_2)
plt.hist(filter_data2, bins=100, density=True, alpha=0.7, color='blue', edgecolor='black', label='Data Histogram')
plt.plot(x, y, color='red', linewidth=2, label='Normal Distribution')
plt.xlabel('DHW of CP')
plt.ylabel('Density')
plt.show()



plt.figure(figsize=(12, 8))        
filter_data3 = (np.array(speedFor) )**2
m_3 = np.mean(filter_data3)
de_3= np.sqrt(np.var(filter_data3))
x = np.linspace(filter_data3.min() - 0.1, filter_data3.max() + 0.1, 1000)
y = stats.norm.pdf(x, loc = m_3, scale = de_3)
plt.hist(filter_data3, bins=100, density=True, alpha=0.7, color='blue', edgecolor='black', label='Data Histogram')
plt.plot(x, y, color='red', linewidth=2, label='Normal Distribution')

plt.xlabel('Speed of TP')
plt.ylabel('Density')
plt.legend()

plt.show()


plt.figure(figsize=(12, 8))        
filter_data4 = np.log(np.array(LTCstartForwardDHW)+60)

filter_data4 = filter_data4.reshape(-1, 1)

num_components = 2 
means4, covariances4, weights4 = expectation_maximization(filter_data4, num_components)
plt.hist(filter_data4, bins=100, density=True, alpha=0.7, color='gray', edgecolor='black', label='Data Histogram')

x = np.linspace(min(filter_data4), max(filter_data4), 1000)
for i in range(num_components):
    pdf = multivariate_normal.pdf(x, mean=means4[i, 0], cov=covariances4[i][0, 0])
    plt.plot(x, pdf * weights4[i], label=f'Gaussian {i+1}')
plt.xlabel('DHW of TP')
plt.ylabel('Density')
plt.legend()

plt.show()


plt.figure(figsize=(12, 8))
filter_data5 = (np.array(speedBack)) ** 2
m_5 = np.mean(filter_data5)
de_5= np.sqrt(np.var(filter_data5))
x = np.linspace(filter_data5.min() - 0.1, filter_data5.max() + 0.1, 1000)
y = stats.norm.pdf(x, loc = m_5, scale = de_5)
plt.hist(filter_data5, bins=100, density=True, alpha=0.7, color='blue', edgecolor='black', label='Data Histogram')
plt.plot(x, y, color='red', linewidth=2, label='Normal Distribution')

plt.xlabel('Speed of TF')
plt.ylabel('Density')
plt.legend()
plt.show()


plt.figure(figsize=(12, 8))
filter_data6 = np.log(np.array(LTCstartBackwardDHW)+60)
m_6 = np.mean(filter_data6)
de_6= np.sqrt(np.var(filter_data6))
x = np.linspace(filter_data6.min() - 0.1, filter_data6.max() + 0.1, 1000)
y = stats.norm.pdf(x, loc = m_6, scale = de_6)
plt.hist(filter_data6, bins=100, density=True, alpha=0.7, color='blue', edgecolor='black', label='Data Histogram')
plt.plot(x, y, color='red', linewidth=2, label='Normal Distribution')

plt.xlabel('DHW of TF')
plt.ylabel('Density')
plt.legend()


plt.show()


plt.figure(figsize=(12, 8))
filter_data7 = (np.array(speedEgo))**2
m_7 = np.mean(filter_data7)
de_7= np.sqrt(np.var(filter_data7))
x = np.linspace(filter_data7.min() - 0.1, filter_data7.max() + 0.1, 1000)
y = stats.norm.pdf(x, loc = m_7, scale = de_7)
plt.hist(filter_data7, bins=100, density=True, alpha=0.7, color='blue', edgecolor='black', label='Data Histogram')
plt.plot(x, y, color='red', linewidth=2, label='Normal Distribution')

plt.xlabel('Speed of EGO')
plt.ylabel('Density')
plt.legend()


plt.show()








np.random.seed(0)

C1 = 0.20016125574406818
C2 = 0.2525091705905732
C3 = 0.27031317360178175
C4 = 0.27701640006357686

mean1 = -0.8493905719016642
dev1 = 0.18528538158017796
mean2 = -0.6006179853968538
dev2 = 0.1650662422455657
mean3 = -0.5640301173510307
dev3 = 0.16995023213692617
mean4 = -0.6890284057644792
dev4 = 0.23534783666783257

num_samples = 20000

samples1 = []
samples4 = []

for _ in range(num_samples):
    # 
    component = np.random.choice(len(weights1), p=weights1)  
    # 
    sample1 = np.random.multivariate_normal(means1[component], covariances1[component])
    # 
    samples1.append(sample1)
    
    component = np.random.choice(len(weights4), p=weights4)  
    # 
    sample4 = np.random.multivariate_normal(means4[component], covariances4[component])
    # 
    samples4.append(sample4)


# 
v1 = np.sqrt(np.random.normal(m_7, de_7, 20000))
v2 = np.sqrt(np.array(samples1))
v3 = np.sqrt(np.random.normal(m_3, de_3, 20000))
v4 = np.sqrt(np.random.normal(m_5, de_5, 20000))
d1 = np.exp(np.sqrt(np.random.normal(m_2, de_2, 20000)))-60
d2 = np.exp(np.array(samples4))
d3 = np.exp(np.sqrt(np.random.normal(m_6, de_6, 20000)))-60

plt.figure(figsize=(15, 10))

nn = 0



for i in range(100):
    # print(f"v1: {v1[i]}, v2: {v2[i]}, v3: {v3[i]}, v4: {v4[i]}, d1: {d1[i]}, d2: {d2[i]}, d3: {d3[i]}")

    t1 = -((v1[i]**2-277)+(mean1-3*dev1)*(v2[i]**2-277+8*d1[i]))/(8*(v2[i]-v1[i])*(mean1-3*dev1))
    t2 = -((v1[i]**2-277)+(mean2-3*dev2)*(v3[i]**2-277+8*d2[i]))/(8*(v3[i]-v1[i])*(mean2-3*dev2))
    t3 = -(2*v1[i]**3+(2*v3[i]**3+d2[i]*v3[i]**2)*(mean3-3*dev3))/(v3[i]**2*(v3[i]-v1[i])*(mean3-3*dev3))
    t4 = -(2*v4[i]**3+(2*v1[i]**3+d3[i]*v1[i]**2)*(mean4-3*dev4))/(v1[i]**2*(v1[i]-v4[i])*(mean4-3*dev4))

    t5_2 = -(v3[i]**2-277+8*d2[i])/(8*(v3[i]-v1[i]))
    t5_3 = -(2*v3[i]+d2[i])/(v3[i]-v1[i])
    t5_4 = -(2*v1[i]+d3[i])/(v1[i]-v4[i])
    
    if v1[i] >= v2[i]:
        t1_a = -200
        t1_b = t1
    else:
        t1_a = t1
        t1_b = 200
        
    if v1[i] >= v3[i]:
        t2_a = -200
        t2_b = min(t2,t5_2)
        t3_a = -200
        t3_b = min(t3,t5_3)
    else:
        t2_a = max(t2,t5_2)
        t2_b = 200
        t3_a = max(t2,t5_3)
        t3_b = 200
    
    if v4[i] >= v1[i]:
        t4_a = -200
        t4_b = min(t4,t5_4)
    else:
        t4_a = max(t4,t5_4)
        t4_b = 200
    
    T_a = max(t1_a,t2_a,t3_a,t4_a)
    T_b = min(t1_b,t2_b,t3_b,t4_b)
    

    if T_a < T_b - 0.1 and T_a > -200 and T_b < 200:
        nn = nn + 1
        values = []
        d_1 = np.array([])
        d_2 = np.array([])
        d_3 = np.array([])
        
    
        values = np.arange(math.floor(T_a*10)/10, math.ceil(T_b*10)/10, 0.1)
        
        d_1 = d1[i] + values * (v2[i] - v1[i])
        d_2 = d2[i] + values * (v3[i] - v1[i])
        d_3 = d3[i] + values * (v1[i] - v4[i])

        X_1 = -(v1[i]**2 - 277.778) / (v2[i]**2 - 277.778 + 8 * d_1)
        X_2 = -(v1[i]**2 - 277.778) / (v3[i]**2 - 277.778 + 8 * d_2)
        X_3 = -(v1[i]*2) * (v1[i]**2) / (v3[i] * 2 + d_2) / (v3[i] ** 2)
        X_4 = -(v4[i]*2) * (v4[i]**2) / (v1[i] * 2 + d_3) / (v1[i] ** 2)
 
        Y1 = norm.cdf(X_1, loc=mean1, scale=dev1)
        Y2 = norm.cdf(X_2, loc=mean2, scale=dev2)
        Y3 = norm.cdf(X_3, loc=mean3, scale=dev3)
        Y4 = norm.cdf(X_4, loc=mean4, scale=dev4)
        pro1 = C1/Y1**(1)
        pro2 = C2/Y2**(1)
        pro3 = C3/Y3**(1)
        pro4 = C4/Y4**(1)
        
        Y = (Y1*pro1 + Y2*pro2 + Y3*pro3 + Y4*pro4)/(pro1+pro2+pro3+pro4)
        
        # D_X1 = 8*(v1[i]**2-277.778)*(v2[i]-v1[i])/((v2[i]**2-277.778+8*d_1)**2)
        # D_X2 = 8*(v1[i]**2-277.778)*(v3[i]-v1[i])/((v3[i]**2-277.778+8*d_2)**2)
        # D_X3 = 2*v1[i]**3*(v3[i]-v1[i])/(v3[i]**2)/((2*v3[i]+d_2)**2)
        # D_X4 = 2*v4[i]**3*(v1[i]-v4[i])/(v1[i]**2)/((2*v1[i]+d_3)**2)
                
        # D_Y1 = D_X1/(dev1*math.sqrt(2*math.pi))*np.exp(-(X_1-mean1)**2/(2*dev1**2))
        # D_Y2 = D_X2/(dev2*math.sqrt(2*math.pi))*np.exp(-(X_2-mean2)**2/(2*dev2**2))
        # D_Y3 = D_X3/(dev3*math.sqrt(2*math.pi))*np.exp(-(X_3-mean3)**2/(2*dev3**2))
        # D_Y4 = D_X4/(dev4*math.sqrt(2*math.pi))*np.exp(-(X_4-mean4)**2/(2*dev4**2))        
        # D_Y = (C1+C2+C3+C4)*(C1*D_Y1/Y1**2+C2*D_Y2/Y2**2+C3*D_Y3/Y3**2+C4*D_Y4/Y4**2)/(C1/Y1+C2/Y2+C3/Y3+C4/Y4)**2
        
        data = np.array(Y)
        TIME = (values - math.ceil(T_a*10)/10) / (math.floor(T_b*10)/10 - math.ceil(T_a*10)/10)
        plt.plot(TIME,  Y)      
        
        # plt.plot(TIME,  D_Y)
        # plt.axhline(y=0, color='k', linestyle='--', linewidth=3)   
        plt.xlim(-0.05,1.05)
plt.xlabel('Time window')
# plt.ylabel(r"$U'(t)$")
plt.ylabel(r'$\mathit{U}(t)$')
# plt.title('Data Histogram')
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.rcParams.update({'font.size': 24})