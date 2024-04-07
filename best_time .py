# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 08:05:46 2023

@author: HAN
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import math
import pickle
import os
import scipy.stats as stats



left_right = 0       # left lane change:0  /  right lane change:1
sig_k = 1            # sigma=1 indicates that the risk level is R1；  sigma=2 indicates that the risk level is R2；  sigma=3 indicates that the risk level is R3；
# buchang = 1 * 25
def Extract_data_of_lane_change(l_r,case,nn):       
    """ Extract lane change trajectories from the HighD data set
    left_right=0：left lane change；left_right=1：right lane change
    case = 0: There are 3 vehicles around
    case = 1: CPV
    case = 2: TPV
    case = 3: TFV    """
    
    cnt = 1   #
    n = 0
    Lc_tracks = {}
    for i in range(0,nn):
        Lc_complete = {}
        temp_name = "LC_vehcile_features_" + str(cnt) + ".pkl"
        cnt = cnt + 1

        prexfix_path = "C:/Users/HAN/.spyder-py3/data_pkl"  ## Please replace the path
        pkl_path = os.path.join(prexfix_path, temp_name)
        data = {}
        f=open(pkl_path,'rb')
        data=pickle.load(f)
        
        for key in data.keys():
            Lc_vehicle = {}   
            
            if (data[key]['is_complete_start'] == True): #
                Lc_start = data[key]['LC_action_start_local_frame']
                
                if np.array(data[key]['s_Velocity'])[Lc_start] > 60/3.6:    # Delete data with a speed lower than 60km/h
  
                    if np.array(data[key]['left_right_maveuver']) == l_r:
                        # Extract lane change data when there are other vehicles around
                        if isinstance(data[key]['LC_interaction']['action_ego_preceding_S_velocity'], np.float64) and isinstance(data[key]['LC_interaction']['action_ego_LTCpreceding_S_velocity'], np.float64) and isinstance(data[key]['LC_interaction']['action_ego_LTCfollowing_S_velocity'], np.float64) and case == 0:
                            
                            Lc_vehicle['id'] = data[key]['vehicle_id'] 
                            Lc_vehicle['class'] = data[key]['class']
                            Lc_vehicle['left_right'] = data[key]['left_right_maveuver']
                            Lc_vehicle['start_forward_TTC'] = np.array(data[key]['LC_interaction']['action_ttc'])     
                            Lc_vehicle['start_forward_DHW'] = np.array(data[key]['LC_interaction']['action_ego_preceding_S_distance'])                          
                            Lc_vehicle['speed_ego'] = np.array(data[key]['s_Velocity'])[Lc_start]
                            Lc_vehicle['speed_lead'] = Lc_vehicle['speed_ego'] - np.array(data[key]['LC_interaction']['action_ego_preceding_S_velocity'])
                            Lc_vehicle['LTC_start_forward_TTC'] = np.array(data[key]['LC_interaction']['action_LTC_forward_TTC'])     
                            Lc_vehicle['LTC_start_forward_DHW'] = np.array(data[key]['LC_interaction']['action_ego_LTCpreceding_S_distance'])                          
                            Lc_vehicle['speed_forward'] = Lc_vehicle['speed_ego'] - np.array(data[key]['LC_interaction']['action_ego_LTCpreceding_S_velocity'])
                            Lc_vehicle['LTC_start_backward_TTC'] = np.array(data[key]['LC_interaction']['action_LTC_backward_TTC'])     
                            Lc_vehicle['LTC_start_backward_DHW'] = np.array(data[key]['LC_interaction']['action_ego_LTCfollowing_S_distance'])                          
                            Lc_vehicle['speed_backward'] = Lc_vehicle['speed_ego'] - np.array(data[key]['LC_interaction']['action_ego_LTCfollowing_S_velocity']) 
                            Lc_vehicle['i'] = i+1
                            Lc_vehicle['start_frame'] = data[key]['LC_action_start_global_frame']
                            Lc_vehicle['moment_frame'] = data[key]['LC_moment_global']
                            
                            Lc_complete[n] = Lc_vehicle   
                            n = n + 1
                        
                        if isinstance(data[key]['LC_interaction']['action_ego_preceding_S_velocity'], np.float64) and case == 1:
                            
                            Lc_vehicle['id'] = data[key]['vehicle_id'] 
                            Lc_vehicle['class'] = data[key]['class']
                            Lc_vehicle['left_right'] = data[key]['left_right_maveuver']
                            Lc_vehicle['start_forward_TTC'] = np.array(data[key]['LC_interaction']['action_ttc'])     
                            Lc_vehicle['start_forward_DHW'] = np.array(data[key]['LC_interaction']['action_ego_preceding_S_distance'])                          
                            Lc_vehicle['speed_ego'] = np.array(data[key]['s_Velocity'])[Lc_start]
                            Lc_vehicle['speed_lead'] = Lc_vehicle['speed_ego'] - np.array(data[key]['LC_interaction']['action_ego_preceding_S_velocity'])
                           
                            Lc_complete[n] = Lc_vehicle   
                            n = n + 1
                        
                        if isinstance(data[key]['LC_interaction']['action_ego_LTCpreceding_S_velocity'], np.float64) and case == 2:
                            
                            Lc_vehicle['id'] = data[key]['vehicle_id'] 
                            Lc_vehicle['class'] = data[key]['class']
                            Lc_vehicle['left_right'] = data[key]['left_right_maveuver']
                            Lc_vehicle['LTC_start_forward_TTC'] = np.array(data[key]['LC_interaction']['action_LTC_forward_TTC'])     
                            Lc_vehicle['LTC_start_forward_DHW'] = np.array(data[key]['LC_interaction']['action_ego_LTCpreceding_S_distance'])                          
                            Lc_vehicle['speed_ego'] = np.array(data[key]['s_Velocity'])[Lc_start]
                            Lc_vehicle['speed_forward'] = Lc_vehicle['speed_ego'] - np.array(data[key]['LC_interaction']['action_ego_LTCpreceding_S_velocity'])
                           
                            Lc_complete[n] = Lc_vehicle   
                            n = n + 1
                            
                        if isinstance(data[key]['LC_interaction']['action_ego_LTCfollowing_S_velocity'], np.float64) and case == 3:
                            
                            Lc_vehicle['id'] = data[key]['vehicle_id'] 
                            Lc_vehicle['class'] = data[key]['class']
                            Lc_vehicle['left_right'] = 0
                            Lc_vehicle['LTC_start_backward_TTC'] = np.array(data[key]['LC_interaction']['action_LTC_backward_TTC'])     
                            Lc_vehicle['LTC_start_backward_DHW'] = np.array(data[key]['LC_interaction']['action_ego_LTCfollowing_S_distance'])                          
                            Lc_vehicle['speed_ego'] = np.array(data[key]['s_Velocity'])[Lc_start]
                            Lc_vehicle['speed_backward'] = Lc_vehicle['speed_ego'] - np.array(data[key]['LC_interaction']['action_ego_LTCfollowing_S_velocity'])
                          
                            Lc_complete[n] = Lc_vehicle   
                            n = n + 1        
        Lc_tracks[i] = Lc_complete
    return Lc_tracks

Lc_data_0 = Extract_data_of_lane_change(left_right,0,57)
Lc_data_1 = Extract_data_of_lane_change(left_right,1,57)
Lc_data_2 = Extract_data_of_lane_change(left_right,2,57)
Lc_data_3 = Extract_data_of_lane_change(left_right,3,57)



def compute_cp(data=Lc_data_1):  #cpv
    
    startForwardTTC = []
    startForwardTHW = []
    startForwardDHW = []
    
    speedEgo = []
    speedLead = []
    safeModel = []
    
    for key0 in data.keys():
        for key in data[key0].keys():
            
            if data[key0][key]['start_forward_DHW'] > 5:
              
                SFT = data[key0][key]['start_forward_TTC']  
                SFD = data[key0][key]['start_forward_DHW']    
                SE = data[key0][key]['speed_ego'] 
                SL = data[key0][key]['speed_lead']         
                SFTH = SFD / SE
                SM = -(SE * SE - 277.778) / (SL * SL-277.778 + 8 * SFD)
        
                startForwardTTC.append(SFT)      
                startForwardTHW.append(SFTH)      
                startForwardDHW.append(SFD)
                speedEgo.append(SE)
                speedLead.append(SL)     
                safeModel.append(SM)
    return safeModel,startForwardTTC,startForwardTHW,startForwardDHW,speedEgo,speedLead

def compute_tp(data=Lc_data_2):  # tpv

    LTCstartForwardTTC = []
    LTCstartForwardTHW = []
    LTCstartForwardDHW = []
    
    speedEgo = []
    speedFor = []
    safeModelFor = []
    effiModelFor =[]
    
    for key0 in data.keys():
        for key in data[key0].keys():
            
            if data[key0][key]['LTC_start_forward_DHW'] != 0:
                LSFT = data[key0][key]['LTC_start_forward_TTC']        
                LSFD = data[key0][key]['LTC_start_forward_DHW']   
                SE = data[key0][key]['speed_ego'] 
                SF = data[key0][key]['speed_forward'] 
                LSFTH = LSFD/SE      
                   
                EMF = -(SE * SE - 277.778) / (SF * SF-277.778 + 8 * LSFD)
                SMF = -((2  *SE) / (LSFD + 2 * SF)) * SE * SE / SF / SF
        
                LTCstartForwardTTC.append(LSFT)
                LTCstartForwardTHW.append(LSFTH)
                LTCstartForwardDHW.append(LSFD)
        
                speedEgo.append(SE)
                speedFor.append(SF)     
                safeModelFor.append(SMF)
                effiModelFor.append(EMF)
    return effiModelFor,safeModelFor,LTCstartForwardTTC,LTCstartForwardTHW,LTCstartForwardDHW,speedEgo,speedFor

def compute_tf(data=Lc_data_3):  # tfv
    LTCstartBackwardTTC = []
    LTCstartBackwardTHW = []
    LTCstartBackwardDHW = []
    
    speedEgo = []
    speedBack = []
    safeModelBack = []
    for key0 in data.keys():
        for key in data[key0].keys():
            
            if data[key0][key]['LTC_start_backward_DHW'] != 0:
                LSBT = data[key0][key]['LTC_start_backward_TTC']        
                LSBD = data[key0][key]['LTC_start_backward_DHW']   
                SE = data[key0][key]['speed_ego'] 
                SB = data[key0][key]['speed_backward'] 
                LSBTH = LSBD/SB      
                
        
                SMB =  -(SB*2)/(LSBD+SE*2)*(SB/SE)*(SB/SE)
                
                LTCstartBackwardTTC.append(LSBT)
                LTCstartBackwardTHW.append(LSBTH)
                LTCstartBackwardDHW.append(LSBD)
        
                speedEgo.append(SE)
                speedBack.append(SB)     
                safeModelBack.append(SMB)
    return safeModelBack,LTCstartBackwardTTC,LTCstartBackwardTHW,LTCstartBackwardDHW,speedEgo,speedBack


def miu_sigama(SM,EMF,SMF,SMB):   # Calculate the mean and variance of the indices
    mean1 = np.mean(np.array(SM))
    dev1 = np.sqrt(np.var(np.array(SM)))
    mean2 = np.mean(np.array(EMF))
    dev2 = np.sqrt(np.var(np.array(EMF)))
    mean3 = np.mean(np.array(SMF))
    dev3 = np.sqrt(np.var(np.array(SMF)))
    mean4 = np.mean(np.array(SMB))
    dev4 = np.sqrt(np.var(np.array(SMB)))
    MIU_SIGAMA = [mean1,dev1, mean2,dev2, mean3,dev3, mean4,dev4,]
    return MIU_SIGAMA


def compute_all(data = Lc_data_0):  # safeModel:X1  effiModelFor: X2   safeModelFor:X3   safeModelBack:X4

    safeModel = []   
    safeModelFor = []
    effiModelFor = []
    safeModelBack = []
    for key0 in data.keys():
        for key in data[key0].keys():
            # 剔除前车距离为负的个例
            if data[key0][key]['start_forward_DHW']>5 and data[key0][key]['LTC_start_forward_DHW'] != 0 and data[key0][key]['LTC_start_backward_DHW'] != 0:
              
                SFD = data[key0][key]['start_forward_DHW']    
                SE = data[key0][key]['speed_ego'] 
                SL = data[key0][key]['speed_lead']                   
                SM = -(SE*SE-277.778)/(SL*SL-277.778+8*SFD)               
                safeModel.append(SM)
    
                LSFD = data[key0][key]['LTC_start_forward_DHW']   
                SF = data[key0][key]['speed_forward'] 
                EMF = -(SE*SE-277.778)/(SF*SF-277.778+8*LSFD)
                SMF = -((2*SE)/(LSFD+2*SF))*SE*SE/SF/SF
                effiModelFor.append(EMF)
                safeModelFor.append(SMF)
                
                LSBD = data[key0][key]['LTC_start_backward_DHW']   
                SB = data[key0][key]['speed_backward'] 
                SMB =  -(SB*SB-277.778)/(SE*SE-277.778+8*LSBD)   
                safeModelBack.append(SMB)
                x_all = [safeModel,effiModelFor,safeModelFor,safeModelBack]
    return x_all

X_all = compute_all()

MiuSigama = miu_sigama(compute_cp()[0], compute_tp()[0], compute_tp()[1], compute_tf()[0])


def y_pro(data = X_all ,argument = MiuSigama):     # Calculated probability distribution
    probabilities1 = norm.cdf(np.array(data[0]), loc=argument[0], scale=argument[1])
    probabilities2 = norm.cdf(np.array(data[1]), loc=argument[2], scale=argument[3])
    probabilities3 = norm.cdf(np.array(data[2]), loc=argument[4], scale=argument[5])
    probabilities4 = norm.cdf(np.array(data[3]), loc=argument[6], scale=argument[7])
    probabilities = [probabilities1, probabilities2, probabilities3, probabilities4]
    return probabilities

pro_y = y_pro()


def min_max_scaling(data = pro_y):         # standardization
    min1 = min(np.array(data[0]))
    max1 = max(np.array(data[0]))
    min2 = min(np.array(data[1]))
    max2 = max(np.array(data[1])) 
    min3 = min(np.array(data[2]))
    max3 = max(np.array(data[2])) 
    min4 = min(np.array(data[3]))
    max4 = max(np.array(data[3]))   
    scaled1 = (data[0] - min1) / (max1 - min1) + 0.0001
    scaled2 = (data[1] - min2) / (max2 - min2) + 0.0001
    scaled3 = (data[2] - min3) / (max3 - min3) + 0.0001
    scaled4 = (data[3] - min4) / (max4 - min4) + 0.0001
    y_scaled = [scaled1,scaled2,scaled3,scaled4]
    return y_scaled

scaled_y = (min_max_scaling())

def entropy_weight(data = scaled_y):        # Entropy weight
    y1 = data[0] / np.sum(data[0])
    y2 = data[1] / np.sum(data[1])
    y3 = data[2] / np.sum(data[2])
    y4 = data[3] / np.sum(data[3])
    
    k = -1/np.log(len(data[0]))
    
    e1 = k*sum(y1*np.log(y1))
    e2 = k*sum(y2*np.log(y2))
    e3 = k*sum(y3*np.log(y3))
    e4 = k*sum(y4*np.log(y4))
    
    c1 = (1 - e1) / (4 - (e1+e2+e3+e4))
    c2 = (1 - e2) / (4 - (e1+e2+e3+e4))
    c3 = (1 - e3) / (4 - (e1+e2+e3+e4))
    c4 = (1 - e4) / (4 - (e1+e2+e3+e4))
    
    c = [c1,c2,c3,c4]
    return c

C = entropy_weight()


def filtered1(data,arg_id):  # Filter more data from highD dataset
    out_filtered = data[data['id'] == arg_id]
    return out_filtered

def first_last(data,arg_a,arg_b): # Gets the serial number of the first and last frames
    out_data = data[(data['frame'] >= arg_a) & (data['frame'] <= arg_b)][['frame','x', 'xVelocity']]
    return out_data
      
def sign_com(v):  # Judge the direction of travel
    if v < 0:
        sign = -1
    else:
        sign = 1
    return sign
 
def y_sum(y1,y2,y3,y4,argment = C):   # emotional utility
    pr1 = C[0]/y1
    pr2 = C[1]/y2
    pr3 = C[2]/y3
    pr4 = C[3]/y4
    Y_sum = (y1*pr1 + y2*pr2 + y3*pr3 + y4*pr4)/(pr1+pr2+pr3+pr4)
    return Y_sum


def dynamic_predict(dcp,dtp,dtf,vego,vcp,vtp,vtf,argument2 = MiuSigama, argument3 = C): # Prediction of emotional utility
    NN = math.floor((len(dcp)-1)/25)+1
    l = len(dcp)  
    g_k = np.array([])    
    for current in range(NN):
        D_CP_PRE = np.array([])
        D_TP_PRE = np.array([])
        D_TF_PRE = np.array([])
        Speed_ego_PRE = np.array([])
        Speed_cp_PRE = np.array([])
        Speed_tp_PRE = np.array([])
        Speed_tf_PRE = np.array([])
       
        t_current = current*25  
        
        for i in range(l):
            if i <= t_current:
                D_CP_PRE = np.append(D_CP_PRE,dcp[i])
                D_TP_PRE = np.append(D_TP_PRE,dtp[i])
                D_TF_PRE = np.append(D_TF_PRE,dtf[i])
                Speed_ego_PRE = np.append(Speed_ego_PRE,vego[i])
                Speed_cp_PRE = np.append(Speed_cp_PRE,vcp[i])
                Speed_tp_PRE = np.append(Speed_tp_PRE,vtp[i])
                Speed_tf_PRE = np.append(Speed_tf_PRE,vtf[i])
            else:
                D_CP_PRE = np.append(D_CP_PRE , D_CP_PRE[-1] + (vcp[t_current] - vego[t_current]) / 25)
                D_TP_PRE = np.append(D_TP_PRE , D_TP_PRE[-1] + (vtp[t_current] - vego[t_current]) / 25)
                D_TF_PRE = np.append(D_TF_PRE , D_TF_PRE[-1] + (vego[t_current] - vtf[t_current]) / 25)
                Speed_ego_PRE = np.append(Speed_ego_PRE,vego[t_current])
                Speed_cp_PRE = np.append(Speed_cp_PRE,vcp[t_current])
                Speed_tp_PRE = np.append(Speed_tp_PRE,vtp[t_current])
                Speed_tf_PRE = np.append(Speed_tf_PRE,vtf[t_current])
        
        # print(D_TF_PRE[t_current],Speed_ego_PRE[t_current],Speed_tf_PRE[t_current])
        # 
        X_1_PRE = -(Speed_ego_PRE**2 - 277) / (Speed_cp_PRE**2 - 277 + 8 * D_CP_PRE)
        X_2_PRE = -(Speed_ego_PRE**2 - 277) / (Speed_tp_PRE**2 - 277 + 8 * D_TP_PRE)
        X_3_PRE = -(Speed_ego_PRE*2) * (Speed_ego_PRE**2) / (Speed_tp_PRE * 2 + D_TP_PRE) / (Speed_tp_PRE**2)
        X_4_PRE = -(Speed_tf_PRE*2) * (Speed_tf_PRE**2) / (Speed_ego_PRE * 2 + D_TF_PRE) / (Speed_ego_PRE**2)
            
        # 计算绩效
        Y_1_PRE = norm.cdf(X_1_PRE, loc=argument2[0], scale=argument2[1])
        Y_2_PRE = norm.cdf(X_2_PRE, loc=argument2[2], scale=argument2[3])
        Y_3_PRE = norm.cdf(X_3_PRE, loc=argument2[4], scale=argument2[5])
        Y_4_PRE = norm.cdf(X_4_PRE, loc=argument2[6], scale=argument2[7])
       
    
        Y_SUM_PRE = y_sum(Y_1_PRE,Y_2_PRE,Y_3_PRE,Y_4_PRE)
        
        g_k = np.append(g_k,Y_SUM_PRE[t_current])
    
        # time window
        if vcp[t_current] == vego[t_current] or vtp[t_current] == vego[t_current] or vtf[t_current] == vego[t_current]:
            vego[t_current] = vego[t_current] + 0.01
  
        t1 = -((vego[t_current]**2-277)+(argument2[0]-sig_k*argument2[1])*(vcp[t_current]**2-277+8*dcp[t_current]))/(8*(vcp[t_current]-vego[t_current])*(argument2[0]-sig_k*argument2[1]))
        t2 = -((vego[t_current]**2-277)+(argument2[2]-sig_k*argument2[3])*(vtp[t_current]**2-277+8*dtp[t_current]))/(8*(vtp[t_current]-vego[t_current])*(argument2[2]-sig_k*argument2[3]))
        t3 = -(2*vego[t_current]**3+(2*vtp[t_current]**3+dtp[t_current]*vtp[t_current]**2)*(argument2[4]-sig_k*argument2[5]))/(vtp[t_current]**2*(vtp[t_current]-vego[t_current])*(argument2[4]-sig_k*argument2[5]))
        t4 = -(2*vtf[t_current]**3+(2*vego[t_current]**3+dtf[t_current]*vego[t_current]**2)*(argument2[6]-sig_k*argument2[7]))/(vego[t_current]**2*(vego[t_current]-vtf[t_current])*(argument2[6]-sig_k*argument2[7]))
        
        t5_2 = -(vtp[t_current]**2-277+8*dtp[t_current])/(8*(vtp[t_current]-vego[t_current]))
        t5_3 = -(2*vtp[t_current]+dtp[t_current])/(vtp[t_current]-vego[t_current])
        t5_4 = -(2*vego[t_current]+dtf[t_current])/(vego[t_current]-vtf[t_current])
    
        
        if vego[t_current] >= vcp[t_current]:
            t1_a = 0
            t1_b = max(t1,0)
        else:
            t1_a = max(t1,0)
            t1_b = 10000
            
        if vego[t_current] >= vtp[t_current]:
            t2_a = 0
            
            t2_b = min(max(t2,0),max(t5_2,0))
            t3_a = 0
            t3_b = min(max(t3,0),max(t5_3,0))
        else:
            t2_a = max(t2,0,t5_2)
            t2_b = 10000   
            t3_a = max(t2,0,t5_3)
            t3_b = 10000   
        
        if vtf[t_current] >= vego[t_current]:
            t4_a = 0
            t4_b = min(max(t4,0),max(t5_4,0))
        else:
            t4_a = max(t4,0,t5_4)
            t4_b = 10000
        
        T_a = max(t1_a,t2_a,t3_a,t4_a)
        T_b = min(t1_b,t2_b,t3_b,t4_b)

    
        if T_a >= T_b and current < NN - 1:
            continue 
        elif T_a >= T_b and current == NN - 1:
            T_m = 100
            G_m = 100
            break   
        
        elif T_a > (l-current*25)/25 and T_a < T_b and current == NN - 1:
            T_m = 100
            G_m = 1000
            break  
        
        elif T_a < T_b and current <= NN - 1:
            if math.ceil(T_a*25) < math.floor(T_b*25) and math.ceil(T_a*25)+t_current < l: 
                max_index = np.argmax(Y_SUM_PRE[math.ceil(T_a*25)+t_current:min(t_current+math.floor(T_b*25),l)])  # 找到最大值的索引
                max_value = Y_SUM_PRE[math.ceil(T_a*25)+t_current:min(t_current+math.floor(T_b*25),l)][max_index]  # 找到最大值
            elif math.ceil(T_a*25) == math.floor(T_b*25) and math.ceil(T_a*25)+t_current < l: 
                max_index = t_current  # Index of maximum value
                max_value = Y_SUM_PRE[t_current]  # maximum value      
            else:
                continue
            
            
            if max_index <= (25-T_a*25):
                if current > 0:
                    if g_k[current] > g_k[current-1]:
                        T_m = max_index/25 + T_a + +current
                        G_m = max_value
                        break
                    else:
                        T_m = T_a + current
                        G_m = g_k[current]
                        break
                else:
                    T_m = max_index/25 + T_a +current
                    G_m = max_value
                    break  
            else: 
                if current > 0:
                    if g_k[current] > g_k[current-1]:
                        T_m = max_index/25 + T_a + +current
                        G_m = max_value
                        continue
                    else:
                        T_m = T_a + current
                        G_m = g_k[current]
                        break
                else:
                    T_m = max_index/25 + T_a +current
                    G_m = max_value
                    continue
    return T_m, G_m,T_a,T_b   # T_m:Best time to change lanes


Lc_data_n = Extract_data_of_lane_change(left_right,0,57)

def stca(dcp,dtp,dtf,vego,vcp,vtp,vtf):  
    vego_des = vego[0]
    l = len(dcp)
    for i in range(l):
        if (dcp[i] < (vego_des + 1) and vcp[i] < vego_des) and (dtp[i] > dcp[i] and vtp[i] > vcp[i]) and (dtf[i] > vtf[i] + 1):
            lane_change = True
            t_LC = i/25
            break
        else:
            lane_change = False
            t_LC = 100
    return lane_change,t_LC
                              


def filtered_data_all(data = Lc_data_n,argument1 = left_right, argument2 = MiuSigama, argument3 = C):
    i = 0 
    Tm_Gm = {}
    n = 0
    for key0 in data.keys():       
        df = []
        CNT = i+1
        i = i+1
        temp_name = str(CNT) + "_tracks.csv"
        prexfix_path = "H:/data/data"  ## Please replace the path
        pkl_path = os.path.join(prexfix_path, temp_name)                         
        df = pd.read_csv(pkl_path)   
        
        for key in data[key0].keys():
            out_data = {}
            frame_start = data[key0][key]['start_frame'] 
            frame_moment = data[key0][key]['moment_frame']             
            # print(frame_start,frame_moment)
            EGO_ID = data[key0][key]['id']    
            
            filtered_ego = filtered1(df,EGO_ID)   # ego
            first_ego = filtered_ego['frame'].iloc[0]
            last_ego = filtered_ego['frame'].iloc[-1] 
            Width_ego = filtered_ego['width'].iloc[0]
    
            # if i == 21 and EGO_ID == 115:
            #     continue     
            CP_ID = filtered_ego[filtered_ego['frame'] == frame_start]['precedingId'].iloc[0]   # ID of CPV、TPV、TFV
                 
            if argument1 == 0:
                TP_ID = filtered_ego[filtered_ego['frame'] == frame_moment]['leftPrecedingId'].iloc[0]
                TF_ID = filtered_ego[filtered_ego['frame'] == frame_moment]['leftFollowingId'].iloc[0]
            else:
                TP_ID = filtered_ego[filtered_ego['frame'] == frame_moment]['rightPrecedingId'].iloc[0]
                TF_ID = filtered_ego[filtered_ego['frame'] == frame_moment]['rightFollowingId'].iloc[0]        
            
            # print(EGO_ID,CP_ID,TP_ID,TF_ID)
            # print(i)
           
            
            if CP_ID == 0 or TP_ID == 0 or TF_ID == 0:
                continue

            filtered_cp = filtered1(df,CP_ID)   # cpv 
            first_cp = filtered_cp['frame'].iloc[0]
            
            last_cp = filtered_cp['frame'].iloc[-1] 
            Width_cp = filtered_cp['width'].iloc[0]
            
            filtered_tp = filtered1(df,TP_ID)   # tpv 
            first_tp = filtered_tp['frame'].iloc[0]
            last_tp = filtered_tp['frame'].iloc[-1] 
            Width_tp = filtered_tp['width'].iloc[0]
            
            filtered_tf = filtered1(df,TF_ID)   # tfv
            first_tf = filtered_tf['frame'].iloc[0]
            last_tf = filtered_tf['frame'].iloc[-1]                    
           
            first_frame = max(first_ego,first_cp,first_tp,first_tf,frame_start-100)  
            last_frame = min(last_ego,last_cp,last_tp,last_tf,frame_start+100)
            
            sign = sign_com(np.array(filtered_ego['xVelocity'])[0])
           
            # Fetch data from first_frame to last_frame
            Ego_data = first_last(filtered_ego,first_frame,last_frame)
            Cp_data = first_last(filtered_cp,first_frame,last_frame)
            Tp_data = first_last(filtered_tp,first_frame,last_frame)
            Tf_data = first_last(filtered_tf,first_frame,last_frame)
            Speed_ego = sign * np.array(Ego_data['xVelocity'])
            Local_ego = sign * np.array(Ego_data['x'])
            Speed_cp = sign * np.array(Cp_data['xVelocity'])
            Local_cp = sign * np.array(Cp_data['x'])
            Speed_tp= sign * np.array(Tp_data['xVelocity'])
            Local_tp = sign * np.array(Tp_data['x'])
            Speed_tf = sign * np.array(Tf_data['xVelocity'])
            Local_tf = sign * np.array(Tf_data['x'])

            # 
            D_CP = Local_cp - Local_ego - Width_cp
            D_TP = Local_tp - Local_ego - Width_tp
            D_TF = Local_ego - Local_tf - Width_ego
            
            # 计算指标
            # X_1 = -(Ego_data[1]**2 - 277.778) / (Cp_data[1]**2 - 277.778 + 8 * D_CP)
            # X_2 = -(Ego_data[1]**2 - 277.778) / (Tp_data[1]**2 - 277.778 + 8 * D_TP)
            # X_3 = -(Ego_data[1]*2) * (Ego_data[1]**2) / (Tp_data[1] * 2 + D_TP) / (Tp_data[1]**2)
            # X_4 = -(Tf_data[1]*2) * (Tf_data[1]**2) / (Ego_data[1]*2 + D_TF) / (Ego_data[1]**2)             
            # 计算绩效
            # Y_1 = norm.cdf(X_1, loc=argument2[0], scale=argument2[1])
            # Y_2 = norm.cdf(X_2, loc=argument2[2], scale=argument2[3])
            # Y_3 = norm.cdf(X_3, loc=argument2[4], scale=argument2[5])
            # Y_4 = norm.cdf(X_4, loc=argument2[8], scale=argument2[7])
            
            # Y_SUM = y_sum(Y_1,Y_2,Y_3,Y_4)                       
            Results = dynamic_predict(D_CP,D_TP,D_TF,Speed_ego,Speed_cp,Speed_tp,Speed_tf,argument2 = MiuSigama, argument3 = C)    
            T_fact = (frame_start-first_frame)/25
            T_total = (last_frame-first_frame)/25
            if Results[0] != 100:
                out_data['TM'] = Results[0]
                out_data['GM'] = Results[1]
                out_data['Ta'] = Results[2]
                out_data['Tb'] = Results[3]
                out_data['T_fact'] = T_fact
                out_data['T_total'] = T_total            
                
                out_data['lanechange'] = stca(D_CP,D_TP,D_TF,Speed_ego,Speed_cp,Speed_tp,Speed_tf)[0]
                out_data['t_LC'] = stca(D_CP,D_TP,D_TF,Speed_ego,Speed_cp,Speed_tp,Speed_tf)[1]
              
                Tm_Gm[n] = out_data 
                n = n+1  
            else:
                print(EGO_ID,CP_ID,TP_ID,TF_ID,Results[0],Results[1],Results[2],Results[3])
                print(i)                    
            
    return Tm_Gm


results = filtered_data_all()    
            
t_m_data = np.array([])
t_window_data = np.array([])
t_m_data1 = np.array([])
t_m_stca = np.array([])
t_window_data1 = np.array([])
for key in results.keys():
    t_m_data = np.append(t_m_data,results[key]['TM']-results[key]['T_fact'])
    t_window_data = np.append(t_window_data,results[key]['Tb']-results[key]['Ta'])
    if (results[key]['lanechange'] == True):
        t_m_stca = np.append(t_m_stca,results[key]['t_LC']-results[key]['T_fact'])
        t_m_data1 = np.append(t_m_data1,results[key]['TM']-results[key]['T_fact'])
        t_window_data1 = np.append(t_window_data1,results[key]['Tb']-results[key]['Ta'])
    else:
        continue
    
window_data = t_window_data[t_window_data <= 60]   
mean_window1 = np.mean(abs(window_data))
std_dev_window1 = np.std(window_data)  

 
    
mean_mine = np.mean(abs(t_m_data))
std_dev_mine = np.std(t_m_data)  
mean_stca = np.mean(abs(t_m_stca))
std_dev_stca = np.std(t_m_stca)
mean_mine1 = np.mean(abs(t_m_data1))
std_dev_mine1 = np.std(t_m_data1)    
mean_window = np.mean(abs(t_window_data))
std_dev_window = np.std(t_window_data)                                                           
mean_window2 = np.mean(abs(t_window_data1))
std_dev_window2 = np.std(t_window_data1)               


    


t_m_Ta = np.array([])
t_m_Tb = np.array([])
for key in results.keys():
    if results[key]['TM'] <= results[key]['T_fact']:        
        t_m_Ta = np.append(t_m_Ta,results[key]['T_fact']-results[key]['TM'])
    else:
        t_m_Tb = np.append(t_m_Tb,results[key]['TM']-results[key]['T_fact'])
    


 
    
mean_mine_ta = np.mean(t_m_Ta)
std_dev_mine_ta = np.std(t_m_Ta)  
mean_mine_tb = np.mean(t_m_Tb)
std_dev_mine_tb = np.std(t_m_Tb) 









