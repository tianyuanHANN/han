# -*- coding: utf-8 -*-
"""

"""

# ID = 949
# CP = 945  
# TP = 950
# TF = 951


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import math

frame_start = 9931
frame_moment = 10024

Left_right = 0
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


df = pd.read_csv('H:/data/data/1_tracks.csv')

# ----------------------------------------------------------------------------------

filtered_ego = df[df['id'] == 482]

first_ego = filtered_ego['frame'].iloc[0]
last_ego = filtered_ego['frame'].iloc[-1]


Width_ego = filtered_ego['width'].iloc[0]


CP_ID = filtered_ego[filtered_ego['frame'] == frame_start]['precedingId'].iloc[0]
TP_ID = filtered_ego[filtered_ego['frame'] == frame_moment]['leftPrecedingId'].iloc[0]
TF_ID = filtered_ego[filtered_ego['frame'] == frame_moment]['leftFollowingId'].iloc[0]

print(CP_ID,TP_ID,TF_ID)


filtered_cp = df[df['id'] == CP_ID]

first_cp = filtered_cp['frame'].iloc[0]
last_cp = filtered_cp['frame'].iloc[-1]

Width_cp = filtered_cp['width'].iloc[0]


filtered_tp = df[df['id'] == TP_ID]

first_tp = filtered_tp['frame'].iloc[0]
last_tp = filtered_tp['frame'].iloc[-1]

Width_tp = filtered_tp['width'].iloc[0]


filtered_tf = df[df['id'] == TF_ID]

first_tf = filtered_tf['frame'].iloc[0]
last_tf = filtered_tf['frame'].iloc[-1]

first_frame = max(first_ego,first_cp,first_tp,first_tf,frame_start-100)
last_frame = min(last_ego,last_cp,last_tp,last_tf,frame_start+100)

# ----------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------

Ego_data = filtered_ego[(filtered_ego['frame'] >= first_frame) & (filtered_ego['frame'] <= last_frame)][['frame','x', 'xVelocity']]
if np.array(filtered_ego['xVelocity'])[0]<0:
    sign = -1
else:
    sign = 1
Speed_ego = sign * np.array(Ego_data['xVelocity'])
Local_ego = sign * np.array(Ego_data['x'])

Cp_data = filtered_cp[(filtered_cp['frame'] >= first_frame) & (filtered_cp['frame'] <= last_frame)][['frame','x', 'xVelocity']]
Speed_cp = sign * np.array(Cp_data['xVelocity'])
Local_cp = sign * np.array(Cp_data['x'])

Tp_data = filtered_tp[(filtered_tp['frame'] >= first_frame) & (filtered_tp['frame'] <= last_frame)][['frame','x', 'xVelocity']]
Speed_tp = sign * np.array(Tp_data['xVelocity'])
Local_tp = sign * np.array(Tp_data['x'])

Tf_data = filtered_tf[(filtered_tf['frame'] >= first_frame) & (filtered_tf['frame'] <= last_frame)][['frame','x', 'xVelocity']]
Speed_tf = sign * np.array(Tf_data['xVelocity'])
Local_tf = sign * np.array(Tf_data['x'])


D_CP = Local_cp - Local_ego - Width_cp
D_TP = Local_tp - Local_ego - Width_tp
D_TF = Local_ego - Local_tf - Width_ego



TIME = np.arange(len(Speed_cp))

X_1 = -(Speed_ego*Speed_ego - 277.778) / (Speed_cp*Speed_cp - 277.778 + 8 * D_CP)
X_2 = -(Speed_ego*Speed_ego - 277.778) / (Speed_tp*Speed_tp - 277.778 + 8 * D_TP)
X_3 = -(Speed_ego*2) * (Speed_ego*Speed_ego) / (Speed_tp * 2 + D_TP) / (Speed_tp*Speed_tp)
X_4 = -(Speed_tf*2) * (Speed_tf*Speed_tf) / (Speed_ego * 2 + D_TF) / (Speed_ego*Speed_ego)


Y_1 = norm.cdf(X_1, loc=mean1, scale=dev1)
Y_2 = norm.cdf(X_2, loc=mean2, scale=dev2)
Y_3 = norm.cdf(X_3, loc=mean3, scale=dev3)
Y_4 = norm.cdf(X_4, loc=mean4, scale=dev4)

pr1 = C1/Y_1**(0.5)
pr2 = C2/Y_2**(0.5)
pr3 = C3/Y_3**(0.5)
pr4 = C4/Y_4**(0.5)




plt.figure(figsize=(15, 10))
Y_5 = (Y_1*pr1 + Y_2*pr2 + Y_3*pr3 + Y_4*pr4)/(pr1+pr2+pr3+pr4)
#------------------------------------------------------------------------------


TIME = np.arange(len(Speed_cp))
buchang = math.floor((len(Speed_cp)-1)/25)+1
g_k = np.array([])

#----------------------------------------------------------

l = len(D_CP)                                                
for current in range(5):
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
            D_CP_PRE = np.append(D_CP_PRE,D_CP[i])
            D_TP_PRE = np.append(D_TP_PRE,D_TP[i])
            D_TF_PRE = np.append(D_TF_PRE,D_TF[i])
            Speed_ego_PRE = np.append(Speed_ego_PRE,Speed_ego[i])
            Speed_cp_PRE = np.append(Speed_cp_PRE,Speed_cp[i])
            Speed_tp_PRE = np.append(Speed_tp_PRE,Speed_tp[i])
            Speed_tf_PRE = np.append(Speed_tf_PRE,Speed_tf[i])
        else:
            D_CP_PRE = np.append(D_CP_PRE , D_CP_PRE[-1] + (Speed_cp[t_current] - Speed_ego[t_current]) / 25)
            D_TP_PRE = np.append(D_TP_PRE , D_TP_PRE[-1] + (Speed_tp[t_current] - Speed_ego[t_current]) / 25)
            D_TF_PRE = np.append(D_TF_PRE , D_TF_PRE[-1] + (Speed_ego[t_current] - Speed_tf[t_current]) / 25)
            Speed_ego_PRE = np.append(Speed_ego_PRE,Speed_ego[t_current])
            Speed_cp_PRE = np.append(Speed_cp_PRE,Speed_cp[t_current])
            Speed_tp_PRE = np.append(Speed_tp_PRE,Speed_tp[t_current])
            Speed_tf_PRE = np.append(Speed_tf_PRE,Speed_tf[t_current])
    
    # print(D_TF_PRE[t_current],Speed_ego_PRE[t_current],Speed_tf_PRE[t_current])

    X_1_PRE = -(Speed_ego_PRE**2 - 277) / (Speed_cp_PRE**2 - 277 + 8 * D_CP_PRE)
    X_2_PRE = -(Speed_ego_PRE**2 - 277) / (Speed_tp_PRE**2 - 277 + 8 * D_TP_PRE)
    X_3_PRE = -(Speed_ego_PRE*2) * (Speed_ego_PRE**2) / (Speed_tp_PRE * 2 + D_TP_PRE) / (Speed_tp_PRE**2)
    X_4_PRE = -(Speed_tf_PRE*2) * (Speed_tf_PRE**2) / (Speed_ego_PRE * 2 + D_TF_PRE) / (Speed_ego_PRE**2)
    
  
    D_X1 = 8*(Speed_ego_PRE**2-277)*(Speed_cp_PRE-Speed_ego_PRE)/((Speed_cp_PRE**2-277+8*D_CP_PRE)**2)
    D_X2 = 8*(Speed_ego_PRE**2-277)*(Speed_tp_PRE-Speed_ego_PRE)/((Speed_tp_PRE**2-277+8*D_TP_PRE)**2)
    D_X3 = 2*Speed_ego_PRE**3*(Speed_tp_PRE-Speed_ego_PRE)/(Speed_tp_PRE**2)/((2*Speed_tp_PRE+D_TP_PRE)**2)
    D_X4 = 2*Speed_tf_PRE**3*(Speed_ego_PRE-Speed_tf_PRE)/(Speed_ego_PRE**2)/((2*Speed_ego_PRE+D_TF_PRE)**2)
    
    
    Y_1_PRE = norm.cdf(X_1_PRE, loc=mean1, scale=dev1)
    Y_2_PRE = norm.cdf(X_2_PRE, loc=mean2, scale=dev2)
    Y_3_PRE = norm.cdf(X_3_PRE, loc=mean3, scale=dev3)
    Y_4_PRE = norm.cdf(X_4_PRE, loc=mean4, scale=dev4)
   

    D_Y1 = D_X1/(dev1*math.sqrt(2*math.pi))*np.exp(-(X_1_PRE-mean1)**2/(2*dev1**2))
    D_Y2 = D_X2/(dev2*math.sqrt(2*math.pi))*np.exp(-(X_2_PRE-mean2)**2/(2*dev2**2))
    D_Y3 = D_X3/(dev3*math.sqrt(2*math.pi))*np.exp(-(X_3_PRE-mean3)**2/(2*dev3**2))
    D_Y4 = D_X4/(dev4*math.sqrt(2*math.pi))*np.exp(-(X_4_PRE-mean4)**2/(2*dev4**2))
    
    
    D_Y = (C1+C2+C3+C4)*(C1*D_Y1/Y_1_PRE**2+C2*D_Y2/Y_2_PRE**2+C3*D_Y3/Y_3_PRE**2+C4*D_Y4/Y_4_PRE**2)/(C1/Y_1_PRE+C2/Y_2_PRE+C3/Y_3_PRE+C4/Y_4_PRE)**2
    
    pro1 = C1/Y_1_PRE**(0.5)
    pro2 = C2/Y_2_PRE**(0.5)
    pro3 = C3/Y_3_PRE**(0.5)
    pro4 = C4/Y_4_PRE**(0.5)
    
    
    Y_5_PRE = (Y_1_PRE*pro1 + Y_2_PRE*pro2 + Y_3_PRE*pro3 + Y_4_PRE*pro4)/(pro1+pro2+pro3+pro4)
    g_k = np.append(g_k,Y_5_PRE[t_current])
    plt.plot(TIME/25, Y_5_PRE, linewidth=3) 
   
    t1 = -((Speed_ego[t_current]**2-277)+(mean1-3*dev1)*(Speed_cp[t_current]**2-277+8*D_CP[t_current]))/(8*(Speed_cp[t_current]-Speed_ego[t_current])*(mean1-3*dev1))
    t2 = -((Speed_ego[t_current]**2-277)+(mean2-3*dev2)*(Speed_tp[t_current]**2-277+8*D_TP[t_current]))/(8*(Speed_tp[t_current]-Speed_ego[t_current])*(mean2-3*dev2))
    t3 = -(2*Speed_ego[t_current]**3+(2*Speed_tp[t_current]**3+D_TP[t_current]*Speed_tp[t_current]**2)*(mean3-3*dev3))/(Speed_tp[t_current]**2*(Speed_tp[t_current]-Speed_ego[t_current])*(mean3-3*dev3))
    t4 = -(2*Speed_tf[t_current]**3+(2*Speed_ego[t_current]**3+D_TF[t_current]*Speed_ego[t_current]**2)*(mean4-3*dev4))/(Speed_ego[t_current]**2*(Speed_ego[t_current]-Speed_tf[t_current])*(mean4-3*dev4))
    
    t5_2 = -(Speed_tp[t_current]**2-277+8*D_TP[t_current])/(8*(Speed_tp[t_current]-Speed_ego[t_current]))
    t5_3 = -(2*Speed_tp[t_current]+D_TP[t_current])/(Speed_tp[t_current]-Speed_ego[t_current])
    t5_4 = -(2*Speed_ego[t_current]+D_TF[t_current])/(Speed_ego[t_current]-Speed_tf[t_current])

    
    
    if Speed_ego[t_current] >= Speed_cp[t_current]:
        t1_a = 0
        t1_b = max(t1,0)
    else:
        t1_a = max(t1,0)
        t1_b = 10000
        
    if Speed_ego[t_current] >= Speed_tp[t_current]:
        t2_a = 0
        
        t2_b = max(min(t2,t5_2),0)
        t3_a = 0
        t3_b = max(min(t3,t5_3),0)
    else:
        t2_a = max(t2,0,t5_2)
        t2_b = 10000   
        t3_a = max(t2,0,t5_3)
        t3_b = 10000   
    
    if Speed_tf[t_current] >= Speed_ego[t_current]:
        t4_a = 0
        t4_b = max(min(t4,t5_4),0)
    else:
        t4_a = max(t4,0,t5_4)
        t4_b = 10000
        
        T_a = max(t1_a,t2_a,t3_a,t4_a)
        T_b = min(t1_b,t2_b,t3_b,t4_b)    
    T_a = max(t1_a,t2_a,t3_a,t4_a,0)
    T_b = min(t1_b,t2_b,t3_b,t4_b)
    
    plt.axvline(T_a+current,  color='b', linestyle='--')
    plt.axvline(T_b+current,  color='r', linestyle='--')    
    
    if T_a >= T_b and current < buchang - 1:
        continue 
    elif T_a >= T_b and current == buchang - 1:
        T_m = 100
        G_m = 100
        break   
    
    elif T_a > (l-current*25)/25 and T_a < T_b and current == buchang - 1:
        T_m = 100
        G_m = 1000
        break  
    
    elif T_a < T_b and current <= buchang - 1:
        if math.ceil(T_a*25) < math.floor(T_b*25) and math.ceil(T_a*25)+t_current < l: 
            max_index = np.argmax(Y_5_PRE[math.ceil(T_a*25)+t_current:min(t_current+math.floor(T_b*25),l)]) 
            max_value = Y_5_PRE[math.ceil(T_a*25)+t_current:min(t_current+math.floor(T_b*25),l)][max_index] 
        elif math.ceil(T_a*25) == math.floor(T_b*25) and math.ceil(T_a*25)+t_current < l: 
            max_index = t_current  
            max_value = Y_5_PRE[t_current]      
        else:
            continue
        

        if max_index <= (25-T_a*25):
            if current > 0:
                if g_k[current] > g_k[current-1]:
                    T_m = max_index/25 + T_a + current
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


     
print(T_m,G_m)

plt.plot(TIME/25, Y_5, color='k')
plt.axvline((frame_start-first_frame)/25,  color='k', linestyle='--')
plt.axvline(T_m,  color='k', linewidth=3)
plt.xlim(-0.5,10.5)


plt.xlabel('Time')
plt.ylabel(r'$\mathit{U}(t)$')
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.rcParams.update({'font.size': 24})


