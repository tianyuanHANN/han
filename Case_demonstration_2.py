# -*- coding: utf-8 -*-
"""
@author: HAN
"""


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
Y_5 = C1*Y_1+C2*Y_2+C3*Y_3+C4*Y_4
    
pr1 = C1/Y_1
pr2 = C2/Y_2
pr3 = C3/Y_3
pr4 = C4/Y_4

Y_1

buchang = math.floor((len(Speed_cp)-1)/25)+1
g_k = np.array([])

#----------------------------------------------------------
l = len(D_CP)                                                
for current in range(4):
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
    
  

    X_1_PRE = -(Speed_ego_PRE**2 - 277) / (Speed_cp_PRE**2 - 277 + 8 * D_CP_PRE)
    X_2_PRE = -(Speed_ego_PRE**2 - 277) / (Speed_tp_PRE**2 - 277 + 8 * D_TP_PRE)
    X_3_PRE = -(Speed_ego_PRE*2) * (Speed_ego_PRE**2) / (Speed_tp_PRE * 2 + D_TP_PRE) / (Speed_tp_PRE**2)
    X_4_PRE = -(Speed_tf_PRE*2) * (Speed_tf_PRE**2) / (Speed_ego_PRE * 2 + D_TF_PRE) / (Speed_ego_PRE**2)
    X_5_PRE = C1*X_1_PRE+C2*X_2_PRE+C3*X_3_PRE+C4*X_4_PRE
    
    
    D_X1 = 8*(Speed_ego_PRE**2-277)*(Speed_cp_PRE-Speed_ego_PRE)/((Speed_cp_PRE**2-277+8*D_CP_PRE)**2)
    D_X2 = 8*(Speed_ego_PRE**2-277)*(Speed_tp_PRE-Speed_ego_PRE)/((Speed_tp_PRE**2-277+8*D_TP_PRE)**2)
    D_X3 = 2*Speed_ego_PRE**3*(Speed_tp_PRE-Speed_ego_PRE)/(Speed_tp_PRE**2)/((2*Speed_tp_PRE+D_TP_PRE)**2)
    D_X4 = 2*Speed_tf_PRE**3*(Speed_ego_PRE-Speed_tf_PRE)/(Speed_ego_PRE**2)/((2*Speed_ego_PRE+D_TF_PRE)**2)
    
    
    Y_1_PRE = norm.cdf(X_1_PRE, loc=mean1, scale=dev1)
    Y_2_PRE = norm.cdf(X_2_PRE, loc=mean2, scale=dev2)
    Y_3_PRE = norm.cdf(X_3_PRE, loc=mean3, scale=dev3)
    Y_4_PRE = norm.cdf(X_4_PRE, loc=mean4, scale=dev4)
    Y_5_PRE = C1* Y_1_PRE + C2*Y_2_PRE + C3*Y_3_PRE + C4*Y_4_PRE

    D_Y1 = D_X1/(dev1*math.sqrt(2*math.pi))*np.exp(-(X_1_PRE-mean1)**2/(2*dev1**2))
    D_Y2 = D_X2/(dev2*math.sqrt(2*math.pi))*np.exp(-(X_2_PRE-mean2)**2/(2*dev2**2))
    D_Y3 = D_X3/(dev3*math.sqrt(2*math.pi))*np.exp(-(X_3_PRE-mean3)**2/(2*dev3**2))
    D_Y4 = D_X4/(dev4*math.sqrt(2*math.pi))*np.exp(-(X_4_PRE-mean4)**2/(2*dev4**2))
    
    
    D_Y = (C1+C2+C3+C4)*(C1*D_Y1/Y_1_PRE**2+C2*D_Y2/Y_2_PRE**2+C3*D_Y3/Y_3_PRE**2+C4*D_Y4/Y_4_PRE**2)/(C1/Y_1_PRE+C2/Y_2_PRE+C3/Y_3_PRE+C4/Y_4_PRE)**2
    
    pro1 = C1/Y_1_PRE
    pro2 = C2/Y_2_PRE
    pro3 = C3/Y_3_PRE
    pro4 = C4/Y_4_PRE
    
    Y_PRE_1 = C1/(pro1+pro2+pro3+pro4)
    Y_PRE_2 = C2/(pro1+pro2+pro3+pro4)
    Y_PRE_3 = C3/(pro1+pro2+pro3+pro4)
    Y_PRE_4 = C4/(pro1+pro2+pro3+pro4)
    Y_PRE_5 = Y_PRE_1 +Y_PRE_2+Y_PRE_3+Y_PRE_4


def find_max_index(arr):
    max_index = np.argmax(arr)
    return max_index


plt.figure(figsize=(10, 8))
TIME = np.arange(len(Speed_cp))
plt.plot(TIME/25, Y_1_PRE, label= r'$v(x_1)$', linewidth=3)
plt.plot(TIME/25, Y_2_PRE, label= r'$v(x_2)$', linewidth=3)
plt.plot(TIME/25, Y_3_PRE, label= r'$v(x_3)$', linewidth=3)
plt.plot(TIME/25, Y_4_PRE, label= r'$v(x_4)$', linewidth=3)


plt.axvline(3,  color='b', linewidth=1.25, linestyle='--')
# plt.axvline(find_max_index(X_5_PRE)/25,  label='$t_p$ of RR',  color='k', linewidth=1.25, linestyle='-.')
plt.axvline(find_max_index(Y_5_PRE)/25,  color='k', linewidth=1.5, linestyle='--')
plt.axvline(find_max_index(Y_PRE_5)/25,  color='k', linewidth=1.5)
# plt.axvline(find_max_index(Y_PRE_5)/25,  label='$t_p$ of CUM',  color='k', linewidth=1.25, linestyle='--' )
plt.legend(frameon=False)
plt.xlabel('Time')
plt.ylabel('Benefit')

plt.xlim(-0.5,8.5)
plt.show()


plt.figure(figsize=(10, 8))
TIME = np.arange(len(Speed_cp))
plt.plot(TIME/25, Y_5_PRE, color = 'k',  linewidth=3)
plt.axvline(3,  color='b', linewidth=1.25, linestyle='--')
# plt.axvline(find_max_index(X_5_PRE)/25,  label='$t_p$ of RR',  color='k', linewidth=1.25, linestyle='-.')
plt.axvline(find_max_index(Y_5_PRE)/25,  color='k', linewidth=1.5, linestyle='--')
plt.axvline(find_max_index(Y_PRE_5)/25,  color='k', linewidth=1.5)
# plt.axvline(find_max_index(Y_PRE_5)/25,  label='$t_p$ of CUM',  color='k', linewidth=1.25, linestyle='--' )
plt.legend(frameon=False)
plt.xlabel('Time')
plt.ylabel('Benefit')

plt.xlim(-0.5,8.5)
plt.show()



plt.figure(figsize=(10, 8))
plt.plot(TIME/25, Y_PRE_1, label= r'$u(x_1)$', linewidth=3)
plt.plot(TIME/25, Y_PRE_2, label= r'$u(x_2)$', linewidth=3)
plt.plot(TIME/25, Y_PRE_3, label= r'$u(x_3)$', linewidth=3)
plt.plot(TIME/25, Y_PRE_4, label= r'$u(x_4)$', linewidth=3)
plt.axvline(3,  color='b', linewidth=1.25, linestyle='--')
# plt.axvline(find_max_index(Y_5_PRE)/25,  label='$t_p$ of RRM',  color='k', linewidth=1.5, linestyle=':')
plt.axvline(find_max_index(Y_5_PRE)/25,  color='k', linewidth=1.5, linestyle='--')
plt.axvline(find_max_index(Y_PRE_5)/25,  color='k', linewidth=1.5)
plt.legend(frameon=False)
plt.xlabel('Time')
plt.ylabel('Utility')
plt.xlim(-0.5,8.5)
plt.show()


plt.figure(figsize=(10, 8))

plt.plot(TIME/25, Y_PRE_5, color = 'k',  linewidth=3)
plt.axvline(3,  color='b', linewidth=1.25, linestyle='--')

# plt.axvline(find_max_index(Y_5_PRE)/25,  label='$t_p$ of RRM',  color='k', linewidth=1.5, linestyle=':')
plt.axvline(find_max_index(Y_5_PRE)/25,  color='k', linewidth=1.5, linestyle='--')
plt.axvline(find_max_index(Y_PRE_5)/25,  color='k', linewidth=1.5)
plt.legend(frameon=False)
plt.xlabel('Time')
plt.ylabel('Utility')
plt.xlim(-0.5,8.5)
plt.show()




# print(find_max_index(X_5_PRE)/25)
print(find_max_index(Y_5_PRE)/25)
print(find_max_index(Y_PRE_5)/25)
# print(X_5_PRE[find_max_index(X_5_PRE)])

print(Y_1_PRE[find_max_index(Y_PRE_5)])
print(Y_2_PRE[find_max_index(Y_PRE_5)])
print(Y_3_PRE[find_max_index(Y_PRE_5)])
print(Y_4_PRE[find_max_index(Y_PRE_5)])