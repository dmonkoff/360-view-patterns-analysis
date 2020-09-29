# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 16:30:00 2018

@author: monakhov
"""
import numpy as np
import matplotlib.pyplot as plt
from savitzky_golay import savitzky_golay
arr_test_subjects = ['Lasse', 'Ramin', 'Jani', 'Antti', 'xingyang ni','Nahid', 'Vida', 'Ling yu', 
                     'Jari', 'Pouria','Sujeet','emre', 'Adriana', 'kashyap', 'You yu', 'Peter',  
                     'Kimmo', 'Yuri', 'Kalle', 'Toni']


# =============================================================================
# clip='armor'
# cuts = [8, 26.026, 38.672, 50.684]
# clip_length_sec = 66
# =============================================================================




# =============================================================================
# clip='martial'
# cuts = [12.56, 21.76, 44.76, 54.76]
# clip_length_sec = 62
# arr_test_subjects = ['Lasse', 'Ramin', 'Jani', 'Antti', 'xingyang ni','Nahid', 'Vida', 'Ling yu', 
#                     'Jari', 'Sujeet', 'Adriana', 'kashyap', 'You yu', 'Peter',  
#                     'Kimmo', 'Yuri', 'Kalle', 'Toni']
# =============================================================================



clip='lion'
cuts = [11.011, 32.866, 55.122, 67.034]
clip_length_sec = 76


cuts_small = np.array([int(x*1000/50) for x in cuts])
res = {}
#%%
seg_length = 250
w = seg_length/1000
folder_name = 'yaw_diff_max/'+str(seg_length)+'/'
######################################################################
seconds_arr = np.arange(0,clip_length_sec,w)*1000
#####################################################
seconds_cuts = []
for cut in cuts:
        seconds_cuts.append(np.argmin(np.abs(seconds_arr-cut*1000)))#array of indeces for cuts  
        #print(second)
tmp_indexes = []
jjj=0
for (ind,ttt) in enumerate(data['Vida'][clip]['time']):
    if ttt/1000>seconds_cuts[jjj]:
        tmp_indexes.append(ind)
        jjj += 1
    if jjj >= len(seconds_cuts):
        break
#%%
#seconds_cuts = [19,39,59,79,99,119,139]
res = np.zeros((len(arr_test_subjects),len(seconds_arr)))
for cur_iter,user in enumerate(arr_test_subjects):
    tmp_clip = []
    tmp_arr = []
    current_sec_barrier = 0
    i = 0
    #current_trajectory = savitzky_golay(np.array(data[user][clip]['yaw']), window_size=11, order=3,deriv=0)
    current_trajectory = np.array(data[user][clip]['yaw'])
    while (i < len(data[user][clip]['time']) and current_sec_barrier < len(seconds_arr) ):
        if (data[user][clip]['time'][i] >= seconds_arr[current_sec_barrier] and tmp_clip != []):
#           abs_diff = np.abs([s-tmp_clip[0] for s in tmp_clip])
#           shortest_abs_diff = np.minimum(abs_diff,360-abs_diff)
#           shortest_abs_diff = abs_diff
            shortest_abs_diff = np.max(tmp_clip) - np.min(tmp_clip)
            #shortest_abs_diff = tmp_clip[-1] - tmp_clip[0]
            tmp_arr.append(np.max(shortest_abs_diff))
            current_sec_barrier += 1
            tmp_clip = []
#           if seconds_arr[current_sec_barrier] >= 159*1000:
#               break
        tmp_clip.append(current_trajectory[i])
        i += 1
    #if (len(tmp_clip) != 0):
     #   abs_diff = np.abs([s-tmp_clip[0] for s in tmp_clip])
      #  shortest_abs_diff = np.minimum(abs_diff,360-abs_diff)
       # tmp_arr.append(np.max(shortest_abs_diff))
       # tmp_clip = []
    #tmp_arr = tmp_arr/np.max(tmp_arr)
    res[cur_iter,:] = tmp_arr
#%%
#folder_name='range/martial/'
#plt.ioff()
for cur_iter,user in enumerate(arr_test_subjects):
    plt.figure()
    plt.bar(seconds_arr/1000,res[cur_iter,:],width=(-1)*w,align='edge')
    plt.bar(seconds_arr[seconds_cuts]/1000,res[cur_iter,seconds_cuts],width=(-1)*w,align='edge')
    plt.title('exploration range: '+user)
    plt.xlabel('time, sec')
    plt.ylabel('exploration range, degrees')
    #plt.ylim( (0, 100) )
#    plt.savefig(folder_name+user+'.png',dpi=200)
#    plt.close()
#plt.ion()
#%%
#plt.ioff()
plt.figure()
meanRange = np.mean(res,axis=0)
plt.bar(seconds_arr/1000,meanRange,width=(-1)*w,align='edge')
plt.bar(seconds_arr[seconds_cuts]/1000,meanRange[seconds_cuts],width=(-1)*w,align='edge', label='scene cut')
#plt.title('Average exploration range over users:')
plt.xlabel('Time (s)', fontsize=16)
plt.ylabel('Exploration range (degrees)', fontsize=16)
plt.legend()
plt.ylim( (0, 20) )
#plt.savefig(folder_name+'1average.png',dpi=200)
#plt.close()
#plt.ion()
#plt.savefig('SRM_Lions.jpg',dpi=600)
#%%
####################################
#Velocity values
##########################################


#clip='helsinki_na'
#cuts = []
#clip_length_sec = data['Vida'][clip]['time'][-1]/1000


seg_length = 250
w = seg_length/1000
folder_name = 'velocity/'+str(seg_length)+'/'
######################################################################
seconds_arr = np.arange(0,clip_length_sec,w)*1000
#####################################################
seconds_cuts = []
for cut in cuts:
        seconds_cuts.append(np.argmin(np.abs(seconds_arr-cut*1000)))      
        #print(second)
#%%
res2 = np.zeros((len(arr_test_subjects),len(seconds_arr)))
tmp_velocities = []
for cur_iter,user in enumerate(arr_test_subjects):
    tmp_clip = []
    tmp_arr = []
    current_sec_barrier = 0
    i = 0
    current_trajectory = savitzky_golay(np.array(data[user][clip]['yaw']), window_size=11, order=3,deriv=0)
    velocity = np.gradient(current_trajectory, data[user][clip]['time'])*1000
    tmp_velocities.append(velocity)
    while (i < len(data[user][clip]['time']) and current_sec_barrier < len(seconds_arr) ):
        if (data[user][clip]['time'][i] >= seconds_arr[current_sec_barrier] and tmp_clip != []):
           tmp_arr.append(np.mean(np.abs(tmp_clip)))
           current_sec_barrier += 1
           tmp_clip = []
#           if current_sec_barrier >= 159:
#               break
        tmp_clip.append(velocity[i])
        if current_sec_barrier == 0:
            tmp_clip[-1] = 0
        i += 1
    #if (len(tmp_clip) != 0):
     #   abs_diff = np.abs([s-tmp_clip[0] for s in tmp_clip])
      #  shortest_abs_diff = np.minimum(abs_diff,360-abs_diff)
       # tmp_arr.append(np.max(shortest_abs_diff))
       # tmp_clip = []
    res2[cur_iter,:] = tmp_arr 
#%%
folder_name='velocity/martial/'
plt.ioff()
for cur_iter,user in enumerate(arr_test_subjects):
    plt.figure()
    plt.bar(seconds_arr/1000,res2[cur_iter,:],width=-w,align='edge')
    plt.bar(seconds_arr[seconds_cuts]/1000,res2[cur_iter,seconds_cuts],width=-w,align='edge')
    plt.title('Angular speed: '+user)
    plt.xlabel('time, sec')
    plt.ylabel('angular speed, degrees/sec')
    plt.ylim( (0, 250) )
    plt.savefig(folder_name+user+'.png',dpi=200)
    plt.close()
plt.ion()
#%%
#plt.ioff()
meanRange = np.mean(res2,axis=0)
plt.bar(seconds_arr/1000,meanRange,width=-w,align='edge')
plt.bar(seconds_arr[seconds_cuts]/1000,meanRange[seconds_cuts],width=-w,align='edge', label='scene cut')
#plt.title('Angular speed average: Lions')
#plt.xlabel('time, sec')
#plt.ylabel('angular speed, degrees/sec')
plt.ylim( (0, 70) )
plt.xlabel('Time (s)', fontsize=16)
plt.legend()
plt.ylabel('Angular speed (degrees per sec)', fontsize=16)
#plt.savefig(folder_name+'1average'+'.png',dpi=200)
#plt.close()
#plt.ion()

#%%
tmp_velocities = np.abs(np.array(tmp_velocities))
plt.plot(np.arange(tmp_velocities.shape[1])*50/1000,np.mean(tmp_velocities,axis=0))
plt.vlines(cuts_small*50/1000,ymin=0,ymax=60,color='orange', label='scene cut')
plt.xlabel('time, sec',fontsize=14)
plt.ylabel('angular speed, degrees/sec',fontsize=14)
plt.legend(fontsize=14)
####################################
#Acceleration values
##########################################
#%%
seg_length = 250
w = seg_length/1000
folder_name = 'acceleration/'+str(seg_length)+'_mean/'
######################################################################
seconds_arr = np.arange(0,clip_length_sec,w)*1000
#####################################################
seconds_cuts = []
for cut in cuts:
        seconds_cuts.append(np.argmin(np.abs(seconds_arr-cut*1000)))      
        #print(second)
#%%
res3 = np.zeros((len(arr_test_subjects),len(seconds_arr)))
for cur_iter,user in enumerate(arr_test_subjects):
    tmp_clip = []
    tmp_arr = []
    current_sec_barrier = 0
    i = 0
    current_trajectory = savitzky_golay(np.array(data[user][clip]['yaw']), window_size=11, order=3,deriv=0)
    velocity = np.gradient(current_trajectory, data[user][clip]['time'])*1000
    acceleration = np.gradient(velocity, data[user][clip]['time'])*1000
    while (i < len(data[user][clip]['time']) and current_sec_barrier < len(seconds_arr) ):
        if (data[user][clip]['time'][i] >= seconds_arr[current_sec_barrier] and tmp_clip != []):
            tmp_arr.append(np.median(np.abs(tmp_clip)))
#            upper_quartile = np.percentile(np.abs(tmp_clip), 95,interpolation='nearest')
#            lower_quartile = np.percentile(np.abs(tmp_clip), 5,interpolation='nearest')
#            tmp_arr.append(upper_quartile)
#            tmp_arr.append(lower_quartile)
            current_sec_barrier += 1
            tmp_clip = []
#           if current_sec_barrier >= 159:
#               break
        tmp_clip.append(acceleration[i])
        if current_sec_barrier == 0:
            tmp_clip[-1] = 0
        i += 1
    res3[cur_iter,:] = tmp_arr#/np.max(tmp_arr)
#%%
folder_name='acceleration/martial/'
plt.ioff()   
for cur_iter,user in enumerate(arr_test_subjects):
    plt.bar(seconds_arr/1000,res3[cur_iter,:],width=-w,align='edge')
    plt.bar(seconds_arr[seconds_cuts]/1000,res3[cur_iter,seconds_cuts],width=-w,align='edge')
    plt.title('Angular acceleration: '+user)
    plt.xlabel('time, sec')
    plt.ylabel('angular acceleration, degrees/sec^2')
    plt.ylim( (0, np.max(res3) ))
    plt.savefig(folder_name+user+'.png',dpi=200)
    plt.close()
plt.ion()
#%%
#plt.ioff()   
meanRange = np.mean(res3,axis=0)
plt.bar(seconds_arr/1000,meanRange,width=-w,align='edge')
plt.bar(seconds_arr[seconds_cuts]/1000,meanRange[seconds_cuts],width=-w,align='edge', label='scene cut')
#plt.title('Angular acceleration mean: Martial training')
#plt.xlabel('time, sec')
#plt.ylabel('angular acceleration, degrees/sec^2')
plt.ylim( (0, 280) )
plt.xlabel('Time (s)', fontsize=16)
plt.ylabel('Angular acceleration (degrees per $sec^2$)', fontsize=16)
plt.legend()
#plt.savefig(folder_name+'1average'+'.png',dpi=200)
#plt.close()
#plt.ion()   
###############################################
#computing number of head turns
#%%
seg_length = 2000
folder_name = 'num_of_turns/1000/'
######################################################################
seconds_arr = np.arange(2,160,2)*1000
#####################################################
seconds_cuts = []
for cur_iter,second in enumerate(seconds_arr):
    if second%20000 == 0:
        seconds_cuts.append(cur_iter)       
#%%
#seconds_cuts = [19,39,59,79,99,119,139]
res4 = np.zeros((len(arr_test_subjects),len(seconds_arr)))
for cur_iter,user in enumerate(arr_test_subjects):
    tmp_clip = []
    tmp_arr = []
    current_sec_barrier = 0
    i = 0
    turns_buffer = [0,0,0,0,0,0,0,0,0,0]
    prev_value = 0
    num_of_turns = 0
    while (i < len(data[user][clip]['time']) and current_sec_barrier < len(seconds_arr) ):
        if (data[user][clip]['time'][i] >= seconds_arr[current_sec_barrier]):
            tmp_arr.append(num_of_turns)
            current_sec_barrier += 1
            num_of_turns = 0
#           if seconds_arr[current_sec_barrier] >= 159*1000:
#               break
        if data[user][clip]['yaw'][i] > prev_value:
            turns_buffer.pop(0)
            turns_buffer.append(1)
        else:
            turns_buffer.pop(0)
            turns_buffer.append(-1)
        prev_value = data[user][clip]['yaw'][i]
        i += 1
        if (sum(turns_buffer[:5])== 5 and sum(turns_buffer[5:]) == -5) or (sum(turns_buffer[:5])== -5 and sum(turns_buffer[5:]) == 5):
            num_of_turns+=1
    res4[cur_iter,:] = tmp_arr
#%%
for cur_iter,user in enumerate(arr_test_subjects):
    plt.figure()
    plt.bar(seconds_arr/1000,res4[cur_iter,:],width=-1,align='edge')
    plt.bar(seconds_arr[seconds_cuts]/1000,res4[cur_iter,seconds_cuts],width=-1,align='edge')
    plt.title('number of turns: '+user)
#    plt.ylim( (0, 600) )
#%%
meanRange = np.median(res4,axis=0)
plt.bar(seconds_arr/1000,meanRange,width=-1,align='edge')
plt.bar(seconds_arr[seconds_cuts]/1000,meanRange[seconds_cuts],width=-1,align='edge')
plt.title('mean number of turns')
#%%

#%%
#############################
##############################
#correlation
##############################
###############################
seg_length = 500
w = seg_length/1000
folder_name = 'correlation/'+str(seg_length)+'/'
######################################################################
seconds_arr = np.arange(0,160,w)*1000
#####################################################
seconds_cuts = []
for cur_iter,second in enumerate(seconds_arr):
    if second%20000 == 0:
        seconds_cuts.append(cur_iter)       
#%%
#seconds_cuts = [19,39,59,79,99,119,139]
res = np.zeros((len(arr_test_subjects),len(seconds_arr)))
for cur_iter,user in enumerate(arr_test_subjects):
    tmp_clip = []
    tmp_arr = []
    current_sec_barrier = 0
    i = 0
    current_trajectory = savitzky_golay(np.array(data[user][clip]['yaw']), window_size=11, order=3,deriv=0)
    while (i < len(data[user][clip]['time']) and current_sec_barrier < len(seconds_arr) ):
        if (data[user][clip]['time'][i] >= seconds_arr[current_sec_barrier] and tmp_clip != []):
#           abs_diff = np.abs([s-tmp_clip[0] for s in tmp_clip])
#           shortest_abs_diff = np.minimum(abs_diff,360-abs_diff)
#           shortest_abs_diff = abs_diff
            shortest_abs_diff = np.max(tmp_clip) - np.min(tmp_clip)
            tmp_arr.append(np.max(shortest_abs_diff))
            current_sec_barrier += 1
            tmp_clip = []
#           if seconds_arr[current_sec_barrier] >= 159*1000:
#               break
        tmp_clip.append(current_trajectory[i])
        i += 1
    #if (len(tmp_clip) != 0):
     #   abs_diff = np.abs([s-tmp_clip[0] for s in tmp_clip])
      #  shortest_abs_diff = np.minimum(abs_diff,360-abs_diff)
       # tmp_arr.append(np.max(shortest_abs_diff))
       # tmp_clip = []
    res[cur_iter,:] = tmp_arr
#%%
plt.ioff()
for cur_iter,user in enumerate(arr_test_subjects):
    plt.figure()
    tmp = res[cur_iter,:]
    tmp = tmp - np.mean(tmp)
    norm = np.sum(tmp ** 2)
    cur_corr = (np.correlate(tmp, tmp, mode='full')/norm)[len(tmp)-1:]
    plt.bar(seconds_arr/1000,cur_corr,width=(-1)*w,align='edge')
    plt.bar(seconds_arr[seconds_cuts]/1000,cur_corr[seconds_cuts],width=(-1)*w,align='edge')
    plt.title('correlation: '+user)
    plt.xlabel('shift, sec')
    plt.ylabel('correlation value')
    plt.savefig(folder_name+user+'.png',dpi=200)
    plt.close()
plt.ion()
#%%
plt.ioff()
#plt.figure()
meanRange = np.mean(res,axis=0)
tmp = meanRange - np.mean(meanRange)
norm = np.sum(tmp ** 2)
cur_corr = (np.correlate(tmp, tmp, mode='full')/norm)[len(tmp)-1:]
plt.bar(seconds_arr/1000,cur_corr,width=(-1)*w,align='edge')
plt.bar(seconds_arr[seconds_cuts]/1000,cur_corr[seconds_cuts],width=(-1)*w,align='edge')
plt.title('average correlation')
plt.xlabel('shift, sec')
plt.ylabel('correlation value')
plt.savefig(folder_name+'1average.png',dpi=200)
plt.close()
plt.ion()
#%%