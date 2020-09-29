# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 15:58:32 2018

@author: monakhov
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy import stats
from matplotlib.collections import PatchCollection

def SRM_general(data,axis,clip_name,ring_radius,draw_plot=False,users=None,ring_width=10):
    #ring_width = 50
    if users == None:
        users = list(data.keys())
    users_scores = {}
    timestamps = np.arange(0,160000,len(data[users[0]][clip_name][axis]))
    for user in users:
        users_scores[user] = 0
    clip_length = len(data[users[0]][clip_name][axis])
    num_of_tested_users = len(users)
    result_arr = []
    mode_arr = []
    for i in range(0,clip_length-ring_width,ring_width):
        tmp_arr = []
        for user in users:
            tmp_arr.append(data[user][clip_name][axis][i:i+ring_width+1])
        tmp_arr = np.array(tmp_arr)
        tmp_arr = tmp_arr.flatten()
        #ring_mode = stats.mode(tmp_arr,axis=None).mode[0]
        ring_mode = np.median(tmp_arr)
        mode_arr.append(ring_mode)
        for j in range(i,i+ring_width):
            check_if_in_ring_arr = []
            for user in users:
                if (data[user][clip_name][axis][j]>=ring_mode-ring_radius and
                    data[user][clip_name][axis][j]<=ring_mode+ring_radius):
                    check_if_in_ring_arr.append(0)
                    #users_scores[user] += 1
                elif (ring_mode+ring_radius > 180):
                    diff = ring_mode+ring_radius - 180
                    if (data[user][clip_name][axis][j]<=-180+diff and 
                       data[user][clip_name][axis][j]>=-180):
                        check_if_in_ring_arr.append(0)
                        #users_scores[user] += 1
                    else:
                        users_scores[user] += 1
                        check_if_in_ring_arr.append(1)
                elif (ring_mode-ring_radius < -180):
                    diff = 180 - (ring_mode-ring_radius)
                    if (data[user][clip_name][axis][j]<=180 and 
                        data[user][clip_name][axis][j]>= 180-diff):
                        check_if_in_ring_arr.append(0)
                        #users_scores[user] += 1
                    else:
                        users_scores[user] += 1
                        check_if_in_ring_arr.append(1)                
                else:
                    users_scores[user] += 1
                    check_if_in_ring_arr.append(1)
                result_arr.append(np.sum(check_if_in_ring_arr)/num_of_tested_users)
    res = (1-np.sum(result_arr)/len(result_arr))*100
    if (draw_plot == True):
        ringboxes = []
        for i in range(len(mode_arr)):
            rect =patches.Rectangle((ring_width*i,mode_arr[i]-ring_radius),ring_width,2*ring_radius,hatch='/',fill=False)
            ringboxes.append(rect)
        pc = PatchCollection(ringboxes,match_original=True,hatch='//')
        fig, ax = plt.subplots(1)
        for user in users:
            ax.plot(data[user][clip_name][axis])
        #ax.set_xticklabels([0,0,10000,20000,30000,400000,50000,60000,70000])
        ax.add_collection(pc)
#        plt.title(clip_name)
        plt.ylim( (-180, 180) )
        plt.show()
        plt.title('Similarity Ring Metric')
        plt.ylabel('yaw values, degrees')
        plt.xlabel('timestamp, ms')
        #plt.xticks(np.arange(0,160000,len(data[user][clip_name][axis])))
        #plt.savefig('byClip/'+clip_name[:-3]+'1.png')
        #plt.xticks(plt.xticks()[0], np.linspace(0,160000,9))
    return res,mode_arr,users_scores

def SRM_pair(clip_ref,clip_test,ring_radius,draw_plot=False,title=None):
    clip_length = len(clip_ref)
    res_arr = []
    for i in range(clip_length):
        if (clip_test[i]>=clip_ref[i]-ring_radius and 
           clip_test[i]<=clip_ref[i]+ring_radius):
            res_arr.append(1)
        elif (clip_ref[i]+ring_radius > 180):
            diff = clip_ref[i]+ring_radius - 180
            if (clip_test[i]<=-180+diff and 
                clip_test[i]>=-180):
                res_arr.append(1)
            else:
                res_arr.append(0)
        elif (clip_ref[i]-ring_radius < -180):
            diff = 180 - (clip_ref[i]-ring_radius)
            if (clip_test[i]<=180 and 
                clip_test[i]>= 180-diff):
                res_arr.append(1)
            else:
                res_arr.append(0)
        else:
            res_arr.append(0)
    res = np.sum(res_arr)/len(res_arr)*100
    if (draw_plot == True):
        fig, ax = plt.subplots(1)
        ax.plot(clip_ref)
        ax.plot(clip_test)
        ringboxes = []
        for i in range(clip_length):
            rect =patches.Rectangle((i,clip_ref[i]-ring_radius),2,2*ring_radius)
            ringboxes.append(rect)
        pc = PatchCollection(ringboxes,match_original=True,alpha=0.1,edgecolor="none")
        ax.add_collection(pc)
        plt.ylim( (-180, 180) )
        if title != None:
            plt.title(title)
    return res

def SRM_merge(datasets,axis,clip_names,ring_radius,draw_plot=False):
    ring_width = 50
    clip_length = len(datasets[0][list(datasets[0].keys())[0]][clip_names[0]][axis])
    num_of_tested_users = len(datasets[0].keys())
    full_data_arr = []
    for user in datasets[0].keys():
        for i in range(len(datasets)):
            full_data_arr.append(datasets[i][user][clip_names[i]][axis])
    result_arr = []
    mode_arr = []
    for i in range(0,clip_length-ring_width,ring_width):
        tmp_arr = []
        for traj_num in range(len(full_data_arr)):
            tmp_arr.append(full_data_arr[traj_num][i:i+ring_width+1])
        tmp_arr = np.array(tmp_arr)
        tmp_arr = tmp_arr.flatten()
        #ring_mode = stats.mode(tmp_arr,axis=None).mode[0]
        ring_mode = np.median(tmp_arr)
        mode_arr.append(ring_mode)
        for j in range(i,i+ring_width):
            check_if_in_ring_arr = []
            for traj_num in range(len(full_data_arr)):
                if (full_data_arr[traj_num][j]>=ring_mode-ring_radius and
                    full_data_arr[traj_num][j]<=ring_mode+ring_radius):
                    check_if_in_ring_arr.append(0)
                else:
                    check_if_in_ring_arr.append(1)
                result_arr.append(np.sum(check_if_in_ring_arr)/num_of_tested_users)
    res = (1-np.sum(result_arr)/len(result_arr))*100
    if (draw_plot == True):
        ringboxes = []
        for i in range(len(mode_arr)):
            rect =patches.Rectangle((ring_width*i,mode_arr[i]-ring_radius),ring_width,2*ring_radius,hatch='/',fill=False)
            ringboxes.append(rect)
        pc = PatchCollection(ringboxes,match_original=True,hatch='//')
        fig, ax = plt.subplots(1)
        for traj_num in range(len(full_data_arr)):
            ax.plot(full_data_arr[traj_num])
        ax.add_collection(pc)
        plt.title(clip_names[0][:-3])
        plt.show()
    return res,mode_arr