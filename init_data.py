# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import xlrd
import numpy as np
import matplotlib.pyplot as plt
#dict_test_subjects = {'Vida':'B','Mikko':'H','Ramin':'N','Aleksei':'T','Pekka':'Z',
#                      'Peetu':'AF','Henrik':'AL','Kalle':'AR','Marja':'AX','James':'BD',
#                      'Payman':'BJ','Alizera':'BP','Sujeet':'BV','Tony':'CB','Nahid':'CH',
#                      'Ali':'CN'}
arr_test_subjects = ['Lasse', 'Ramin', 'Jani', 'Antti', 'xingyang ni','Nahid', 'Vida', 'Ling yu', 
                     'Jari', 'Pouria','Sujeet','emre', 'Adriana', 'kashyap', 'You yu', 'Peter',  
                     'Kimmo', 'Yuri', 'Kalle', 'Toni']
unroll_flag = True
dict_clips = {'armor':[2,1321],'martial':[1323,2563], 'lion':[2565,4084]}
test_subj_nof = 6
start_col = 1
wb = xlrd.open_workbook('orient_cam_switch_20.xls')
sh = wb.sheet_by_index(0)
data = {}
max_length = 3200
for i in range(len(arr_test_subjects)):
    data[arr_test_subjects[i]] = {}
    for (key,clip) in dict_clips.items():
        tmp_yaw = []
        tmp_pitch = []
        tmp_roll = []
        tmp_time = []
        data[arr_test_subjects[i]][key] = {}
        k=0
        yaw_prev = 0
        for j in range(clip[0],clip[1]):
            time_val = sh.cell_value(rowx=j, colx=(start_col+i*test_subj_nof))
            yaw_val = sh.cell_value(rowx=j, colx=(start_col+i*test_subj_nof+1))
            if (yaw_val == ''):
                yaw_val = 0
            pitch_val = sh.cell_value(rowx=j, colx=(start_col+i*test_subj_nof+2))
            if (pitch_val == ''):
                pitch_val = 0
            roll_val = sh.cell_value(rowx=j, colx=(start_col+i*test_subj_nof+3))
            if (roll_val == ''):
                roll_val = 0
            if (time_val == ''):
                if (tmp_time != []):
                    time_val = tmp_time[-1]+1
                else:
                    time_val = 0
# =============================================================================
#             if (j>clip[0] and np.abs(yaw_val-tmp_yaw[-1])>90):
#                 yaw_val = -yaw_val
# =============================================================================

            if unroll_flag:
                if yaw_val+k*360-yaw_prev < -320:
                    k=k+1
                elif yaw_val+k*360-yaw_prev > 320:
                    k=k-1

            yaw_prev = yaw_val+k*360
            #tmp_yaw.append(yaw_val)
            tmp_yaw.append(yaw_prev)
            tmp_pitch.append(pitch_val)
            tmp_roll.append(roll_val)
            tmp_time.append(time_val)
        data[arr_test_subjects[i]][key]['yaw'] = tmp_yaw[:max_length]
        data[arr_test_subjects[i]][key]['pitch'] = tmp_pitch[:max_length]
        data[arr_test_subjects[i]][key]['roll'] = tmp_roll[:max_length]
        data[arr_test_subjects[i]][key]['time'] = tmp_time[:max_length]
#%%
for i in range(len(arr_test_subjects)):
    plt.plot(data[arr_test_subjects[i]]['armor']['yaw'],label=arr_test_subjects[i])
#plt.plot(mode_arr2,linewidth=4)
plt.show()
#%%
#from SRM_main import SRM_general
#SRM_general(data2,'yaw','bar_dance_sa',120)