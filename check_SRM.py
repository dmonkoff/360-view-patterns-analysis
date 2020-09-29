# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 15:06:29 2018

@author: monakhov
"""
#Computes similarity ring metric for each user
from SRM_main import SRM_general
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy import stats
from matplotlib.collections import PatchCollection
#dict_clips = {'armor':[2,1321],'martial':[1323,2563], 'lion':[2565,4084]}
# =============================================================================
# users = ['Lasse', 'Ramin', 'Jani', 'Antti', 'xingyang ni','Nahid', 'Vida', 'Ling yu', 
#                      'Jari', 'Sujeet', 'Adriana', 'kashyap', 'You yu', 'Peter',  
#                      'Kimmo', 'Yuri', 'Kalle', 'Toni']
# =============================================================================
users = ['Lasse', 'Ramin', 'Jani', 'Antti', 'xingyang ni','Nahid', 'Vida', 'Ling yu', 
                     'Jari', 'Pouria','Sujeet','emre', 'Adriana', 'kashyap', 'You yu', 'Peter',  
                     'Kimmo', 'Yuri', 'Kalle', 'Toni']

users2 = ['subj 1', 'subj 2', 'subj 3', 'subj 4', 'subj 5', 'subj 6', 'subj 7', 'subj 8', 'subj 9', 'subj 10',
         'subj 11', 'subj 12', 'subj 13', 'subj 14', 'subj 15', 'subj 16', 'subj 17', 'subj 18', 'subj 19', 'subj 20' ]
clip='lion'
ring_width=10
res,mode_arr,users_scores = SRM_general(data,'yaw',clip,50,True,users)
user_scores_norm = np.array(list(users_scores.values()))/len(data['Nahid'][clip]['yaw'])
plt.locator_params(nbins=6)
plt.locator_params('y',nbins=20)
plt.xticks(plt.xticks()[0], [str(j) for j in ['',0,12,24,36,48,60]])
plt.title('')
plt.ylabel('Yaw (degrees)', fontsize=16)
plt.xlabel('Time (s)', fontsize=16)

#plt.title('Similarity Ring Metric: Lions')
#%%
ringboxes = []
ring_width=10
ring_radius=60
for i in range(len(mode_arr)):
    rect =patches.Rectangle((ring_width*i,mode_arr[i]-ring_radius),ring_width,2*ring_radius,hatch='/',fill=False)
    ringboxes.append(rect)
pc = PatchCollection(ringboxes,match_original=True,hatch='//')
fig, ax = plt.subplots(1)
ax.plot(data['Nahid'][clip]['yaw'])
ax.add_collection(pc)   
#%%
plt.figure()
user_scores_norm = np.array(list(users_scores.values()))/len(data['Nahid'][clip]['yaw'])
plt.bar(range(1,21),(1-user_scores_norm)*100,align='center')
plt.xticks(range(1,21), users2, fontsize=12)
plt.xticks(rotation=45)
plt.bar([10,12],[(1-user_scores_norm[9])*100,(1-user_scores_norm[11])*100],align='center',color='r')
plt.hlines(50,0,21,'g')
plt.ylabel('Individual SRM scores (%)', fontsize=16)
plt.xlabel('Test subjects', fontsize=16)
#plt.plot(users,[0.6 for user in users],'red')
#plt.savefig('SRM_Lions.jpg',dpi=600)
#%%
clip_names = ["'Lions'","'Armor'","'Martial'"]
for i,elem in enumerate(['lion', 'armor', 'martial']):
    res,mode_arr,users_scores = SRM_general(data,'yaw',elem,50,False,users)
    plt.subplot(1,3,i+1)
    user_scores_norm = np.array(list(users_scores.values()))/len(data['Nahid'][clip]['yaw'])
    plt.bar(range(1,21),(1-user_scores_norm)*100,align='center',color='#1f77b4')
#    plt.xticks(range(1,21), users2, fontsize=12)
    plt.xticks(rotation=45)
    idices_under = np.argwhere((1-user_scores_norm)*100 < 50)
    if idices_under.size != 0:
        plt.bar(idices_under[0]+1,(1-user_scores_norm[idices_under[0]])*100,align='center',color='r')
    plt.hlines(50,0,21,'g')
    plt.ylim([0,110])
    plt.ylabel('Individual SRM scores (%)', fontsize=16)
    plt.xlabel(clip_names[i]+' clip', fontsize=16)
    plt.tick_params(\
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='off') # labels along the bottom edge are off
    