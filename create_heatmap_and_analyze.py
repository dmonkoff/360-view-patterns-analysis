# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 14:17:02 2018

@author: monakhov
"""

import numpy as np
import scipy.misc as misc
from matplotlib import pyplot as plt
from matplotlib import cm
from AnglesToCoords import YawToX, PitchToY
import scipy.stats as st
import os, sys
from scipy import signal
from sklearn.metrics import roc_curve, auc
from scipy.ndimage.morphology import binary_dilation
def gkern(kernlen=10, nsig=3):
    """Returns a 2D Gaussian kernel array."""

    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    return kernel


from matplotlib.colors import ListedColormap
# Choose colormap
cmap1 = cm.autumn_r

# Get the colormap colors
my_cmap = cmap1(np.arange(cmap1.N))

# Set alpha
my_cmap[:,-1] = np.linspace(0, 1, cmap1.N)

# Create new colormap
my_cmap = ListedColormap(my_cmap)
def CollectFixationMap(data,clip,timestamps,subjects,hor_res,ver_res):
    res = np.zeros((ver_res,hor_res))
    for timestamp in timestamps:
        for subject in subjects:
            tmp = data[subject][clip]['yaw'][timestamp]
            x = YawToX(tmp,hor_res)
            tmp = data[subject][clip]['pitch'][timestamp]
            y = PitchToY(tmp,ver_res)
            #print(x)
            #print(y)
            res[y,x] += 1
    return res

def GetMinMaxValue(n,scaler,hor_res,ver_res,kernel):
    tmp = np.zeros((ver_res,hor_res))
    tmp[int(ver_res/2),int(hor_res/2)] = n
    res_gauss = signal.fftconvolve(tmp*scaler, kernel, mode='same')
    res_max = np.max(res_gauss)
    tmp[int(ver_res/2),int(hor_res/2)] = 1
    res_gauss = signal.fftconvolve(tmp*scaler, kernel, mode='same')
    res_min = np.max(res_gauss)/2
    return (res_min,res_max)

arr_test_subjects =  ['Nahid', 'Ramin', 'Ling Yu', 'Lasse','Vida', 'Alizera', 'Jani Kartelu', 
                     'Aleksei', 'Ali','Francesco','Mikko Pekkarinen', 'Peter', 'Jani', 'Henri', 'Sujeet',  
                     'Roimila', 'Antti', 'Jari', 'Payman']

clip_name='scene_cut_test'
#%%
test = CollectFixationMap(data,'wife_carry_na',[300],users,3840,1920)
#%%
#plt.imshow(test,cmap='gray')
scale_coeff = 100
test_gauss = signal.fftconvolve(test*scale_coeff, gkern(250,3), mode='same')
#%%
plt.imshow(test_gauss,cmap=my_cmap)
#%%

#%%
my_dpi = 125.37
scale_coeff = 100
w=3840
h=1920
#(min_val,max_val) = GetMinMaxValue(len(users),100,3840,1920, gkern(600,3))
plt.ioff()
if not os.path.exists('mapTest3'):
    os.makedirs('mapTest3')
g_kern = gkern(30,3)
users =  ['Nahid', 'Ramin', 'Pouria', 'Ling Yu', 'Lasse','Vida', 'Alizera', 'Jani Kartelu', 
                     'Aleksei', 'Ali','Francesco','Mikko Pekkarinen', 'Peter', 'Jani', 'Henri', 'Sujeet',  
                     'Roimila', 'Antti', 'Jari', 'Payman']
for i in range(len(data['Vida'][clip_name]['yaw'])):
    test = CollectFixationMap(data,clip_name,[i],users,480,240)
    test_gauss = signal.convolve2d(test, g_kern, mode='same')
    fig = plt.gcf()
    DPI = fig.get_dpi()
    fig.set_size_inches(w/float(DPI),h/float(DPI))
    test_gauss2 = misc.imresize(test_gauss,(1920,3840),interp='cubic',mode='F')
    img  = plt.imshow(test_gauss,cmap=my_cmap)#,vmin=min_val,vmax=max_val
    plt.axis('off')
    plt.savefig('salMap/'+'scene_cut_test'+str(i).zfill(4)+'.png',transparent=True)
    plt.close()
    print(i)
plt.ion()
#%%
my_dpi = 125.37
w=3840
h=1920
window_size = 300
sigma = 3
t = (((window_size - 1)/2)-0.5)/sigma
dir_name = 'salMap'
#os.mkdir(dir_name)
plt.ioff()
users_forward=['Vida', 'Mikko', 'Ramin', 'Aleksei', 'Pekka','Peetu', 'Henrik', 'Kalle', 'Marja','Alizera']
users_backward=['James','Payman','Sujeet','Tony', 'Nahid', 'Ali','Jani','Teena','Teemu Lahtela','Erja','jouko']
for i in range(len(data1['Vida']['helsinki_mult_views_na']['yaw'])):
    test = CollectFixationMap(data1,'helsinki_mult_views_na',[i],users_forward,3840,1920)
    #test2 = CollectFixationMap(data3,'helsinki_mult_views_spa',[i],users_backward,3840,1920)
    #test = test1 + test2
    test_gauss = signal.fftconvolve(test, gkern(300,3), mode='same')
    #test_gauss = gaussian_filter(test, sigma=sigma, truncate=t)
    fig = plt.gcf()
    DPI = fig.get_dpi()
    fig.set_size_inches(w/float(DPI),h/float(DPI))
    img  = plt.imshow(test_gauss,cmap=my_cmap)
    plt.axis('off')
    plt.savefig(dir_name+'/helsinki_mult_views_'+str(i).zfill(4)+'.png',transparent=True)
    plt.close()
    print(i)
plt.ion()



#%%
res_corr = []
eps=1e-16
for j in range(4,8):
    res_tmp = []
    NA_SPA_ALL = []
    SA_SPA_ALL = []
    NA_SA_ALL = []
    NA_SPA_FIRST = []
    SA_SPA_BACKWARD = []
    NA_SA_BACKWARD = []
    SA_SPA_FORWARD = []
    NA_SA_FORWARD = []
    for i in range(0,len(data1['Vida'][clip_names[j][0]]['yaw'])):
        test_na = CollectFixationMap(data1,clip_names[j][0],[i],users,480,240)
        map_na_all = signal.fftconvolve(test_na, gkern(75,3), mode='same')
        test_spa = CollectFixationMap(data3,clip_names[j][2],[i],users,480,240)
        map_spa_all = signal.fftconvolve(test_spa, gkern(75,3), mode='same')
        test_sa = CollectFixationMap(data2,clip_names[j][1],[i],users,480,240)
        map_sa_all = signal.fftconvolve(test_sa, gkern(75,3), mode='same')
        
        test_na = CollectFixationMap(data1,clip_names[j][0],[i],users_forward,480,240)
        map_na_forward = signal.fftconvolve(test_na, gkern(75,3), mode='same')
        test_spa = CollectFixationMap(data3,clip_names[j][2],[i],users_forward,480,240)
        map_spa_forward = signal.fftconvolve(test_spa, gkern(75,3), mode='same')
        test_sa = CollectFixationMap(data2,clip_names[j][1],[i],users_forward,480,240)
        map_sa_forward = signal.fftconvolve(test_sa, gkern(75,3), mode='same')
        
        test_na = CollectFixationMap(data1,clip_names[j][0],[i],users_backward,480,240)
        map_na_backward = signal.fftconvolve(test_na, gkern(75,3), mode='same')
        test_spa = CollectFixationMap(data3,clip_names[j][2],[i],users_backward,480,240)
        map_spa_backward = signal.fftconvolve(test_spa, gkern(75,3), mode='same')
        test_sa = CollectFixationMap(data2,clip_names[j][1],[i],users_backward,480,240)
        map_sa_backward = signal.fftconvolve(test_sa, gkern(75,3), mode='same')
        
        NA_SPA_ALL.append(np.corrcoef([map_na_all.ravel(),map_spa_all.ravel()])[0][1])
        SA_SPA_ALL.append(np.corrcoef([map_sa_all.ravel(),map_spa_all.ravel()])[0][1])
        NA_SA_ALL.append(np.corrcoef([map_sa_all.ravel(),map_na_all.ravel()])[0][1])
        NA_SPA_FIRST.append(np.corrcoef([map_na_forward.ravel(),map_spa_backward.ravel()])[0][1])
        SA_SPA_BACKWARD.append(np.corrcoef([map_sa_backward.ravel(),map_spa_backward.ravel()])[0][1])
        NA_SA_BACKWARD.append(np.corrcoef([map_sa_backward.ravel(),map_na_backward.ravel()])[0][1])
        SA_SPA_FORWARD.append(np.corrcoef([map_spa_forward.ravel(),map_sa_forward.ravel()])[0][1])
        NA_SA_FORWARD.append(np.corrcoef([map_na_forward.ravel(),map_sa_forward.ravel()])[0][1])
        print(i)
    res_corr.append([np.mean(NA_SPA_ALL), np.mean(SA_SPA_ALL), np.mean(NA_SA_ALL),
                     np.mean(NA_SPA_FIRST), np.mean(SA_SPA_BACKWARD), np.mean(NA_SA_BACKWARD),
                     np.mean(SA_SPA_FORWARD), np.mean(NA_SA_FORWARD),])
pickle.dump( res_corr, open( "res_corr.bin", "wb" ) )
#%%

#%%
res_corr = []
eps=1e-16
arr1 = []
arr2 = []
for i in range(0,len(data1['Vida']['helsinki_mult_views_na']['yaw'])):
     test_na = CollectFixationMap(data1,'helsinki_mult_views_na',[i],users_forward,960,480)
     map_na_forward = signal.fftconvolve(test_na, gkern(150,3), mode='same')
     test_spa = CollectFixationMap(data3,'helsinki_mult_views_spa',[i],users_backward,960,480)
     map_spa_backward = signal.fftconvolve(test_spa, gkern(150,3), mode='same')
     arr1.append(np.corrcoef([map_spa_backward.ravel(),map_na_forward.ravel()])[0][1])
     X = np.vstack((map_na_forward.ravel(),map_spa_backward.ravel()))
     arr2.append(np.corrcoef(X)[0][1])
     print(i)
#%%
my_dpi = 125.37
w=3840
h=1920
window_size = 300
sigma = 3
t = (((window_size - 1)/2)-0.5)/sigma
dir_name = 'salMap2'
#os.mkdir(dir_name)
plt.ioff()
users =  ['Nahid', 'Ramin', 'Ling Yu', 'Lasse','Vida', 'Alizera', 'Jani Kartelu', 
                     'Aleksei', 'Ali','Francesco','Mikko Pekkarinen', 'Peter', 'Jani', 'Henri', 'Sujeet',  
                     'Roimila', 'Antti', 'Jari', 'Payman']
for i in range(len(data['Aleksei'][clip_name]['yaw'])):
    test = CollectFixationMap(data, clip_name, [i],['Aleksei'],3840,1920)
    #test2 = CollectFixationMap(data3,'helsinki_mult_views_spa',[i],users_backward,3840,1920)
    #test = test1 + test2
    dil_size = 150
    mask = np.where(test==1)
    y = mask[0][0]
    x = mask[1][0]
    left = x-dil_size 
#    if left < 0:
#        left = 0
    right =x+dil_size 
#    if right > 3839:
#        right = 3839
    top =y-dil_size
#    if top < 0:
#        top = 0
    bottom = y+dil_size
#    if bottom > 1919:
#        bottom = 1919
    test[top:bottom,left:right] = 1
    test_gauss = test
    #test_gauss = gaussian_filter(test, sigma=sigma, truncate=t)
    fig = plt.gcf()
    DPI = fig.get_dpi()
    fig.set_size_inches(w/float(DPI),h/float(DPI))
    img  = plt.imshow(test_gauss,cmap=my_cmap)
    plt.axis('off')
    plt.savefig(dir_name+'/Aleksei/frame'+str(i).zfill(4)+'.png')
    plt.close()
    print(i)
plt.ion()