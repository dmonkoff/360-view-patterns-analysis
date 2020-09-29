# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 16:30:27 2018

@author: monakhov
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import pickle
from scipy.ndimage.filters import convolve1d
#represents one collision box with two corners
class TileElem():
    def __init__(self, yaw_min,yaw_max,pitch_min,pitch_max):
        self.yaw_min = yaw_min
        self.yaw_max = yaw_max
        self.pitch_min = pitch_min
        self.pitch_max = pitch_max
#tile is a set of collision boxes, with functions to check intersections
class Tile():
    def __init__(self, yaw_min,yaw_max,pitch_min,pitch_max):
        self.tile_list =[]
        if isinstance(yaw_min,list) and isinstance(yaw_max,list) and isinstance(pitch_min,list) and isinstance(pitch_max,list):
            for i in range(len(yaw_min)):
                self.tile_list.append(TileElem(yaw_min[i],yaw_max[i],pitch_min[i],pitch_max[i]))
        else:
            self.tile_list.append(TileElem(yaw_min,yaw_max,pitch_min,pitch_max))
    #compute intersection area between this tile collision box and some other tile collision box
    def ComputeIntersectionArea(self, otherTilelem, tileElem):
        leftx = np.max((tileElem.yaw_min,otherTilelem.yaw_min))
        rightx = np.min((tileElem.yaw_max,otherTilelem.yaw_max))
        topy = np.min((tileElem.pitch_max,otherTilelem.pitch_max))
        bottomy = np.max((tileElem.pitch_min,otherTilelem.pitch_min))
        return (rightx-leftx)*(topy-bottomy)
    
    #check if the tile is intersected enough with some other tile
    def CheckIfIntersectEnough(self, otherTile, threshold = 300):
        intersect_area = 0
        for tileElem in self.tile_list:
            for otherTileElem in otherTile.tile_list:
                if (otherTileElem.yaw_min > tileElem.yaw_max or otherTileElem.yaw_max < tileElem.yaw_min or
                    otherTileElem.pitch_min > tileElem.pitch_max or otherTileElem.pitch_max < tileElem.pitch_min):
                    continue
                else:
                    intersect_area = intersect_area + self.ComputeIntersectionArea(otherTileElem,tileElem)
        if intersect_area >= threshold:
            return True
        else:
            return False
    
    #checks if one tile is inside the other tile
    def CheckIfInside(self,otherTile):
        res = True
        for otherTileElem in otherTile.tile_list:
            tmp = False
            for tileElem in self.tile_list:
                tmp = tmp or (otherTileElem.yaw_min >= tileElem.yaw_min and otherTileElem.yaw_max <= tileElem.yaw_max and
                    otherTileElem.pitch_min >= tileElem.pitch_min and otherTileElem.pitch_max <= tileElem.pitch_max)
            res = res and tmp
        return res
#class which manages which tiles are loaded in a moment
class TileObserver():
    def __init__(self, num_of_tiles, dash_duration=1000,dash_duration_short = 600):
        self.dash_timestamps = np.ones(num_of_tiles)*-1e16 #array of timestamps for each tile, if 0 - lq tile is used, !0 - hq tile is loaded with legth until the given timestamp
        self.dash_timestamps_lq_tiles = np.ones(num_of_tiles)*-1e16
        self.dash_quality = np.zeros(num_of_tiles) # what quality each tile is
        self.dash_duration=dash_duration #defines duration of hq tile
        self.prev_tiles = []
        self.dash_duration_short = dash_duration_short
        self.dash_duration_long = dash_duration
        self.download_buffer = {}
        self.all_downloaded_tiles = []
    def GetClosestTSSwitch(self):
        return np.min(self.dash_timestamps)
    def SetDashDuration(self, dash_duration):
        self.dash_duration=dash_duration
    def Reset(self):
        self.dash_timestamps[...] = -1e16
        self.dash_timestamps_lq_tiles[...] = -1e16
        self.dash_quality[...] = 0
        self.prev_tiles = []
        self.download_buffer = {}
        self.all_downloaded_tiles = []
        self.numUselessTiles = 0
    def CountCurrentTiles(self):
        return np.sum(self.dash_quality > 0)
    def ReplaceTileSet(self, num_of_tiles,dash_duration=1000):
        self.dash_timestamps = np.ones(num_of_tiles)*-1e16 #array of timestamps for each tile, if 0 - lq tile is used, !0 - hq tile is loaded with legth until the given timestamp
        self.dash_timestamps_lq_tiles = np.ones(num_of_tiles)*-1e16
        self.dash_quality = np.zeros(num_of_tiles) # what quality each tile is
        self.dash_duration=dash_duration #defines duration of hq tile
        self.prev_tiles = []
        self.download_buffer = {}
        self.all_downloaded_tiles = []
        self.numUselessTiles = 0
    def LoadFirstDashSegment(self,new_tiles_list,cur_ts):
        list_hq_tiles = []
        list_lq_tiles = []
        for new_tile in new_tiles_list:
                self.dash_quality[new_tile] = 1
                segment_index = int(cur_ts//self.dash_duration)
                self.dash_timestamps[new_tile] = segment_index*self.dash_duration+self.dash_duration
                list_hq_tiles.append(new_tile)
        for new_tile in range(len(self.dash_timestamps)):
            if new_tile not in new_tiles_list:
                segment_index = int(cur_ts//self.dash_duration)
                self.dash_timestamps_lq_tiles[new_tile] = segment_index*self.dash_duration+self.dash_duration
                list_lq_tiles.append(new_tile)
        return list_hq_tiles,list_lq_tiles
    def UpdateTilesHQ(self,new_tiles_list, cur_ts):
        res= 0
        list_hq_tiles = []
        list_hq_tiles_short = []
        for new_tile in new_tiles_list:
            if  (new_tile not in self.download_buffer.keys()):
                segment_index_tmp = int(cur_ts//self.dash_duration)
                next_tile_ts = np.max([self.dash_timestamps_lq_tiles[new_tile],self.dash_timestamps[new_tile]])
                
#                assert next_tile_ts+self.dash_duration < segment_index_tmp*self.dash_duration
                
                if next_tile_ts % dash_duration == 0:
                    self.download_buffer[new_tile] = {'start_ts':next_tile_ts,
                                        'end_ts':next_tile_ts+self.dash_duration}
                    list_hq_tiles.append(new_tile)
                else:
                    self.download_buffer[new_tile] = {'start_ts':next_tile_ts,
                                        'end_ts':next_tile_ts+self.dash_duration_short}
                    list_hq_tiles_short.append(new_tile)
                res += 1
                    
        self.prev_tiles = new_tiles_list
        return res,list_hq_tiles,list_hq_tiles_short 
    def UpdateTilesLQ(self, cur_ts):
        res = 0
        list_lq_tiles = []
        list_lq_tiles_short = []
        segment_index = int(cur_ts//self.dash_duration)
        for new_tile in range(len(self.dash_timestamps)):
            if (self.dash_quality[new_tile] != 1):
                ts_tmp = np.max([self.dash_timestamps_lq_tiles[new_tile],self.dash_timestamps[new_tile]])
                if cur_ts >= ts_tmp:
                    if ts_tmp % dash_duration == 0:
                        self.dash_timestamps_lq_tiles[new_tile] = ts_tmp + self.dash_duration
                        res += 1
                        list_lq_tiles.append(new_tile)
                    else:
                        self.dash_timestamps_lq_tiles[new_tile] = ts_tmp + self.dash_duration_short
                        res += 1
                        list_lq_tiles_short.append(new_tile)
        return res,list_lq_tiles,list_lq_tiles_short
    def CheckBuffer(self,ts):
        clear_tiles_flag = True
        useless_tiles = []
        buffer_items = list(self.download_buffer.keys())
        for tile in buffer_items:
            if ts >= self.download_buffer[tile]['start_ts']:
                if clear_tiles_flag:
                    clear_tiles_flag = False
                    useless_tiles = self.all_downloaded_tiles
                    self.all_downloaded_tiles = []
                self.dash_quality[tile] = 1
                self.all_downloaded_tiles.append(tile)
                self.dash_timestamps[tile] = self.download_buffer[tile]['end_ts']
                self.download_buffer.pop(tile)
        return useless_tiles
    def CheckWhatTilesUseless(self,HQtiles):
        for tile in HQtiles:
            if tile in self.all_downloaded_tiles:
                self.all_downloaded_tiles.remove(tile)
    def RemoveExpiredHQTiles(self,cur_ts):
        for i in range(len(self.dash_timestamps)):
            if cur_ts >= self.dash_timestamps[i]:
                self.dash_quality[i] = 0
        return
#    def ReturnNewTilesDelta(self,new_tiles_list):
#        res = 0
#        for new_tile in new_tiles_list:
#            if self.dash_timestamps[new_tile] == 0:
#                res += 1
#        return res
    def ReturnPrevCurTilesDelta(self, cur_tiles):
        res = 0
        for cur_tile in cur_tiles:
            if cur_tile not in self.prev_tiles:
                res += 1
        return res


class BoundBoxes():
    
    def __init__(self,yaw,pitch,width,height):
        self.yaw = yaw
        self.pitch = pitch
        self.width=width
        self.height=height
        self.area = self.width*self.height
        self.boxes = None
        self.setBoxes()
    
    def UpdateLocation(self,yaw,pitch,width,height):
        self.yaw = yaw
        self.pitch = pitch
        self.width=width
        self.height=height
        self.setBoxes()
    
    def setBoxes(self):
        y_top = self.pitch+self.height/2
        if y_top > 90:
            y_top = 90
        y_bot = self.pitch-self.height/2
        if y_bot < -90:
            y_bot = -90
#        y_top = 90 #very wrong to do it like that, think of other way later
#        y_bot = -90
        x_right = self.yaw+self.width/2
        x_left = self.yaw-self.width/2
        if x_right > 180:
            self.boxes=Tile([x_left,-180],[180,-360+x_right],[y_bot,y_bot],[y_top,y_top])
    
        elif x_left < -180:
            self.boxes=Tile([-180,360+x_left],[x_right,180],[y_bot,y_bot],[y_top,y_top])
        else:
            self.boxes=Tile(x_left,x_right,y_bot,y_top)
       
    def CheckIfOtherBBoxIsInside(self,other_bbox):
        if self.area >= other_bbox.area:
            return self.boxes.CheckIfInside(other_bbox.boxes)
        else:
            return False

    def SelectTiles(self,tileset):
        selected_tiles = []
        for i in range(len(tileset)):
                if self.boxes.CheckIfIntersectEnough(tileset[i]):
                    selected_tiles.append(i)
        return selected_tiles

#%%

#%% 6x4 tiles
tileset_normal = []
tile_angular_width = 360/8
tile_angular_height = 180/6
for y in range(6):
    for x in range(8):
        tileset_normal.append(Tile(-180+x*tile_angular_width,-180+(x+1)*tile_angular_width,
                                   90-(y+1)*tile_angular_height,90-y*tile_angular_height))


trajectories = data
seg_length = 250
w = seg_length/1000

# =============================================================================
# clip_name = 'scene_cut_test'
# seconds_arr = np.arange(0,160,w)*1000
# scene_cuts = [20000,40000,60000,80000,100000,120000,140000]
# =============================================================================


# =============================================================================
# clip_name = 'armor'
# seconds_arr = np.arange(0,66,w)*1000
# scene_cuts = [8000, 26026, 38672, 50684]
# 
# =============================================================================
clip_name = 'lion'
seconds_arr = np.arange(0,76,w)*1000
scene_cuts = [11011, 32866, 55122, 67034]

# =============================================================================
# clip_name = 'martial'
# seconds_arr = np.arange(0,62,w)*1000
# scene_cuts = [12560, 21760, 44760, 54760]
# =============================================================================


seconds_cuts = []
for cut in scene_cuts:
        seconds_cuts.append(np.argmin(np.abs(seconds_arr-cut)))#array of indeces for cuts  
        #print(second)



#trajectories = {}
#trajectories['Nahid'] = {}
#trajectories['Nahid'][clip_name] = {}
#trajectories['Nahid'][clip_name]['yaw'] = np.hstack((np.linspace(0,180,48),np.zeros(48)))
#trajectories['Nahid'][clip_name]['pitch'] = np.zeros(96)
#trajectories['Nahid'][clip_name]['time'] = np.linspace(0,2000,96)

#%%
dash_duration_1 = 1200
dash_duration_2 = 600


tileSizes = {}
tileSizes['6x8_1200ms'] = {}
with open('lions_tile_sizes/lions-6x8_QP22_1200ms.bin','rb') as f:
    tileSizes['6x8_1200ms']['high'] = pickle.load(f)
    tileSizes['6x8_1200ms']['high'] = tileSizes['6x8_1200ms']['high']*8/10**6
with open('lions_tile_sizes/lions-6x8_QP28_1200ms.bin','rb') as f:
    tileSizes['6x8_1200ms']['low'] = pickle.load(f)
    tileSizes['6x8_1200ms']['low'] = tileSizes['6x8_1200ms']['low']*8/10**6
tileSizes['6x8_1200ms']['ts'] = np.array(list(range(tileSizes['6x8_1200ms']['high'].shape[1])))*1200


tileSizes['6x8_600ms'] = {}
with open('lions_tile_sizes/lions-6x8_QP28_600ms.bin','rb') as f:
    tileSizes['6x8_600ms']['low'] = pickle.load(f)
    tileSizes['6x8_600ms']['low'] = tileSizes['6x8_600ms']['low']*8/10**6
with open('lions_tile_sizes/lions-6x8_QP22_600ms.bin','rb') as f:
    tileSizes['6x8_600ms']['high'] = pickle.load(f)
    tileSizes['6x8_600ms']['high'] = tileSizes['6x8_600ms']['high']*8/10**6
tileSizes['6x8_600ms']['ts'] = np.array(list(range(tileSizes['6x8_600ms']['high'].shape[1])))*600

#tileSizes['4x6']['ts'] = []
#%%
plt.figure()
tmp_sizes = tileSizes['6x8_300ms']['high'][2,:]
plt.plot(tileSizes['6x8_300ms']['ts']/1000,tmp_sizes,'-', label='size of segment, kbits')
plt.vlines(np.array(scene_cuts)/1000,np.zeros(len(scene_cuts)),np.ones(len(scene_cuts))*1,'r',label='scene cut')
#plt.ylim(0,6)
#plt.xlim(0,20)
plt.legend()
#%%
tmp_bitrate = convolve1d(tmp_sizes,[1,1,1,1,0,0,0])
plt.plot(tileSizes['6x8_300ms']['ts']/1000,tmp_bitrate,'-', label='size of segment, mbits/sec')
plt.vlines(np.array(scene_cuts)/1000,np.zeros(len(scene_cuts)),np.ones(len(scene_cuts))*10,'r',label='scene cut')
#plt.ylim(0,6)
#plt.xlim(0,20)
plt.legend()
#%%
#%%
dash_duration_normal = 600
dash_duration_cut = 300
tileset_tag_normal = "6x8_"+str(dash_duration_normal)+"ms"
tileset_tag_cut = "6x8_"+str(dash_duration_cut)+"ms"
dash_duration = dash_duration_normal
HMD_width = 100
HQA_width = 120 
HQA_width_sc = 120
box_height = 100
exploration_start = 0
exploration_end = 0 
HQA_width_cur = HQA_width
bitrate_stats = np.zeros((len(list(trajectories.keys())), len(trajectories['Nahid'][clip_name]['yaw'])))
useless_stats = np.zeros((len(list(trajectories.keys())), len(trajectories['Nahid'][clip_name]['yaw'])))
total_bitrate_arr = []
tileset = tileset_normal
observer = TileObserver( len(tileset),dash_duration_normal,dash_duration_cut)
all_users = list(trajectories.keys())
#random.shuffle(all_users)
for indx,user in enumerate(all_users):
    total_bitrate = 0
    useless_bitrate = []
    useless_tmp = 0
    cutIndx = 1
    scene_cut_ts = scene_cuts[0]
    tileset = tileset_normal
    print(indx)
    tmp = []
    tmp_useless = []
    HMD = BoundBoxes(trajectories[user][clip_name]['yaw'][0],trajectories[user][clip_name]['pitch'][0],HMD_width,box_height)
    HQA = BoundBoxes(trajectories[user][clip_name]['yaw'][0],trajectories[user][clip_name]['pitch'][0],HQA_width,box_height)
    selected_tiles = HQA.SelectTiles(tileset)
    tileset_tag = tileset_tag_normal
    dash_duration = dash_duration_normal
#    tileset_tag = tileset_tag_cut
#    dash_duration = dash_duration_cut
    observer.ReplaceTileSet(len(tileset),dash_duration_normal)
    list_of_new_hq_tiles,list_of_new_lq_tiles = observer.LoadFirstDashSegment(selected_tiles,trajectories[user][clip_name]['time'][0])
    list_of_new_hq_tiles_short = []
    list_lq_tiles_short = []
    flag_scene_cut = 0
    flag_scene_cut_start = 0
    for i in range(len(trajectories[user][clip_name]['yaw'])):
        if i != 0:
            num_hq_tiles = 0
            list_of_new_hq_tiles = []
            list_of_new_hq_tiles_short = []
            list_lq_tiles = []
            list_lq_tiles_short = []
#        print(trajectories[user][clip_name]['time'][i])
        HMD.UpdateLocation(trajectories[user][clip_name]['yaw'][i],trajectories[user][clip_name]['pitch'][i],HMD_width,box_height)
        observer.RemoveExpiredHQTiles(trajectories[user][clip_name]['time'][i])
        useless_tiles_list = observer.CheckBuffer(trajectories[user][clip_name]['time'][i])
        if not  HQA.CheckIfOtherBBoxIsInside(HMD):
            HQA.UpdateLocation(trajectories[user][clip_name]['yaw'][i],trajectories[user][clip_name]['pitch'][i],HQA_width_cur,box_height)

        selected_tiles = HQA.SelectTiles(tileset)
#        selected_tiles = list(range(48))
        observer.CheckWhatTilesUseless(selected_tiles)
        num_hq_tiles,list_of_new_hq_tiles,list_of_new_hq_tiles_short = observer.UpdateTilesHQ(selected_tiles,trajectories[user][clip_name]['time'][i])
        all_hq_tiles = list_of_new_hq_tiles.copy()
        all_hq_tiles.extend(list_of_new_hq_tiles_short)
        num_lq_tiles,list_lq_tiles,list_lq_tiles_short = observer.UpdateTilesLQ(trajectories[user][clip_name]['time'][i])
        
        
        full_bitrate_size = 0
        useless_size = 0
        for new_tile_lq in list_lq_tiles:
            if (new_tile_lq not in list_of_new_hq_tiles) and (new_tile_lq not in list_of_new_hq_tiles_short):                
                full_bitrate_size += tileSizes[tileset_tag]['low'][new_tile_lq,int(trajectories[user][clip_name]['time'][i]//dash_duration)]
        
        for new_tile_lq in list_lq_tiles_short:
            if (new_tile_lq not in list_of_new_hq_tiles) and (new_tile_lq not in list_of_new_hq_tiles_short):                
                full_bitrate_size += tileSizes[tileset_tag_cut]['low'][new_tile_lq,int(trajectories[user][clip_name]['time'][i]//dash_duration_cut)]       
        
        for new_tile_hq in list_of_new_hq_tiles:
            full_bitrate_size += tileSizes[tileset_tag]['high'][new_tile_hq,int(trajectories[user][clip_name]['time'][i]//dash_duration)]
        
        for new_tile_hq in list_of_new_hq_tiles_short:
            full_bitrate_size += tileSizes[tileset_tag_cut]['high'][new_tile_hq,int(trajectories[user][clip_name]['time'][i]//dash_duration_cut)]
        total_bitrate += full_bitrate_size
        tmp.append(full_bitrate_size)

        for useless_tile in useless_tiles_list:
            useless_size += tileSizes[tileset_tag]['high'][useless_tile,int(trajectories[user][clip_name]['time'][i]//dash_duration)]
        tmp_useless.append(useless_size)
    bitrate_stats[indx,...] = tmp
    useless_stats[indx,...] = tmp_useless
    total_bitrate_arr.append(total_bitrate)
    
#print(np.mean(total_bitrate_arr))#%%
seg_length = 1000
w = seg_length/1000
res_bitrate_secondwise = []
n_elems = 0
seconds_arr = np.arange(0,76,w)*1000

seconds_cuts = []
for cut in scene_cuts:
        seconds_cuts.append(np.argmin(np.abs(seconds_arr-cut)))#array of indeces for cuts  
        #print(second)
for indx,user in enumerate(trajectories.keys()):
    tmp_ts = seg_length
    tmp_seg = 0
    tmp_arr = []
    tmp_count = 0
    for i in range(len(trajectories[user][clip_name]['yaw'])):
        if trajectories[user][clip_name]['time'][i] < tmp_ts:
            tmp_seg= tmp_seg+bitrate_stats[indx,i]
            tmp_count += 1
        else:
#            tmp_seg= tmp_seg+bitrate_stats[indx,i]
            tmp_arr.append(tmp_seg)
            tmp_ts += seg_length
            tmp_count = 1
            tmp_seg = bitrate_stats[indx,i]
    if tmp_count != 0:
        tmp_arr.append(tmp_seg)
    if n_elems == 0:
        n_elems = len(tmp_arr)
    else:
        tmp_arr = tmp_arr[:n_elems]
    res_bitrate_secondwise.append(tmp_arr)
res_bitrate_secondwise = np.array(res_bitrate_secondwise)

######################################################################
#seconds_arr = np.arange(0,160,w)*1000

#####################################################

        #print(second)
#%%
plt.figure()
plt.plot(np.array(trajectories[all_users[0]][clip_name]['time'])/1000,bitrate_stats[0,...],label='downloaded data')
plt.vlines(np.array(scene_cuts)/1000,0,40,label='scene cuts',color='red')
plt.ylim(0,40)
plt.legend()
#%%
plt.figure()
plt.plot(np.array(trajectories[all_users[0]][clip_name]['time'])/1000,np.cumsum(bitrate_stats[0,...]),label='downloaded data')
plt.vlines(np.array(scene_cuts)/1000,0,1750,label='scene cuts',color='red')
#plt.ylim(0,40)
plt.legend()
#%%
meanVals = np.mean(res_bitrate_secondwise,axis=0)
#meanRange = np.mean(res,axis=0)
plt.figure()
plt.bar(seconds_arr/1000,meanVals/1000,width=(-1)*w,align='edge')
plt.bar(seconds_arr[seconds_cuts]/1000,meanVals[seconds_cuts]/1000,width=(-1)*w,align='edge', label='scene cut')
#plt.bar(seconds_arr/1000,meanRange,width=(-1)*w,align='edge')
plt.xlabel('time, s')
plt.ylabel('bitrate, mbits/sec')
plt.legend()
#plt.ylim(0,1700)
plt.title('bitrate tileset switching ')
#plt.ylim(0,2.5)
