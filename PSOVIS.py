'''
Created on Sep 22, 2016
@author: Diako

------------------------------------------------------ License

This file is part of PSOVIS - an interactive code for extracting post-saccadic eye movements from the eye tracking data
Copyright (C) 2016-2019  Diako Mardanbegi

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>

------------------------------------------------------
ASSUMPTIONS:

- Only monocular data, having both eyes ("RIGHT_..." and "LEFT_...") or only one
- column titles in the data file:
-     Eye data: all eye-related columns should be started with either "RIGHT_..." OR "LEFT_...":
-         "RIGHT_GAZE_X","RIGHT_GAZE_Y","RIGHT_VELOCITY_X", "RIGHT_IN_SACCADE", "RIGHT_PUPIL_SIZE" (optional) , "SAMPLE_MESSAGE" (optional. set event MSGs in the code)
-     other columns:
-         "TIMESTAMP" (integer and measured in milisecond), 
- See the constant values used in the GetPSO function
- Change screen values in the pixels_to_degrees() to match your setup

TODO: (search for TODO name to find out where you can add your implementation)
- TODO_1: adding saccade and fixation detection in the absence of "_IN_SACCADE" 
- TODO_2: adding velocity and acceleration column if it's not provided in the file

'''






import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import *
import matplotlib.ticker as ticker
import matplotlib.patches as patches
from matplotlib.widgets import RadioButtons
from matplotlib.text import Text
from pylab import *
import colorsys
import csv
import numpy as np
from scipy.spatial import distance
from itertools import cycle
from scipy.signal import argrelmax, argrelmin,argrelextrema
from collections import defaultdict
import os
import time
import pandas as pd
from lib import PSOVIS_tools as PSOVIS_tools

# import seaborn as sns


Experiment='pso'

folder_data='./data/' + Experiment + '/'
folder_tables='./data/' + Experiment + '/processed_tables/'
folder_results='./results/' + Experiment + '/'

PSOVIS_tools.Create_Folder(folder_data)
PSOVIS_tools.Create_Folder(folder_tables)
PSOVIS_tools.Create_Folder(folder_results)

PSO_SIGNAL_WINDOW=(-25,50) # in ms relative to when PSO min happens (first bump)

include_right=200 # [in ms]. how many ms after the end of the saccade should be included in the data

offset_right=40  # [in ms]. This defines how many ms after the end of the saccade we should take the fixation for spatial alignment

fixation_window=30 ## [in ms] 



datafile_prefix='.xls'
delimiter="\t"

# target_onset_msg='Target_display'#OR leave it empty ''
# target_timeout_msg='Target_timeout'#OR leave it empty ''
target_onset_msg=''#OR leave it empty ''
target_timeout_msg=''#OR leave it empty ''



Show_Visualizer=True
IgnoreErrors=False

## List of files in the data folder
files=PSOVIS_tools.listdir_fullpath_file(folder_data)
participants= [f[:-4] for f in files if (f[-len(datafile_prefix):]==datafile_prefix) ]
participants.sort()

## Filter files
# participants=participants[3:4] #prevent from processing all the files
# participants=['Vids_cp191mm']


print ("List of all %s participants in the data folder: %s"%( size(participants),participants))
print(participants)



print ("processing %s files among total %s files --> %s"%(size(participants),size(participants),participants))



for idx, participant in enumerate(participants):
    print ('participant: ', participant)
    start = time.time()
    print(str(idx+1) + ' of ' + str(size(participants)) +  '    (' + participant + ')' )
    

        
    ## Load saccades table for subject if exists
    participant_table_file=folder_tables + participant + '_table.pkl'
    saccades=None
    if os.path.isfile(participant_table_file):
        print('loading the processed table')
        saccades= pd.read_pickle(participant_table_file)
        
    else:
        print('extracting saccades for %s'%participant)

        saccades=PSOVIS_tools.ExtractSaccades(participant,folder_data+participant+datafile_prefix,Experiment,'control',delimiter,target_onset_msg,target_timeout_msg,include_right,offset_right,fixation_window)
        saccades.loc[:,'angle_group_selected']=1
        saccades.loc[:,'amp_deg_selected']=1
        saccades.loc[:,'vel_av_selected']=1
        saccades.loc[:,'acc_av_selected']=1
        saccades.to_pickle(participant_table_file)
        


    if saccades.empty:
        print('no PSO found for %s'%participant)
        continue
        
#     print(saccades.head(2))
        
    
            
    
        #-----------------------------------------Loading parameters (selection criteria)

    ##Load parameters if the file exists
    if os.path.isfile(folder_results + 'params.npz'):#os.path.exists(folder_results + 'params.npz')
        done,[Amplitude_range, Velocity_range,Acceleration_range,Angle_group]=PSOVIS_tools.LoadParams(folder_results+'params.npz')     
        if done: 
            print('params loaded')
            Amplitude_range=tuple(Amplitude_range)
            Velocity_range=tuple(Velocity_range)
            Acceleration_range=tuple(Acceleration_range)
            Angle_group=Angle_group
    else:
        # If not create the file
###         open(folder_results + 'params.npz', 'w').close()
#         Amplitude_range=(saccades['amp_deg'].min(),saccades['amp_deg'].max())
#         Velocity_range=(saccades['vel_av'].min(),saccades['vel_av'].max())
#         Acceleration_range=(saccades['acc_av'].min(),saccades['acc_av'].max())
#         Angle_group=[True,True,True]
        Amplitude_range=(1,15)
        Velocity_range=(saccades['vel_av'].min(),saccades['vel_av'].max())
        Acceleration_range=(saccades['acc_av'].min(),saccades['acc_av'].max())
        Angle_group=[True,False,False]
        
        
        
        PSOVIS_tools.SaveParams(folder_results+'params.npz', Amplitude_range,Velocity_range,Acceleration_range,Angle_group)
    

    if not ('highlighted' in saccades.columns.values):
        saccades.loc[:,'highlighted' ]=0
        

    if not ('excluded' in saccades.columns.values):
        saccades.loc[:,'excluded' ]=0
    
    ## Update the selection columns according to params
    direction_histogram_values=[]
    saccades.loc[:,'angle_group_selected']=0
    saccades.loc[:,'amp_deg_selected']=0
    saccades.loc[:,'vel_av_selected']=0
    saccades.loc[:,'acc_av_selected']=0
    saccades.loc[saccades['amp_deg'].between(Amplitude_range[0],Amplitude_range[1]),'amp_deg_selected']=1
    saccades.loc[saccades['vel_av'].between(Velocity_range[0],Velocity_range[1]),'vel_av_selected']=1
    saccades.loc[saccades['acc_av'].between(Acceleration_range[0],Acceleration_range[1]),'acc_av_selected']=1
    
    if Angle_group[0]:
        saccades.loc[saccades['angle_group']==1,'angle_group_selected']=1
    if Angle_group[1]:
        saccades.loc[saccades['angle_group']==2,'angle_group_selected']=1
    if Angle_group[2]:
        saccades.loc[saccades['angle_group']==3,'angle_group_selected']=1    
    
    saccades.loc[:,'selected']=0
    saccades.loc[(saccades['angle_group_selected']==1) & (saccades['amp_deg_selected'] == 1) & (saccades['vel_av_selected'] == 1)& (saccades['acc_av_selected'] == 1),'selected']=1
        
    
#     f=open('./results/' + Experiment + '/' + SetNameForRangesFolder() + '/log.txt',"a+b")
#     f.write(msg)
        
        
    ## Load participant original data to access gaze data
    print('wait...')

    participant_data = pd.ExcelFile(folder_data+participant+datafile_prefix)
    participant_data = participant_data.parse(participant_data.sheet_names[0])
    participant_data=participant_data.replace('.', np.NaN)
    participant_data=participant_data.replace(np.NaN, 0) # this also converts the columns data types to float if possible
    participant_data=participant_data.apply(pd.to_numeric, errors='ignore')
   




    Gaze_x=participant_data[saccades.eye_tracked.iloc[0] +'_GAZE_X']
    Gaze_y=participant_data[saccades.eye_tracked.iloc[0]  +'_GAZE_Y']
    Gaze_x=Gaze_x.replace(0, np.nan)
    Gaze_y=Gaze_y.replace(0, np.nan)
    Gaze_x= np.array(Gaze_x).astype(float)
    Gaze_y= np.array(Gaze_y).astype(float)

    TIMESTAMP=participant_data['TIMESTAMP']

    
    timestamp_interval=TIMESTAMP.iloc[2]-TIMESTAMP.iloc[1]
    include_right=int(include_right//timestamp_interval) # conver to frames
    offset_right=int(offset_right//timestamp_interval)
    fixation_window=int(fixation_window//timestamp_interval)

    
    
    
        
    
    ##-----------------------------------------Show interactive plots 
    
    
    
    ##Defining a distinct color for each saccade (one color per index)
    N = len(saccades)
    HSV_tuples = [(x*1.0/N, 1.0, 0.5) for x in range(N)]
    RGB_tuples = list(map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples))
    
    
    
    ##Prepare subplots
    fig= plt.figure(1,figsize=(12,7))
    
    gs = gridspec.GridSpec(5, 4,width_ratios=[0.5,1,1,1],height_ratios=[1,1,1,1,1])
    
    
    ax_ang_group=plt.subplot(gs[1,0])
    ax_amp_hist=plt.subplot(gs[2,0])
    ax_vel_hist=plt.subplot(gs[3,0])
    ax_acc_hist=plt.subplot(gs[4,0])
    
    
    ax_gaze_x=plt.subplot(gs[0:2,1:4])
    ax_gaze_y=plt.subplot(gs[2:3,1:4])
    
    ax_screen=plt.subplot(gs[3:5,1:2])
    ax_PSO_ch1=plt.subplot(gs[3:5,2:3])
    ax_PSO_ch2=plt.subplot(gs[3:5,3:4])

    ax_hide=[]


    ##..........................ax_ang_group: angle selection
    

    
    plt.sca(ax_ang_group)
    labels = ['Hor','Ver', 'Obl']
    direction_histogram_values,XX=np.histogram(list(saccades['angle_group']), [0,1,2,3], normed=False)
    
    ax1_bars=ax_ang_group.bar(XX[:-1],direction_histogram_values,color='grey',width=XX[1]-XX[0],picker=True)
    
    def ax1_UpdateColors():    
        ax1_bars[0].set_color('green' if Angle_group[0] else 'grey')
        ax1_bars[1].set_color('green' if Angle_group[1] else 'grey')
        ax1_bars[2].set_color('green' if Angle_group[2] else 'grey')
        
    plt.title('direction')
    ax1_UpdateColors()
    plt.xticks( [0.5,1.5,2.5], labels, fontsize=8, rotation='vertical')
    plt.yticks( fontsize=8)
    
    
    ##..........................ax_amp_hist: Amplitude
    plt.sca(ax_amp_hist)
    n, bins, patches = plt.hist(list(saccades['amp_deg']), 50, normed=False, facecolor='gray', alpha=0.75)
    plt.title('amplitudes')
    plt.xticks( fontsize=8, rotation='vertical')
    plt.yticks( fontsize=8)
    plt.grid(True)
    ax_amp_hist_vspan=ax_amp_hist.axvspan(Amplitude_range[0], Amplitude_range[1], facecolor='g', alpha=0.5)

    
    ##..........................ax_vel_hist: Velocity
    plt.sca(ax_vel_hist)
    n, bins, patches = plt.hist(list(saccades['vel_av']), 50, normed=False, facecolor='gray', alpha=0.75)
    plt.title('velocities')
    plt.xticks( fontsize=8, rotation='vertical')
    plt.yticks( fontsize=8)
    plt.grid(True)
    ax_vel_hist_vspan=ax_vel_hist.axvspan(Velocity_range[0], Velocity_range[1], facecolor='g', alpha=0.5)
    
    
    ##..........................ax_acc_hist: acceleration
    plt.sca(ax_acc_hist)
    n, bins, patches = plt.hist(list(saccades['acc_av']), 50, normed=False, facecolor='gray', alpha=0.75)
    plt.title('Acceleration')
    plt.xticks( fontsize=8, rotation='vertical')
    plt.yticks( fontsize=8)
    plt.grid(True)
    ax_acc_hist_vspan=ax_acc_hist.axvspan(Acceleration_range[0], Acceleration_range[1], facecolor='g', alpha=0.5)
    
    
    
    
    
    
    ##..........................Selected Saccades + PSO Signals
    ##Preparing and Re-adjusting the subplots
    
    curves_ax_gaze_x=[0]*len(saccades)
    curves_ax_gaze_y=[0]*len(saccades)
    
    
    scatter_ax_screen=[0]*len(saccades)
    curves_ax_screen=[0]*len(saccades)
    text0_ax_screen=[0]*len(saccades)
    text1_ax_screen=[0]*len(saccades)
    
    
    curves_ax_PSO_ch1=[0]*len(saccades)
    scatter_ax_PSO_ch1=[0]*len(saccades)
    
    curve_lumped_ax_PSO_ch1=[]
    curve_model_ax_PSO_ch1=[]

    curves_ax_PSO_ch2=[0]*len(saccades)
    scatter_ax_PSO_ch2=[0]*len(saccades)
    curve_lumped_ax_PSO_ch2=[]
    curve_model_ax_PSO_ch2=[]
    
    c2='grey'
    alpha_highlight=1.0
    alpha_normal=0.4
    alpha_excluded=0.2  
    
    linewidth_highlight=2.0
    linewidth_normal=0.7
    linewidth_excluded=0.5  
    linestyle_normal='-'
    
    

    ##***********************************
    ##***********************************
    ##***********************************
    ##***********************************
    
    
    
    
    def exclude_saccade(axes, id):
        temp=False

        for object in axes.get_children():
            if object.get_gid()==id:
                temp=True               
                object.set_alpha(alpha_excluded)                
                object.set_color('red')
                if type(object)==plt.Line2D:
                    object.set_linewidth(linewidth_excluded)
                    object.set_linestyle(':')
        return temp      
    
    def include_saccade(axes,id):
        temp=False
        for object in axes.get_children():
            if object.get_gid()==id:
                temp=True     
                object.set_alpha(alpha_normal) 
                object.set_color(c2)#RGB_tuples[id])
                object.set_zorder(len(saccades))
                if type(object)==plt.Line2D:
                    object.set_linewidth(linewidth_normal)
                    object.set_linestyle(linestyle_normal)
#                
        return temp      
                
    def highlight_OFF(axes,id):     
        for object in axes.get_children():
            if object.get_gid()==id:
                object.set_alpha(alpha_normal) 
                object.set_color(c2)#RGB_tuples[id])
                if type(object)==plt.Line2D:                    
                    object.set_linewidth(linewidth_normal)
        return True
    
    def highlight_ON(axes,id):
        for object in axes.get_children():
            if object.get_gid()==id:                    
                object.set_alpha(alpha_highlight) 
                object.set_color(RGB_tuples[id])
                object.set_zorder(len(saccades))#on top
                if type(object)==plt.Line2D:
                    object.set_linewidth(linewidth_highlight)
#     
        return True 
      
    def Highlight(id,Force=False):
        global saccades
        if (not Force) & saccades.loc[id,'highlighted']==1:#already highlighted

            highlight_OFF(ax_gaze_x,id)
            highlight_OFF(ax_gaze_y,id)
            
            highlight_OFF(ax_screen,id)
            highlight_OFF(ax_PSO_ch1,id)
            highlight_OFF(ax_PSO_ch2,id)
            saccades.loc[id,'highlighted']=0
            
            print ('saccade %s highlight off' % id)
                    
        elif  saccades.loc[id,'highlighted']==0: #something else is already highlighted
            if (len(saccades.loc[saccades.highlighted==1,'highlighted'])>0):
                id_other=saccades.loc[saccades.highlighted==1,'highlighted'].iloc[0]
                highlight_OFF(ax_gaze_x,id_other)
                highlight_OFF(ax_gaze_y,id_other)
                
                highlight_OFF(ax_screen,id_other)
                highlight_OFF(ax_PSO_ch1,id_other)
                highlight_OFF(ax_PSO_ch2,id_other)     
                print ('saccade %s was highlighted before. that is off now' % id_other) 
            highlight_ON(ax_gaze_x,id)
            highlight_ON(ax_gaze_y,id)                
            highlight_ON(ax_screen,id)
            highlight_ON(ax_PSO_ch1,id)
            highlight_ON(ax_PSO_ch2,id)
            
            
            saccades.loc[id,'highlighted']=1
            print ('saccade %s highlight on' % id)
            
    def Exclude(id,Force=False):  
        global saccades
        if (not Force) & saccades.loc[id,'excluded']==1:## already excluded
            temp=include_saccade(ax_screen,id)
            include_saccade(ax_PSO_ch1,id) 
            include_saccade(ax_PSO_ch2,id)                 
            saccades.loc[id,'excluded']=0
            if temp:
                print ('saccade %s included' % id)
            
        else:
            if saccades.loc[id,'highlighted']==1:
                highlight_OFF(ax_gaze_x,id)
                highlight_OFF(ax_gaze_y,id)
                highlight_OFF(ax_screen,id)
                highlight_OFF(ax_PSO_ch1,id)
                highlight_OFF(ax_PSO_ch2,id)
                saccades.loc[id,'highlighted']=0
                print ('saccade %s highlight off' % id)
            

            saccades.loc[id,'excluded']=1
            temp=exclude_saccade(ax_screen, id)
            exclude_saccade(ax_PSO_ch1, id)
            exclude_saccade(ax_PSO_ch2, id)
            if temp:
                print ('saccade %s excluded' % id)    
    
        

    def Highlight_and_exclude():
        print('Highlight_and_exclude')
        for sac_ind, sac in saccades[saccades['excluded']==1].iterrows():
            print('excluding saccade %s'%sac.name)
            Exclude(sac.name,Force=True)
        
        if len(saccades[saccades['highlighted']==1])==1:
            indx=saccades[saccades['highlighted']==1].iloc[0].name
            print('highlighting saccade %s'%indx)
            Highlight(indx,Force=True)
    
    
    def LumpPSOs():
        global curve_lumped_ax_PSO_ch1,curve_lumped_ax_PSO_ch2,curve_model_ax_PSO_ch1,curve_model_ax_PSO_ch2
               
    
        ##calculating the average curve
    
        
        valid_saccades=saccades.dropna(subset=['PSO_ch1'])
        LumpPSOs_ch1=PSOVIS_tools.AverageCurves(valid_saccades.loc[valid_saccades['excluded']==0],PSO_SIGNAL_WINDOW,timestamp_interval,channel='PSO_ch1') 
        LumpPSOs_ch2=PSOVIS_tools.AverageCurves(valid_saccades.loc[valid_saccades['excluded']==0],PSO_SIGNAL_WINDOW,timestamp_interval,channel='PSO_ch2')
    
 
        curve_lumped_ax_PSO_ch1.set_data(LumpPSOs_ch1.index.tolist(),list(LumpPSOs_ch1['median']))
        curve_lumped_ax_PSO_ch2.set_data(LumpPSOs_ch2.index.tolist(),list(LumpPSOs_ch2['median']))
        
        if 'model' in LumpPSOs_ch1.columns.values:
            curve_model_ax_PSO_ch1.set_data(LumpPSOs_ch1.index.tolist(),list(LumpPSOs_ch1['model']))
        else:
            curve_model_ax_PSO_ch1.set_data(np.nan,np.nan)

        if 'model' in LumpPSOs_ch2.columns.values:
            curve_model_ax_PSO_ch2.set_data(LumpPSOs_ch2.index.tolist(),list(LumpPSOs_ch2['model']))
        else:
            curve_model_ax_PSO_ch2.set_data(np.nan,np.nan)

        
    #         print 'num of included saccades: ', len(channel_1_all)
    #         print '******** new mean: ', list(Av)
         
        
        return LumpPSOs_ch1,LumpPSOs_ch2
    
        
    
    
    
    def UpdateFigureTitle():       
        fig.canvas.set_window_title(str(participant + '_' +  PSOVIS_tools.SetNameForRangesFolder(Amplitude_range,Velocity_range,Acceleration_range,Angle_group)))   
    
            
    
    
        
    def Update():
        global saccades
        ## '', '','',''
        saccades.loc[:,'selected']=0
        saccades.loc[(saccades['angle_group_selected']==1) & (saccades['amp_deg_selected'] == 1) & (saccades['vel_av_selected'] == 1)& (saccades['acc_av_selected'] == 1),'selected']=1
        
        saccades.to_pickle(participant_table_file)   
        
        ##Saving the current ranges in the main folder so that the visualizer starts with those values next time
        PSOVIS_tools.SaveParams(folder_results+'params.npz', Amplitude_range,Velocity_range,Acceleration_range,Angle_group) 
        

        ##Update Figure title
        UpdateFigureTitle()

    #         print str(idx+1)+ "--> participant name: " + participant +'    |    total accepted saccades: ', len(saccades_selection)
        
        global scatter_ax_screen,curves_ax_screen,text0_ax_screen,text1_ax_screen,curves_ax_PSO_ch1,scatter_ax_PSO_ch1,curves_ax_PSO_ch2,scatter_ax_PSO_ch2,curve_lumped_ax_PSO_ch1,curve_lumped_ax_PSO_ch2,curve_model_ax_PSO_ch1,curve_model_ax_PSO_ch2
        global Gaze_x,Gaze_y,data
        global PSO_dict
        global ax_gaze_x,ax_gaze_y,ax_screen,ax_PSO_ch1,ax_PSO_ch2
        
        ##-----------------General updates in the plots
  

        PSOVIS_tools.PlotGazeChannel(ax_gaze_x,participant_data,Gaze_x[:],'Gaze_x')
        PSOVIS_tools.PlotGazeChannel(ax_gaze_y,participant_data,Gaze_y[:],'Gaze_y')
        
    
        plt.sca(ax_screen)
        ax_screen.cla()
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)  
        ax_screen.set_title('saccades in 2D',fontsize=9)
        ax_screen.set_aspect('equal', 'datalim')
    #         ax_screen.set_ylim(0,1000)
    #         ax_screen.set_xlim(0,1000)        
        
        plt.sca(ax_PSO_ch1)
        ax_PSO_ch1.cla()
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8) 
        ax_PSO_ch1.set_title('PSO signals ch1',fontsize=9)  
        ax_PSO_ch1.set_ylabel('Gaze',fontsize=8) 
        ax_PSO_ch1.set_ylim(-30,-30+Amplitude_range[1]) 
        ax_PSO_ch1.set_ylim(-30,50) 
        
        ax_PSO_ch1.set_xlim(PSO_SIGNAL_WINDOW[0],PSO_SIGNAL_WINDOW[1])
        ax_PSO_ch1.axvline(0, color='grey',linestyle='--',linewidth=0.5, alpha=0.5)
    
        curve_lumped_ax_PSO_ch1,=ax_PSO_ch1.plot([],[], 'k', linewidth=2,color='red',linestyle=linestyle_normal,alpha=0.6,zorder=len(saccades)+1)
        curve_model_ax_PSO_ch1,=ax_PSO_ch1.plot([],[], 'k', linewidth=2,color='blue',linestyle=linestyle_normal,alpha=0.6,zorder=len(saccades)+1)
      
        plt.sca(ax_PSO_ch2)
        ax_PSO_ch2.cla()
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8) 
        ax_PSO_ch2.set_title('PSO signals ch2',fontsize=9)
        ax_PSO_ch2.set_ylabel('Gaze',fontsize=8)
        ax_PSO_ch2.set_ylim(-30,-30+Amplitude_range[1])
        ax_PSO_ch2.set_ylim(-30,50) 
        ax_PSO_ch2.set_xlim(PSO_SIGNAL_WINDOW[0],PSO_SIGNAL_WINDOW[1])

        curve_lumped_ax_PSO_ch2,=ax_PSO_ch2.plot([],[], 'k', linewidth=2,color='red',linestyle=linestyle_normal,alpha=0.6,zorder=len(saccades)+1)
        curve_model_ax_PSO_ch2,=ax_PSO_ch2.plot([],[], 'k', linewidth=2,color='blue',linestyle=linestyle_normal,alpha=0.6,zorder=len(saccades)+1)
      
        
    
        valid_saccades=saccades.dropna(subset=['PSO_ch1'])

        valid_selected_saccades=valid_saccades[valid_saccades['selected']==1]
        for sac_ind, sac in valid_selected_saccades.iterrows():


            c=RGB_tuples[sac.name]
            
              

            ##..........................ax6: All Saccades

            curves_ax_gaze_x[sac.name],=ax_gaze_x.plot( TIMESTAMP[sac['start_row']:sac['end_row']+include_right],Gaze_x[sac['start_row']:sac['end_row']+include_right], linestyle_normal,linewidth=1.0,color=c,alpha=0.8,gid=sac.name,zorder=int(sac.name),picker=True) 

            ##........................main channel of the  Signal (X for horizontal saccades and Y for vertical saccades)
            

            plt.sca(ax_PSO_ch1)

            
#                 scatter_ax_PSO_ch1[sac.name],=plt.plot( xx,yy, 'o',markersize=1.0,color='black', alpha=alpha_normal,gid=sac.name,zorder=sac.name)
#                 curves_ax_PSO_ch1[sac.name],=plt.plot(xx,yy, 'k', linewidth=linewidth_normal,color=c2,alpha=alpha_normal,gid=sac.name,zorder=sac.name,picker=True)
#         

            ## color by velocity

            def floatRgb(mag, cmin, cmax):
                try:
                    # normalize to [0,1]
                    x = float(mag-cmin)/float(cmax-cmin)
                except:
                    # cmax = cmin
                    x = 0.5
                blue = min((max((4*(0.75-x), 0.)), 1.))
                red  = min((max((4*(x-0.25), 0.)), 1.))
                green= min((max((4*math.fabs(x-0.5)-1., 0.)), 1.))
                return (red, green, blue)  
            c3=floatRgb(sac['vel_av'],Velocity_range[0],Velocity_range[1])
            c3=c
            
            
            #c3=floatRgb(s['amplitude'],50,300)
            
            scatter_ax_PSO_ch1[sac.name],=plt.plot(sac['PSO_ch1'][:,0],sac['PSO_ch1'][:,1], 'o',markersize=0.5,color=c3, alpha=alpha_normal,gid=sac.name,zorder=sac.name)
            curves_ax_PSO_ch1[sac.name],=plt.plot(sac['PSO_ch1'][:,0],sac['PSO_ch1'][:,1], 'k', linewidth=linewidth_normal,color=c2,alpha=alpha_normal,gid=sac.name,zorder=sac.name,picker=True)


            ##........................channel 2
            
            plt.sca(ax_PSO_ch2)     
            
#             scatter_ax_PSO_ch2[sac.name],=plt.plot(sac['PSO_ch2'][:,0],sac['PSO_ch2'][:,1], 'o',markersize=0.5,color='black', alpha=alpha_normal,gid=sac.name,zorder=sac.name)
            curves_ax_PSO_ch2[sac.name],=plt.plot(sac['PSO_ch2'][:,0],sac['PSO_ch2'][:,1], 'k', linewidth=linewidth_normal,color=c2,alpha=alpha_normal,gid=sac.name,zorder=sac.name,picker=True)
    
            
            
            ##........................Selected Saccades
            plt.sca(ax_screen)
            
            x=Gaze_x[sac['start_row']-include_right:sac['end_row']+include_right]
            y=Gaze_y[sac['start_row']-include_right:sac['end_row']+include_right]
 
#             if sac.name==618:
#                 print(sac['start_row'],sac['end_row'])
#                 print(participant_data.loc[sac['start_row']:sac['end_row'],[   saccades.eye_tracked.iloc[0] +'_GAZE_X' ,   saccades.eye_tracked.iloc[0] +'_GAZE_Y',   saccades.eye_tracked.iloc[0] +'_IN_SACCADE']])

                
#             scatter_ax_screen[sac.name],=plt.plot(x, y,  'o',linewidth=0,markersize=1.0,color=c, alpha=0.0,gid=sac.name,zorder=sac.name)
            curves_ax_screen[sac.name],=plt.plot(x, y, 'k', linewidth=linewidth_normal,color=c,alpha=alpha_normal,gid=sac.name,zorder=sac.name,picker=True)
#             text0_ax_screen[sac.name]=plt.text(x[0], y[0],'0',fontsize=12,color=c,alpha=0.0,gid=sac.name,zorder=sac.name,picker=True)
#             text1_ax_screen[sac.name]=plt.text(x[-1], y[-1],'1',fontsize=12,color=c,alpha=0.0,gid=sac.name,zorder=sac.name,picker=True)        

#                 for t in targets_rect:
#                     ax_screen.add_patch(Rectangle((t[0], t[1]),t[2],t[3],fill=False,linewidth=0.3,linestyle='dashed'))
                                                        
      
  
        
        print(Amplitude_range,Velocity_range,Acceleration_range,Angle_group)
        print( '---total selected saccades %s/%s'%(len(valid_selected_saccades),len(valid_saccades)))
        PSO_ch1_lumped,PSO_ch2_lumped=LumpPSOs()   
        
        

        fig.canvas.draw()
        
        saccades.to_csv(folder_results + participant +  '_saccades.csv', sep='\t',index=True)
        PSO_ch1_lumped.to_csv(folder_results + participant +  '_PSOs.csv', sep='\t',index=True)
        PSO_ch2_lumped.to_csv(folder_results + participant +  '_PSOs.csv', sep='\t',index=True)

        fig.savefig(folder_results + participant + '.png')
        
        
        
    
    ##..........................Put all the matplotlib event handlers here
    
    ##....disable zoom and pan on the select-range figures
    def enter_axes(event):
        if event.inaxes==ax_ang_group or event.inaxes==ax_amp_hist or event.inaxes==ax_vel_hist or event.inaxes==ax_acc_hist:
            thismanager = get_current_fig_manager()
            if thismanager.toolbar.mode == 'zoom rect':
                thismanager.toolbar.zoom()
                print ("zoom rect disabled!")
                
            elif thismanager.toolbar.mode == 'pan/zoom':
                thismanager.toolbar.pan()
                print ("pan/zoom disabled")
    #             event.inaxes.patch.set_facecolor('yellow')
    

    
    def onpick(event):
        global Angle_group,saccades
        

        if isinstance(event.artist, Rectangle):
            patch = event.artist
            Angle_group[ int(patch.get_x()+0.5)]=not Angle_group[int( patch.get_x()+0.5)]
            print( 'selected directions updated to [Hor, Ver, Obl]=', Angle_group)
            saccades['angle_group_selected']=0
            if Angle_group[0]:
                saccades.loc[saccades['angle_group']==1,'angle_group_selected']=1
            if Angle_group[1]:
                saccades.loc[saccades['angle_group']==2,'angle_group_selected']=1
            if Angle_group[2]:
                saccades.loc[saccades['angle_group']==3,'angle_group_selected']=1    
            

        
        
            ax1_UpdateColors()
            Update()
            Highlight_and_exclude()

        
        if isinstance(event.artist, Line2D):
            id= int(event.artist.get_gid())
            print ('you have picked saccade number ', id)
            
            if (event.artist==curves_ax_gaze_x[id] or event.artist==curves_ax_screen[id] or event.artist==scatter_ax_screen[id]) and  (saccades.loc[id,'excluded']==0): # when you pick a saccade from ax_screen                        
                Highlight(id)               
            elif (event.artist==curves_ax_PSO_ch1[id] or event.artist==scatter_ax_PSO_ch1[id]) or (event.artist==curves_ax_PSO_ch2[id] or event.artist==scatter_ax_PSO_ch2[id]):
                Exclude(id)   
             
#             print(saccades.loc[id,:])
                
    
        fig.canvas.draw()
        
                    

        
    def onselect_ax_amp_hist(xmin, xmax):
        global Amplitude_range,saccades
        print ('on select ax_amp_hist')  
        ax_amp_hist_vspan.set_xy([[ xmin ,  0.  ],[xmin,  1.  ], [xmax,  1.  ], [ xmax,  0.  ]])
        Amplitude_range=(int(xmin),int(xmax))
        #Update ylim for ax_PSO_ch1 and ax_PSO_ch2 accordingly
        ax_PSO_ch1.set_ylim(-30,-30+Amplitude_range[1]) 
        ax_PSO_ch2.set_ylim(-30,-30+Amplitude_range[1]) 
        
        saccades.loc[:,'amp_deg_selected']=0
        saccades.loc[saccades['amp_deg'].between(xmin,xmax),'amp_deg_selected']=1
        Update()
        Highlight_and_exclude()
    
    def onselect_ax_vel_hist(xmin, xmax):
        global Velocity_range,saccades
        print ('on select ax_vel_hist') 
        ax_vel_hist_vspan.set_xy([[ xmin ,  0.  ],[xmin,  1.  ], [xmax,  1.  ], [ xmax,  0.  ]])
        Velocity_range=(int(xmin),int(xmax))
        
        saccades.loc[:,'vel_av_selected']=0
        saccades.loc[saccades['vel_av'].between(xmin,xmax),'vel_av_selected']=1
        Update()
        Highlight_and_exclude()
    def onselect_ax_acc_hist(xmin, xmax):
        global Acceleration_range,saccades
        print ('on select ax_acc_hist') 
        ax_acc_hist_vspan.set_xy([[ xmin ,  0.  ],[xmin,  1.  ], [xmax,  1.  ], [ xmax,  0.  ]])
        Acceleration_range=(int(xmin),int(xmax))
        saccades.loc[:,'acc_av_selected']=0
        saccades.loc[saccades['acc_av'].between(xmin,xmax),'acc_av_selected']=1
        Update()
        Highlight_and_exclude()

    ##***********************************
    ##***********************************
    ##***********************************
    ##***********************************
    
    span_ax_acc_hist=SpanSelector(ax_acc_hist, onselect_ax_acc_hist, 'horizontal', useblit=PSOVIS_tools.blit,rectprops=dict(alpha=0.5, facecolor='red'))
    span_ax_vel_hist=SpanSelector(ax_vel_hist, onselect_ax_vel_hist, 'horizontal', useblit=PSOVIS_tools.blit,rectprops=dict(alpha=0.5, facecolor='red'))
    span_ax_amp_hist=SpanSelector(ax_amp_hist, onselect_ax_amp_hist, 'horizontal', useblit=PSOVIS_tools.blit,rectprops=dict(alpha=0.5, facecolor='red'))
    
    
    MPL=fig.canvas.mpl_connect('pick_event', onpick)
    fig.canvas.mpl_connect('axes_enter_event', enter_axes)
    
  
 
        
    Update()
    Highlight_and_exclude()
   

    ##...............................................................................
    
    
    UpdateFigureTitle()
    fig.set_tight_layout(True)
    
    if Show_Visualizer:
        plt.show()
    
    
    
    end = time.time()
    print  ('elapsed time',  end - start)



    