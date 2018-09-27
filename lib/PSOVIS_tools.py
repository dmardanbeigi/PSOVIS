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
'''

import numpy as np



import pyximport
pyximport.install(setup_args={"include_dirs":np.get_include()})
from lib import cython_code

import math
# from skimage.external.tifffile.tifffile import astype
blit=False
from pylab import *


from scipy.spatial import distance
from scipy.signal import argrelmax, argrelmin,argrelextrema
import matplotlib
from collections import defaultdict
import pandas as pd
from scipy.optimize import leastsq
import matplotlib.pyplot as plt
import os

def W(s): 
    """This function returns the index of the last occurrence of '_' in the file name"""  
    last=0
    for i in re.finditer('_', s):
        last= i.end() #go until the last occurrence
    return last

def listdir_fullpath_dir(d): 
    """This function returns a list of tuples containing the name and path for all files that exist in folder d"""
    return [(f,os.path.join(d, f)) for f in os.listdir(d) if os.path.isdir(os.path.join(d, f))]


def listdir_fullpath_file(d):
    """This function returns a list of file names that exist in folder d"""
    return [f for f in os.listdir(d) if os.path.isfile(os.path.join(d, f))]

def DataFiles_in_folder(folder):
    return [[f[W(f[:-4]):-4],folder+f] for f in listdir_fullpath_file(folder) if (f[-3:]=='csv') ]

def pixels_to_degrees(x,distance_to_screen=55,resolution=(1680.0 ,1050.0),monitor_size=(61.0,53.0)):
    pixel_equalent_for_one_degree= distance_to_screen*math.tan(np.pi/180.0)*resolution[0]/monitor_size[0]
    out=np.array(x)/pixel_equalent_for_one_degree
    return out

def CalculateAngle(x,y):
    m,b = polyfit(x, y, 1)
    angle=math.degrees(np.arctan(m))
    return angle

def CalculateAngle2(p1,p2):
    angle=math.degrees(math.atan2(p2[1]-p1[1], p2[0]-p1[0]))
    if angle<0:
        angle=-angle
    else:
        angle=(360-angle)
    return angle


# GetPSO
def Create_Folder(dir,print_msg=True):
    if(print_msg):
        print ('creating result folder: '+ dir)   
    if not os.path.exists(dir):
        os.makedirs(dir)

def SetNameForRangesFolder(Amplitude_range,Velocity_range,Acceleration_range,Angle_range):
    return str('Amp' + str(Amplitude_range) +
             '_Vel'+ str(Velocity_range) + 
             '_Acc'+str(Acceleration_range)+
             'Angle' + str(Angle_range))


def PlotGazeChannel(ax,data,X,label):
    plt.sca(ax)
    ax.cla()
    ax.set_xlabel('time [ms]',fontsize=8)
    ax.set_ylabel(label ,fontsize=8)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)     
    ax.plot( data['TIMESTAMP'],X, '-',color='black',linewidth=0.2,alpha=0.8) 
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, pos: '{0:g}'.format((x/1000))))  
    
      
def LoadParams(file):
    Am =None
    V =None
    Ac=None
    D=None
    try: 
        params = np.load(file)              
        Am =params['Amplitude_range']
        V =params['Velocity_range']
        Ac=params['Acceleration_range']
        D=params['Angle_range']
        return True,[Am,V,Ac,D]
    except:
        os.remove(file) 
        print ('    >>>>>>>>>>>> ERROR LOADING PARAMS FILE <<<<<<<<<<<<<<<<    ')  
        return False,[Am,V,Ac,D]

def SaveParams(file, Amplitude_range,Velocity_range,Acceleration_range,Angle_range):
    np.savez(file, Amplitude_range=Amplitude_range,Velocity_range=Velocity_range,Acceleration_range=Acceleration_range,Angle_range=Angle_range)        
    
    
def FillGaps(b):
    b = np.array(b)
    mask = np.isnan(b)
    b[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), b[~mask])
    return(b)


def GetPSO(TIMESTAMP,Gaze_x,Gaze_y,saccade,include_right,offset_right,fixation_window):

    time_correction_multiplier=TIMESTAMP.diff().median()

    ## pre-saccade fixation
    A_x=(np.nanmedian(Gaze_x[saccade['start_row']-fixation_window:saccade['start_row']]))
    A_y=(np.nanmedian(Gaze_y[saccade['start_row']-fixation_window:saccade['start_row']]))
    A=(A_x,A_y)

    ## define pivot point for data rotation (different than A)
    Piv_x=np.nanmedian(Gaze_x[saccade['start_row']-5:saccade['start_row']+5])
    Piv_y=np.nanmedian(Gaze_y[saccade['start_row']-5:saccade['start_row']+5])
    Piv=[Piv_x,Piv_y]
    
    ## define rotation matrix
    ang=np.deg2rad(saccade['angle'])
    c, s = np.cos(ang), np.sin(ang)
    R = np.matrix([[c, -s], [s, c]])

    
    ## post_saccade fixation 
    B_x=np.nanmedian(Gaze_x[saccade['end_row']+offset_right:saccade['end_row']+offset_right+fixation_window])
    B_y=np.nanmedian(Gaze_y[saccade['end_row']+offset_right:saccade['end_row']+offset_right+fixation_window])
    B=(B_x,B_y)
    ## rotate point B (translate to pivot>rotate>translate back)
    tmp=np.array(np.dot([B_x-Piv_x , B_y-Piv_y], R.T))
    (B_x_r,B_y_r)=(tmp[0,:][0]+Piv_x,  tmp[0,:][1]+Piv_y)
    B_rotated=(B_x_r,B_y_r)
    
#     print(A,B,Piv,B_rotated)
    if np.isnan(ang) | np.isnan(Piv_x) | np.isnan(Piv_y) | np.isnan(A_x)|np.isnan(A_y)| np.isnan(B_x)|np.isnan(B_y):
#         print(Gaze_x[saccade['start_row']-5:saccade['end_row']+5])
        return {'Zero_row':np.nan, 'channel_1':np.nan,'channel_2':np.nan,'gaze_at_start':np.nan,'gaze_at_end':np.nan}
    
    
    ## gaze data
    GAZE=Gaze_x[saccade['start_row']-fixation_window:saccade['end_row']+include_right].to_frame(name='gaze_x')
    GAZE['gaze_y']=Gaze_y[saccade['start_row']-fixation_window:saccade['end_row']+include_right]
    GAZE['TIMESTAMP']=TIMESTAMP[saccade['start_row']-fixation_window:saccade['end_row']+include_right]


    gaze_ch1=[]
    gaze_ch2=[]
    
    
    
#     for row in GAZE.itertuples(index=True, name='Pandas'):
# #         print(R.T)
#         print(getattr(row,'gaze_x'))         
#         gaze_ch1.append(cython_code.Transfer(getattr(row,'gaze_x'), getattr(row,'gaze_y'),Piv,R.T))
#         gaze_ch2.append(cython_code.Transfer(getattr(row,'gaze_x'), getattr(row,'gaze_y'),Piv,R.T))
# # 
    gaze_ch1=cython_code.Transfer(list(GAZE.gaze_x),list(GAZE.gaze_y) ,Piv,[[c, -s], [s, c]],0)


    gaze_ch2=cython_code.Transfer(list(GAZE.gaze_x),list(GAZE.gaze_y) ,Piv,[[c, -s], [s, c]],1)


    GAZE['gaze_ch1']=gaze_ch1
    GAZE['gaze_ch2']=gaze_ch2

    ## Rotate Gaze and add them to a new column
#     GAZE['gaze_ch1']=GAZE.apply(lambda row: -(np.dot([row.gaze_x-Piv_x,row.gaze_y-Piv_y] , R.T)[0,0]+ Piv_x)  , axis=1)
#     GAZE['gaze_ch2']=GAZE.apply(lambda row: -(np.dot([row.gaze_x-Piv_x,row.gaze_y-Piv_y] , R.T)[0,1]+ Piv_y) , axis=1)
    
    

   ##.....Temporal alignment (Shifting the curves horizontally)
    
    
    min_frame_index=np.nan
    index_ignore_before= saccade['start_row'] + 0.2* (saccade['end_row']-saccade['start_row'])
    
    ##....................METHOD1 for temporal alignment
    mins=argrelextrema(FillGaps(GAZE['gaze_ch1'].values), np.less_equal,order=5)[0]
    
    mins=[GAZE.iloc[x].name for x in mins] ## convert min locations to original dataframe index
    

    for min_ind in mins:  
        if min_ind >index_ignore_before:## don't take the min if it's before the first 20% of each saccede. 
            min_frame_index =GAZE.loc[ min_ind].name 
            break
    
    
    ##....................METHOD2 for temporal alignment
    GAZE['gaze_ch1_diff']=GAZE['gaze_ch1'].diff()
    
    mins=GAZE.loc[index_ignore_before: ,:]## don't take the min if it's before the first 20% of each saccede.
    
    mins_type1_signal=mins[mins.gaze_ch1_diff>0]
    mins_type2_signal=mins[mins.gaze_ch1_diff>-0.2]## be less harsh
        
    if len(mins_type1_signal)>0:
        min_frame_index =mins_type1_signal.iloc[0].name
    elif len(mins_type2_signal)>0:
        min_frame_index =mins_type2_signal.iloc[0].name
    
     ##......................
    



    #Align Spatially
    # Redefine fixation based on pso min and not saccade end
        ## post_saccade fixation 
    B_x=np.nanmedian(Gaze_x[min_frame_index +offset_right:min_frame_index+offset_right+fixation_window])
    B_y=np.nanmedian(Gaze_y[min_frame_index +offset_right:min_frame_index+offset_right+fixation_window])
    B=(B_x,B_y)
    ## rotate point B (translate to pivot>rotate>translate back)
    tmp=np.array(np.dot([B_x-Piv_x , B_y-Piv_y], R.T))
    (B_x_r,B_y_r)=(tmp[0,:][0]+Piv_x,  tmp[0,:][1]+Piv_y)
    B_rotated=(B_x_r,B_y_r)

    t_0=GAZE.iloc[0]['TIMESTAMP']
    gaze_ch1=[]
    gaze_ch2=[]
    time=[]
    for row in GAZE.itertuples(index=True, name='Pandas'):     
        gaze_ch1.append(getattr(row,'gaze_ch1')+B_rotated[0] )
        gaze_ch2.append(getattr(row,'gaze_ch2')+B_rotated[1] )
        t1=getattr(row,'TIMESTAMP')-t_0
        time.append(t1)
    GAZE['gaze_ch1']=gaze_ch1


    ## Align spatially
    GAZE['gaze_ch2']=gaze_ch2
    GAZE['time']=time
    t_min= GAZE.loc[min_frame_index-1,'time']
 

    
    GAZE['time']=GAZE.apply(lambda row: row['TIMESTAMP']-GAZE.iloc[0]['TIMESTAMP']  , axis=1)
    

    
    time2=[]
    for row in GAZE.itertuples(index=True, name='Pandas'):
        if np.isnan(min_frame_index):
            time2.append(getattr(row,'time'))
        else:
            time2.append(getattr(row,'time')- t_min)
    GAZE['time2']=time2
    




    ## -------------------------  detect pso extrema
    ##-----------------------------------------------


    # # we limit the pso window to pso_period
    # pso_df_limited=pso_df[pso_df.time.between(pso_period[0],pso_period[1])]

    n=3 # number of points to be checked before and after 
    # Find local peaks
    GAZE['min'] = GAZE.iloc[argrelextrema(GAZE.gaze_ch1.values, np.less_equal, order=n)[0]]['gaze_ch1']
    GAZE['max'] = GAZE.iloc[argrelextrema(GAZE.gaze_ch1.values, np.greater_equal, order=n)[0]]['gaze_ch1']

    GAZE.loc[~GAZE['min'].isnull(),'extrema_type']= -1 # stands for minimum
    GAZE.loc[~GAZE['max'].isnull(),'extrema_type']= +1 # for maximum

    # plt.scatter(GAZE.time2, GAZE['min'], c='r')
    # plt.scatter(GAZE.time2, GAZE['max'], c='g')
    # plt.plot(GAZE.time2, GAZE['gaze_ch1'])
    # plt.show()



    
    ch1=[]
    ch2=[]
    for row in GAZE.itertuples(index=True, name='Pandas'):
        
        if np.isnan(min_frame_index): ## return nan if no min is found
            ch1.append([getattr(row,'time'),np.nan,np.nan] )
            ch2.append([getattr(row,'time'),np.nan ,np.nan] )
            print('no min found for pso for saccade %s'%saccade['saccade_index'])
        else:
            ch1.append([getattr(row,'time2'),getattr(row,'gaze_ch1'),getattr(row,'extrema_type')  ] )
            ch2.append([getattr(row,'time2'),getattr(row,'gaze_ch2'),getattr(row,'extrema_type') ] )
                        
            

    GAZE['CH1']=ch1
    GAZE['CH2']=ch2
        




    # GAZE.loc[~GAZE['extrema_type'].isnull(),['extrema_type','time','signal']].values.tolist()




    return {'Zero_row':GAZE[GAZE.time2==0].index.values[0] ,'channel_1':np.array(GAZE['CH1'].tolist()),'channel_2':np.array(GAZE['CH2'].tolist()),'gaze_at_start':A,'gaze_at_end':B}











def pixels_to_degrees_res_based(dx=None,dy=None,resxy_av=None):
    '''
    converting amplitude measured in pixels to degrees using res data according to EyeLink manual
    '''
    if type(dx)!=list:
        out=math.hypot(dx/resxy_av[0], dy/resxy_av[1]) 
    else:
        out=[]
        for x_i,x in dx:   
            out=math.hypot(x/resxy_av[x_i][0], dy[x_i]/resxy_av[x_i][1])
        out=np.array(out)
        
    return out


def ExtractSaccades(name,file,participant_group=None,delimiter='\t',target_onset_msg="",target_timeout_msg="",PSO_include_right=200,PSO_fixation_offset_right=40,PSO_fixation_duration=30):
    '''
    PSO_include_right [ms] how many ms after the end of the saccade should be included in the data
    PSO_fixation_offset_right [ms] how many ms after the first peak of the pso should be included in the data
    PSO_fixation_duration [ms] fixation defined by taking the median of PSO_fixation_duration ms


    '''



    columns=['name','group','eye_tracked','saccade_index',
             'reaction_time','amp_pix','amp_deg','vel_av','peak_vel','peak_vel_row','acc_av','pupil_size','angle',
             'trial_index','start_row','end_row',
             'PSO_ch1','PSO_ch2','PSO_RES','PSO_zero_row']

    
    print('loading data file ...')
    
    participant_data = pd.ExcelFile(file)
    participant_data = participant_data.parse(participant_data.sheet_names[0])
    participant_data=participant_data.replace('.', np.NaN)
    participant_data=participant_data.replace(np.NaN, 0) # this also converts the columns data types to float if possible
    participant_data=participant_data.apply(pd.to_numeric, errors='ignore')
   

   




    participant_data=participant_data.reset_index(drop=True)

    ## Determining which eye was tracked for this participant
    Eye=['RIGHT','LEFT']
    trackerdEye=''    

    re=('RIGHT_GAZE_X' in participant_data.columns)
    le=('LEFT_GAZE_X' in participant_data.columns)
    if re & le:        
        R=sum(participant_data['RIGHT_GAZE_X']!=0)
        L=sum(participant_data['LEFT_GAZE_X']!=0)
    
        if R>L:
            trackerdEye=str(Eye[0])
        else:
            trackerdEye=str(Eye[1])
    elif (re) & (not le):
        trackerdEye=str(Eye[0])
    elif (le) & (not re):
        trackerdEye=str(Eye[1])
            


    TIMESTAMP=participant_data['TIMESTAMP']

    Gaze_x=participant_data[trackerdEye +'_GAZE_X']
    Gaze_y=participant_data[trackerdEye +'_GAZE_Y']
    
    blink=participant_data[trackerdEye +'_IN_BLINK']

    res_x=participant_data['RESOLUTION_X']
    res_y=participant_data['RESOLUTION_Y']
    

    Gaze_x=Gaze_x.replace(0, np.nan)
    Gaze_y=Gaze_y.replace(0, np.nan)

    # print('adding vel column ... ')
    ## TODO_2: calculate velocity and acceleration if they don't exist in the file
    vel_col=[]
    for row in participant_data.itertuples(index=True, name='Pandas'):
        vel_col.append(np.linalg.norm([getattr(row, trackerdEye +'_VELOCITY_X'),getattr(row, trackerdEye +'_VELOCITY_Y')]))
    participant_data.loc[:,'vel']=vel_col
    Vel=participant_data.vel

    # participant_data.loc[:,'vel']=participant_data.apply(lambda x: np.linalg.norm([x[[trackerdEye +'_VELOCITY_X']],x[[trackerdEye +'_VELOCITY_Y']]]) , axis=1)
    # Vel=participant_data.vel



    Acc_x=participant_data[trackerdEye +'_ACCELERATION_X']
    Acc_y=participant_data[trackerdEye +'_ACCELERATION_Y']
    

    ## finding saccades
    ## TODO_1: add your own saccade detection here and add the _IN_SACCADE column to the table
    in_saccades=participant_data[trackerdEye +'_IN_SACCADE']




    saccade_start_end_indices=list(zip(list(participant_data[participant_data[str(trackerdEye +'_IN_SACCADE')].diff()==1].index.values),list(participant_data[participant_data[str(trackerdEye +'_IN_SACCADE')].diff()==-1].index.values-1)))


    
    ## process all saccades 
    TABLE_subject = pd.DataFrame( index =range(0,len(saccade_start_end_indices)),columns=columns)
    PSOs_ch1=[]
    PSOs_ch2=[]
    

    gaze_at_start_all=[]
    gaze_at_end_all=[]


    if ('SAMPLE_MESSAGE' in participant_data.columns) &  (target_onset_msg!="" ) & ( target_timeout_msg!=""):
        
        
        TABLE_subject.loc[:,'reaction_time']=np.nan
        ## reaction time (only sets RT for target saccades)
        rows_of_target_display_occurrence=participant_data.loc[participant_data['SAMPLE_MESSAGE']==target_onset_msg].index
        rows_of_Target_timeout=participant_data.loc[participant_data['SAMPLE_MESSAGE']==target_timeout_msg].index
    
        for i,td in enumerate(rows_of_target_display_occurrence):
            ##METHOD1: search on a window of w frames around the event for the longest saccade towards the target
            search_window=300   
            ci=np.array(saccade_start_end_indices[:,0])
            ci= np.bitwise_and((td-search_window)<ci,ci <td+search_window)
            try:# couldn't find the reason for the error
                cd=np.array(Gaze_x[np.array(saccade_start_end_indices[:,0])])-np.array(Gaze_x[np.array(saccade_start_end_indices[:,0])])
      
                cd[np.invert( ci)]=0
    
    
                ##Pick the longest one
                cd= abs(cd)
                xmax = argmax(cd)    
                RT=TIMESTAMP[saccade_start_end_indices[:,0][xmax]]-TIMESTAMP[td]
                TABLE_subject.loc[xmax,'reaction_time']=RT
            except:
                pass

    
    

    
    print('wait')
    for sac_i,sac in enumerate(saccade_start_end_indices):
#         print('saccade %s'%sac_i)
        TABLE_subject.loc[sac_i,'saccade_index']=participant_data.loc[sac[0]+1,trackerdEye +'_SACCADE_INDEX']
        

        saccade_gaze_x=Gaze_x[sac[0]:sac[1]]
        saccade_gaze_y=Gaze_y[sac[0]:sac[1]]

        
        saccade_vel=Vel[sac[0]:sac[1]]

        saccade_acc_x=Acc_x[sac[0]:sac[1]]
        saccade_acc_y=Acc_y[sac[0]:sac[1]]


#         if np.any(blink[sac[0]:sac[1]]==1):
#             print('saccades %s:%s:%s is a blink saccade'%(sac_i,participant_data.loc[sac[0]+1,trackerdEye +'_SACCADE_INDEX'],sac))
            
        ## ignore blink saccades
        if (len(saccade_gaze_x)==0) | (len(saccade_gaze_y)==0) | np.any(blink[sac[0]:sac[1]]==1):
            gaze_at_start_all.append((np.nan,np.nan))
            gaze_at_end_all.append((np.nan,np.nan))
            PSOs_ch1.append(np.nan)
            PSOs_ch2.append(np.nan)

            
            continue


        TABLE_subject.loc[sac_i,('name')]=name
        TABLE_subject.loc[sac_i,('group')]=participant_group

        TABLE_subject.loc[sac_i,('eye_tracked')]=trackerdEye

        
        ## amplitude

        if (np.isnan( saccade_gaze_x.iloc[0]) | np.isnan(saccade_gaze_y.iloc[0]) | np.isnan(saccade_gaze_x.iloc[len(saccade_gaze_x)-1]) | np.isnan(saccade_gaze_y.iloc[len(saccade_gaze_y)-1])):
            TABLE_subject.loc[sac_i,('amp_pix')]=np.nan
        else:

            TABLE_subject.loc[sac_i,('amp_pix')]=distance.euclidean([saccade_gaze_x.iloc[0],saccade_gaze_y.iloc[0]],[saccade_gaze_x.iloc[len(saccade_gaze_x)-1],saccade_gaze_y.iloc[len(saccade_gaze_y)-1]])

    
        ## angle                                                  
        (Ax,Ay)=(np.nanmedian(Gaze_x[sac[0]:sac[0]+5]),np.nanmedian(Gaze_y[sac[0]:sac[0]+5]))
        (Bx,By)=(np.nanmedian(Gaze_x[sac[1]:sac[1]+5]),np.nanmedian(Gaze_y[sac[1]:sac[1]+5]))

            
        RES_av=[np.mean([res_x.loc[sac[0]],res_x.loc[sac[1]]]),np.mean([res_y.loc[sac[0]],res_y.loc[sac[1]]])]
        saccade_amp=pixels_to_degrees_res_based(dx= Ax-Bx,dy= Ay-By,resxy_av=RES_av) 


   
        TABLE_subject.loc[sac_i,('amp_deg')]= saccade_amp # (TIMESTAMP[ends[s_i]] - TIMESTAMP[starts[s_i]] + 1.0)/1000 * np.nanmean(vel_norm) if len(vel_norm)>0 else np.nan

        ## saccade duration
        TABLE_subject.loc[sac_i,('duration')]= (TIMESTAMP[sac[1]] - TIMESTAMP[sac[0]] + 1.0)


        
        
        ##  vel    
        TABLE_subject.loc[sac_i,('vel_av')]=np.nanmean(saccade_vel)     if len(saccade_vel)>0 else np.nan                     
        TABLE_subject.loc[sac_i,('peak_vel')]=np.max(saccade_vel)
        TABLE_subject.loc[sac_i,('peak_vel_row')]=saccade_vel.idxmax()

        ## acc
        acc_norm=[ np.linalg.norm(vvv) for vvv in zip(saccade_acc_x,saccade_acc_y)]
        TABLE_subject.loc[sac_i,('acc_av')]=np.nanmean(acc_norm)         if len(acc_norm)>0 else np.nan                 


        ## pupil size 
        if (trackerdEye +'_PUPIL_SIZE') in participant_data.columns:
            ps=participant_data[trackerdEye +'_PUPIL_SIZE']
            TABLE_subject.loc[sac_i,('pupil_size')]=np.nanmean(ps[sac[0]:sac[1]])   if len(ps[sac[0]:sac[1]])>0 else np.nan



        
        TABLE_subject.loc[sac_i,('angle')]=CalculateAngle2((Ax,Ay),(Bx,By))

        def AngleGroup(a):
            if (0 <= a <=22 ) | (343<= a <=365) | (158<=a<=202):
                return 1
            elif (68 <= a <= 112) | (248 <= a <= 292) :
                return 2
            elif (22 < a < 68) | (112 < a < 158)| (202 < a < 248) | (292 <a<343) :
                return 3  
            else:
                pass
            
        
        TABLE_subject.loc[sac_i,('angle_group')]=AngleGroup(CalculateAngle2((Ax,Ay),(Bx,By)))

        

        if 'TRIAL_INDEX' in participant_data.columns:
            TABLE_subject.loc[sac_i,'trial_index']= participant_data['TRIAL_INDEX'][sac[0]:sac[1]].iloc[0]

        
        

        ## end row and start row
        TABLE_subject.loc[sac_i,('start_row')]= int(sac[0])
        TABLE_subject.loc[sac_i,('end_row')]= int(sac[1])

        timestamp_interval=TIMESTAMP[2]-TIMESTAMP[1]


        PSO=GetPSO(TIMESTAMP,Gaze_x,Gaze_y,
                   {'start_row':sac[0],'end_row':sac[1],'angle':TABLE_subject.loc[sac_i,('angle')]},
                   int(PSO_include_right//timestamp_interval),
                   int(PSO_fixation_offset_right//timestamp_interval),
                   int(PSO_fixation_duration//timestamp_interval))



        TABLE_subject.loc[sac_i,('PSO_RES')]= np.nan
        TABLE_subject.loc[sac_i,('PSO_zero_row')]= PSO['Zero_row']




        if ('RESOLUTION_X' in participant_data.columns) and ('RESOLUTION_Y' in participant_data.columns):
            if not np.isnan( PSO['Zero_row'] ):              
                RES=np.linalg.norm([res_x.loc[PSO['Zero_row']],res_y.loc[PSO['Zero_row']]])
                TABLE_subject.loc[sac_i,('PSO_RES')]=RES

                # TABLE_subject.loc[sac_i,('PSO_amp_deg')]=pixels_to_degrees_res_based(dx= PSO_max_point[0]-PSO_min_point[0],dy= PSO_max_point[1]-PSO_min_point[1],resxy_av=RES_av)



        
        gaze_at_start_all.append(PSO['gaze_at_start'])
        gaze_at_end_all.append(PSO['gaze_at_end'])
        PSOs_ch1.append(PSO['channel_1'])
        PSOs_ch2.append(PSO['channel_2'])


    
    TABLE_subject['PSO_ch1']= PSOs_ch1
    TABLE_subject['PSO_ch2']= PSOs_ch2

    TABLE_subject['A']= gaze_at_start_all
    TABLE_subject['B']= gaze_at_end_all
    

    not_valid=len(TABLE_subject)-len(TABLE_subject.dropna(subset=['PSO_ch1']))
    
    if not_valid==1:
        print('%s saccade was not valid!' %(not_valid ))
    elif not_valid>1:
        print('%s saccades were not valid among %s' %(not_valid,len(TABLE_subject) ))
                
    TABLE_subject=TABLE_subject.dropna(subset=['PSO_ch1'])#TABLE_subject
    TABLE_subject.reset_index( drop=True, inplace=True)
    
    return TABLE_subject


def UnderDamped(t,p):
    '''
    p[0]: amplitude
    p[1]: (\gamma), damping coefficient is <1 for Underdamped.  the exponential decay of the underdamped harmonic oscillator is given by \lambda =\omega_{0}\zeta . Q factor is related to the damping ratio by the equation Q=1/(2\gamma) 
    p[2]:\omega _{1}=sqrt(\omega _{0} ^{2}- \gamma^{2})
    p[3]: phase
    '''
    if len(p)!=4:
        print ('ERROR')
        return None
    else:
        return p[0] * np.exp(-p[1] * t) * np.cos(p[2] * t+p[3])
   
def Get_PSO_func(t,p):
    ''' 
    PUPIL:  p[1] * np.exp(-p[2] * t) * np.cos(p[3] * t+p[4])
    ISIR:   p[5] * np.exp(-p[6] * t) * np.cos(p[7] * t+p[8])
    ''' 
    
    ''' 
    Assuming both have the same phase
    PUPIL:  p[1] * np.exp(-p[2] * t) * np.cos(p[3] * t+p[4])
    ISIR:   p[5] * np.exp(-p[6] * t) * np.cos(p[7] * t-p[4])
    ''' 
#     p[8]=-p[4]
    
    ''' 
    No phase!
    PUPIL:  p[1] * np.exp(-p[2] * t) * np.cos(p[3] * t)
    ISIR:   p[5] * np.exp(-p[6] * t) * np.cos(p[7] * t)
    ''' 
    p[4]=p[8]=0
    
    ''' 
    PUPIL:  p[1] * np.exp(-p[2] * t) * np.cos(p[3] * t+p[4])
    ''' 
#     p[5]=p[6]=p[7]=p[8]=0
    
    PUPIL=UnderDamped(t,[p[1],p[2],p[3],p[4]])
    IRIS=UnderDamped(t,[p[5],p[6],p[7],p[8]])
    
    
    ## p[8]:offset from a stationary position
    PSO=p[0]+ PUPIL+IRIS
    return [PSO, PUPIL,IRIS]
    
def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx,array[idx] 
def ModelPSO(df):
    
    t2=arange(df.iloc[0].name,df.iloc[len(df)-1].name,0.01)
    
    def residuals(p,data,t):
        model=Get_PSO_func(t,p)
        err = data-model[0]
        return err
    
    minErr=100
    minParams=[]
    
    rn=np.linspace(0, 1, num=11)-0.5
    
    data=df['median']
    t=np.array(df.index.tolist()).astype(np.int32)
    
    for i in rn:      
        p0 = np.ones(9)*i#-0.1#0.1 # initial guesses
        
        pbest = leastsq(residuals,p0,args=(data,t),full_output=1)
        if sum(abs(residuals(pbest[0],data,t)))<minErr:
            minErr=sum(abs(residuals(pbest[0],data,t)))
            minParams=pbest[0]
            
            
    bestparams =minParams#pbest[0]
#         cov_x = pbest[1]
    print ('bestparams', bestparams)
     

#         if abs(bestparams[1])>1000:
#             continue
         
    if (len( bestparams)==0) | (minErr>60) :
        print ('Model fitting unsuccessful')
        model=np.nan
        t2=np.nan
        
    else:
        print ('best fit parameters:', [float('{:.2f}'.format(i)) for i in bestparams])
        print ('comp1 signal: %.2f e**(-%.2f t) cos(%.2f t + %.2f)'%(bestparams[1],bestparams[2],bestparams[3],bestparams[4]))
        print ('comp2 signal: %.2f e**(-%.2f t) cos(%.2f t + %.2f)'%(bestparams[5],bestparams[6],bestparams[7],bestparams[8]))


        model= Get_PSO_func(t2,bestparams)
    

        df['model']=df.apply(lambda row: model[0][ find_nearest(t2,  row.name)[0]]  , axis=1)
    
#     print(list(zip(t2,model[0])))
#     print(fff)
#     df.loc[:,'model']=
    
    return df
    
def AverageCurves(selected_saccades,SIGNAL_WINDOW,timestamp_interval,channel='PSO_ch1'):
    ## This function for each participants returns a vector that contains <mean> and <STD> for each time within SIGNAL_WINDOW.
    ## When the number of signals per each instant of time is less than 3, zero is returned for <mean> and <STD>
    
   
    
    w1=SIGNAL_WINDOW[1]+ SIGNAL_WINDOW[1]%2
    w0=SIGNAL_WINDOW[0]+SIGNAL_WINDOW[0]%2
    ind=np.array(range(0,int(w1-w0),int(timestamp_interval)))+w0
    


   
    df=pd.DataFrame(index=ind)
    

    
    for sac_ind, sac in selected_saccades.iterrows():
#         print(type(sac[channel]))
        for pair in sac[channel]:

            if (int(pair[0]) in df.index):

                df.loc[pair[0],sac.name]=pair[1]

    df['mean'] = df.mean(axis=1)
    df['median'] = df.median(axis=1)
    df['std'] = df.std(axis=1)
    df['min'] = df.min(axis=1)
    df['max'] = df.max(axis=1)


    df=ModelPSO(df)

   

    return df


    
