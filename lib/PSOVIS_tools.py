# import matplotlib
# from math import atan2
import math
from skimage.external.tifffile.tifffile import astype
blit=False
from pylab import *
# import csv
import numpy as np
# import matplotlib.pyplot as plt
# 
from scipy.spatial import distance
# from itertools import cycle
from scipy.signal import argrelmax, argrelmin,argrelextrema
import matplotlib
# import matplotlib.gridspec as gridspec
# from matplotlib.widgets import *
# from matplotlib.patches import Rectangle
from collections import defaultdict
# import re
# import colorsys
import pandas as pd

# from collections import OrderedDict
# import math
from scipy.optimize import leastsq
# from pandas import Series, DataFrame
# 
# import numpy as np
# from pandas import Series, DataFrame
# import pandas as pd
import matplotlib.pyplot as plt
# import math as math
# from pylab import *
# from scipy.optimize import curve_fit

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
def Create_Folder(dir):
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



    ## pre-saccade fixation
    A_x=(np.nanmedian(Gaze_x[saccade['start_row']-fixation_window:saccade['start_row']]))
    A_y=(np.nanmedian(Gaze_y[saccade['start_row']-fixation_window:saccade['start_row']]))
    A=(A_x,A_y)

    ## define pivot point for data rotation (different than A)
    Piv_x=np.nanmedian(Gaze_x[saccade['start_row']-5:saccade['start_row']+5])
    Piv_y=np.nanmedian(Gaze_y[saccade['start_row']-5:saccade['start_row']+5])
    Piv=(Piv_x,Piv_y)
    
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
        return {'PSO_max_row':np.nan,'PSO_min_row':np.nan , 'amp_pix_ch1':np.nan, 'amp_deg':np.nan ,'channel_1':np.nan,'channel_2':np.nan,'gaze_at_start':np.nan,'gaze_at_end':np.nan}
    
    
    ## gaze data
    GAZE=Gaze_x[saccade['start_row']-fixation_window:saccade['end_row']+include_right].to_frame(name='gaze_x')
    GAZE['gaze_y']=Gaze_y[saccade['start_row']-fixation_window:saccade['end_row']+include_right]
    GAZE['TIMESTAMP']=TIMESTAMP[saccade['start_row']-fixation_window:saccade['end_row']+include_right]




    ## Rotate Gaze and add them to a new column
    GAZE['gaze_ch1']=GAZE.apply(lambda row: -(np.dot([row.gaze_x-Piv_x,row.gaze_y-Piv_y] , R.T)[0,0]+ Piv_x)  , axis=1)
    GAZE['gaze_ch2']=GAZE.apply(lambda row: -(np.dot([row.gaze_x-Piv_x,row.gaze_y-Piv_y] , R.T)[0,1]+ Piv_y) , axis=1)
    


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
    




    ## Align spatially
    GAZE['gaze_ch1']=GAZE.apply(lambda row: row.gaze_ch1+B_rotated[0]  , axis=1)
    GAZE['gaze_ch2']=GAZE.apply(lambda row: row.gaze_ch2+B_rotated[1] , axis=1)
 

    
    GAZE['time']=GAZE.apply(lambda row: row['TIMESTAMP']-GAZE.iloc[0]['TIMESTAMP']  , axis=1)
    

    
    if np.isnan(min_frame_index):
        GAZE['time2']= GAZE.time
    else:
        GAZE['time2']= GAZE.apply(lambda row: row.time - GAZE.loc[min_frame_index,'time'], axis=1)
    
    
    
    GAZE['nan']=np.nan
    

    
    ## prepare output 
    if np.isnan(min_frame_index): ## return nan if no min is found
        GAZE['CH1']=GAZE.loc[:,['time','nan']] .apply(tuple, axis=1)
        GAZE['CH2']=GAZE.loc[:,['time','nan']] .apply(tuple, axis=1)
   
    else:
        GAZE['CH1']=GAZE.loc[:,['time2','gaze_ch1']] .apply(tuple, axis=1)
        GAZE['CH2']=GAZE.loc[:,['time2','gaze_ch2']] .apply(tuple, axis=1)
        
    GAZE['CH1']= GAZE.apply(lambda row: list(row.CH1), axis=1)
    GAZE['CH2']= GAZE.apply(lambda row: list(row.CH2), axis=1)
    

   
    ## get PSO amp
    GAZE_slice=GAZE[GAZE.time2.between(0,15,inclusive=True)]

    
    PSO_max_row=GAZE_slice.gaze_ch1.idxmax()
    PSO_min_row=GAZE_slice.gaze_ch1.idxmin()
     
    
    PSO_amp= GAZE_slice.gaze_ch1.max()-GAZE_slice.gaze_ch1.min()

    PSO_amp_deg = pixels_to_degrees(PSO_amp)

    return {'PSO_max_row':PSO_max_row,'PSO_min_row':PSO_min_row , 'amp_pix_ch1':PSO_amp, 'amp_deg':PSO_amp_deg ,'channel_1':np.array(GAZE['CH1'].tolist()),'channel_2':np.array(GAZE['CH2'].tolist()),'gaze_at_start':A,'gaze_at_end':B}
    

# ExtractSaccades
def ExtractSaccades(name,file,exp,participant_group,delimiter,target_onset_msg,target_timeout_msg,include_right,offset_right,fixation_window):
    
    


    columns=['name','group','eye_tracked','reaction_time','amp_pix','amp_deg','vel_av','peak_vel','acc_av','pupil_size','angle','trial_index','start_row','end_row','PSO_ch1','PSO_ch2','PSO_amp_deg']

    
    
    participant_data = pd.read_csv(file, delimiter=delimiter, na_values=['.'], low_memory=True)
    
    
    participant_data=participant_data.replace(np.NaN, 0) # this also converts the columns data types to float if possible




    print('wait...')
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
    elif (lr) & (not re):
        trackerdEye=str(Eye[1])
            


    TIMESTAMP=participant_data['TIMESTAMP']

    Gaze_x=participant_data[trackerdEye +'_GAZE_X']
    Gaze_y=participant_data[trackerdEye +'_GAZE_Y']
    

    Gaze_x=Gaze_x.replace(0, np.nan)
    Gaze_y=Gaze_y.replace(0, np.nan)

 


    ## TODO_2: calculate velocity and acceleration if they don't exist in the file
    Vel_x=participant_data[trackerdEye +'_VELOCITY_X']
    Vel_y=participant_data[trackerdEye +'_VELOCITY_Y']

    Acc_x=participant_data[trackerdEye +'_ACCELERATION_X']
    Acc_y=participant_data[trackerdEye +'_ACCELERATION_Y']
    

    ## finding saccades
    ## TODO_1: add your own saccade detection here and add the _IN_SACCADE column to the table
    in_saccades=participant_data[trackerdEye +'_IN_SACCADE']


    
    starts= [ind+1 for ind, (a, b) in enumerate(zip(in_saccades, in_saccades[1:])) if a-b==-1]
    ends= [ind for ind, (a, b) in enumerate(zip(in_saccades, in_saccades[1:])) if a-b==1]


    
    ## process all saccades 
    TABLE_subject = pd.DataFrame( index =range(0,len(ends)),columns=columns)
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
            ci=np.array(starts)
            ci= np.bitwise_and((td-search_window)<ci,ci <td+search_window)
            try:# couldn't find the reason for the error
                cd=np.array(Gaze_x[np.array(starts)])-np.array(Gaze_x[np.array(ends)])
      
                cd[np.invert( ci)]=0
    
    
                ##Pick the longest one
                cd= abs(cd)
                xmax = argmax(cd)    
                RT=TIMESTAMP[starts[xmax]]-TIMESTAMP[td]
                TABLE_subject.loc[xmax,'reaction_time']=RT
            except:
                pass

    
    
    
    
            

    for s_i in range(len(ends)):

 
        saccade_gaze_x=Gaze_x[starts[s_i]:ends[s_i]]
        saccade_gaze_y=Gaze_y[starts[s_i]:ends[s_i]]

        
        saccade_vel_x=Vel_x[starts[s_i]:ends[s_i]]
        saccade_vel_y=Vel_y[starts[s_i]:ends[s_i]]

        saccade_acc_x=Acc_x[starts[s_i]:ends[s_i]]
        saccade_acc_y=Acc_y[starts[s_i]:ends[s_i]]


        TABLE_subject.loc[s_i,('name')]=name
        TABLE_subject.loc[s_i,('group')]=participant_group

        TABLE_subject.loc[s_i,('eye_tracked')]=trackerdEye


        ## amplitude
        if (np.isnan( saccade_gaze_x.iloc[0]) | np.isnan(saccade_gaze_y.iloc[0]) | np.isnan(saccade_gaze_x.iloc[len(saccade_gaze_x)-1]) | np.isnan(saccade_gaze_y.iloc[len(saccade_gaze_y)-1])):
            TABLE_subject.loc[s_i,('amp_pix')]=np.nan
        else:

            TABLE_subject.loc[s_i,('amp_pix')]=distance.euclidean([saccade_gaze_x.iloc[0],saccade_gaze_y.iloc[0]],[saccade_gaze_x.iloc[len(saccade_gaze_x)-1],saccade_gaze_y.iloc[len(saccade_gaze_y)-1]])
        
        ## Method1:
#         TABLE_subject.loc[s_i,('amp_deg')]=  pixels_to_degrees(TABLE_subject.loc[s_i,('amp_pix')])
        ## Method2:
        vel_norm=[ np.linalg.norm(vvv) for vvv in zip(saccade_vel_x,saccade_vel_y)]

        TABLE_subject.loc[s_i,('amp_deg')]= (TIMESTAMP[ends[s_i]] - TIMESTAMP[starts[s_i]] + 1.0)/1000 * np.nanmean(vel_norm)

        ##  vel    
        TABLE_subject.loc[s_i,('vel_av')]=np.nanmean(vel_norm)                         
        TABLE_subject.loc[s_i,('peak_vel')]=np.max(vel_norm)


        ## acc
        acc_norm=[ np.linalg.norm(vvv) for vvv in zip(saccade_acc_x,saccade_acc_y)]
        TABLE_subject.loc[s_i,('acc_av')]=np.nanmean(acc_norm)                         


        ## pupil size 
        if (trackerdEye +'_PUPIL_SIZE') in participant_data.columns:
            ps=participant_data[trackerdEye +'_PUPIL_SIZE']
            TABLE_subject.loc[s_i,('pupil_size')]=np.nanmean(ps[starts[s_i]:ends[s_i]])


        ## angle                                                  
        (Ax,Ay)=(np.nanmedian(Gaze_x[starts[s_i]:starts[s_i]+5]),np.nanmedian(Gaze_y[starts[s_i]:starts[s_i]+5]))
        (Bx,By)=(np.nanmedian(Gaze_x[ends[s_i]:ends[s_i]+5]),np.nanmedian(Gaze_y[ends[s_i]:ends[s_i]+5]))

        
        TABLE_subject.loc[s_i,('angle')]=CalculateAngle2((Ax,Ay),(Bx,By))

        def AngleGroup(a):
            if (0 <= a <=22 ) | (343<= a <=365) | (158<=a<=202):
                return 1
            elif (68 <= a <= 112) | (248 <= a <= 292) :
                return 2
            elif (22 < a < 68) | (112 < a < 158)| (202 < a < 248) | (292 <a<343) :
                return 3  
            else:
                pass
            
        TABLE_subject.loc[s_i,('angle_group')]=AngleGroup(CalculateAngle2((Ax,Ay),(Bx,By)))



        if 'TRIAL_INDEX' in participant_data.columns:
            TABLE_subject.loc[s_i,('trial_index')]= participant_data['TRIAL_INDEX'][starts[s_i]:ends[s_i]].iloc[0]

        


        ## end row and start row
        TABLE_subject.loc[s_i,('start_row')]= int(starts[s_i])
        TABLE_subject.loc[s_i,('end_row')]= int(ends[s_i])

        timestamp_interval=TIMESTAMP[2]-TIMESTAMP[1]


        PSO=GetPSO(TIMESTAMP,Gaze_x,Gaze_y,{'start_row':starts[s_i],'end_row':ends[s_i],'angle':TABLE_subject.loc[s_i,('angle')]},int(include_right//timestamp_interval),int(offset_right//timestamp_interval),int(fixation_window//timestamp_interval))


        ## method1
        TABLE_subject.loc[s_i,('PSO_amp_deg')]= PSO['amp_deg']

        ## method2
        if ('RESOLUTION_X' in participant_data.columns) and ('RESOLUTION_Y' in participant_data.columns):
            if not np.isnan( PSO['PSO_max_row']):              
                PSO_max_point=(participant_data.loc[PSO['PSO_max_row'],trackerdEye +'_GAZE_X'] ,participant_data.loc[PSO['PSO_max_row'],trackerdEye +'_GAZE_Y'])
                PSO_min_point=(participant_data.loc[PSO['PSO_min_row'],trackerdEye +'_GAZE_X'] ,participant_data.loc[PSO['PSO_min_row'],trackerdEye +'_GAZE_Y'])

                PSO.update({"amp_pix_in2D":  distance.euclidean(PSO_max_point,PSO_min_point)})## This may not be equal to PSO['amp_pix_ch1']
                TABLE_subject.loc[s_i,('PSO_amp_pix_in2D')]= PSO['amp_pix_in2D']

            ## according to the Eyelink manual 
                PSO_ang_x=(PSO_max_point[0]-PSO_min_point[0])/np.mean([participant_data.loc[PSO['PSO_max_row'] ,'RESOLUTION_X'],participant_data.loc[PSO['PSO_min_row'] ,'RESOLUTION_X']])
                PSO_ang_y=(PSO_max_point[1]-PSO_min_point[1])/np.mean([participant_data.loc[PSO['PSO_max_row'] ,'RESOLUTION_Y'],participant_data.loc[PSO['PSO_min_row'] ,'RESOLUTION_Y']])
                TABLE_subject.loc[s_i,('PSO_amp_deg')]=np.linalg.norm((PSO_ang_x,PSO_ang_y))
        
            
            
        TABLE_subject.loc[s_i,('PSO_amp_pix_ch1')]= PSO['amp_pix_ch1']

        
        gaze_at_start_all.append(PSO['gaze_at_start'])
        gaze_at_end_all.append(PSO['gaze_at_end'])
        PSOs_ch1.append(PSO['channel_1'])
        PSOs_ch2.append(PSO['channel_2'])


    


    TABLE_subject['PSO_ch1']= PSOs_ch1
    TABLE_subject['PSO_ch2']= PSOs_ch2
    TABLE_subject['A']= gaze_at_start_all
    TABLE_subject['B']= gaze_at_end_all
    

    
    print('%s saccades were not valid!' %(len(TABLE_subject)-len(TABLE_subject.dropna(subset=['PSO_ch1'])) ))

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


    