![alt text](https://github.com/dmardanbeigi/PSOVIS/blob/master/results/exp1/Vids_ad138fb.png)

# PSOVIS

This a python code for extracitng post saccadic oscillations from eye movement data. It's currenlty tested on SR Eyelink 1000 tracker but it should work with the eye data recorded from other trackers. 

ASSUMPTIONS:

- Only monocular data, having both eyes ("RIGHT_..." and "LEFT_...") or only one
- column titles in the data file:
+ Eye data: all eye-related columns should be started with either "RIGHT_..." OR "LEFT_...":
+ "RIGHT_GAZE_X","RIGHT_GAZE_Y","RIGHT_VELOCITY_X", "RIGHT_IN_SACCADE", "RIGHT_PUPIL_SIZE" (optional) , "SAMPLE_MESSAGE" (optional. set event MSGs in the code)
+ other columns:
+ "TIMESTAMP" (integer and measured in milisecond), 
- See the constant values used in the GetPSO function
- Change screen values in the pixels_to_degrees() to match your setup

TODO: (search for TODO name to find out where you can add your implementation)
- TODO_1: adding saccade and fixation detection in the absence of "_IN_SACCADE" 
- TODO_2: adding velocity and acceleration column if it's not provided in the file

# More details:
http://cogain2017.cogain.org/camready/talk6-Mardanbegi.pdf


# License

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
