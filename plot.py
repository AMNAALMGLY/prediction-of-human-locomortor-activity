# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 20:29:29 2018

@author: MUHAMMED ALI
"""
import numpy as np
import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import time
from datetime import timedelta
from sklearn import metrics

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

from sklearn.utils import resample

plt.style.use('seaborn-whitegrid')
post = pd.read_csv('C:\\Users\\MUHAMMED ALI\\Desktop\\final year project\\ENABL3S\\AB156\\Raw\\AB156_Circuit_036_raw.csv')
plt.figure()


colors = ['pink','orange', 'turquoise' , 'green', 'goldenrod']
for i,c in zip([0,1,2,5,6],colors):
#for i in (0,1,3,4,6,7):
    mod_post_mode= post[post.Mode == i]
    mod_post_mode.Right_Knee = mod_post_mode.Right_Knee *(-1)
    mod_post_mode.Left_Knee = mod_post_mode.Left_Knee *(-1)

    #plt.subplot(121)

    plt.plot( y=mod_post_mode.Right_Shank_Gy ,linewidths=0.00000001 , c=c)
    plt.autoscale()
plt.title("Right_Shank_Ax")
plt.legend(['Sitting','Level Walking','Ramp Descent','Stair Ascent','SittingToWalking','WalkingToSitting'])
plt.show()
