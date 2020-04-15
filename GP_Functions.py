# -*- coding: utf-8 -*-
"""
Created on Wed May 23 15:03:19 2018

"""
#----------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

import itertools
import time
from datetime import timedelta
import pywt
from scipy import signal
from scipy import stats

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score ,GridSearchCV ,StratifiedKFold ,train_test_split , ShuffleSplit , StratifiedShuffleSplit ,learning_curve, RandomizedSearchCV
from sklearn import metrics

from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif

from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis , QuadraticDiscriminantAnalysis
from sklearn.decomposition import PCA

from keras.layers import Dense ,LSTM ,Flatten
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from sklearn import metrics
from sklearn.utils import resample

from statsmodels.tsa.ar_model import AR

from Denoising_functions import wden



#---------------------------------------------------------------------------------------------------------------------------------------------------------------------

def WTDenoise(df_raw, columns_to_denoise, wavelet = 'db7', plot=False , print_time=False) :
    
    df_den=df_raw.copy()
    for i in columns_to_denoise:
       
        start_time = time.monotonic()
        df_den.loc[:,i] = wden(df_den.loc[:,i],'sqtwolog','soft','one',4, wavelet)
        end_time = time.monotonic()
        
        if print_time == True :
            print('Denoising '+ i +' took : ' , timedelta(seconds=end_time - start_time))
        
        if plot == True:
            plt.figure()
            df_raw.loc[:,i].plot(color='r', alpha=0.5)
            df_den.loc[:,i].plot(color='b',alpha = 0.5)
            plt.show()

    rmse= np.sqrt(((df_raw - df_den)**2).mean())
    return df_den , rmse

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------

def BWDenoise(df_raw, columns_to_denoise, plot=False , print_time=False) :
    
    def butter_lowpass(lowcut, fs, order=6):
        nyq = 0.5 * fs
        low = lowcut / nyq
    
        b, a = signal.butter(order,low, btype='low')
        return b, a


    def butter_lowpass_filter(data, lowcut, fs=100, order=6):
        b, a = butter_lowpass(lowcut, fs, order=order)
        zi = signal.lfilter_zi(b, a)
        y,_ = signal.lfilter(b, a, data,zi=zi*data[0] )
        return y

    df_denoised= df_raw.copy()
    for column in columns_to_denoise:
        
        start_time = time.monotonic()
        
        if column == 'Right_Knee' and column == 'Left_Knee' :
            df_denoised[column]= butter_lowpass_filter(df_raw[column], 10)
        else:
            df_denoised[column]= butter_lowpass_filter(df_raw[column], 25)
            
            end_time = time.monotonic()
        
        if print_time == True :
            print('Denoising '+ column +' took : ' , timedelta(seconds=end_time - start_time))
            
        
        if plot == True:
            plt.figure()
            plt.subplot(2,1,1)
            plt.plot(df_raw[column], color='r', alpha=0.5)
            plt.subplot(2,1,2)
            plt.plot(df_denoised[column], color='b', alpha=0.5)
            plt.show()

    rmse= np.sqrt(((df_raw - df_denoised)**2).mean())
    return df_denoised , rmse

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
def Extract_Features(circuit , IMU_mean= False, IMU_min= False,IMU_max = False, IMU_STD=False, IMU_init=False, IMU_finl=False,
                    IMU_AR0= False, IMU_AR1= False, IMU_AR2= False, IMU_AR3= False, IMU_AR4= False, IMU_AR5= False, IMU_AR6= False,
                    IMUs_Fmean= False, IMUs_Fmedian= False, IMUs_MFmean= False, IMUs_MFmedian= False,
                    EMG_MAV= False, EMG_var= False, EMG_RMS= False, EMG_ZCs = False, EMG_WL= False, EMG_SSI= False,
                    EMG_IEMG= False, EMG_Fmean= False, EMG_Fmedian= False, EMG_MFmean= False, EMG_MFmedian= False, WT_cD_Mean= False, WT_cD_RMS= False,WT_cD_IWT= False,WT_cD_MAV= False,WT_cD_VAR= False,
                    WT_cD_ZCs = False, WT_D2_Mean = False, WT_D2_RMS  = False, WT_D2_IWT  = False, WT_D2_MAV  = False, WT_D2_VAR= False ,WT_D2_ZCs= False,
                    wavelet = 'db7', window=1500 , slide =1) :
                
    circuit_features_R = pd.DataFrame()
    circuit_features_L = pd.DataFrame()      
             
    for i in range(0,int(np.around(len(circuit)/(slide*window)))):
        
        if int(np.around((i+1)*window - i*(window-slide*window))) <= len(circuit) :
            
            chunk = pd.DataFrame()
            chunk = chunk.append(circuit.iloc[int(i*slide*window) : int(np.around((i+1)*window - i*(window-slide*window))) ,:] , ignore_index=True)

            features_R = pd.DataFrame()
            features_L = pd.DataFrame()
            
            # Mode
            Mode_mode = stats.mode(chunk.Our_Mode)
            if (Mode_mode[1]/window) >= 0.80 :
#                features_R.loc[i,'Mode']= Mode_mode[0]
                features_L.loc[i,'Mode']= Mode_mode[0]
        
                #------------------------------------------------------------------
                
                Right_IMUs = ['Right_Shank_Ax','Right_Shank_Ay','Right_Shank_Az','Right_Shank_Gx','Right_Shank_Gy','Right_Shank_Gz',
                              'Right_Thigh_Ax','Right_Thigh_Ay','Right_Thigh_Az','Right_Thigh_Gx','Right_Thigh_Gy','Right_Thigh_Gz',
                              'Right_Knee','Right_Ankle']
                
                Left_IMUs = ['Left_Shank_Ax','Left_Shank_Ay','Left_Shank_Az','Left_Shank_Gx','Left_Shank_Gy','Left_Shank_Gz',
                              'Left_Thigh_Ax','Left_Thigh_Ay','Left_Thigh_Az','Left_Thigh_Gx','Left_Thigh_Gy','Left_Thigh_Gz',
                              'Left_Knee','Left_Ankle']
                
                Right_EMGs = ['Right_TA','Right_MG','Right_SOL','Right_BF','Right_ST','Right_VL','Right_RF']
                
                Left_EMGs = ['Left_TA','Left_MG','Left_SOL','Left_BF','Left_ST','Left_VL','Left_RF']
                
                #------------------------------------------------------------------
                
                ###     IMUs Time domain
                
                if IMU_mean or IMU_min or IMU_max or IMU_STD or IMU_init or IMU_finl == True :
                
                        
                    mean_cols = ['Shank_Ax_Mean','Shank_Ay_Mean','Shank_Az_Mean','Shank_Gx_Mean','Shank_Gy_Mean','Shank_Gz_Mean',
                                 'Thigh_Ax_Mean','Thigh_Ay_Mean','Thigh_Az_Mean','Thigh_Gx_Mean','Thigh_Gy_Mean','Thigh_Gz_Mean',
                                 'Knee_Mean','Ankle_Mean']
                    min_cols = ['Shank_Ax_Min','Shank_Ay_Min','Shank_Az_Min','Shank_Gx_Min','Shank_Gy_Min','Shank_Gz_Min',
                                'Thigh_Ax_Min','Thigh_Ay_Min','Thigh_Az_Min','Thigh_Gx_Min','Thigh_Gy_Min','Thigh_Gz_Min',
                                'Knee_Min','Ankle_Min']
                    max_cols = ['Shank_Ax_Max','Shank_Ay_Max','Shank_Az_Max','Shank_Gx_Max','Shank_Gy_Max','Shank_Gz_Max',
                                'Thigh_Ax_Max','Thigh_Ay_Max','Thigh_Az_Max','Thigh_Gx_Max','Thigh_Gy_Max','Thigh_Gz_Max',
                                'Knee_Max','Ankle_Max']
                    std_cols = ['Shank_Ax_STD','Shank_Ay_STD','Shank_Az_STD','Shank_Gx_STD','Shank_Gy_STD','Shank_Gz_STD',
                                'Thigh_Ax_STD','Thigh_Ay_STD','Thigh_Az_STD','Thigh_Gx_STD','Thigh_Gy_STD','Thigh_Gz_STD',
                                'Knee_STD','Ankle_STD']
                    init_cols = ['Shank_Ax_Init','Shank_Ay_Init','Shank_Az_Init','Shank_Gx_Init','Shank_Gy_Init','Shank_Gz_Init',
                                 'Thigh_Ax_Init','Thigh_Ay_Init','Thigh_Az_Init','Thigh_Gx_Init','Thigh_Gy_Init','Thigh_Gz_Init',
                                 'Knee_Init','Ankle_Init']
                    finl_cols = ['Shank_Ax_Finl','Shank_Ay_Finl','Shank_Az_Finl','Shank_Gx_Finl','Shank_Gy_Finl','Shank_Gz_Finl',
                                 'Thigh_Ax_Finl','Thigh_Ay_Finl','Thigh_Az_Finl','Thigh_Gx_Finl','Thigh_Gy_Finl','Thigh_Gz_Finl',
                                 'Knee_Finl','Ankle_Finl']
                    
                
                
                    for mean_col ,min_col, max_col ,std_col, init_col ,finl_col, Right_IMU, Left_IMU in zip(
                        mean_cols,min_cols,max_cols,std_cols,init_cols,finl_cols,Right_IMUs,Left_IMUs) :
                        
                            #   Mean
                            if IMU_mean == True:
                                features_R.loc[i,'R_'+mean_col]= chunk[Right_IMU].mean()
                                features_L.loc[i,'L_'+mean_col]= chunk[Left_IMU].mean() 
                                
                            #   Minimum value
                            if IMU_min == True:    
                                features_R.loc[i,'R_'+min_col]= chunk[Right_IMU].min()
                                features_L.loc[i,'L_'+min_col]= chunk[Left_IMU].min()
                                
                            #   Maximum value
                            if IMU_max == True :
                                features_R.loc[i,'R_'+max_col]= chunk[Right_IMU].max()
                                features_L.loc[i,'L_'+max_col]= chunk[Left_IMU].max()
                            
                            #   Standard deviation
                            if IMU_STD == True:
                                features_R.loc[i,'R_'+std_col]= chunk[Right_IMU].std()
                                features_L.loc[i,'L_'+std_col]= chunk[Left_IMU].std()
                
                            #   Initial value
                            if IMU_init == True:
                                features_R.loc[i,'R_'+init_col]= chunk[Right_IMU][0]
                                features_L.loc[i,'L_'+init_col]= chunk[Left_IMU][0]
                
                            #   Final value
                            if IMU_finl == True :
                                features_R.loc[i,'R_'+finl_col]= chunk[Right_IMU][len(chunk)-1]
                                features_L.loc[i,'L_'+finl_col]= chunk[Left_IMU][len(chunk)-1]
                                
                #------------------------------------------------------------------
                
                ###     IMUs Autoregrissive coeficients
                
                if IMU_AR0 or IMU_AR1 or IMU_AR2 or IMU_AR3 or IMU_AR4 or IMU_AR5 or IMU_AR6 == True :
                    
                    cols =['Shank_Ax_AR','Shank_Ay_AR','Shank_Az_AR','Shank_Gx_AR','Shank_Gy_AR','Shank_Gz_AR',
                           'Thigh_Ax_AR','Thigh_Ay_AR','Thigh_Az_AR','Thigh_Gx_AR','Thigh_Gy_AR','Thigh_Gz_AR',
                           'Knee_AR','Ankle_AR']
                    
                    for col ,Right_IMU, Left_IMU in zip(
                        cols,Right_IMUs,Left_IMUs) :
                        
                        AR_R = AR(chunk[Right_IMU].values).fit(6).params
                        AR_L = AR(chunk[Left_IMU].values).fit(6).params
                        
                        for o in range(7):
                            features_R.loc[i,'R_'+col+str(o)]= AR_R[o]
                            features_L.loc[i,'L_'+col+str(o)]= AR_L[o]
                            
                            
                #------------------------------------------------------------------
                
                ###     IMUs , Frequency domain
                
                
                if IMUs_Fmean or IMUs_Fmedian or IMUs_MFmean or IMUs_MFmedian == True :
                    
                    Fmean_cols   = ['Shank_Ax_FMean','Shank_Ay_FMean','Shank_Az_FMean','Shank_Gx_FMean','Shank_Gy_FMean','Shank_Gz_FMean',
                                    'Thigh_Ax_FMean','Thigh_Ay_FMean','Thigh_Az_FMean','Thigh_Gx_FMean','Thigh_Gy_FMean','Thigh_Gz_FMean',
                                    'Knee_FMean','Ankle_FMean']
                    Fmedian_cols = ['Shank_Ax_FMedian','Shank_Ay_FMedian','Shank_Az_FMedian','Shank_Gx_FMedian','Shank_Gy_FMedian','Shank_Gz_FMedian',
                                    'Thigh_Ax_FMedian','Thigh_Ay_FMedian','Thigh_Az_FMedian','Thigh_Gx_FMedian','Thigh_Gy_FMedian','Thigh_Gz_FMedian',
                                    'Knee_FMedian','Ankle_FMedian']
                    MFmean_cols   = ['Shank_Ax_MFMean','Shank_Ay_MFMean','Shank_Az_MFMean','Shank_Gx_MFMean','Shank_Gy_MFMean','Shank_Gz_MFMean',
                                    'Thigh_Ax_MFMean','Thigh_Ay_MFMean','Thigh_Az_MFMean','Thigh_Gx_MFMean','Thigh_Gy_MFMean','Thigh_Gz_MFMean',
                                    'Knee_MFMean','Ankle_MFMean']
                    MFmedian_cols = ['Shank_Ax_MFMedian','Shank_Ay_MFMedian','Shank_Az_MFMedian','Shank_Gx_MFMedian','Shank_Gy_MFMedian','Shank_Gz_MFMedian',
                                    'Thigh_Ax_MFMedian','Thigh_Ay_MFMedian','Thigh_Az_MFMedian','Thigh_Gx_MFMedian','Thigh_Gy_MFMedian','Thigh_Gz_MFMedian',
                                    'Knee_MFMedian','Ankle_MFMedian']
                    
                    for Fmean_col, Fmedian_col, MFmean_col, MFmedian_col, Right_IMU, Left_IMU in zip(
                        Fmean_cols,Fmedian_cols,MFmean_cols,MFmedian_cols,Right_IMUs,Left_IMUs):
                        
                        #   Power Spectrum Density claculations:
                        freqs_R , PSD_R = signal.welch(chunk[Right_IMU], 1000,nperseg=window)
                        freqs_L , PSD_L = signal.welch(chunk[Left_IMU], 1000,nperseg=window)
                    
                        #   Frequency Mean
                        if IMUs_Fmean == True :
                            features_R.loc[i,'R_'+Fmean_col]= ((PSD_R * freqs_R).sum())/PSD_R.sum()
                            features_L.loc[i,'L_'+Fmean_col]= ((PSD_L * freqs_L).sum())/PSD_L.sum()
                    
                        #   Frequency Median
                        if IMUs_Fmedian == True :
                            features_R.loc[i,'R_'+Fmedian_col]= 0.5 * PSD_R.sum()
                            features_L.loc[i,'L_'+Fmedian_col]= 0.5 * PSD_L.sum()
                            
                        #   Modefied Frequency Mean
                        if IMUs_MFmean == True :
                            features_R.loc[i,'R_'+MFmean_col]= ((np.sqrt(PSD_R) * freqs_R).sum())/np.sqrt(PSD_R).sum()
                            features_L.loc[i,'L_'+MFmean_col]= ((np.sqrt(PSD_L) * freqs_L).sum())/np.sqrt(PSD_L).sum()
                    
                        #   Modefied Frequency Median
                        if IMUs_MFmedian == True :
                            features_R.loc[i,'R_'+MFmedian_col]= 0.5 * np.sqrt(PSD_R).sum()
                            features_L.loc[i,'L_'+MFmedian_col]= 0.5 * np.sqrt(PSD_L).sum()
           
                    
                #------------------------------------------------------------------
                
                ###     EMGs , Time domain
                
                if EMG_MAV or EMG_var or EMG_RMS or EMG_ZCs or EMG_WL or EMG_SSI or EMG_IEMG  == True :
                    mav_cols = ['TA_MAV','MG_MAV','SOL_MAV','BF_MAV','ST_MAV','VL_MAV','RF_MAV']
                    var_cols = ['TA_VAR','MG_VAR','SOL_VAR','BF_VAR','ST_VAR','VL_VAR','RF_VAR']
                    rms_cols = ['TA_RMS','MG_RMS','SOL_RMS','BF_RMS','ST_RMS','VL_RMS','RF_RMS']
                    zcs_cols = ['TA_ZCs','MG_ZCs','SOL_ZCs','BF_ZCs','ST_ZCs','VL_ZCs','RF_ZCs']
                    WL_cols  = ['TA_WL','MG_WL','SOL_WL','BF_WL','ST_WL','VL_WL','RF_WL']
                    ssi_cols = ['TA_SII','MG_SII','SOL_SII','BF_SII','ST_SII','VL_SII','RF_SII']
                    iemg_cols = ['TA_IEMG','MG_IEMG','SOL_IEMG','BF_IEMG','ST_IEMG','VL_IEMG','RF_IEMG']
                    
                    
                    for mav_col ,var_col, rms_col, zcs_col, WL_col, ssi_col, iemg_col, Right_EMG , Left_EMG in zip(
                        mav_cols,var_cols,rms_cols,zcs_cols,WL_cols,ssi_cols,iemg_cols,Right_EMGs,Left_EMGs):
                        
                        # Mean abdolute value
                        if EMG_MAV == True:
                            features_R.loc[i,'R_'+mav_col]= chunk[Right_EMG].abs().mean()
                            features_L.loc[i,'L_'+mav_col]= chunk[Left_EMG].abs().mean()
                
                        # Variance
                        if EMG_var == True :
                            features_R.loc[i,'R_'+var_col]= chunk[Right_EMG].var()
                            features_L.loc[i,'L_'+var_col]= chunk[Left_EMG].var()
                            
                
                        # Root Mean Square
                        if EMG_RMS == True :
                            features_R.loc[i,'R_'+rms_col]= np.sqrt((chunk[Right_EMG]**2).mean())
                            features_L.loc[i,'L_'+rms_col]= np.sqrt((chunk[Left_EMG]**2).mean())
                
                        #   Number of zero crossings
                        if EMG_ZCs == True :
                            features_R.loc[i,'R_'+zcs_col]= np.sum(np.diff(np.sign(chunk[Right_EMG])[np.sign(chunk[Right_EMG])!=0])!= 0)
                            features_L.loc[i,'L_'+zcs_col]= np.sum(np.diff(np.sign(chunk[Left_EMG])[np.sign(chunk[Left_EMG])!=0])!= 0)
                
                        #   Waveform length
                        if EMG_WL == True :
                            chunk2 = pd.DataFrame()
                            chunk2 = chunk2.append(chunk[[Right_EMG,Left_EMG]], ignore_index=True)
                            chunk2 = chunk2.iloc[1:,:]
                            chunk2 = chunk2.append(chunk2.iloc[-1,:] , ignore_index=True)
                    
                            features_R.loc[i,'R_'+WL_col]= (chunk2[Right_EMG] - chunk[Right_EMG]).abs().sum()
                            features_L.loc[i,'L_'+WL_col]= (chunk2[Left_EMG] - chunk[Left_EMG]).abs().sum()
    
                
                        #   Simple Square Integral
                        if EMG_SSI == True :
                            features_R.loc[i,'R_'+ssi_col]= (chunk[Right_EMG].abs()**2).sum()
                            features_L.loc[i,'L_'+ssi_col]= (chunk[Left_EMG].abs()**2).sum()
                
                        #   Integrated EMG
                        if EMG_IEMG == True :
                            features_R.loc[i,'R_'+iemg_col]= chunk[Right_EMG].abs().sum()
                            features_L.loc[i,'L_'+iemg_col]= chunk[Left_EMG].abs().sum()
                    
                
                #------------------------------------------------------------------
                
                ###     EMGs , Frequency domain
                
                
                if EMG_Fmean or EMG_Fmedian or EMG_MFmean or EMG_MFmedian == True :
                    
                    fmean_cols   = ['TA_FMean','MG_FMean','SOL_FMean','BF_FMean','ST_FMean','VL_FMean','RF_FMean']
                    fmedian_cols = ['TA_FMedian','MG_FMedian','SOL_FMedian','BF_FMedian','ST_FMedian','VL_FMedian','RF_FMedian']
                    mfmean_cols   = ['TA_MFMean','MG_MFMean','SOL_MFMean','BF_MFMean','ST_MFMean','VL_MFMean','RF_MFMean']
                    mfmedian_cols = ['TA_MFMedian','MG_MFMedian','SOL_MFMedian','BF_MFMedian','ST_MFMedian','VL_MFMedian','RF_MFMedian']
                    
                    for fmean_col, fmedian_col, mfmean_col, mfmedian_col, Right_EMG, Left_EMG in zip(
                        fmean_cols,fmedian_cols,mfmean_cols,mfmedian_cols,Right_EMGs,Left_EMGs):
                        
                        #   Power Spectrum Density claculations:
                        freqs_R , PSD_R = signal.welch(chunk[Right_EMG], 1000,nperseg=window)
                        freqs_L , PSD_L = signal.welch(chunk[Left_EMG], 1000,nperseg=window)
                    
                        #   Frequency Mean
                        if EMG_Fmean == True :
                            features_R.loc[i,'R_'+fmean_col]= ((PSD_R * freqs_R).sum())/PSD_R.sum()
                            features_L.loc[i,'L_'+fmean_col]= ((PSD_L * freqs_L).sum())/PSD_L.sum()
                    
                        #   Frequency Median
                        if EMG_Fmedian == True :
                            features_R.loc[i,'R_'+fmedian_col]= 0.5 * PSD_R.sum()
                            features_L.loc[i,'L_'+fmedian_col]= 0.5 * PSD_L.sum()
                            
                        #   Modefied Frequency Mean
                        if EMG_MFmean == True :
                            features_R.loc[i,'R_'+mfmean_col]= ((np.sqrt(PSD_R) * freqs_R).sum())/np.sqrt(PSD_R).sum()
                            features_L.loc[i,'L_'+mfmean_col]= ((np.sqrt(PSD_L) * freqs_L).sum())/np.sqrt(PSD_L).sum()
                    
                        #   Modefied Frequency Median
                        if EMG_MFmedian == True :
                            features_R.loc[i,'R_'+mfmedian_col]= 0.5 * np.sqrt(PSD_R).sum()
                            features_L.loc[i,'L_'+mfmedian_col]= 0.5 * np.sqrt(PSD_L).sum()
           
                #------------------------------------------------------------------
                
                ###     EMGs , Wavelet
                
                
                if WT_cD_Mean or WT_cD_RMS or WT_cD_IWT or WT_cD_MAV or WT_cD_VAR or WT_cD_ZCs or WT_D2_Mean or WT_D2_RMS or WT_D2_IWT or WT_D2_MAV or WT_D2_VAR or WT_D2_ZCs == True:
                    
                    ###-----------------------------------------------
                    def wrcoef(coef_type, X, coeffs, wavename, level):
                        N = np.array(X).size
                        a, ds = coeffs[0], list(reversed(coeffs[1:]))
                        if coef_type =='a':
                            return pywt.upcoef('a', a, wavename, level=level)[:N]
                        elif coef_type == 'd':
                            return pywt.upcoef('d', ds[level-1], wavename, level=level)[:N]
                        else:
                            raise ValueError("Invalid coefficient type: {}".format(coef_type))
                    #-------------------------------------------------
                    
                    
                    cD4_mean_cols = ['TA_WT_cD4_Mean', 'MG_WT_cD4_Mean', 'SOL_WT_cD4_Mean', 'BF_WT_cD4_Mean', 'ST_WT_cD4_Mean', 'VL_WT_cD4_Mean', 'RF_WT_cD4_Mean']
                    cD3_mean_cols = ['TA_WT_cD3_Mean', 'MG_WT_cD3_Mean', 'SOL_WT_cD3_Mean', 'BF_WT_cD3_Mean', 'ST_WT_cD3_Mean', 'VL_WT_cD3_Mean', 'RF_WT_cD3_Mean']
                    cD2_mean_cols = ['TA_WT_cD2_Mean', 'MG_WT_cD2_Mean', 'SOL_WT_cD2_Mean', 'BF_WT_cD2_Mean', 'ST_WT_cD2_Mean', 'VL_WT_cD2_Mean', 'RF_WT_cD2_Mean']
                    cD1_mean_cols = ['TA_WT_cD1_Mean', 'MG_WT_cD1_Mean', 'SOL_WT_cD1_Mean', 'BF_WT_cD1_Mean', 'ST_WT_cD1_Mean', 'VL_WT_cD1_Mean', 'RF_WT_cD1_Mean']
                    cD4_rms_cols = ['TA_WT_cD4_RMS', 'MG_WT_cD4_RMS', 'SOL_WT_cD4_RMS', 'BF_WT_cD4_RMS', 'ST_WT_cD4_RMS', 'VL_WT_cD4_RMS', 'RF_WT_cD4_RMS']
                    cD3_rms_cols = ['TA_WT_cD3_RMS', 'MG_WT_cD3_RMS', 'SOL_WT_cD3_RMS', 'BF_WT_cD3_RMS', 'ST_WT_cD3_RMS', 'VL_WT_cD3_RMS', 'RF_WT_cD3_RMS']
                    cD2_rms_cols = ['TA_WT_cD2_RMS', 'MG_WT_cD2_RMS', 'SOL_WT_cD2_RMS', 'BF_WT_cD2_RMS', 'ST_WT_cD2_RMS', 'VL_WT_cD2_RMS', 'RF_WT_cD2_RMS']
                    cD1_rms_cols = ['TA_WT_cD1_RMS', 'MG_WT_cD1_RMS', 'SOL_WT_cD1_RMS', 'BF_WT_cD1_RMS', 'ST_WT_cD1_RMS', 'VL_WT_cD1_RMS', 'RF_WT_cD1_RMS']
                    cD4_iwt_cols = ['TA_WT_cD4_IWT', 'MG_WT_cD4_IWT', 'SOL_WT_cD4_IWT', 'BF_WT_cD4_IWT', 'ST_WT_cD4_IWT', 'VL_WT_cD4_IWT', 'RF_WT_cD4_IWT']
                    cD3_iwt_cols = ['TA_WT_cD3_IWT', 'MG_WT_cD3_IWT', 'SOL_WT_cD3_IWT', 'BF_WT_cD3_IWT', 'ST_WT_cD3_IWT', 'VL_WT_cD3_IWT', 'RF_WT_cD3_IWT']
                    cD2_iwt_cols = ['TA_WT_cD2_IWT', 'MG_WT_cD2_IWT', 'SOL_WT_cD2_IWT', 'BF_WT_cD2_IWT', 'ST_WT_cD2_IWT', 'VL_WT_cD2_IWT', 'RF_WT_cD2_IWT']
                    cD1_iwt_cols = ['TA_WT_cD1_IWT', 'MG_WT_cD1_IWT', 'SOL_WT_cD1_IWT', 'BF_WT_cD1_IWT', 'ST_WT_cD1_IWT', 'VL_WT_cD1_IWT', 'RF_WT_cD1_IWT']
                    cD4_mav_cols = ['TA_WT_cD4_MAV', 'MG_WT_cD4_MAV', 'SOL_WT_cD4_MAV', 'BF_WT_cD4_MAV', 'ST_WT_cD4_MAV', 'VL_WT_cD4_MAV', 'RF_WT_cD4_MAV']
                    cD3_mav_cols = ['TA_WT_cD3_MAV', 'MG_WT_cD3_MAV', 'SOL_WT_cD3_MAV', 'BF_WT_cD3_MAV', 'ST_WT_cD3_MAV', 'VL_WT_cD3_MAV', 'RF_WT_cD3_MAV']
                    cD2_mav_cols = ['TA_WT_cD2_MAV', 'MG_WT_cD2_MAV', 'SOL_WT_cD2_MAV', 'BF_WT_cD2_MAV', 'ST_WT_cD2_MAV', 'VL_WT_cD2_MAV', 'RF_WT_cD2_MAV']
                    cD1_mav_cols = ['TA_WT_cD1_MAV', 'MG_WT_cD1_MAV', 'SOL_WT_cD1_MAV', 'BF_WT_cD1_MAV', 'ST_WT_cD1_MAV', 'VL_WT_cD1_MAV', 'RF_WT_cD1_MAV']
                    cD4_var_cols = ['TA_WT_cD4_VAR', 'MG_WT_cD4_VAR', 'SOL_WT_cD4_VAR', 'BF_WT_cD4_VAR', 'ST_WT_cD4_VAR', 'VL_WT_cD4_VAR', 'RF_WT_cD4_VAR']
                    cD3_var_cols = ['TA_WT_cD3_VAR', 'MG_WT_cD3_VAR', 'SOL_WT_cD3_VAR', 'BF_WT_cD3_VAR', 'ST_WT_cD3_VAR', 'VL_WT_cD3_VAR', 'RF_WT_cD3_VAR']
                    cD2_var_cols = ['TA_WT_cD2_VAR', 'MG_WT_cD2_VAR', 'SOL_WT_cD2_VAR', 'BF_WT_cD2_VAR', 'ST_WT_cD2_VAR', 'VL_WT_cD2_VAR', 'RF_WT_cD2_VAR']
                    cD1_var_cols = ['TA_WT_cD1_VAR', 'MG_WT_cD1_VAR', 'SOL_WT_cD1_VAR', 'BF_WT_cD1_VAR', 'ST_WT_cD1_VAR', 'VL_WT_cD1_VAR', 'RF_WT_cD1_VAR']
                    cD4_zcs_cols = ['TA_WT_cD4_ZCs', 'MG_WT_cD4_ZCs', 'SOL_WT_cD4_ZCs', 'BF_WT_cD4_ZCs', 'ST_WT_cD4_ZCs', 'VL_WT_cD4_ZCs', 'RF_WT_cD4_ZCs']
                    cD3_zcs_cols = ['TA_WT_cD3_ZCs', 'MG_WT_cD3_ZCs', 'SOL_WT_cD3_ZCs', 'BF_WT_cD3_ZCs', 'ST_WT_cD3_ZCs', 'VL_WT_cD3_ZCs', 'RF_WT_cD3_ZCs']
                    cD2_zcs_cols = ['TA_WT_cD2_ZCs', 'MG_WT_cD2_ZCs', 'SOL_WT_cD2_ZCs', 'BF_WT_cD2_ZCs', 'ST_WT_cD2_ZCs', 'VL_WT_cD2_ZCs', 'RF_WT_cD2_ZCs']
                    cD1_zcs_cols = ['TA_WT_cD1_ZCs', 'MG_WT_cD1_ZCs', 'SOL_WT_cD1_ZCs', 'BF_WT_cD1_ZCs', 'ST_WT_cD1_ZCs', 'VL_WT_cD1_ZCs', 'RF_WT_cD1_ZCs']
                    D2_mean_cols = ['TA_WT_D4_Mean', 'MG_WT_D4_Mean', 'SOL_WT_D4_Mean', 'BF_WT_D4_Mean', 'ST_WT_D4_Mean', 'VL_WT_D4_Mean', 'RF_WT_D4_Mean']
                    D2_rms_cols = ['TA_WT_D4_RMS', 'MG_WT_D4_RMS', 'SOL_WT_D4_RMS', 'BF_WT_D4_RMS', 'ST_WT_D4_RMS', 'VL_WT_D4_RMS', 'RF_WT_D4_RMS']
                    D2_iwt_cols = ['TA_WT_D4_IWT', 'MG_WT_D4_IWT', 'SOL_WT_D4_IWT', 'BF_WT_D4_IWT', 'ST_WT_D4_IWT', 'VL_WT_D4_IWT', 'RF_WT_D4_IWT']
                    D2_mav_cols = ['TA_WT_D4_MAV', 'MG_WT_D4_MAV', 'SOL_WT_D4_MAV', 'BF_WT_D4_MAV', 'ST_WT_D4_MAV', 'VL_WT_D4_MAV', 'RF_WT_D4_MAV']
                    D2_var_cols = ['TA_WT_D4_VAR', 'MG_WT_D4_VAR', 'SOL_WT_D4_VAR', 'BF_WT_D4_VAR', 'ST_WT_D4_VAR', 'VL_WT_D4_VAR', 'RF_WT_D4_VAR']
                    D2_zcs_cols = ['TA_WT_D4_ZCs', 'MG_WT_D4_ZCs', 'SOL_WT_D4_ZCs', 'BF_WT_D4_ZCs', 'ST_WT_D4_ZCs', 'VL_WT_D4_ZCs', 'RF_WT_D4_ZCs']
                    
                    for cD4_mean_col, cD3_mean_col, cD2_mean_col, cD1_mean_col, cD4_rms_col, cD3_rms_col, cD2_rms_col, cD1_rms_col, cD4_iwt_col, cD3_iwt_col, cD2_iwt_col, cD1_iwt_col, cD4_mav_col, cD3_mav_col, cD2_mav_col, cD1_mav_col, cD4_var_col, cD3_var_col, cD2_var_col, cD1_var_col, cD4_zcs_col, cD3_zcs_col, cD2_zcs_col, cD1_zcs_col ,D2_mean_col, D2_rms_col, D2_iwt_col, D2_mav_col, D2_var_col, D2_zcs_col, Right_EMG, Left_EMG in zip(
                        cD4_mean_cols,cD3_mean_cols,cD2_mean_cols,cD1_mean_cols,cD4_rms_cols,cD3_rms_cols,cD2_rms_cols,cD1_rms_cols,cD4_iwt_cols,cD3_iwt_cols,cD2_iwt_cols,cD1_iwt_cols,cD4_mav_cols,cD3_mav_cols,cD2_mav_cols,cD1_mav_cols,cD4_var_cols,cD3_var_cols,cD2_var_cols,cD1_var_cols,cD4_zcs_cols,cD3_zcs_cols,cD2_zcs_cols,cD1_zcs_cols,D2_mean_cols,D2_rms_cols,D2_iwt_cols,D2_mav_cols,D2_var_cols,D2_zcs_cols,Right_EMGs,Left_EMGs):
                        
                        cA_R,cD4_R,cD3_R,cD2_R,cD1_R =pywt.wavedec(chunk[Right_EMG], wavelet, level=4)
                        cA_L,cD4_L,cD3_L,cD2_L,cD1_L =pywt.wavedec(chunk[Left_EMG], wavelet, level=4)
                        
                        #   cD Mean
                        if WT_cD_Mean == True :
                            features_R.loc[i,'R_'+cD4_mean_col]= cD4_R.mean()
                            features_L.loc[i,'L_'+cD4_mean_col]= cD4_L.mean()
                            features_R.loc[i,'R_'+cD3_mean_col]= cD3_R.mean()
                            features_L.loc[i,'L_'+cD3_mean_col]= cD3_L.mean()
                            features_R.loc[i,'R_'+cD2_mean_col]= cD2_R.mean()
                            features_L.loc[i,'L_'+cD2_mean_col]= cD2_L.mean()
                            features_R.loc[i,'R_'+cD1_mean_col]= cD1_R.mean()
                            features_L.loc[i,'L_'+cD1_mean_col]= cD1_L.mean()
                    
                        #   cD RMS
                        if WT_cD_RMS == True :
                            features_R.loc[i,'R_'+cD4_rms_col]= np.sqrt((cD4_R**2).mean())
                            features_L.loc[i,'L_'+cD4_rms_col]= np.sqrt((cD4_L**2).mean())
                            features_R.loc[i,'R_'+cD3_rms_col]= np.sqrt((cD3_R**2).mean())
                            features_L.loc[i,'L_'+cD3_rms_col]= np.sqrt((cD3_L**2).mean())
                            features_R.loc[i,'R_'+cD2_rms_col]= np.sqrt((cD2_R**2).mean())
                            features_L.loc[i,'L_'+cD2_rms_col]= np.sqrt((cD2_L**2).mean())
                            features_R.loc[i,'R_'+cD1_rms_col]= np.sqrt((cD1_R**2).mean())
                            features_L.loc[i,'L_'+cD1_rms_col]= np.sqrt((cD1_L**2).mean())
                    
                        #   cD Integrated wavelet
                        if WT_cD_IWT == True : 
                            features_R.loc[i,'R_'+cD4_iwt_col]= np.abs(cD4_R).sum()
                            features_L.loc[i,'L_'+cD4_iwt_col]= np.abs(cD4_L).sum()
                            features_R.loc[i,'R_'+cD3_iwt_col]= np.abs(cD3_R).sum()
                            features_L.loc[i,'L_'+cD3_iwt_col]= np.abs(cD3_L).sum()
                            features_R.loc[i,'R_'+cD2_iwt_col]= np.abs(cD2_R).sum()
                            features_L.loc[i,'L_'+cD2_iwt_col]= np.abs(cD2_L).sum()
                            features_R.loc[i,'R_'+cD1_iwt_col]= np.abs(cD1_R).sum()
                            features_L.loc[i,'L_'+cD1_iwt_col]= np.abs(cD1_L).sum()
                    
                        #   cD MAV
                        if WT_cD_MAV == True :
                            features_R.loc[i,'R_'+cD4_mav_col]= np.abs(cD4_R).mean()
                            features_L.loc[i,'L_'+cD4_mav_col]= np.abs(cD4_L).mean()
                            features_R.loc[i,'R_'+cD3_mav_col]= np.abs(cD3_R).mean()
                            features_L.loc[i,'L_'+cD3_mav_col]= np.abs(cD3_L).mean()
                            features_R.loc[i,'R_'+cD2_mav_col]= np.abs(cD2_R).mean()
                            features_L.loc[i,'L_'+cD2_mav_col]= np.abs(cD2_L).mean()
                            features_R.loc[i,'R_'+cD1_mav_col]= np.abs(cD1_R).mean()
                            features_L.loc[i,'L_'+cD1_mav_col]= np.abs(cD1_L).mean()
                            
                        #   cD VAR
                        if WT_cD_VAR == True :
                            features_R.loc[i,'R_'+cD4_var_col]= cD4_R.var()
                            features_L.loc[i,'L_'+cD4_var_col]= cD4_L.var()
                            features_R.loc[i,'R_'+cD3_var_col]= cD3_R.var()
                            features_L.loc[i,'L_'+cD3_var_col]= cD3_L.var()
                            features_R.loc[i,'R_'+cD2_var_col]= cD2_R.var()
                            features_L.loc[i,'L_'+cD2_var_col]= cD2_L.var()
                            features_R.loc[i,'R_'+cD1_var_col]= cD1_R.var()
                            features_L.loc[i,'L_'+cD1_var_col]= cD1_L.var()
                        
                        #   cD Number Zero Crossings
                        if WT_cD_ZCs == True :
                            features_R.loc[i,'R_'+cD4_zcs_col]= np.sum(np.diff(np.sign(cD4_R)[np.sign(cD4_R)!=0])!= 0)
                            features_L.loc[i,'L_'+cD4_zcs_col]= np.sum(np.diff(np.sign(cD4_L)[np.sign(cD4_L)!=0])!= 0)
                            features_R.loc[i,'R_'+cD3_zcs_col]= np.sum(np.diff(np.sign(cD3_R)[np.sign(cD3_R)!=0])!= 0)
                            features_L.loc[i,'L_'+cD3_zcs_col]= np.sum(np.diff(np.sign(cD3_L)[np.sign(cD3_L)!=0])!= 0)
                            features_R.loc[i,'R_'+cD2_zcs_col]= np.sum(np.diff(np.sign(cD2_R)[np.sign(cD2_R)!=0])!= 0)
                            features_L.loc[i,'L_'+cD2_zcs_col]= np.sum(np.diff(np.sign(cD2_L)[np.sign(cD2_L)!=0])!= 0)
                            features_R.loc[i,'R_'+cD1_zcs_col]= np.sum(np.diff(np.sign(cD1_R)[np.sign(cD1_R)!=0])!= 0)
                            features_L.loc[i,'L_'+cD1_zcs_col]= np.sum(np.diff(np.sign(cD1_L)[np.sign(cD1_L)!=0])!= 0)
                            
                    #--------------------------------------------------------------
                    
                        D2_R = wrcoef('a',chunk[Right_EMG],[cA_R,cD4_R,cD3_R,cD2_R,cD1_R], wavelet ,level=4)
                        D2_L = wrcoef('a',chunk[Left_EMG],[cA_L,cD4_L,cD3_L,cD2_L,cD1_L], wavelet ,level=4)
                    
                        #   D2 Mean
                        if WT_D2_Mean == True :
                            features_R.loc[i,'R_'+D2_mean_col]= D2_R.mean()
                            features_L.loc[i,'L_'+D2_mean_col]= D2_L.mean()
                           
                        #   A RMS
                        if WT_D2_RMS == True :
                            features_R.loc[i,'R_'+D2_rms_col]= np.sqrt((D2_R**2).mean())
                            features_L.loc[i,'L_'+D2_rms_col]= np.sqrt((D2_L**2).mean())
                    
                        #   A Integrated wavelet
                        if WT_D2_IWT == True :
                            features_R.loc[i,'R_'+D2_iwt_col]= np.abs(D2_R).sum()
                            features_L.loc[i,'L_'+D2_iwt_col]= np.abs(D2_L).sum()
                        
                        #   A MAV
                        if WT_D2_MAV == True :
                            features_R.loc[i,'R_'+D2_mav_col]= np.abs(D2_R).mean()
                            features_L.loc[i,'L_'+D2_mav_col]= np.abs(D2_L).mean()
                            
                        #   D2 VAR
                        if WT_D2_VAR == True :
                            features_R.loc[i,'R_'+D2_var_col]= D2_R.var()
                            features_L.loc[i,'L_'+D2_var_col]= D2_L.var()
                            
                        #   D2 Number Zero Crossings
                        if WT_D2_ZCs == True :
                            features_R.loc[i,'R_'+D2_zcs_col]= np.sum(np.diff(np.sign(D2_R)[np.sign(D2_R)!=0])!= 0)
                            features_L.loc[i,'L_'+D2_zcs_col]= np.sum(np.diff(np.sign(D2_L)[np.sign(D2_L)!=0])!= 0)
                        
                #--------------------------------------------------------------------------
                 
                    
                circuit_features_R= circuit_features_R.append(features_R , ignore_index=True)
                circuit_features_L= circuit_features_L.append(features_L , ignore_index=True) 
                
    circuit_features =pd.concat([circuit_features_R,circuit_features_L],axis=1)
            
    return circuit_features

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------

def Evaluate_Classification(copmarision_list=(),cv = 10):
    
    preformance_KNN= []
    preformance_LR = []
    preformance_SVM= []
    
    for j in copmarision_list:
    
        features = pd.read_pickle(j)
        
        print('\nclaculating file : '+str(j))
    
        features_x=features.iloc[:,:features.shape[1]-1]
        features_y=features.Mode.astype('category')
        
        scaler = StandardScaler()
        features_x =scaler.fit_transform(features_x)
        
        x_train , x_test, y_train, y_test = train_test_split(features_x, features_y, test_size=0.3 , random_state=123)
    
        clf_KNN = KNeighborsClassifier()
        clf_LR =LogisticRegression(max_iter=200)
        clf_SVM =svm.SVC(class_weight ='balanced')
    
        param_grid_KNN=dict(n_neighbors=np.arange(1,20))
        param_grid_LR=dict(solver=('lbfgs', 'liblinear'))
        param_grid_SVM=dict(kernel=('linear','poly'), degree=np.arange(1,8))
        
        
        print('Fitting KNN ...')
        start_KNN_time = time.monotonic()
        grid_KNN = GridSearchCV(clf_KNN, param_grid=param_grid_KNN , cv=cv , scoring='accuracy')
        grid_KNN.fit(x_train,y_train)
        predicted_KNN=grid_KNN.predict(x_test)
        end_KNN_time = time.monotonic()
        print('KNN fitting took : ' , timedelta(seconds=end_KNN_time - start_KNN_time))
        
        
        print('Fitting LR ...')
        start_LR_time = time.monotonic()
        grid_LR = GridSearchCV(clf_LR, param_grid=param_grid_LR , cv=cv , scoring='accuracy')
        grid_LR.fit(x_train,y_train)
        predicted_LR=grid_LR.predict(x_test)
        end_LR_time = time.monotonic()
        print('LR fitting took : ' , timedelta(seconds=end_LR_time - start_LR_time))
        
        
        print('Fitting SVM ...')
        start_SVM_time = time.monotonic()
        grid_SVM = GridSearchCV(clf_SVM, param_grid=param_grid_SVM , cv=cv , scoring='accuracy')
        grid_SVM.fit(x_train,y_train)
        predicted_SVM=grid_SVM.predict(x_test)
        end_SVM_time = time.monotonic()
        print('SVM fitting took : ' , timedelta(seconds=end_SVM_time - start_SVM_time))
        
        
        preformance_KNN.append(metrics.f1_score(y_test , predicted_KNN , average = 'weighted'))
        preformance_LR.append(metrics.f1_score(y_test , predicted_LR , average = 'weighted'))
        preformance_SVM.append(metrics.f1_score(y_test , predicted_SVM , average = 'weighted'))
    
    return preformance_KNN, preformance_LR, preformance_SVM

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------

def GridScore(x, y, file_name, algorithm , print_scores = True ):

        
    print('\nStarting...')
    start_time = time.monotonic()    
  
    
    cv = StratifiedShuffleSplit(n_splits=3, test_size=0.15, random_state = 123)
    f1_scorer = metrics.make_scorer(metrics.f1_score, average='weighted')

    
    if algorithm == 'SVM' :
        C_range = np.logspace(-1, 4, 6)
        gamma_range = np.logspace(-5, 0, 6)
        param_grid = dict(C= C_range , gamma= gamma_range)
        grid = RandomizedSearchCV(svm.SVC(cache_size=500, class_weight ='balanced', kernel='rbf'), param_distributions=param_grid,cv=cv , scoring= f1_scorer, n_iter=5)
        grid.fit(x, y)
        end_time = time.monotonic()
        

    if algorithm == 'KNN' :
        k_range= np.arange(1,9)
        param_grid = dict(n_neighbors= k_range)
        grid = RandomizedSearchCV(KNeighborsClassifier(weights='distance', metric='euclidean'), param_distributions=param_grid, cv=cv , scoring= f1_scorer, n_iter=5)
        grid.fit(x, y)
        end_time = time.monotonic()

            
    if algorithm == 'LR' :
        C_range = np.logspace(-2, 10, 13)
        param_grid = dict(C= C_range )
        grid = RandomizedSearchCV(LogisticRegression(multi_class='multinomial', solver='lbfgs'), param_distributions=param_grid, cv=cv , scoring= f1_scorer, n_iter=5)
        grid.fit(x, y)
        end_time = time.monotonic()
            
            
    if algorithm == 'LDA' :
        solvers = ['svd']
        param_grid = dict( solver= solvers)
        grid = GridSearchCV(LinearDiscriminantAnalysis(), param_grid=param_grid, cv=cv , scoring= f1_scorer)
        grid.fit(x, y)
        end_time = time.monotonic()
            
            
    if algorithm == 'QDA' :
        reg_params = [0, 0.0001, 0.001, 0.1]
        param_grid = dict( reg_param= reg_params)
        grid = GridSearchCV(QuadraticDiscriminantAnalysis(), param_grid=param_grid, cv=cv , scoring= f1_scorer)
        grid.fit(x, y)
        end_time = time.monotonic()
            
            
    if algorithm == 'ETC' :
        n_estimators_ = [int(x) for x in range(200,1200,100)]
        min_samples_leaf_=[10,20,30,40,50,60,70,80,90,100]
        param_grid = dict(n_estimators= n_estimators_, min_samples_leaf=min_samples_leaf_ )
        grid = RandomizedSearchCV(ExtraTreesClassifier(random_state=0, max_features=1), param_distributions=param_grid, cv=cv , scoring= f1_scorer, n_iter=5)
        grid.fit(x,y)
        end_time = time.monotonic()
            
    
    print('Time : ', timedelta(seconds=end_time - start_time))
    
    if print_scores:
        print("The best parameters are %s with a score of %0.3f"
        % (grid.best_params_, grid.best_score_))
        print(grid.best_estimator_)

    return grid

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------

def DrawLearningCurve(X ,Y ,file_name , algorithm , best_params):
    
    def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                            n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
        """
        Generate a simple plot of the test and training learning curve.
        Parameters
        ----------
        estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.
        title : string
        Title for the chart.
        X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.
        y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.
        ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.
        cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
        - None, to use the default 3-fold cross-validation,
        - integer, to specify the number of folds.
        - An object to be used as a cross-validation generator.
        - An iterable yielding train/test splits.
        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.
        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.
        n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
        """
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel("Training examples")
        plt.ylabel("Score")
        train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
        
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        
        plt.grid()
        
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
        
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
        
        plt.legend(loc="best")
        return plt
          
    #--------------------------------------------------------------------------

    scaler = StandardScaler()

    X = pd.DataFrame(scaler.fit_transform(X) , columns= X.columns)
    Y = Y.astype('category')
    
    print('\nDrawing Learning Curve...')
    start_time = time.monotonic()    
    
    cv= ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)

    if algorithm == 'LR':
        title = "Learning Curves (LR, Solver ="+str(best_params['solver'])+" , C ="+str(best_params['C'])+')'
        clf = LogisticRegression(multi_class='multinomial' ,C= best_params['C'], solver= best_params['solver'])

    if algorithm == 'SVM':
         title = "Learning Curves (SVM, RBF kernel, $\gamma="+str(best_params['gamma'])+"$ , C ="+str(best_params['C'])+')'
         clf =svm.SVC(cache_size=500,class_weight ='balanced', kernel='rbf', C=best_params['C'], gamma =best_params['gamma'])
         
    if algorithm == 'KNN':
        title = "Learning Curves (KNN, Ball_Tree algorithm, n_neighbors="+str(best_params['n_neighbors'])+" , p ="+str(best_params['p'])+')'
        clf = KNeighborsClassifier(algorithm='ball_tree', weights='distance', metric ='manhattan', p=best_params['p'], n_neighbors=best_params['n_neighbors'])

    plt.figure(file_name+' - Learning Curve')
    plot_learning_curve(clf, title, X, Y, (0.6, 1.01), cv=cv, n_jobs=4)
    plt.show()
    
    end_time = time.monotonic()
    print('End Time : ' , timedelta(seconds=end_time - start_time))
    
    return 0    
    

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------

def NeuralNetwork(X , Y, file_name, Draw_Learning_Curves=True):

    Y= pd.get_dummies(Y)    

     
    print('Starting... ')
    start_time = time.monotonic()
    
    x , x_val, y, y_val = train_test_split(X, Y, test_size=0.15 , random_state = 123)
    
    NN_model = Sequential()
    NN_model.add(Dense(200, activation='sigmoid' , input_dim=x.shape[1]))
    #NN_model.add(Dense(100, activation='sigmoid' ))
    NN_model.add(Dense(y.shape[1], activation='softmax'))
    np.random.seed(87)
    NN_model.compile(optimizer='adam',metrics=['accuracy'], loss= 'categorical_crossentropy')
    history = NN_model.fit(x=x , y=y , epochs=1500, validation_data=[x_val,y_val], shuffle=True ,callbacks= [EarlyStopping(monitor='val_loss', min_delta=0.001,patience=5)])
    
    
    end_time = time.monotonic()
    print('Iteration ended : ' , timedelta(seconds=end_time - start_time))
    
        
    if Draw_Learning_Curves == True:
        plt.style.use('fivethirtyeight')

        plt.figure(file_name+'_Accuracy')
#            plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.ylim(0,1)
        plt.title('Validation Accuracy over training epochs')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
#            plt.legend(['Iteration'+str(i)], loc='best')
        plt.show()
#            
#            plt.figure(path+'_Loss')
##            plt.plot(history.history['loss'])
#            plt.plot(history.history['val_loss'])
#            plt.ylim(0,2)
#            plt.title('Validation Cross-Entropy loss over training epochs')
#            plt.ylabel('loss')
#            plt.xlabel('epoch')
##            plt.legend(['Iteration'+str(i)], loc='best')
#            plt.show()
        
    
    return NN_model , history

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------

def RNN_LSTM(X , Y, file_name, Draw_Learning_Curves=True):

    Y= pd.get_dummies(Y)    

     
    print('Starting... ')
    start_time = time.monotonic()
    
    x , x_val, y, y_val = train_test_split(X, Y, test_size=0.15 , random_state = 123)
    
        #def create_network():
    NN_model = Sequential()
    NN_model.add(LSTM(200, activation='relu' , input_shape=(None,  x.shape[2])))
    NN_model.add(Dense(200, activation='relu' , input_dim=x.shape[1]))
    NN_model.add(Dense(200, activation='relu'))
    NN_model.add(Dense(200, activation='relu'))
    NN_model.add(Dense(200, activation='relu'))
    NN_model.add(Dense(200, activation='relu'))
      #NN_model.add(Flatten())
    NN_model.add(Dense(y.shape[1], activation='softmax'))
    NN_model.compile(optimizer='adam',metrics=['accuracy'], loss= 'categorical_crossentropy')
    #  return NN_model
    
    np.random.seed(87)
    epochs=50
    batch_size=100
    callbacks=EarlyStopping(monitor='val_loss', min_delta=0.001,patience=5)
    history = NN_model.fit(x, y, epochs, batch_size, validation_data=[x_val,y_val],callbacks=[callbacks])
    
    
    end_time = time.monotonic()
    print('Iteration ended : ' , timedelta(seconds=end_time - start_time))
    
        
    
    return NN_model , history


#-------------------------------------------------------------------------------------------------------------------------------------------------------------------

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    np.set_printoptions(precision=2)

    if normalize:
        cm = (cm.astype('float') / cm.sum(axis=1)[:, np.newaxis])*100
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
def uniform(X, Y, random_state):
    min_samples=1000000
    for j in range(8):
        samp=len(Y[Y == j])
        if min_samples > samp:
            min_samples = samp
            
    res_Mode_0_X , res_Mode_0_y = resample(X[Y ==0],Y[Y ==0],n_samples=min_samples ,replace =False, random_state=random_state)
    res_Mode_1_X , res_Mode_1_y = resample(X[Y ==1],Y[Y ==1],n_samples=min_samples ,replace =False, random_state=random_state)
    res_Mode_2_X , res_Mode_2_y = resample(X[Y ==2],Y[Y ==2],n_samples=min_samples ,replace =False, random_state=random_state)
    res_Mode_3_X , res_Mode_3_y = resample(X[Y ==3],Y[Y ==3],n_samples=min_samples ,replace =False, random_state=random_state)
    res_Mode_4_X , res_Mode_4_y = resample(X[Y ==4],Y[Y ==4],n_samples=min_samples ,replace =False, random_state=random_state)
    res_Mode_5_X , res_Mode_5_y = resample(X[Y ==5],Y[Y ==5],n_samples=min_samples ,replace =False, random_state=random_state)
    res_Mode_6_X , res_Mode_6_y = resample(X[Y ==6],Y[Y ==6],n_samples=min_samples ,replace =False, random_state=random_state)
    res_Mode_7_X , res_Mode_7_y = resample(X[Y ==7],Y[Y ==7],n_samples=min_samples ,replace =False, random_state=random_state)
    
    X = pd.concat([res_Mode_0_X,res_Mode_1_X,res_Mode_2_X,res_Mode_3_X,res_Mode_4_X,res_Mode_5_X,res_Mode_6_X,res_Mode_7_X],axis=0)
    Y = pd.concat([res_Mode_0_y,res_Mode_1_y,res_Mode_2_y,res_Mode_3_y,res_Mode_4_y,res_Mode_5_y,res_Mode_6_y,res_Mode_7_y],axis=0)
    
    return X, Y
