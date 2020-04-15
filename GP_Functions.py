# -*- coding: utf-8 -*-
"""
Created on Wed May 23 15:03:19 2018


"""
#----------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import time
from datetime import timedelta
import pywt
from scipy import signal
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score ,GridSearchCV ,StratifiedKFold ,train_test_split , ShuffleSplit , StratifiedShuffleSplit
from sklearn import metrics

from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif

from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis , QuadraticDiscriminantAnalysis
from sklearn.decomposition import PCA

from Denoising_functions import wden

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------

def WTDenoise(df_raw, columns_to_denoise, plot=False , print_time=False) :
    
    df_den=df_raw.copy()
    for i in columns_to_denoise:
       
        start_time = time.monotonic()
        df_den.loc[:,i] = wden(df_den.loc[:,i],'sqtwolog','soft','one',4,'sym5')
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
                    EMG_MAV= False, EMG_var= False, EMG_RMS= False, EMG_ZCs = False, EMG_WL= False, EMG_SSI= False,
                    EMG_IEMG= False, EMG_Fmean= False, EMG_Fmedian= False, WT_cD_Mean= False, WT_cD_RMS= False,WT_cD_IWT= False,WT_cD_MAV= False,WT_cD_VAR= False,
                    WT_cD_ZCs = False, WT_D2_Mean = False, WT_D2_RMS  = False, WT_D2_IWT  = False, WT_D2_MAV  = False, WT_D2_VAR= False ,WT_D2_ZCs= False,
                    wavelet = 'db7', window=1500 , slide =1) :
                
                

    circuit_features = pd.DataFrame()
    
    for i in range(0,int(np.around(len(circuit)/(slide*window)))):
        
        if int(np.around((i+1)*window - i*(window-slide*window))) <= len(circuit) :
            
            chunk = pd.DataFrame()
            chunk = chunk.append(circuit.iloc[int(i*slide*window) : int(np.around((i+1)*window - i*(window-slide*window))) ,:] , ignore_index=True)

            features_R = pd.DataFrame()
            features_L = pd.DataFrame()
            
            #------------------------------------------------------------------
            
            Right_IMUs = ['Right_Shank_Ax','Right_Shank_Ay','Right_Shank_Az','Right_Shank_Gx','Right_Shank_Gy','Right_Shank_Gz',
                          'Right_Thigh_Ax','Right_Thigh_Ay','Right_Thigh_Az','Right_Thigh_Gx','Right_Thigh_Gy','Right_Thigh_Gz',
                          'Right_Knee']
            
            Left_IMUs = ['Left_Shank_Ax','Left_Shank_Ay','Left_Shank_Az','Left_Shank_Gx','Left_Shank_Gy','Left_Shank_Gz',
                          'Left_Thigh_Ax','Left_Thigh_Ay','Left_Thigh_Az','Left_Thigh_Gx','Left_Thigh_Gy','Left_Thigh_Gz',
                          'Left_Knee']
            
            Right_EMGs = ['Right_TA','Right_MG','Right_SOL','Right_BF','Right_ST','Right_VL','Right_RF']
            
            Left_EMGs = ['Left_TA','Left_MG','Left_SOL','Left_BF','Left_ST','Left_VL','Left_RF']
            
            #------------------------------------------------------------------
            
            ###     IMUs
            
            if IMU_mean or IMU_min or IMU_max or IMU_STD or IMU_init or IMU_finl == True :
            
                    
                mean_cols = ['Shank_Ax_Mean','Shank_Ay_Mean','Shank_Az_Mean','Shank_Gx_Mean','Shank_Gy_Mean','Shank_Gz_Mean',
                             'Thigh_Ax_Mean','Thigh_Ay_Mean','Thigh_Az_Mean','Thigh_Gx_Mean','Thigh_Gy_Mean','Thigh_Gz_Mean',
                             'Knee_Mean']
                min_cols = ['Shank_Ax_Min','Shank_Ay_Min','Shank_Az_Min','Shank_Gx_Min','Shank_Gy_Min','Shank_Gz_Min',
                            'Thigh_Ax_Min','Thigh_Ay_Min','Thigh_Az_Min','Thigh_Gx_Min','Thigh_Gy_Min','Thigh_Gz_Min',
                            'Knee_Min']
                max_cols = ['Shank_Ax_Max','Shank_Ay_Max','Shank_Az_Max','Shank_Gx_Max','Shank_Gy_Max','Shank_Gz_Max',
                            'Thigh_Ax_Max','Thigh_Ay_Max','Thigh_Az_Max','Thigh_Gx_Max','Thigh_Gy_Max','Thigh_Gz_Max',
                            'Knee_Max']
                std_cols = ['Shank_Ax_STD','Shank_Ay_STD','Shank_Az_STD','Shank_Gx_STD','Shank_Gy_STD','Shank_Gz_STD',
                            'Thigh_Ax_STD','Thigh_Ay_STD','Thigh_Az_STD','Thigh_Gx_STD','Thigh_Gy_STD','Thigh_Gz_STD',
                            'Knee_STD']
                init_cols = ['Shank_Ax_Init','Shank_Ay_Init','Shank_Az_Init','Shank_Gx_Init','Shank_Gy_Init','Shank_Gz_Init',
                             'Thigh_Ax_Init','Thigh_Ay_Init','Thigh_Az_Init','Thigh_Gx_Init','Thigh_Gy_Init','Thigh_Gz_Init',
                             'Knee_Init']
                finl_cols = ['Shank_Ax_Finl','Shank_Ay_Finl','Shank_Az_Finl','Shank_Gx_Finl','Shank_Gy_Finl','Shank_Gz_Finl',
                             'Thigh_Ax_Finl','Thigh_Ay_Finl','Thigh_Az_Finl','Thigh_Gx_Finl','Thigh_Gy_Finl','Thigh_Gz_Finl',
                             'Knee_Finl']
                
            
            
                for mean_col ,min_col, max_col ,std_col, init_col ,finl_col, Right_IMU, Left_IMU in zip(
                    mean_cols,min_cols,max_cols,std_cols,init_cols,finl_cols,Right_IMUs,Left_IMUs) :
                    
                        #   Mean
                        if IMU_mean == True:
                            features_R.loc[i,mean_col]= chunk[Right_IMU].mean()
                            features_L.loc[i,mean_col]= chunk[Left_IMU].mean() 
                            
                        #   Minimum value
                        if IMU_min == True:    
                            features_R.loc[i,min_col]= chunk[Right_IMU].min()
                            features_L.loc[i,min_col]= chunk[Left_IMU].min()
                            
                        #   Maximum value
                        if IMU_max == True :
                            features_R.loc[i,max_col]= chunk[Right_IMU].max()
                            features_L.loc[i,max_col]= chunk[Left_IMU].max()
                        
                        #   Standard deviation
                        if IMU_STD == True:
                            features_R.loc[i,std_col]= chunk[Right_IMU].std()
                            features_L.loc[i,std_col]= chunk[Left_IMU].std()
            
                        #   Initial value
                        if IMU_init == True:
                            features_R.loc[i,init_col]= chunk[Right_IMU][0]
                            features_L.loc[i,init_col]= chunk[Left_IMU][0]
            
                        #   Final value
                        if IMU_finl == True :
                            features_R.loc[i,finl_col]= chunk[Right_IMU][len(chunk)-1]
                            features_L.loc[i,finl_col]= chunk[Left_IMU][len(chunk)-1]
                            
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
                        features_R.loc[i,mav_col]= chunk[Right_EMG].abs().mean()
                        features_L.loc[i,mav_col]= chunk[Left_EMG].abs().mean()
            
                    # Variance
                    if EMG_var == True :
                        features_R.loc[i,var_col]= chunk[Right_EMG].var()
                        features_L.loc[i,var_col]= chunk[Left_EMG].var()
                        
            
                    # Root Mean Square
                    if EMG_RMS == True :
                        features_R.loc[i,rms_col]= np.sqrt((chunk[Right_EMG]**2).mean())
                        features_L.loc[i,rms_col]= np.sqrt((chunk[Left_EMG]**2).mean())
            
                    #   Number of zero crossings
                    if EMG_ZCs == True :
                        features_R.loc[i,zcs_col]= np.sum(np.diff(np.sign(chunk[Right_EMG])[np.sign(chunk[Right_EMG])!=0])!= 0)
                        features_L.loc[i,zcs_col]= np.sum(np.diff(np.sign(chunk[Left_EMG])[np.sign(chunk[Left_EMG])!=0])!= 0)
            
                    #   Waveform length
                    if EMG_WL == True :
                        chunk2 = pd.DataFrame()
                        chunk2 = chunk2.append(chunk[[Right_EMG,Left_EMG]], ignore_index=True)
                        chunk2 = chunk2.iloc[1:,:]
                        chunk2 = chunk2.append(chunk2.iloc[-1,:] , ignore_index=True)
                
                        features_R.loc[i,WL_col]= (chunk2[Right_EMG] - chunk[Right_EMG]).abs().sum()
                        features_L.loc[i,WL_col]= (chunk2[Left_EMG] - chunk[Left_EMG]).abs().sum()

            
                    #   Simple Square Integral
                    if EMG_SSI == True :
                        features_R.loc[i,ssi_col]= (chunk[Right_EMG].abs()**2).sum()
                        features_L.loc[i,ssi_col]= (chunk[Left_EMG].abs()**2).sum()
            
                    #   Integrated EMG
                    if EMG_IEMG == True :
                        features_R.loc[i,iemg_col]= chunk[Right_EMG].abs().sum()
                        features_L.loc[i,iemg_col]= chunk[Left_EMG].abs().sum()
                
            
            #------------------------------------------------------------------
            
            ###     EMGs , Frequency domain
            
            
            if EMG_Fmean or EMG_Fmedian == True :
                
                fmean_cols   = ['TA_FMean','MG_FMean','SOL_FMean','BF_FMean','ST_FMean','VL_FMean','RF_FMean']
                fmedian_cols = ['TA_FMedian','MG_FMedian','SOL_FMedian','BF_FMedian','ST_FMedian','VL_FMedian','RF_FMedian']
                
                for fmean_col, fmedian_col, Right_EMG, Left_EMG in zip(
                    fmean_cols,fmedian_cols,Right_EMGs,Left_EMGs):
                    
                    #   Power Spectrum Density claculations:
                    freqs_R , PSD_R = signal.welch(chunk[Right_EMG])
                    freqs_L , PSD_L = signal.welch(chunk[Left_EMG])
                
                    #   Frequency Mean
                    if EMG_Fmean == True :
                        features_R.loc[i,fmean_col]= ((PSD_R * freqs_R).sum())/PSD_R.sum()
                        features_L.loc[i,fmean_col]= ((PSD_L * freqs_L).sum())/PSD_L.sum()
                
                    #   Frequency Median
                    if EMG_Fmedian == True :
                        features_R.loc[i,fmedian_col]= 0.5 * PSD_R.sum()
                        features_L.loc[i,fmedian_col]= 0.5 * PSD_L.sum()
       
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
                D2_mean_cols = ['TA_WT_D2_Mean', 'MG_WT_D2_Mean', 'SOL_WT_D2_Mean', 'BF_WT_D2_Mean', 'ST_WT_D2_Mean', 'VL_WT_D2_Mean', 'RF_WT_D2_Mean']
                D2_rms_cols = ['TA_WT_D2_RMS', 'MG_WT_D2_RMS', 'SOL_WT_D2_RMS', 'BF_WT_D2_RMS', 'ST_WT_D2_RMS', 'VL_WT_D2_RMS', 'RF_WT_D2_RMS']
                D2_iwt_cols = ['TA_WT_D2_IWT', 'MG_WT_D2_IWT', 'SOL_WT_D2_IWT', 'BF_WT_D2_IWT', 'ST_WT_D2_IWT', 'VL_WT_D2_IWT', 'RF_WT_D2_IWT']
                D2_mav_cols = ['TA_WT_D2_MAV', 'MG_WT_D2_MAV', 'SOL_WT_D2_MAV', 'BF_WT_D2_MAV', 'ST_WT_D2_MAV', 'VL_WT_D2_MAV', 'RF_WT_D2_MAV']
                D2_var_cols = ['TA_WT_D2_VAR', 'MG_WT_D2_VAR', 'SOL_WT_D2_VAR', 'BF_WT_D2_VAR', 'ST_WT_D2_VAR', 'VL_WT_D2_VAR', 'RF_WT_D2_VAR']
                D2_zcs_cols = ['TA_WT_D2_ZCs', 'MG_WT_D2_ZCs', 'SOL_WT_D2_ZCs', 'BF_WT_D2_ZCs', 'ST_WT_D2_ZCs', 'VL_WT_D2_ZCs', 'RF_WT_D2_ZCs']
                
                for cD4_mean_col, cD3_mean_col, cD2_mean_col, cD1_mean_col, cD4_rms_col, cD3_rms_col, cD2_rms_col, cD1_rms_col, cD4_iwt_col, cD3_iwt_col, cD2_iwt_col, cD1_iwt_col, cD4_mav_col, cD3_mav_col, cD2_mav_col, cD1_mav_col, cD4_var_col, cD3_var_col, cD2_var_col, cD1_var_col, cD4_zcs_col, cD3_zcs_col, cD2_zcs_col, cD1_zcs_col ,D2_mean_col, D2_rms_col, D2_iwt_col, D2_mav_col, D2_var_col, D2_zcs_col, Right_EMG, Left_EMG in zip(
                    cD4_mean_cols,cD3_mean_cols,cD2_mean_cols,cD1_mean_cols,cD4_rms_cols,cD3_rms_cols,cD2_rms_cols,cD1_rms_cols,cD4_iwt_cols,cD3_iwt_cols,cD2_iwt_cols,cD1_iwt_cols,cD4_mav_cols,cD3_mav_cols,cD2_mav_cols,cD1_mav_cols,cD4_var_cols,cD3_var_cols,cD2_var_cols,cD1_var_cols,cD4_zcs_cols,cD3_zcs_cols,cD2_zcs_cols,cD1_zcs_cols,D2_mean_cols,D2_rms_cols,D2_iwt_cols,D2_mav_cols,D2_var_cols,D2_zcs_cols,Right_EMGs,Left_EMGs):
                    
                    cA_R,cD4_R,cD3_R,cD2_R,cD1_R =pywt.wavedec(chunk[Right_EMG], wavelet, level=4)
                    cA_L,cD4_L,cD3_L,cD2_L,cD1_L =pywt.wavedec(chunk[Left_EMG], wavelet, level=4)
                    
                    #   cD Mean
                    if WT_cD_Mean == True :
                        features_R.loc[i,cD4_mean_col]= cD4_R.mean()
                        features_L.loc[i,cD4_mean_col]= cD4_L.mean()
                        features_R.loc[i,cD3_mean_col]= cD3_R.mean()
                        features_L.loc[i,cD3_mean_col]= cD3_L.mean()
                        features_R.loc[i,cD2_mean_col]= cD2_R.mean()
                        features_L.loc[i,cD2_mean_col]= cD2_L.mean()
                        features_R.loc[i,cD1_mean_col]= cD1_R.mean()
                        features_L.loc[i,cD1_mean_col]= cD1_L.mean()
                
                    #   cD RMS
                    if WT_cD_RMS == True :
                        features_R.loc[i,cD4_rms_col]= np.sqrt((cD4_R**2).mean())
                        features_L.loc[i,cD4_rms_col]= np.sqrt((cD4_L**2).mean())
                        features_R.loc[i,cD3_rms_col]= np.sqrt((cD3_R**2).mean())
                        features_L.loc[i,cD3_rms_col]= np.sqrt((cD3_L**2).mean())
                        features_R.loc[i,cD2_rms_col]= np.sqrt((cD2_R**2).mean())
                        features_L.loc[i,cD2_rms_col]= np.sqrt((cD2_L**2).mean())
                        features_R.loc[i,cD1_rms_col]= np.sqrt((cD1_R**2).mean())
                        features_L.loc[i,cD1_rms_col]= np.sqrt((cD1_L**2).mean())
                
                    #   cD Integrated wavelet
                    if WT_cD_IWT == True : 
                        features_R.loc[i,cD4_iwt_col]= np.abs(cD4_R).sum()
                        features_L.loc[i,cD4_iwt_col]= np.abs(cD4_L).sum()
                        features_R.loc[i,cD3_iwt_col]= np.abs(cD3_R).sum()
                        features_L.loc[i,cD3_iwt_col]= np.abs(cD3_L).sum()
                        features_R.loc[i,cD2_iwt_col]= np.abs(cD2_R).sum()
                        features_L.loc[i,cD2_iwt_col]= np.abs(cD2_L).sum()
                        features_R.loc[i,cD1_iwt_col]= np.abs(cD1_R).sum()
                        features_L.loc[i,cD1_iwt_col]= np.abs(cD1_L).sum()
                
                    #   cD MAV
                    if WT_cD_MAV == True :
                        features_R.loc[i,cD4_mav_col]= np.abs(cD4_R).mean()
                        features_L.loc[i,cD4_mav_col]= np.abs(cD4_L).mean()
                        features_R.loc[i,cD3_mav_col]= np.abs(cD3_R).mean()
                        features_L.loc[i,cD3_mav_col]= np.abs(cD3_L).mean()
                        features_R.loc[i,cD2_mav_col]= np.abs(cD2_R).mean()
                        features_L.loc[i,cD2_mav_col]= np.abs(cD2_L).mean()
                        features_R.loc[i,cD1_mav_col]= np.abs(cD1_R).mean()
                        features_L.loc[i,cD1_mav_col]= np.abs(cD1_L).mean()
                        
                    #   cD VAR
                    if WT_cD_VAR == True :
                        features_R.loc[i,cD4_var_col]= cD4_R.var()
                        features_L.loc[i,cD4_var_col]= cD4_L.var()
                        features_R.loc[i,cD3_var_col]= cD3_R.var()
                        features_L.loc[i,cD3_var_col]= cD3_L.var()
                        features_R.loc[i,cD2_var_col]= cD2_R.var()
                        features_L.loc[i,cD2_var_col]= cD2_L.var()
                        features_R.loc[i,cD1_var_col]= cD1_R.var()
                        features_L.loc[i,cD1_var_col]= cD1_L.var()
                    
                    #   cD Number Zero Crossings
                    if WT_cD_ZCs == True :
                        features_R.loc[i,cD4_zcs_col]= np.sum(np.diff(np.sign(cD4_R)[np.sign(cD4_R)!=0])!= 0)
                        features_L.loc[i,cD4_zcs_col]= np.sum(np.diff(np.sign(cD4_L)[np.sign(cD4_L)!=0])!= 0)
                        features_R.loc[i,cD3_zcs_col]= np.sum(np.diff(np.sign(cD3_R)[np.sign(cD3_R)!=0])!= 0)
                        features_L.loc[i,cD3_zcs_col]= np.sum(np.diff(np.sign(cD3_L)[np.sign(cD3_L)!=0])!= 0)
                        features_R.loc[i,cD2_zcs_col]= np.sum(np.diff(np.sign(cD2_R)[np.sign(cD2_R)!=0])!= 0)
                        features_L.loc[i,cD2_zcs_col]= np.sum(np.diff(np.sign(cD2_L)[np.sign(cD2_L)!=0])!= 0)
                        features_R.loc[i,cD1_zcs_col]= np.sum(np.diff(np.sign(cD1_R)[np.sign(cD1_R)!=0])!= 0)
                        features_L.loc[i,cD1_zcs_col]= np.sum(np.diff(np.sign(cD1_L)[np.sign(cD1_L)!=0])!= 0)
                        
                #--------------------------------------------------------------
                
                    D2_R = wrcoef('a',chunk[Right_EMG],[cA_R,cD4_R,cD3_R,cD2_R,cD1_R], wavelet ,level=2)
                    D2_L = wrcoef('a',chunk[Left_EMG],[cA_L,cD4_L,cD3_L,cD2_L,cD1_L], wavelet ,level=2)
                
                    #   D2 Mean
                    if WT_D2_Mean == True :
                        features_R.loc[i,D2_mean_col]= D2_R.mean()
                        features_L.loc[i,D2_mean_col]= D2_L.mean()
                       
                    #   A RMS
                    if WT_D2_RMS == True :
                        features_R.loc[i,D2_rms_col]= np.sqrt((D2_R**2).mean())
                        features_L.loc[i,D2_rms_col]= np.sqrt((D2_L**2).mean())
                
                    #   A Integrated wavelet
                    if WT_D2_IWT == True :
                        features_R.loc[i,D2_iwt_col]= np.abs(D2_R).sum()
                        features_L.loc[i,D2_iwt_col]= np.abs(D2_L).sum()
                    
                    #   A MAV
                    if WT_D2_MAV == True :
                        features_R.loc[i,D2_mav_col]= np.abs(D2_R).mean()
                        features_L.loc[i,D2_mav_col]= np.abs(D2_L).mean()
                        
                    #   D2 VAR
                    if WT_D2_VAR == True :
                        features_R.loc[i,D2_var_col]= D2_R.var()
                        features_L.loc[i,D2_var_col]= D2_L.var()
                        
                    #   D2 Number Zero Crossings
                    if WT_D2_ZCs == True :
                        features_R.loc[i,D2_zcs_col]= np.sum(np.diff(np.sign(D2_R)[np.sign(D2_R)!=0])!= 0)
                        features_L.loc[i,D2_zcs_col]= np.sum(np.diff(np.sign(D2_L)[np.sign(D2_L)!=0])!= 0)
                    
                    
            
            # Mode Mean
            features_R.loc[i,'Mode']= features_L.loc[i,'Mode']= chunk.Mode.mean()
            #--------------------------------------------------------------------------
                
                
            circuit_features= circuit_features.append(features_R , ignore_index=True)
            circuit_features= circuit_features.append(features_L , ignore_index=True)
            
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

def GridScore(features_x,features_y , algorithm , use_PCA=False , HeatMap = True):

    start_time = time.monotonic()
        
    print('\nStarting...')
    start_time = time.monotonic()    
    
    
    if use_PCA == True:
        pca = PCA(0.99)
        pca.fit(features_x)
        PCA_components= pd.DataFrame(pca.components_, columns=list(features_x.columns))
        Columns_from_PCA = features_x.columns[PCA_components.describe().loc['75%',:] >= 0.0001]
        features_x= pd.DataFrame(features_x.loc[:,Columns_from_PCA] , columns = Columns_from_PCA)
        
    # It is usually a good idea to scale the data for SVM training.
    # We are cheating a bit in this example in scaling all of the data,
    # instead of fitting the transformation on the training set and
    # just applying it on the test set.
    #features_x = scaler.fit_transform(features_x)
    
    # Train classifiers
    
    cv = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    f1_scorer = metrics.make_scorer(metrics.f1_score, average='weighted')

    # For an initial search, a logarithmic grid with basis
    # 10 is often helpful. Using a basis of 2, a finer
    # tuning can be achieved but at a much higher cost.
    
    if algorithm == 'SVM' :
        C_range = np.logspace(-2, 10, 13)
        gamma_range = np.logspace(-9, 3, 13)
        param_grid = dict(C= C_range , gamma= gamma_range)
        grid = GridSearchCV(svm.SVC(cache_size=500,class_weight ='balanced', kernel='rbf'), param_grid=param_grid, cv=cv , scoring= f1_scorer)
        grid.fit(features_x, features_y)
        scores = grid.cv_results_['mean_test_score'].reshape(len(gamma_range), len(C_range))
        end_time = time.monotonic()
        if HeatMap == True:
            xlabel, xparam, ylabel, yparam = 'C' , C_range , 'Gamma' , gamma_range

    if algorithm == 'KNN' :
        k_range= np.arange(1,9)
        p_ = [1,2]
        param_grid = dict(n_neighbors= k_range , p= p_)
        grid = GridSearchCV(KNeighborsClassifier(metric='manhattan',algorithm='auto', weights='distance'), param_grid=param_grid, cv=cv , scoring= f1_scorer)
        grid.fit(features_x, features_y)
        scores = grid.cv_results_['mean_test_score'].reshape(len(p_),len(k_range))
        end_time = time.monotonic()
        if HeatMap == True:
            xlabel, xparam, ylabel, yparam ='K' , k_range , 'P' , p_

    if algorithm == 'LR' :
        C_range = np.logspace(-2, 10, 13)
        solvers = ['newton-cg', 'lbfgs']
        param_grid = dict(C= C_range , solver= solvers)
        grid = GridSearchCV(LogisticRegression(multi_class='multinomial' ), param_grid=param_grid, cv=cv , scoring= f1_scorer)
        grid.fit(features_x, features_y)
        scores = grid.cv_results_['mean_test_score'].reshape(len(C_range), len(solvers))
        end_time = time.monotonic()
        if HeatMap == True:
            xlabel, xparam, ylabel, yparam ='Solvers', solvers , 'C', C_range 


    print('Time : ', timedelta(seconds=end_time - start_time))
    print("The best parameters are %s with a score of %0.3f"
    % (grid.best_params_, grid.best_score_))
    print(grid.best_estimator_)
    
# -------- HeatMap --------:
    
    if HeatMap == True :
        # Utility function to move the midpoint of a colormap to be around
        # the values of interest.
        class MidpointNormalize(Normalize):
            def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
                self.midpoint = midpoint
                Normalize.__init__(self, vmin, vmax, clip)
            def __call__(self, value, clip=None):
                x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
                return np.ma.masked_array(np.interp(value, x, y))
            
        # #############################################################################
          
        
        # Draw heatmap of the validation accuracy as a function of gamma and C
        #
        # The score are encoded as colors with the hot colormap which varies from dark
        # red to bright yellow. As the most interesting scores are all located in the
        # 0.92 to 0.97 range we use a custom normalizer to set the mid-point to 0.92 so
        # as to make it easier to visualize the small variations of score values in the
        # interesting range while not brutally collapsing all the low score values to
        # the same color.
        
        plt.figure(figsize=(8, 6))
        plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
        plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot,
        norm=MidpointNormalize(vmin=0.5, midpoint=np.max(scores)-0.05))
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.colorbar()
        plt.xticks(np.arange(len(xparam)), xparam, rotation=45)
        plt.yticks(np.arange(len(yparam)), yparam)
        plt.title(algorithm+' Validation F1 Score')
        plt.show()
        
#---------------------
  
    return scores, grid.best_params_, grid.best_score_

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
