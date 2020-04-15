# -*- coding: utf-8 -*-
"""
Created on Tue May 22 19:36:13 2018


"""
import pandas as pd
import matplotlib.pyplot as plt
import time
from datetime import timedelta
from GP_Functions import WTDenoise , BWDenoise , Extract_Features , Evaluate_Classification , GridScore

#%%

def Candidate_Features_Extraction(path , end_circuit , slide , candidateID):

    columns_to_use= ['Right_Shank_Ax', 'Right_Shank_Ay', 'Right_Shank_Az', 'Right_Shank_Gy',
                     'Right_Shank_Gz', 'Right_Shank_Gx', 'Right_Thigh_Ax', 'Right_Thigh_Ay',
                     'Right_Thigh_Az', 'Right_Thigh_Gy', 'Right_Thigh_Gz', 'Right_Thigh_Gx',
                     'Left_Shank_Ax', 'Left_Shank_Ay', 'Left_Shank_Az', 'Left_Shank_Gy',
                     'Left_Shank_Gz', 'Left_Shank_Gx', 'Left_Thigh_Ax', 'Left_Thigh_Ay',
                     'Left_Thigh_Az', 'Left_Thigh_Gy', 'Left_Thigh_Gz', 'Left_Thigh_Gx',
                     'Right_TA', 'Right_MG', 'Right_SOL', 'Right_BF', 'Right_ST', 'Right_VL',
                     'Right_RF', 'Left_TA', 'Left_MG', 'Left_SOL', 'Left_BF', 'Left_ST',
                     'Left_VL', 'Left_RF', 'Right_Knee', 'Left_Knee', 'Mode']
    
    columns_to_denoise=['Right_Shank_Ax', 'Right_Shank_Ay', 'Right_Shank_Az', 'Right_Shank_Gy',
                        'Right_Shank_Gz', 'Right_Shank_Gx', 'Right_Thigh_Ax', 'Right_Thigh_Ay',
                        'Right_Thigh_Az', 'Right_Thigh_Gy', 'Right_Thigh_Gz', 'Right_Thigh_Gx',
                        'Left_Shank_Ax', 'Left_Shank_Ay', 'Left_Shank_Az', 'Left_Shank_Gy',
                        'Left_Shank_Gz', 'Left_Shank_Gx', 'Left_Thigh_Ax', 'Left_Thigh_Ay',
                        'Left_Thigh_Az', 'Left_Thigh_Gy', 'Left_Thigh_Gz', 'Left_Thigh_Gx',
                        'Right_TA', 'Right_MG', 'Right_SOL', 'Right_BF', 'Right_ST', 'Right_VL',
                        'Right_RF', 'Left_TA', 'Left_MG', 'Left_SOL', 'Left_BF', 'Left_ST',
                        'Left_VL', 'Left_RF', 'Right_Knee', 'Left_Knee']
    
    columns_to_denoise_WT =['Right_TA', 'Right_MG', 'Right_SOL', 'Right_BF', 'Right_ST', 'Right_VL',
                            'Right_RF', 'Left_TA', 'Left_MG', 'Left_SOL', 'Left_BF', 'Left_ST',
                            'Left_VL', 'Left_RF']
    
    
    
    
    
    features_all = pd.DataFrame()
    
    for j in range(1 , end_circuit+1):
        
        start_file_time = time.monotonic()
        circuit = pd.read_csv(path+format(j, "03d")+'_post.csv' , usecols = columns_to_use)
#        circuit = pd.read_csv(path+str(j)+'.csv' , usecols = columns_to_use)
        
        denoised_circuit, WT_RMSE = WTDenoise(circuit, columns_to_denoise_WT)
        denoised_circuit, BW_RMSE = BWDenoise(denoised_circuit, columns_to_denoise)
        
        ###      If you want to not use a spicific sensor reading or not use the left leg, get in the function and change.
        circuit_extracted_features = Extract_Features(denoised_circuit, IMU_mean= True, IMU_min= True,IMU_max = True, IMU_STD=True, IMU_init=True, IMU_finl=True,
#                        EMG_MAV= True, EMG_var= True, EMG_RMS= True, EMG_ZCs = True, EMG_WL= True, EMG_SSI= True, EMG_IEMG= True,
#                        EMG_Fmean = True, EMG_Fmedian = True,
#                        WT_cD_Mean= True, WT_cD_RMS = True, WT_cD_IWT = True, WT_cD_MAV = True, WT_cD_VAR = True, WT_cD_ZCs = True,
#                        WT_D2_Mean= True, WT_D2_RMS = True, WT_D2_IWT = True, WT_D2_MAV = True, WT_D2_VAR = True, WT_D2_ZCs  = True,
                        wavelet = 'sym5', window=1500 , slide =slide)

        end_file_time = time.monotonic()
        print('End of file '+str(j)+' : ' , timedelta(seconds=end_file_time - start_file_time))
    
        features_all = features_all.append(circuit_extracted_features , ignore_index =True)    
        
        features_discrete_Modes = pd.DataFrame()
        for i in (0,1,2,3,4,5,6):
            features_discrete_Modes= features_discrete_Modes.append(features_all.loc[features_all.Mode == i,:], ignore_index=True)
            
        features_discrete_Modes['CandidateID'] = candidateID
        
    return features_discrete_Modes

#------------------------------------------------------------------------------
        

path='C:\\Users\\D\\Desktop\\Graduation Project Stuff\\- Datasets\\ENABL3S\\'

AB156_end_circuit = 50
AB185_end_circuit = 50
AB186_end_circuit = 44
AB188_end_circuit = 40
AB189_end_circuit = 49
AB190_end_circuit = 49
AB191_end_circuit = 42
AB192_end_circuit = 48
AB193_end_circuit = 50
AB194_end_circuit = 50

circuit_ends = [AB156_end_circuit]#, AB185_end_circuit, AB186_end_circuit, AB188_end_circuit, AB189_end_circuit,
#                AB190_end_circuit, AB191_end_circuit, AB192_end_circuit, AB193_end_circuit, AB194_end_circuit]

candidates = ['AB156']# ,'AB185', 'AB186', 'AB188', 'AB189', 'AB190', 'AB191', 'AB192', 'AB193', 'AB194']

for s in range(1,2):#, 0.75, 0.5, 0.25):
    features_for_all= pd.DataFrame()
    for candidate, circuit_end in zip(candidates,circuit_ends):
        print('\nStarting candidate '+candidate)
        start_candidate_time = time.monotonic()
        from_path = path+candidate+'\\Processed\\'+candidate+'_Circuit_'
#        from_path = path+candidate+'\\Our_Processed\\Circuit_'
        features_for_all = features_for_all.append(Candidate_Features_Extraction(from_path, circuit_end , slide=s , candidateID=candidate))
        end_candidate_time = time.monotonic()
        print('End of candidate '+candidate +' ', timedelta(seconds=end_candidate_time - start_candidate_time))
        
    
    features_for_all.to_pickle('C:\\Users\\D\\Desktop\\Graduation Project Stuff\\- Datasets\\ENABL3S\\Candidates Features\\'+'Post_IMU_WT+BW_Window_1500_Slide_'+str(s)+'.pkl')

#%%
path1 = 'C:\\Users\\D\\Desktop\\Graduation Project Stuff\\- Datasets\\ENABL3S\\Candidates Features\\EMG_Wavelet_sym5_Features_Window_1500_Slide_1.pkl'
path2 = 'C:\\Users\\D\\Desktop\\Graduation Project Stuff\\- Datasets\\ENABL3S\\Candidates Features\\EMG_Time_Features_Window_1500_Slide_1.pkl'
path3 = 'C:\\Users\\D\\Desktop\\Graduation Project Stuff\\- Datasets\\ENABL3S\\Candidates Features\\EMG_Frequency_Features_Window_1500_Slide_1.pkl'
path4 = 'C:\\Users\\D\\Desktop\\Graduation Project Stuff\\- Datasets\\ENABL3S\\Candidates Features\\IMU_Features_Window_1500_Slide_1.pkl'
path5 = 'C:\\Users\\D\\Desktop\\Graduation Project Stuff\\- Datasets\\ENABL3S\\Candidates Features\\EMG_Wavelet_sym5_Features_Window_1500_Slide_0.75.pkl'
path6 = 'C:\\Users\\D\\Desktop\\Graduation Project Stuff\\- Datasets\\ENABL3S\\Candidates Features\\EMG_Time_Features_Window_1500_Slide_0.75.pkl'
path7 = 'C:\\Users\\D\\Desktop\\Graduation Project Stuff\\- Datasets\\ENABL3S\\Candidates Features\\EMG_Frequency_Features_Window_1500_Slide_0.75.pkl'
path8 = 'C:\\Users\\D\\Desktop\\Graduation Project Stuff\\- Datasets\\ENABL3S\\Candidates Features\\IMU_Features_Window_1500_Slide_0.75.pkl'
path9 = 'C:\\Users\\D\\Desktop\\Graduation Project Stuff\\- Datasets\\ENABL3S\\Candidates Features\\EMG_Wavelet_sym5_Features_Window_1500_Slide_0.5.pkl'
path10 = 'C:\\Users\\D\\Desktop\\Graduation Project Stuff\\- Datasets\\ENABL3S\\Candidates Features\\EMG_Time_Features_Window_1500_Slide_0.5.pkl'
path11 = 'C:\\Users\\D\\Desktop\\Graduation Project Stuff\\- Datasets\\ENABL3S\\Candidates Features\\EMG_Frequency_Features_Window_1500_Slide_0.5.pkl'
path12 = 'C:\\Users\\D\\Desktop\\Graduation Project Stuff\\- Datasets\\ENABL3S\\Candidates Features\\IMU_Features_Window_1500_Slide_0.5.pkl'
path13 = 'C:\\Users\\D\\Desktop\\Graduation Project Stuff\\- Datasets\\ENABL3S\\Candidates Features\\EMG_Wavelet_sym5_Features_Window_1500_Slide_0.25.pkl'
path14 = 'C:\\Users\\D\\Desktop\\Graduation Project Stuff\\- Datasets\\ENABL3S\\Candidates Features\\EMG_Time_Features_Window_1500_Slide_0.25.pkl'
path15 = 'C:\\Users\\D\\Desktop\\Graduation Project Stuff\\- Datasets\\ENABL3S\\Candidates Features\\EMG_Frequency_Features_Window_1500_Slide_0.25.pkl'
path16 = 'C:\\Users\\D\\Desktop\\Graduation Project Stuff\\- Datasets\\ENABL3S\\Candidates Features\\IMU_Features_Window_1500_Slide_0.25.pkl'

copmarision_list = (path1,path2,path3,path4,path5,path6,path7,path8,path9,path10,path11,path12,path13,path14,path15,path16)
preformance_KNN, preformance_LR, preformance_SVM = Evaluate_Classification(copmarision_list=copmarision_list, cv = 5)

preformance = pd.DataFrame({'KNN' : preformance_KNN , 'LR': preformance_LR , 'SVM' : preformance_SVM },
                           index=('EMG Wavelet_sym5, w=1500,s=1',    'EMG Time, w=1500,s=1',    'EMG Frequency, w=1500,s=1',    'IMU, w=1500,s=1',
                                  'EMG Wavelet_sym5, w=1500,s=0.75', 'EMG Time, w=1500,s=0.75', 'EMG Frequency, w=1500,s=0.75', 'IMU, w=1500,s=0.75',
                                  'EMG Wavelet_sym5, w=1500,s=0.5',  'EMG Time, w=1500,s=0.5',  'EMG Frequency, w=1500,s=0.5',  'IMU, w=1500,s=0.5',
                                  'EMG Wavelet_sym5, w=1500,s=0.25', 'EMG Time, w=1500,s=0.25', 'EMG Frequency, w=1500,s=0.25', 'IMU, w=1500,s=0.25'))


preformance.to_pickle('C:\\Users\\D\\Desktop\\Graduation Project Stuff\\- Datasets\\ENABL3S\\Candidates Features\\Preformance_All_preOptimization')


#%%
plt.title('Different Features')
plt.xlabel("Features")
plt.ylabel("F1 Score")
preformance.KNN.plot()
preformance.LR.plot()
preformance.SVM.plot()
plt.legend()
plt.show()

#%%

path1= 'C:\\Users\\D\\Desktop\\Graduation Project Stuff\\- Datasets\\ENABL3S\\Candidates Features\\Raw_IMU_WT+BW_Window_1500_Slide_1.pkl'
path2= 'C:\\Users\\D\\Desktop\\Graduation Project Stuff\\- Datasets\\ENABL3S\\Candidates Features\\Post_IMU_WT_Window_1500_Slide_1.pkl'
path3= 'C:\\Users\\D\\Desktop\\Graduation Project Stuff\\- Datasets\\ENABL3S\\Candidates Features\\Our_Post_IMU_WT_Window_1500_Slide_1.pkl'

path4= 'C:\\Users\\D\\Desktop\\Graduation Project Stuff\\- Datasets\\ENABL3S\\Candidates Features\\Post_IMU_WT+BW_Window_1500_Slide_1.pkl'
path5= 'C:\\Users\\D\\Desktop\\Graduation Project Stuff\\- Datasets\\ENABL3S\\Candidates Features\\Our_Post_IMU_WT+BW_Window_1500_Slide_1.pkl'

scores = GridScore(path5 , algorithm ='KNN' , use_PCA=False , HeatMap = True)

    
