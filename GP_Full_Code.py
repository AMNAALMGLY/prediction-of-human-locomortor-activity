# -*- coding: utf-8 -*-
"""
Created on Tue May 22 19:36:13 2018

"""
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
import pickle
from datetime import timedelta
from sklearn.model_selection import  StratifiedKFold ,train_test_split ,ShuffleSplit ,StratifiedShuffleSplit ,GroupShuffleSplit ,LeavePGroupsOut
from sklearn import metrics
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

from GP_Functions import WTDenoise , BWDenoise , Extract_Features , Evaluate_Classification , GridScore , DrawLearningCurve, NeuralNetwork,RNN_LSTM, plot_confusion_matrix, uniform

#%%

def Candidate_Features_Extraction(path , end_circuit , candidateID , slide=0.2 , window=1000 ):

    columns_to_use= ['Right_Shank_Ax', 'Right_Shank_Ay', 'Right_Shank_Az', 'Right_Shank_Gy',
                     'Right_Shank_Gz', 'Right_Shank_Gx', 'Right_Thigh_Ax', 'Right_Thigh_Ay',
                     'Right_Thigh_Az', 'Right_Thigh_Gy', 'Right_Thigh_Gz', 'Right_Thigh_Gx',
                     'Left_Shank_Ax', 'Left_Shank_Ay', 'Left_Shank_Az', 'Left_Shank_Gy',
                     'Left_Shank_Gz', 'Left_Shank_Gx', 'Left_Thigh_Ax', 'Left_Thigh_Ay',
                     'Left_Thigh_Az', 'Left_Thigh_Gy', 'Left_Thigh_Gz', 'Left_Thigh_Gx',
                     'Right_TA', 'Right_MG', 'Right_SOL', 'Right_BF', 'Right_ST', 'Right_VL',
                     'Right_RF', 'Left_TA', 'Left_MG', 'Left_SOL', 'Left_BF', 'Left_ST',
                     'Left_VL', 'Left_RF', 'Right_Knee', 'Left_Knee', 'Right_Ankle', 'Left_Ankle',
                     'Our_Mode']
    
    columns_to_denoise_BW=['Right_Shank_Ax', 'Right_Shank_Ay', 'Right_Shank_Az', 'Right_Shank_Gy',
                           'Right_Shank_Gz', 'Right_Shank_Gx', 'Right_Thigh_Ax', 'Right_Thigh_Ay',
                           'Right_Thigh_Az', 'Right_Thigh_Gy', 'Right_Thigh_Gz', 'Right_Thigh_Gx',
                           'Left_Shank_Ax', 'Left_Shank_Ay', 'Left_Shank_Az', 'Left_Shank_Gy',
                           'Left_Shank_Gz', 'Left_Shank_Gx', 'Left_Thigh_Ax', 'Left_Thigh_Ay',
                           'Left_Thigh_Az', 'Left_Thigh_Gy', 'Left_Thigh_Gz', 'Left_Thigh_Gx',
                           'Right_TA', 'Right_MG', 'Right_SOL', 'Right_BF', 'Right_ST', 'Right_VL',
                           'Right_RF', 'Left_TA', 'Left_MG', 'Left_SOL', 'Left_BF', 'Left_ST',
                           'Left_VL', 'Left_RF', 'Right_Knee', 'Left_Knee', 'Right_Ankle', 'Left_Ankle']
    
    columns_to_denoise_WT =['Right_TA', 'Right_MG', 'Right_SOL', 'Right_BF', 'Right_ST', 'Right_VL',
                            'Right_RF', 'Left_TA', 'Left_MG', 'Left_SOL', 'Left_BF', 'Left_ST',
                            'Left_VL', 'Left_RF']
    
    
    
    features_all = pd.DataFrame()
    
    for j in range(1 , end_circuit+1):
        
        start_file_time = time.monotonic()
        circuit = pd.read_csv(path+format(j, "03d")+'_post.csv' , usecols = columns_to_use)

        circuit_extracted_features = Extract_Features(circuit,# IMU_mean= True, IMU_min= True,IMU_max = True, IMU_STD=True, IMU_init=True, IMU_finl=True,
#                        IMU_AR0= True, IMU_AR1= True, IMU_AR2= True, IMU_AR3= True, IMU_AR4= True, IMU_AR5= True, IMU_AR6= True,
#                        IMUs_Fmean= True, IMUs_Fmedian= True, IMUs_MFmean= True, IMUs_MFmedian= True,
#                        EMG_MAV= True, EMG_var= True, EMG_RMS= True, EMG_ZCs = True, EMG_WL= True, EMG_SSI= True, EMG_IEMG= True,
                        EMG_Fmean = True, EMG_Fmedian = True, EMG_MFmean= True, EMG_MFmedian= True,
#                        WT_cD_Mean= True, WT_cD_RMS = True, WT_cD_IWT = True, WT_cD_MAV = True, WT_cD_VAR = True, WT_cD_ZCs = True,
#                        WT_D2_Mean= True, WT_D2_RMS = True, WT_D2_IWT = True, WT_D2_MAV = True, WT_D2_VAR = True, WT_D2_ZCs  = True,
                        wavelet = 'db7', window=window , slide =slide)

        end_file_time = time.monotonic()
        print('End of file '+str(j)+' : ' , timedelta(seconds=end_file_time - start_file_time))
    
        features_all = features_all.append(circuit_extracted_features , ignore_index =True)
        
    Mode_col = features_all.Mode
    features_all.drop(axis=1,columns='Mode',inplace=True)
    features_all=pd.concat([features_all,Mode_col],axis=1)
    
    features_all['CandidateID'] = candidateID
    
    return features_all

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

circuit_ends = [AB156_end_circuit, AB185_end_circuit, AB186_end_circuit, AB188_end_circuit, AB189_end_circuit,
                AB190_end_circuit, AB191_end_circuit, AB192_end_circuit, AB193_end_circuit, AB194_end_circuit]

candidates = ['AB156' ,'AB185', 'AB186', 'AB188', 'AB189', 'AB190', 'AB191', 'AB192', 'AB193', 'AB194']

slide = 0.2
window = 800
   
features_for_all= pd.DataFrame()

for candidate, circuit_end in zip(candidates,circuit_ends):
    print('\nStarting candidate '+candidate)
    start_candidate_time = time.monotonic()
    from_path = path+candidate+'\\Processed\\'+candidate+'_Circuit_'
    features_for_all = features_for_all.append(Candidate_Features_Extraction(from_path, circuit_end , candidateID=candidate , slide=slide , window=window ), ignore_index =True)
    end_candidate_time = time.monotonic()
    print('End of candidate '+candidate +' ', timedelta(seconds=end_candidate_time - start_candidate_time))
    

#features_for_all.to_pickle('C:\\Users\\D\\Desktop\\Graduation Project Stuff\\- Datasets\\ENABL3S\\Candidates Features 4'+'Post_EMG_FDFs_Window_'+str(window)+'_Slide_'+str(slide)+'.pkl')




#%%
#                                                       # Feature Selection - Subject Independent #

file_name ='Post_IMUs+EMG_ALL_Window_800_Slide_0.2'

path= 'C:\\Users\\D\\Desktop\\Graduation Project Stuff\\- Datasets\\ENABL3S\\Candidates Features 4\\'

data = pd.read_pickle(path+file_name+'.pkl')

X = data.iloc[:,:data.shape[1]-2]
Y = data.Mode.astype('category')

features = X.columns
F1_Score=[]


# Equalize
X,Y = uniform(X, Y, random_state=123)
groups = data.CandidateID[X.index.values]

gss = GroupShuffleSplit(n_splits=10, test_size=0.2, random_state=123)
for train_index, test_index in gss.split(X, Y, groups=groups):
#    print('Training On :',groups.iloc[train_index].unique())
    print('Testing On :',groups.iloc[test_index].unique())
    x_train= X.iloc[train_index]
    y_train= Y.iloc[train_index]
    x_test= X.iloc[test_index]
    y_test= Y.iloc[test_index]
    
    # Normalizing
    x_train = pd.DataFrame(scaler.fit_transform(x_train),columns= X.columns)
    x_test = pd.DataFrame(scaler.transform(x_test) ,columns= X.columns)

    clf = ExtraTreesClassifier(random_state=123)
    clf.fit(x_train, y_train)
    model = SelectFromModel(clf,prefit=True)

    features= features.intersection(X.columns[model.get_support()])
    
#    grid = GridScore(x_train[features] , y_train, file_name, 'LDA' , print_scores=False)
#    predicted = grid.predict(x_test[features])
#    F1_Score.append(metrics.f1_score(y_test , predicted , average='weighted'))


#print(F1_Score, np.mean(F1_Score))   
print(features,len(features))

#%%
#                                                       # Feature Selection - Subject dependent #


file_name ='Post_IMUs+EMG_ALL_Window_800_Slide_0.2'

path= 'C:\\Users\\D\\Desktop\\Graduation Project Stuff\\- Datasets\\ENABL3S\\Candidates Features 4\\'

data = pd.read_pickle(path+file_name+'.pkl')

X = data.iloc[:,:data.shape[1]-2]
Y = data.Mode.astype('category')

F1_Score=[]
features = X.columns

Candidates =['AB156','AB185','AB186','AB188','AB189','AB190','AB191','AB192','AB193','AB194']

for candidate in Candidates:
    # Equalize
    X_,Y_ = uniform(X[data.CandidateID == candidate], Y[data.CandidateID == candidate], random_state=123)
    
    x_train , x_test, y_train, y_test = train_test_split(X_, Y_, test_size=0.15 , random_state = 123)
    
    # Normalizing
    x_train = pd.DataFrame(scaler.fit_transform(x_train),columns= X.columns)
    x_test = pd.DataFrame(scaler.transform(x_test) ,columns= X.columns)

    clf = ExtraTreesClassifier(random_state=123)
    clf.fit(x_train, y_train)
    model = SelectFromModel(clf,prefit=True)

    features=X.columns[model.get_support()]
    
    grid = GridScore(x_train[features] , y_train, file_name, 'LDA' , print_scores=False)
    predicted = grid.predict(x_test[features])
    F1_Score.append(metrics.f1_score(y_test , predicted , average='weighted'))


print(F1_Score, np.mean(F1_Score))   
print(features,len(features))


#%%
#                                                   # Classification - Subject Independent #


class_names = ['Sitting','Level Walking' , 'Stair Up', 'Stair Down' ,'Ramp Up' ,'Ramp Down' ,'Sitting to Walking', 'Walking to Sitting']

f1_scorer = metrics.make_scorer(metrics.f1_score, average='weighted')
#------------------------------------------------------------------------------
                        # Change Here #
                        
file_name ='Post_EMG_ALL_Window_800_Slide_0.2'
algorithms =['NN']

#------------------------------------------------------------------------------

path= 'C:\\Users\\D\\Desktop\\Graduation Project Stuff\\- Datasets\\ENABL3S\\Candidates Features 4\\'
data = pd.read_pickle(path+file_name+'.pkl')

with open(path+file_name+'_Subject Independent_Selected Features', "rb") as f:
    features =pickle.load(f)
    
X = data[features[0]]
Y = data.Mode.astype('category')


for algorithm in algorithms:
    print('\nStarting algorithm '+algorithm+'\n')
    
    Accuracy= []
    F1_Score= []
    Classification_Report= []
    Confusion_Matrix= []
    Model_Size= []
    Prediction_time= []
    
    # Equalizing
    X ,Y = uniform(X, Y, random_state=123)
    groups = data.CandidateID[X.index.values]
    
    gss = GroupShuffleSplit(n_splits=10, test_size=0.2, random_state=123)
    for train_index, test_index in gss.split(X, Y, groups=groups):
    #    print('Training On :',groups.iloc[train_index].unique())
        print('Testing On :',groups.iloc[test_index].unique())
        
        x_train= X.iloc[train_index]
        y_train= Y.iloc[train_index]
        x_test= X.iloc[test_index]
        y_test= Y.iloc[test_index]
        
        # Normalizing
        x_train = pd.DataFrame(scaler.fit_transform(x_train),columns= X.columns)
        x_test = pd.DataFrame(scaler.transform(x_test) ,columns= X.columns)  
        
        if algorithm == 'NN':
                # Neural Network Model
            NN_model , history = NeuralNetwork(x_train , y_train, file_name=file_name, Draw_Learning_Curves=False)
            
#            Model_Size.append(sys.getsizeof(pickle.dumps(NN_model))) 
            start_prediction_time = time.clock()
            
            predicted = NN_model.predict_classes(x_test)
            
            end_prediction_time = time.clock()
            Prediction_time.append((end_prediction_time - start_prediction_time)*10**6)
            
        else:
                # SVM, LR, KNN, LDA, QDA, ETC Models
            grid = GridScore(x_train , y_train, file_name, algorithm , print_scores=False)
            
            Model_Size.append(sys.getsizeof(pickle.dumps(grid))) 
            start_prediction_time = time.clock()
            
            predicted = grid.predict(x_test)
            
            end_prediction_time = time.clock()
            Prediction_time.append((end_prediction_time - start_prediction_time)*10**6)
        
        Accuracy.append(metrics.accuracy_score(y_test , predicted))
        F1_Score.append(metrics.f1_score(y_test , predicted , average='weighted'))
        Classification_Report.append(metrics.classification_report(y_test, predicted,target_names=class_names))
        Confusion_Matrix.append(metrics.confusion_matrix(y_test , predicted))
    
        #print('\nAccuracy',Accuracy)
        #print('F1_Score',F1_Score)
        #print('\nClassification_report\n',Classification_report)
        #print('Confusion_Matrix\n',Confusion_Matrix)
    
    results = {'Accuracy':Accuracy ,'F1-Score':F1_Score, 'Classification Report':Classification_Report, 'Confusion Matrix':Confusion_Matrix, 'Model Size':Model_Size , 'Prediction Time':Prediction_time}
    info='Subject Independent , training on 8 candidates and testing on 2. All measures were recorded for each split. corresponding selected features were used.'
    with open(path+file_name+algorithm+'_Subject_Independent_Results', "wb") as f:
        pickle.dump([info , results], f) 
    

#%%
#                                                   # Feature Selection + Classification - Subject dependent #


class_names = ['Sitting','Level Walking' , 'Stair Up', 'Stair Down' ,'Ramp Up' ,'Ramp Down' ,'Sitting to Walking', 'Walking to Sitting']

f1_scorer = metrics.make_scorer(metrics.f1_score, average='weighted')
#------------------------------------------------------------------------------
                        # Change Here #
                        
file_name ='Post_IMUs_ALL_Window_800_Slide_0.2'
algorithms =['NN']

#------------------------------------------------------------------------------

path= 'C:\\Users\\D\\Desktop\\Graduation Project Stuff\\- Datasets\\ENABL3S\\Candidates Features 4\\'
data = pd.read_pickle(path+file_name+'.pkl')

X = data.iloc[:,:data.shape[1]-2]
Y = data.Mode.astype('category')


for algorithm in algorithms:
    print('\nStarting algorithm '+algorithm+'\n')
    
    Accuracy= []
    F1_Score= []
    Classification_Report= []
    Confusion_Matrix= []
    Model_Size= []
    Prediction_time= []
    
    Candidates =['AB156','AB185','AB186','AB188','AB189','AB190','AB191','AB192','AB193','AB194']
    
    for candidate in Candidates:
        # Equalize
        X_,Y_ = uniform(X[data.CandidateID == candidate], Y[data.CandidateID == candidate], random_state=123)
        
        x_train , x_test, y_train, y_test = train_test_split(X_,Y_, test_size=0.15 , random_state = 123)
        
        # Normalizing
        x_train = pd.DataFrame(scaler.fit_transform(x_train),columns= X.columns)
        x_test = pd.DataFrame(scaler.transform(x_test) ,columns= X.columns)
    
        clf = ExtraTreesClassifier(random_state=123)
        clf.fit(x_train, y_train)
        model = SelectFromModel(clf,prefit=True)
    
        features=X.columns[model.get_support()]
        x_train = x_train[features]
        x_test =x_test[features]
        
        if algorithm == 'NN':
                # Neural Network Model
#            x_train = np.array(x_train)[:,np.newaxis,:]
#            x_test = np.array(x_test)[:,np.newaxis,:]
#            NN_model , history = RNN_LSTM(x_train , y_train, file_name=file_name, Draw_Learning_Curves=False)
            NN_model , history = NeuralNetwork(x_train , y_train, file_name=file_name, Draw_Learning_Curves=False)
            
#            Model_Size.append(sys.getsizeof(pickle.dumps(NN_model))) 
            start_prediction_time = time.clock()
            
            predicted = NN_model.predict_classes(x_test)
            
            end_prediction_time = time.clock()
            Prediction_time.append((end_prediction_time - start_prediction_time)*10**6)
            
        else:
                # SVM, LR, KNN, LDA, QDA, ETC Models
            grid = GridScore(x_train , y_train, file_name, algorithm , print_scores=True)
            
            Model_Size.append(sys.getsizeof(pickle.dumps(grid))) 
            start_prediction_time = time.clock()
            
            predicted = grid.predict(x_test)
            
            end_prediction_time = time.clock()
            Prediction_time.append((end_prediction_time - start_prediction_time)*10**6)
        
        Accuracy.append(metrics.accuracy_score(y_test , predicted))
        F1_Score.append(metrics.f1_score(y_test , predicted , average='weighted'))
        Classification_Report.append(metrics.classification_report(y_test, predicted,target_names=class_names))
        Confusion_Matrix.append(metrics.confusion_matrix(y_test , predicted))
    
#        print('\nAccuracy',Accuracy)
#        print('F1_Score',F1_Score)
#        print('\nClassification_report\n',Classification_Report)
#        print('Confusion_Matrix\n',Confusion_Matrix)
    
    results = {'Accuracy':Accuracy ,'F1-Score':F1_Score, 'Classification Report':Classification_Report, 'Confusion Matrix':Confusion_Matrix, 'Model Size':Model_Size , 'Prediction Time':Prediction_time}
#    info='Subject dependent , training and testing on each candidate. All measures were recorded for each one. Feature Selection was saperatly performed for each and used.'
#    with open(path+file_name+algorithm+'_Subject_dependent_Results', "wb") as f:
#        pickle.dump([info , results], f) 




#%%
# subject dependent testing on ++== 900
# subject independent testing on ++== 12000
                                        # Results Gathering #
                                        
                                        

path= 'C:\\Users\\D\\Desktop\\Graduation Project Stuff\\- Datasets\\ENABL3S\\Candidates Features 4\\'
file_name ='Post_IMUs_ALL_Window_800_Slide_0.2'
SB='Subject_Independent_Results'


algorithms =['KNN','LR','LDA','QDA','SVM','ETC','NN']
metrics =['Accuracy', 'F1-Score', 'Model Size', 'Prediction Time']

Results=pd.DataFrame(index=algorithms, columns=metrics)

for algorithm in algorithms:

    with open(path+file_name+algorithm+'_'+SB, "rb") as f:
            info , results =pickle.load(f) 
    
    Results.loc[algorithm,'Accuracy']= np.mean(results['Accuracy'])
    Results.loc[algorithm,'F1-Score']= np.mean(results['F1-Score'])
    Results.loc[algorithm,'Model Size']= np.mean(results['Model Size']) / 1000
    Results.loc[algorithm,'Prediction Time']= np.mean(results['Prediction Time']) /12000
    
    Confusion_Matrix = np.zeros((8,8))
    for i in range(10):
        Confusion_Matrix= Confusion_Matrix + results['Confusion Matrix'][i]
    Confusion_Matrix= Confusion_Matrix/10
    
    class_names = ['Sitting','Level Walking' , 'Stair Up', 'Stair Down' ,'Ramp Up' ,'Ramp Down' ,'Sitting to Walking', 'Walking to Sitting']
    
    plt.figure(file_name+'  '+algorithm+' '+SB)
    plot_confusion_matrix(Confusion_Matrix, classes=class_names, normalize=True,
                          title='Normalized confusion matrix')
    
    plt.show()
    
#Results.to_csv(path+'Results\\'+file_name+'_All_'+SB+'.csv')

#%%

import pickle
import time
from datetime import timedelta
from sklearn.utils import resample

scaler = StandardScaler()


path = 'C:\\Users\\D\\Desktop\\Graduation Project Stuff\\- Datasets\\ENABL3S\\Candidates Features 4\\windows\\'

algorithm = 'LDA'

Accuracy = []
F1_Score= []

windows= [i for i in range(200,1600,100)]
slides = 0.2


for window in windows:  
    
    name_ ='AB156_Post_IMUs_ARFs_'
    name= name_+'Window_'+str(window)+'_Slide_'+str(slide)
    data = pd.read_pickle(path+name+'.pkl')
    X = data.iloc[:,:data.shape[1]-2]
    Y = data.Mode.astype('category')
    
    X , Y = uniform(X, Y, random_state=10)

    X , Y = resample(X,Y , n_samples= 640 ,replace =False )#, random_state=10)
    
    x_train , x_test, y_train, y_test = train_test_split(X, Y, test_size=0.15 )#, random_state = 123)
    x_train = pd.DataFrame(scaler.fit_transform(x_train),columns= X.columns)
    x_test = pd.DataFrame(scaler.transform(x_test) ,columns= X.columns)
    
    print('Starting... window = ', window)
    start_time = time.monotonic()       
    
    grid = GridScore(x_train , y_train, name_, algorithm , print_scores=False)
    predicted = grid.predict(x_test)
    
    end_time = time.monotonic()
    print('Ended : ' , timedelta(seconds=end_time - start_time))
    
    Accuracy.append(metrics.accuracy_score(y_test , predicted))
    F1_Score.append(metrics.f1_score(y_test , predicted , average='weighted'))
    
info = 'This trial was done to measure the effect of changing the window size against recognition performance using post data only from candidate AB156, slide=0.2 , windows [200,1500]. Resampling was done after equalization, 80*8=640 samples were used.'
results = {'Accuracy':Accuracy, 'F1_Score':F1_Score}
#
#with open(path+name_+algorithm+' Windows_Results', "wb") as f:
#        pickle.dump([info , results], f)  
        
#%%
F1_Score_ARFs = F1_Score  
  
#%%
plt.style.use('fivethirtyeight')

plt.figure(path+name_+algorithm+' Windows_Results')
plt.plot(windows,F1_Score_TDFs)
plt.plot(windows,F1_Score_FDFs)
plt.plot(windows,F1_Score)
plt.title('F1-Score vs Window Size')
#plt.ylim(0,1)
plt.ylabel('F1-Score')
plt.xlabel('Window Size')
plt.legend(['TDFs','FDFs','ARFs'])
plt.show()

#%%
import pickle
    
#with open('C:\\Users\\D\\Desktop\\Graduation Project Stuff\\- Datasets\\ENABL3S\\Candidates Features\\EMG_Time_Features_Window_1500_Slide_All_NN_Scores', "rb") as f:
#    EMG_Time_Accuracy , EMG_Time_F1_Score = pickle.load(f)
#with open('C:\\Users\\D\\Desktop\\Graduation Project Stuff\\- Datasets\\ENABL3S\\Candidates Features\\EMG_Wavelet_sym5_Features_Window_1500_Slide_All_NN_Scores', "rb") as f:
#    EMG_wavelet_Accuracy , EMG_wavelet_F1_Score = pickle.load(f)
#with open('C:\\Users\\D\\Desktop\\Graduation Project Stuff\\- Datasets\\ENABL3S\\Candidates Features\\EMG_Frequency_Features_Window_1500_Slide_All_NN_Scores', "rb") as f:
#    EMG_Frequency_Accuracy , EMG_Frequency_F1_Score = pickle.load(f)
with open('C:\\Users\\D\\Desktop\\Graduation Project Stuff\\- Datasets\\ENABL3S\\Candidates Features 2\\EMG_Frequency_WT+BW_Window_All_NN_Scores', "rb") as f:
    NN_Accuracy , NN_F1_Score = pickle.load(f)

with open('C:\\Users\\D\\Desktop\\Graduation Project Stuff\\- Datasets\\ENABL3S\\Candidates Features 2\\EMG_Frequency_WT+BW_Window_All_KNN_Scores', "rb") as f:
    _ , KNN_F1_Score = pickle.load(f)
    
#with open('C:\\Users\\D\\Desktop\\Graduation Project Stuff\\- Datasets\\ENABL3S\\Candidates Features\\EMG_Wavelet_sym5_Features_Window_1500_Slide_All_LR_Scores', "rb") as f:
#    _ , LR_F1_Score = pickle.load(f)

windows =[1000,2000,3000,4000]
#[250,500,750,1000,1250,1500,1750,2000,2250,2500,2750,3000,3250,3500,3750,4000,4250,4500,4750,5000,5250,5500,5750,6000,6250,6500,6750,7000]
slides = [1 , 0.75 , 0.5 , 0.25]

#EMG = pd.DataFrame({'Accuracy' : EMG_Time_Accuracy , 'F1_Score': EMG_Time_F1_Score},
#                           index=slides)
#IMUs = pd.DataFrame({'Accuracy' : IMUs_Accuracy , 'F1_Score': IMUs_F1_Score},
#                           index=slides)
graph = pd.DataFrame({ 'NN' : NN_F1_Score[:4] , 'KNN' : KNN_F1_Score },
                           index=windows)


plt.style.use('fivethirtyeight')

plt.figure('EMG Frequency Features F1 Scores vs Windows')
graph.KNN.plot()
#graph.LR.plot()
graph.NN.plot()
plt.xlabel('Slide')
plt.ylabel('Score')
plt.legend()
plt.show()

#plt.figure('IMUs Accuracy and F1 Score')
#IMUs.Accuracy.plot()
#IMUs.F1_Score.plot()
#plt.xlabel('Window Size')
#plt.ylabel('Score')
#plt.legend()
#plt.show()

#%%

import pickle
    
#with open('C:\\Users\\D\\Desktop\\Graduation Project Stuff\\- Datasets\\ENABL3S\\Candidates Features 2\\EMG_Time_WT+BW_Window_All_slide_1_NN_Scores', "rb") as f:
#    EMG_Time_Accuracy , EMG_Time_F1_Score = pickle.load(f)
with open('C:\\Users\\D\\Desktop\\Graduation Project Stuff\\- Datasets\\ENABL3S\\Candidates Features 2\\IMUs_Window_1500_slide_ALL_NN_Scores', "rb") as f:
    IMUs_Accuracy , IMUs_F1_Score = pickle.load(f)

windows = [250,500,750,1000,1250,1500,1750,2000,2250,2500,2750,3000,3250,3500,3750]
slides= [1,0.75,0.5,0.25]

#EMG_time = pd.DataFrame({'Accuracy' : EMG_Time_Accuracy , 'F1_Score': EMG_Time_F1_Score},
#                           index=windows)
IMUs = pd.DataFrame({'Accuracy' : IMUs_Accuracy , 'F1_Score': IMUs_F1_Score},
                           index=slides)

plt.style.use('fivethirtyeight')
#
#plt.figure('EMG Time Accuracy and F1 Score')
#EMG_time.Accuracy.plot()
#EMG_time.F1_Score.plot()
#plt.xlabel('Window Size')
#plt.ylabel('Score')
#plt.legend()
#plt.show()

plt.figure('IMUs Accuracy and F1 Score')
IMUs.Accuracy.plot()
IMUs.F1_Score.plot()
plt.xlabel('Slide')
plt.ylabel('Score')
plt.ylim(0.90 , 0.95)
plt.legend()
plt.show()


#%%
    
#path = 'C:\\Users\\D\\Desktop\\Graduation Project Stuff\\- Datasets\\ENABL3S\\Candidates Features 2\\EMG_Time_WT+BW_Window_1000_Slide_1.pkl'
path = 'C:\\Users\\D\\Desktop\\Graduation Project Stuff\\- Datasets\\ENABL3S\\AB156\\Raw\\'
name ='AB156_Circuit_001_raw'
data = pd.read_csv(path+name+'.csv')

#Accuracy , F1_Score  = NeuralNetwork(data.iloc[:,:data.shape[1]-2], data.Our_Mode, name, uniform=False , print_metrics = False , Draw_Learning_Curves=False)
F1_Score , best_score , best_params , best_estimator  = GridScore(data.iloc[:,:data.shape[1]-2], data.Our_Mode, name, 'LR' , uniform=False , use_PCA=False , HeatMap = False )



#%%

raw_WT_BW ='C:\\Users\\D\\Desktop\\Graduation Project Stuff\\- Datasets\\ENABL3S\\Candidates Features\\Raw vs Post vs OPost\\'+'Post_IMU_WT+BW_Window_1500_Slide_1.pkl'
raw_WT ='C:\\Users\\D\\Desktop\\Graduation Project Stuff\\- Datasets\\ENABL3S\\Candidates Features\\Raw vs Post vs OPost\\'+'Post_IMU_WT_Window_1500_Slide_1.pkl'

algorithm ='KNN'


scores_raw_WT_BW , best_score_raw_WT_BW , best_params_raw_WT_BW , best_estimator_raw_WT_BW = GridScore(raw_WT_BW , algorithm = algorithm , use_PCA=False , HeatMap = False )
DrawLearningCurve(raw_WT_BW , algorithm = algorithm , best_params= best_params_raw_WT_BW)
#scores_raw_raw_WT , best_score_raw_WT , best_params_raw_WT , best_estimator_raw_WT = GridScore(raw_WT , algorithm = algorithm , use_PCA=False , HeatMap = False)

#print(algorithm+' raw_WT :' ,best_score_raw_WT)
print(algorithm+' raw_WT_BW :' ,best_score_raw_WT_BW)

#%%
from scipy import signal
raw =pd.read_csv('C:\\Users\\D\\Desktop\\Graduation Project Stuff\\- Datasets\\ENABL3S\\AB185\\Raw\\AB185_Circuit_050_raw.csv')
post =pd.read_csv('C:\\Users\\D\\Desktop\\Graduation Project Stuff\\- Datasets\\ENABL3S\\AB185\\Processed\\AB185_Circuit_050_post.csv')

col='Left_TA'


#rawDen, _ = BWDenoise(raw, [col])

#post, _ = WTDenoise(post, [col])

def notch_filter(data, freq, fs=100):
    Q = 1# Quality factor
    w0 = freq/(fs/2) # Normalized Frequency
    # Design notch filter
    b, a = signal.iirnotch(w0, Q)
    zi = signal.lfilter_zi(b, a)
    y,_ = signal.lfilter(b, a, data,zi=zi*data[0] )
    return y

#rawDen[col] = notch_filter(raw[col], 300, fs=1000)
rawDen, _ = WTDenoise(raw, [col])

f, Pxx_den = signal.welch(post[col], 1000,nperseg=1024)
f_, Pxx_den_ = signal.welch(raw[col], 1000,nperseg=1024)
f__, Pxx_den__ = signal.welch(rawDen[col], 1000,nperseg=1024)

plt.subplot(3,1,1)
plt.title("Raw")
plt.semilogy(f_, Pxx_den_)
#plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')

plt.subplot(3,1,2)
plt.title("Raw Denoised")
plt.semilogy(f__, Pxx_den__)
#plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')

plt.subplot(3,1,3)
plt.title("Post")
plt.semilogy(f, Pxx_den)
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')

plt.show()

#%%
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.arima_model import ARMA 
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.ar_model import AR
import numpy as np

post =pd.read_csv('C:\\Users\\D\\Desktop\\Graduation Project Stuff\\- Datasets\\ENABL3S\\AB185\\Raw\\AB185_Circuit_049_raw.csv')


#%%

#plot_pacf(post['Left_Shank_Ax'], lags=20)
#plt.show()

# Fit the data to an AR(p) for p = 0,...,6 , and save the 

BIC = []
for p in range(20):
    mod = ARMA(post.Right_Knee.values, order=(p,0))
    try:
        res = mod.fit()
    except:
        continue
# Save BIC for AR(p)    
    BIC.append(res.bic)
    
# Plot the BIC as a function of p
plt.plot(range(len(BIC)), BIC, marker='o')
plt.xlabel('Order of AR Model')
plt.ylabel('Baysian Information Criterion')
plt.show()

#%%

for col in ['Left_Shank_Ax','Left_Shank_Ay','Left_Shank_Az','Left_Shank_Gx','Left_Shank_Gy','Left_Shank_Gz',
                              'Left_Thigh_Ax','Left_Thigh_Ay','Left_Thigh_Az','Left_Thigh_Gx','Left_Thigh_Gy','Left_Thigh_Gz',
                              'Left_Knee','Left_Ankle']:
    BIC = []
    AIC =[]
    for p in range(20):
        res = AR(post[col].values).fit(p)
    # Save BIC for AR(p)    
        BIC.append(res.bic)
        AIC.append(res.aic)
        
    # Plot the BIC as a function of p
    plt.figure(col)
    plt.plot([i for i in range(len(BIC))], BIC, marker='o')
    plt.plot([i for i in range(len(AIC))], AIC, marker='o')
    plt.xticks([i for i in range(len(BIC))])
    plt.xlabel('Order of AR Model')
    plt.ylabel('Baysian Information Criterion')
    plt.legend()
    plt.show()
    
#%%

for mode in [0,1,3,4,6]:
    BIC = []
    AIC =[]
    for p in range(20):
        res = AR(post.Right_Knee[post.Mode == mode].values).fit(p)
    # Save BIC for AR(p)    
        BIC.append(res.bic)
        AIC.append(res.aic)
        
    # Plot the BIC as a function of p
    plt.figure(mode)
    plt.plot([i for i in range(len(BIC))], BIC, marker='o')
    plt.plot([i for i in range(len(AIC))], AIC, marker='o')
    plt.xticks([i for i in range(len(BIC))])
    plt.xlabel('Order of AR Model')
    plt.ylabel('Baysian Information Criterion')
    plt.legend()
    plt.show()
    
