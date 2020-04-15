# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 18:06:50 2018

@author: Rawan
"""
#----------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score ,GridSearchCV ,StratifiedKFold
from sklearn.metrics import f1_score, make_scorer, confusion_matrix

from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif

from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from keras.layers import Dense
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras import optimizers

#----------------------------------

path_1 = 'C:\\Users\\D\\Desktop\\Graduation Project Stuff\\- Datasets\\ENABL3S\\Candidates Features\\EMG_Wavelet_sym5_Features_Window_1500_Slide_1.pkl'

#---------------------------------------------------------------------------------------------------------------------------------------------------

standarize = True
selectKbest = False


features = pd.read_pickle(path_1)

#features_x=features.iloc[:,:features.shape[1]-1]
features_x=features.loc[:,:'RF_WT_D2_ZCs']
features_y=features.Mode.astype('category')
#features_y=pd.get_dummies(features.Mode)

#------------------------------------------------------------------------------

if selectKbest == True :
    sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
    features_x = sel.fit_transform(features_x)
    features_x = SelectKBest(mutual_info_classif, k=60).fit_transform(features_x, features_y)
  
#------------------------------------------------------------------------------

if standarize == True :
    scaler = StandardScaler()
    features_x =scaler.fit_transform(features_x)

#-------------------------------------------------------------------------------------------------------------
    
def f1_Score(y_true, y_pred):
    confusion_Matrix = confusion_matrix(y_true, y_pred)
    #   Multiclass
    tp, fp, fn =[],[],[]
    #   weighting
    for i in range(7):
        confusion_Matrix[i,:] = (confusion_Matrix[i,:]/np.sum(confusion_Matrix[i,:]))*100
    
    for i in range(7):
        tp.append(confusion_Matrix[i,i])
        fp.append(np.sum(confusion_Matrix[:,i]) - tp[i])
        fn.append(np.sum(confusion_Matrix[i,:]) - tp[i])
        
    tp_avg, fp_avg, fn_avg = np.average(tp), np.average(fp), np.average(fn)
    
    precision = tp_avg / tp_avg + fp_avg
    recall = tp_avg / tp_avg + fn_avg
    
#    # Uniclass
#    tp = confusion_Matrix[0,0]
#    fp = confusion_Matrix[1,0]
#    fn = confusion_Matrix[0,1]
#    
#    precision = tp / tp + fp
#    recall = tp / tp + fn
    
    return (2*precision*recall) / (precision + recall)

def tp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 0]
def tn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 0]
def fp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 0]
def fn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 1]

scoring = {'tp' : make_scorer(tp), 'tn' : make_scorer(tn),
           'fp' : make_scorer(fp, greater_is_better=False), 'fn' : make_scorer(fn, greater_is_better=False)}
#------------------------------------------------------------------------------    

x_train , x_test, y_train, y_test = train_test_split(features_x, features_y, test_size=0.3 , random_state=123)

clf_KNN = KNeighborsClassifier()
clf_LR =LogisticRegression()
clf_SVM =svm.SVC()
#clf_LDA = LinearDiscriminantAnalysis()

#------------------------------------------------------------------------------


#mean_val_scores =[result.mean_validation_score for result in grid.grid_scores_]
#plt.plot(range(1,31) , mean_val_scores)
#plt.show()

param_grid_KNN=dict(n_neighbors=np.arange(1,15))
param_grid_LR=dict(solver=('lbfgs', 'liblinear'))
param_grid_SVM=dict(kernel=('linear','poly'), degree=np.arange(1,8))
param_grid_LDA=dict(solver=('lsqr','eigen'), shrinkage=('auto',None))


print('KNN starting...')
grid_KNN = OneVsRestClassifier(GridSearchCV(clf_KNN, param_grid=param_grid_KNN , cv=5 , scoring = scoring, refit = False))
grid_KNN.fit(x_train,y_train)
predicted_KNN=grid_KNN.predict(x_test)
print('LR starting...')
grid_LR = OneVsRestClassifier(GridSearchCV(clf_LR, param_grid=param_grid_LR , cv=5 , scoring= scoring, refit = False))
grid_LR.fit(x_train,y_train)
predicted_LR=grid_LR.predict(x_test)
print('SVM starting...')
grid_SVM = OneVsRestClassifier(GridSearchCV(clf_SVM, param_grid=param_grid_SVM , cv=5 , scoring= scoring, refit = False))
grid_SVM.fit(x_train,y_train)
predicted_SVM=grid_SVM.predict(x_test)

#grid_LDA = GridSearchCV(clf_LDA, param_grid=param_grid_LDA , cv=10 , scoring=['accuracy', 'precision'])
#grid_LDA.fit(x_train,y_train)
#predicted_LDA =grid_LDA.predict(x_test)

print('KNN F1 score: ' , np.round(f1_score(y_test , predicted_KNN , average = 'weighted'),4))
print('LR F1 score: ' , np.round(f1_score(y_test , predicted_LR , average = 'weighted'),4))
print('SVM F1 score: ' , np.round(f1_score(y_test , predicted_SVM , average = 'weighted'),4))
#print('LDA F1 score: ' , np.round(f1_score(y_test , predicted_LDA , average = 'weighted'),4))

print('KNN accuracy: ' , np.round(grid_KNN.best_score_,4))
print('LR accuracy: ' , np.round(grid_LR.best_score_,4))
print('SVM accuracy: ' , np.round(grid_SVM.best_score_,4))
#print('LDA accuracy: ' , np.round(grid_LDA.best_score_,4))

# using scoring = 'accuracy' :
#    KNN F1 score:  0.704
#    LR F1 score:  0.7582
#    SVM F1 score:  0.7689
#    KNN accuracy:  0.7237
#    LR accuracy:  0.7648
#    SVM accuracy:  0.7723

# using scoring = make_scorer(f1_Score) (multiclass) :
#    KNN F1 score:  0.6712
#    LR F1 score:  0.7568
#    SVM F1 score:  0.6266

# using scoring = make_scorer(f1_Score) (OvR):
#    KNN F1 score:  0.6871
#    LR F1 score:  0.7582
#    SVM F1 score:  0.7387

# using scoring = = 'accuracy' (OvR):
#    KNN F1 score:  0.721
#    LR F1 score:  0.7575
#    SVM F1 score:  0.7447