# -*- coding: utf-8 -*-
"""
Created on Wed May  9 19:30:25 2018

@author: Rawan
"""

#----------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

import time
from datetime import timedelta

from sklearn.model_selection import train_test_split , ShuffleSplit , StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score ,GridSearchCV ,StratifiedKFold , learning_curve
from sklearn import metrics

from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif

from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis , QuadraticDiscriminantAnalysis
from sklearn.decomposition import PCA

#----------------------------------

path_1 = 'C:\\Users\\D\\Desktop\\Graduation Project Stuff\\- Datasets\\ENABL3S\\Candidates Features\\EMG_Wavelet_sym5_Features_Window_1500_Slide_1.pkl'
path_2 = 'C:\\Users\\D\\Desktop\\Graduation Project Stuff\\- Datasets\\ENABL3S\\Candidates Features\\EMG_Time_Features_Window_1500_Slide_1.pkl'
path_3 = 'C:\\Users\\D\\Desktop\\Graduation Project Stuff\\- Datasets\\ENABL3S\\Candidates Features\\EMG_Frequency_Features_Window_1500_Slide_1.pkl'
path_4 = 'C:\\Users\\D\\Desktop\\Graduation Project Stuff\\- Datasets\\ENABL3S\\Candidates Features\\IMU_Features_Window_1500_Slide_1.pkl'

path_0 = 'C:\\Users\\D\\Desktop\\Graduation Project Stuff\\- Datasets\\ENABL3S\\Algorithms Preformances and Parameters\\'

#---------------------------------------------------------------------------------------------------------------------------------------------------

preformance_KNN= []
preformance_LR = []
preformance_SVM= []
preformance_NuSVM= []
preformance_LDA= []
preformance_QDA= []
preformance_SGD= []

scaler = StandardScaler()

#%%

features = pd.read_pickle(path_1)
features_x= features.iloc[:,:features.shape[1]-1]
features_x= pd.DataFrame(scaler.fit_transform(features_x) , columns= features_x.columns)
features_y= features.Mode.astype('category')

#pca = PCA(0.99)
#pca.fit(features_x)
#PCA_components= pd.DataFrame(pca.components_, columns=list(features_x.columns))
#Columns_from_PCA = features_x.columns[PCA_components.describe().loc['75%',:] >= 0.00001]
#features_x= pd.DataFrame(features_x.loc[:,Columns_from_PCA] , columns = Columns_from_PCA)

print('\nStarted...')

for i in range(5):   
    
    start_time = time.monotonic() 

    x_train , x_test, y_train, y_test = train_test_split(features_x, features_y, test_size=0.3 , random_state=123)

    clf_KNN = KNeighborsClassifier(algorithm='ball_tree', weights='distance', metric ='manhattan',p=1, n_neighbors=4, leaf_size=10)
    
    clf_KNN.fit(x_train,y_train)
    predicted_KNN=clf_KNN.predict(x_test)
 
    preformance_KNN.append(metrics.f1_score(y_test , predicted_KNN , average = 'weighted'))
    
    
    end_time = time.monotonic()
    print('End of iteration '+str(i)+' : ' , timedelta(seconds=end_time - start_time))

print('F1-Score',np.mean(preformance_KNN),'\n')

#%%
    
features = pd.read_pickle(path_1)

features_x=features.iloc[:,:features.shape[1]-1]
features_x= pd.DataFrame(scaler.fit_transform(features_x) , columns= features_x.columns)
features_y=features.Mode.astype('category')

#pca = PCA(0.99, whiten=True)
#pca.fit(features_x)
#PCA_components= pd.DataFrame(pca.components_, columns=list(features_x.columns))
#Columns_from_PCA = features_x.columns[PCA_components.describe().loc['75%',:] >= 0.00001]
#features_x= pd.DataFrame(features_x.loc[:,Columns_from_PCA] , columns = Columns_from_PCA)

print('\nStarted...')

for i in range(5):   
    
    start_time = time.monotonic() 
    
    x_train , x_test, y_train, y_test = train_test_split(features_x, features_y, test_size=0.3 , random_state=123)
    
    clf_LR =LogisticRegression(solver='newton-cg' , C=100 , multi_class='multinomial')
    
    clf_LR.fit(x_train,y_train)
    predicted_LR= clf_LR.predict(x_test)
            
    preformance_LR.append(metrics.f1_score(y_test , predicted_LR , average = 'weighted'))
    
    end_time = time.monotonic()
    print('End of iteration '+str(i)+' : ' , timedelta(seconds=end_time - start_time))


print('F1-Score',np.mean(preformance_LR),'\n')
    
#%%
    
features = pd.read_pickle(path_1)

features_x=features.iloc[:,:features.shape[1]-1]
features_x= pd.DataFrame(scaler.fit_transform(features_x) , columns= features_x.columns)
features_y=features.Mode.astype('category')

pca = PCA(0.99)
pca.fit(features_x)
PCA_components= pd.DataFrame(pca.components_, columns=list(features_x.columns))
Columns_from_PCA = features_x.columns[PCA_components.describe().loc['75%',:] >= 0.00001]
features_x= pd.DataFrame(features_x.loc[:,Columns_from_PCA] , columns = Columns_from_PCA)

print('\nStarted...')

for i in range(5):   
    
    start_time = time.monotonic() 
    
    x_train , x_test, y_train, y_test = train_test_split(features_x, features_y, test_size=0.3 , random_state=123)
    
    clf_SVM =svm.SVC(cache_size=500,class_weight ='balanced', kernel='rbf', C=10, gamma =0.01)
    
    clf_SVM.fit(x_train,y_train)
    predicted_SVM=clf_SVM.predict(x_test)
    
    preformance_SVM.append(metrics.f1_score(y_test , predicted_SVM , average = 'weighted'))
    
    end_time = time.monotonic()
    print('End of iteration '+str(i)+' : ' , timedelta(seconds=end_time - start_time))


print('F1-Score',np.mean(preformance_SVM),'\n')

#%%
    
for j in (path_1 , path_2 , path_3 , path_4 ):
    
    features = pd.read_pickle(j)
    
    print('\nclaculating file : '+str(j))
    start_time = time.monotonic()    

    features_x=features.iloc[:,:features.shape[1]-1]
    features_y=features.Mode.astype('category')
    
    scaler = StandardScaler()
    features_x =scaler.fit_transform(features_x)
    
    x_train , x_test, y_train, y_test = train_test_split(features_x, features_y, test_size=0.3 , random_state=123)
    
    clf_NuSVM =svm.NuSVC()

    param_grid_NuSVM=dict(nu=(0.01,0.025,0.05,0.075,0.09,0.1))
    
    grid_NuSVM = GridSearchCV(clf_NuSVM, param_grid=param_grid_NuSVM , cv=10 , scoring='accuracy')
    grid_NuSVM.fit(x_train,y_train)
    predicted_NuSVM=grid_NuSVM.predict(x_test)

    preformance_NuSVM.append(metrics.f1_score(y_test , predicted_NuSVM , average = 'weighted'))
    
    end_time = time.monotonic()
    print('End of file '+str(j)+' : ' , timedelta(seconds=end_time - start_time))
    
    print(grid_NuSVM.best_estimator_)
    print('F1-Score',preformance_NuSVM[-1],'\n')
    
#%%
    
for j in (path_1 , path_2 , path_3 , path_4 ):
    
    features = pd.read_pickle(j)
    
    print('\nclaculating file : '+str(j))
    start_time = time.monotonic()    

    features_x=features.iloc[:,:features.shape[1]-1]
    features_y=features.Mode.astype('category')
    
    scaler = StandardScaler()
    features_x =scaler.fit_transform(features_x)
    
    x_train , x_test, y_train, y_test = train_test_split(features_x, features_y, test_size=0.3 , random_state=123)
    
    clf_LDA = LinearDiscriminantAnalysis()
    
    grid_LDA = GridSearchCV(clf_LDA, param_grid={'solver':('svd','lsqr')} , cv=10 , scoring='accuracy')
    grid_LDA.fit(x_train,y_train)
    
    predicted_LDA=grid_LDA.predict(x_test)

    preformance_LDA.append(metrics.f1_score(y_test , predicted_LDA , average = 'weighted'))
    
    end_time = time.monotonic()
    print('End of file '+str(j)+' : ' , timedelta(seconds=end_time - start_time))
    
    print(grid_LDA.best_estimator_)
    print('F1-Score',preformance_LDA[-1],'\n')
    
#%%
    
for j in (path_1 , path_2 , path_3 , path_4 ):
    
    features = pd.read_pickle(j)
    
    print('\nclaculating file : '+str(j))
    start_time = time.monotonic()    

    features_x=features.iloc[:,:features.shape[1]-1]
    features_y=features.Mode.astype('category')
    
    scaler = StandardScaler()
    features_x =scaler.fit_transform(features_x)
    
    x_train , x_test, y_train, y_test = train_test_split(features_x, features_y, test_size=0.3 , random_state=123)
    
    clf_QDA = QuadraticDiscriminantAnalysis()
    
    grid_QDA = GridSearchCV(clf_QDA, param_grid={'reg_param':(0, 0.0001, 0.001, 0.1)} , cv=10 , scoring='accuracy')
    grid_QDA.fit(x_train,y_train)
    
    predicted_QDA=grid_QDA.predict(x_test)

    preformance_QDA.append(metrics.f1_score(y_test , predicted_QDA , average = 'weighted'))
    
    end_time = time.monotonic()
    print('End of file '+str(j)+' : ' , timedelta(seconds=end_time - start_time))
    
    print(grid_QDA.best_estimator_)
    print('F1-Score',preformance_QDA[-1],'\n')
    
#%%
    
for j in (path_1 , path_2 , path_3 , path_4 ):
    
    features = pd.read_pickle(j)
    
    print('\nclaculating file : '+str(j))
    start_time = time.monotonic()    

    features_x=features.iloc[:,:features.shape[1]-1]
    features_y=features.Mode.astype('category')
    
    scaler = StandardScaler()
    features_x =scaler.fit_transform(features_x)
    
    x_train , x_test, y_train, y_test = train_test_split(features_x, features_y, test_size=0.3 , random_state=123)
    
    clf_SGD = SGDClassifier(tol=1e-3, max_iter=1000)
    
    grid_SGD = GridSearchCV(clf_SGD, param_grid={'loss':('hinge','modified_huber','log'),'penalty':('l1','l2','elasticnet'), 'average':(True,False)} , cv=10 , scoring='accuracy')
    grid_SGD.fit(x_train,y_train)
    
    predicted_SGD= grid_SGD.predict(x_test)

    preformance_SGD.append(metrics.f1_score(y_test , predicted_SGD , average = 'weighted'))
    
    end_time = time.monotonic()
    print('End of file '+str(j)+' : ' , timedelta(seconds=end_time - start_time))
    
    print(grid_SGD.best_estimator_)
    print('F1-Score',preformance_SGD[-1],'\n')
    
    
#%%
    
#preformance = pd.DataFrame({'KNN' : preformance_KNN , 'LR': preformance_LR , 'SVM' : preformance_SVM },
#                           index=(1/4, 1/3, 1/2, 2/3, 1))
#
#plt.title('Different slides, window size = 1.5')
#plt.xlabel("Window Size")
#plt.ylabel("F1 Score")
#preformance.KNN.plot()
#preformance.LR.plot()
#preformance.SVM.plot()
#plt.legend()
#plt.show()
    
    
    SVM:
    0.7950096558941934,
    0.8055247809483382,
    0.693643246209745,
    0.9395665254420007
     
    SVC(C=100, cache_size=500, class_weight='balanced', coef0=0.0,
             decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
             max_iter=-1, probability=False, random_state=None, shrinking=True,
             tol=0.001, verbose=False)
    NuSVM:
    0.8016538694036559,
    0.8192868033784346,
    0.7031495995373367,
    0.941096268427481
    
    NuSVC(cache_size=200, class_weight=None, coef0=0.0,
               decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
               max_iter=-1, nu=0.1, probability=False, random_state=None,
               shrinking=True, tol=0.001, verbose=False)
     
     
     KNN:
     0.7849194862644131,
     0.8022559879742616,
     0.6988231377219171,
     0.9098424094073567
     
     KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='manhattan',
               metric_params=None, n_jobs=-1, n_neighbors=4, p=1,
               weights='distance')
     
     
     LR:
     0.7675165122225368,
     0.6963741805974868,
     0.6069337666437019,
     0.9277194740070576
     
     LogisticRegression(C=(1,100,1000), class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='multinomial',
              n_jobs=1, penalty='l2', random_state=None, solver='lbfgs',
              tol=0.0001, verbose=0, warm_start=False)
     
     LDA:
     0.7514477688803938,
     0.6447327918369256,
     0.5296250269784996,
     0.9240746072175784
     
     LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,
              solver='svd', store_covariance=False, tol=0.0001)
     
     QDA:
     0.617835184965448,
     0.5362539616341327,
     0.48557179411651047,
     0.8983957648460461
     
     QuadraticDiscriminantAnalysis(priors=None, reg_param=(0.1, 0.0001, 0),
               store_covariance=False, store_covariances=None, tol=0.0001)
     
     SGD:
     0.7409588644220708,
     0.6488129045512308,
     0.5887034124866779,
     0.9219196257179891
     
     SGDClassifier(alpha=0.0001, average=(True,False), class_weight=None, epsilon=0.1,
               eta0=0.0, fit_intercept=True, l1_ratio=0.15,
               learning_rate='optimal', loss=(all), max_iter=1000, n_iter=None,
               n_jobs=1, penalty=(all), power_t=0.5, random_state=None,
               shuffle=True, tol=0.001, verbose=0, warm_start=False)        # NEXT: Change alpha.
     

#%%
     
#-------------------------- Scores: -------------------------------
     
start_time = time.monotonic()

features = pd.read_pickle(path_1)
    
print('\nStarting...')
start_time = time.monotonic()    

features_x=features.iloc[:,:features.shape[1]-1]
features_x= pd.DataFrame(scaler.fit_transform(features_x) , columns= features_x.columns)
features_y=features.Mode.astype('category')

#pca = PCA(0.99)
#pca.fit(features_x)
#PCA_components= pd.DataFrame(pca.components_, columns=list(features_x.columns))
#Columns_from_PCA = features_x.columns[PCA_components.describe().loc['75%',:] >= 0.0001]
#features_x= pd.DataFrame(features_x.loc[:,Columns_from_PCA] , columns = Columns_from_PCA)
#    
# It is usually a good idea to scale the data for SVM training.
# We are cheating a bit in this example in scaling all of the data,
# instead of fitting the transformation on the training set and
# just applying it on the test set.
#scaler = StandardScaler()
#features_x = scaler.fit_transform(features_x)
# #############################################################################
# Train classifiers
#
# For an initial search, a logarithmic grid with basis
# 10 is often helpful. Using a basis of 2, a finer
# tuning can be achieved but at a much higher cost.

cv = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
#
#C_range = np.logspace(-2, 10, 13)
#gamma_range = np.logspace(-9, 3, 13)
#param_grid = dict(C= C_range , gamma= gamma_range)
#grid = GridSearchCV(svm.SVC(cache_size=500,class_weight ='balanced', kernel='rbf'), param_grid=param_grid, cv=cv)

#k_range= np.arange(1,9)
#leaf_size_range = np.arange(10,105, 5, dtype=int)
#param_grid = dict(n_neighbors= k_range , leaf_size= leaf_size_range)
#grid = GridSearchCV(KNeighborsClassifier(algorithm='ball_tree', weights='distance', metric ='manhattan', p=1), param_grid=param_grid, cv=cv)

f1_scorer = metrics.make_scorer(metrics.f1_score, average='weighted')

C_range = np.logspace(-2, 10, 13)
solvers = ['newton-cg', 'lbfgs', 'sag', 'saga']
param_grid = dict(C= C_range , solver= solvers)
grid = GridSearchCV(LogisticRegression(multi_class='multinomial' ), param_grid=param_grid, cv=cv , scoring= f1_scorer)

grid.fit(features_x, features_y)
print("The best parameters are %s with a score of %0.3f"
% (grid.best_params_, grid.best_score_))
print(grid.best_estimator_)

#scores = grid.cv_results_['mean_test_score'].reshape(len(gamma_range), len(C_range))
#scores = grid.cv_results_['mean_test_score'].reshape(len(leaf_size_range),len(k_range))
scores = grid.cv_results_['mean_test_score'].reshape(len(C_range), len(solvers))


end_time = time.monotonic()
print('Time : ', timedelta(seconds=end_time - start_time))

#%%

#--------------------------- HeatMap: -------------------------------

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
norm=MidpointNormalize(vmin=0.5, midpoint=0.75))
plt.xlabel('Solver')
plt.ylabel('C')
plt.colorbar()
plt.xticks(np.arange(len(solvers)), solvers, rotation=45)
plt.yticks(np.arange(len(C_range)), C_range)
plt.title('Validation accuracy')
plt.show()

#%%

#-------------------------- Learning Curve: -------------------------------
    
features = pd.read_pickle(path_1)

print('\nStarted...')
start_time = time.monotonic()    

features_x=features.iloc[:,:features.shape[1]-1]
features_y=features.Mode.astype('category')
features_x= pd.DataFrame(scaler.fit_transform(features_x) , columns= features_x.columns)

#C = 100
#gamma = 0.1

#leaf_size = 10
#n_neighbors = 4

C = 1
solver = 'newton-cg'

# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.

#title = "Learning Curves (SVM, RBF kernel, $\gamma="+str(gamma)+"$ , C ="+str(C)
#title = "Learning Curves (KNN, Ball_Tree algorithm, n_neighbors="+str(n_neighbors)+" , leaf_size ="+str(leaf_size)
title = "Learning Curves (LR, Solver ="+str(solver)+" , C ="+str(C)
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)

#clf_SVM =svm.SVC(cache_size=500,class_weight ='balanced', kernel='rbf', C=C, gamma =gamma)
#clf_KNN = KNeighborsClassifier(algorithm='ball_tree', weights='distance', metric ='manhattan', p=1, n_neighbors=n_neighbors, leaf_size=leaf_size)
clf_LR = LogisticRegression(multi_class='multinomial' ,C= C, solver= solver)

plot_learning_curve(clf_LR, title, features_x, features_y, (0.6, 1.01), cv=cv, n_jobs=4)
plt.show()

end_time = time.monotonic()
print('End Time : ' , timedelta(seconds=end_time - start_time))

#%%
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
    plt.figure()
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
