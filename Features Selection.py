# -*- coding: utf-8 -*-
"""
Created on Mon May  7 17:17:16 2018

this code uses different algorithms for features selection to figure out what algorithm is the best
"""
#----------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.feature_selection import SelectKBest , f_classif , chi2 , RFECV , VarianceThreshold , SelectFromModel

from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression , LassoCV

from sklearn.model_selection import StratifiedKFold

#----------------------------------

path_1 = 'C:\\Users\\D\\Desktop\\Graduation Project Stuff\\- Datasets\\ENABL3S\\Candidates Features\\EMG_Wavelet_sym5_Features_Window_1500_Slide_1.pkl'
path_2 = 'C:\\Users\\D\\Desktop\\Graduation Project Stuff\\- Datasets\\ENABL3S\\Candidates Features\\EMG_Time_Features_Window_1500_Slide_1.pkl'
path_3 = 'C:\\Users\\D\\Desktop\\Graduation Project Stuff\\- Datasets\\ENABL3S\\Candidates Features\\EMG_Frequency_Features_Window_1500_Slide_1.pkl'
path_4 = 'C:\\Users\\D\\Desktop\\Graduation Project Stuff\\- Datasets\\ENABL3S\\Candidates Features\\IMU_Features_Window_1500_Slide_1.pkl'

#---------------------------------------------------------------------------------------------------------------------------------


scaler = StandardScaler()

features = pd.read_pickle(path_2)
features_X=features.iloc[:,:features.shape[1]-1]
features_y =features.Mode

#%%
#features_X =scaler.fit_transform(features_X)
#
#for i in range(features_X.shape[1]):
#    plt.figure()
#    sns.boxplot(x=features_y , y =features.iloc[:,i])
#    plt.savefig('C:\\Users\\D\\Desktop\\Graduation Project Stuff\\- Datasets\\ENABL3S\\Feature Variance\\'+str(i)+'. '+features.columns[i]+'.png')

#%%
#             T TEST              #

PValues=[]
for j in range(features_X.shape[1]):
    pval=[]
    for i in range(7):
        stat=stats.ttest_ind(features_X[features_y==i].iloc[:,j],features_X[features_y!=i].iloc[:,j])
        pval.append(stat.pvalue)
    PValues.append(np.mean(pval))

T_test = pd.DataFrame({'Feature':features.columns[:features_X.shape[1]],'Average_P_Value' : PValues})
sig = T_test[T_test.Average_P_Value <=0.05]

#%%
#            Extra tree classifier           #
clf = ExtraTreesClassifier()
clf = clf.fit(features_X, features_y)
imp=clf.feature_importances_



#%%
#            PCA       #
from sklearn import linear_model, decomposition, datasets
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

logistic = linear_model.LogisticRegression()
pca = decomposition.PCA()
pipe = Pipeline(steps=[('pca', pca), ('logistic', logistic)])

digits = datasets.load_digits()
X_digits = digits.data
y_digits = digits.target

# Plot the PCA spectrum
pca.fit(X_digits)
plt.figure(1, figsize=(4, 3))
plt.clf()
plt.axes([.2, .2, .7, .7])
plt.plot(pca.explained_variance_, linewidth=2)
plt.axis('tight')
plt.xlabel('n_components')
plt.ylabel('explained_variance_')

# Prediction
n_components = [20, 40, 64]
Cs = np.logspace(-4, 4, 3)

# Parameters of pipelines can be set using ‘__’ separated parameter names:
estimator = GridSearchCV(pipe, dict(pca__n_components=n_components, logistic__C=Cs))
estimator.fit(X_digits, y_digits)

plt.axvline(estimator.best_estimator_.named_steps['pca'].n_components,
linestyle=':', label='n_components chosen')
plt.legend(prop=dict(size=12))
plt.show()

#%%

features = pd.read_pickle(path_4)
features_X=features.iloc[:,:features.shape[1]-1]
features_X =pd.DataFrame(scaler.fit_transform(features_X), columns= features_X.columns)

pca = PCA(0.99)
pca.fit(features_X)
PCA_components= pd.DataFrame(pca.components_, columns=list(features_X.columns))
Columns_from_PCA = list(features_X.columns[PCA_components.describe().loc['75%',:] >= 0.001])
features_x= pd.DataFrame(features_X.loc[:,Columns_from_PCA] , columns = Columns_from_PCA)

print(PCA_components.shape , features_x.shape)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Variance retained')
plt.ylim(0,1)
plt.show() 

#%%
#           LDA              #

features = pd.read_pickle(path_1)
features_X=features.iloc[:,:features.shape[1]-1]
features_y=features.loc[:,'Mode']
features_X =pd.DataFrame(scaler.fit_transform(features_X), columns= features_X.columns)

pca = PCA(n_components=2)
X_r = pca.fit(features_X).transform(features_X)

lda = LinearDiscriminantAnalysis(n_components=2)
X_r2 = lda.fit(features_X, features_y).transform(features_X)

target_names = ('Sitting','Level walking' , 'Stair up', 'Stair down' ,'Ramp up' ,'Ramp down' , 'Standing')

print('explained variance ratio (first two components): %s'
      % str(pca.explained_variance_ratio_))

plt.figure()
colors = ['navy','yellow', 'turquoise' , 'blue', 'green', 'darkorange', 'purple']
lw = 2


for color, i, target_name in zip(colors, [0, 1, 2 ,3 ,4 ,5 ,6], target_names ):
    plt.scatter(X_r[features_y == i, 0], X_r[features_y == i, 1], color=color, alpha=.8, lw=lw,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA')
plt.figure()
for color, i, target_name in zip(colors, [0, 1, 2 ,3 ,4 ,5 ,6], target_names):
    plt.scatter(X_r2[features_y == i, 0], X_r2[features_y == i, 1], alpha=.8, color=color,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('LDA')

#%%

#ANOVA F-Values and Chi-squared statistics

features = pd.read_pickle(path_1)
features_X=features.iloc[:,:features.shape[1]-1]
features_y=features.loc[:,'Mode']
features_X =pd.DataFrame(scaler.fit_transform(features_X), columns= features_X.columns)

## two best ANOVA F-Values
#selector = SelectKBest(f_classif, k=2)

# two highest chi-squared statistics
selector = SelectKBest(chi2, k=2)

X_kbest = selector.fit_transform(features_X.abs(), features_y)
print('Original number of features:', features_X.shape[1])
print('Reduced number of features:', X_kbest.shape[1])
scores= selector.scores_
pvalues = selector.pvalues_

#%%

# Drop Highly Correlated Features

features = pd.read_pickle(path_3)
features_X=features.iloc[:,:features.shape[1]-1]
features_y=features.loc[:,'Mode']
features_X =pd.DataFrame(scaler.fit_transform(features_X), columns= features_X.columns)

corr_matrix = features_X.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
print(to_drop)

#%%

# Recursive Feature Elimination

# Suppress an annoying but harmless warning
import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

features = pd.read_pickle(path_3)
features_X=features.iloc[:,:features.shape[1]-1]
features_y=features.loc[:,'Mode']
features_X =pd.DataFrame(scaler.fit_transform(features_X), columns= features_X.columns)

# Create the RFE object and compute a cross-validated score.
#clf = SVC(kernel="linear")
clf = LogisticRegression()
# The "accuracy" scoring is proportional to the number of correct
# classifications
rfecv = RFECV(estimator=clf, step=1, cv=StratifiedKFold(5))# , scoring='neg_mean_squared_error'
rfecv.fit(features_X, features_y)

print("Optimal number of features : %d" % rfecv.n_features_)

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()

#%%

# Variance Thresholding For Feature Selection

features = pd.read_pickle(path_3)
features_X=features.iloc[:,:features.shape[1]-1]
features_y=features.loc[:,'Mode']
features_X =pd.DataFrame(scaler.fit_transform(features_X), columns= features_X.columns)

thresholder = VarianceThreshold(threshold=.5)
X_high_variance = thresholder.fit_transform(features_X)
