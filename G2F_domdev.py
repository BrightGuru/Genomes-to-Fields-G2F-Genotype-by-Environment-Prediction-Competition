# installing basic libraries
import datatable as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from skopt import BayesSearchCV


from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.model_selection import LeaveOneGroupOut
import sklearn.metrics as metrics
from scipy.stats import pearsonr

import xgboost as xgb
import time
import pickle
import random

# Set random seed for reproducibility

random.seed(99)
np.random.seed(99)

XGBr_clf= xgb.XGBRegressor(random_state = 99)

###for Train data

# Metal data
Meta_data = dt.fread("/usr/users/osatohanmwen/G2F_competition_2024/PhenoMetal_data_final.csv")
Meta_data=Meta_data.to_pandas()
print(Meta_data.shape)

#Weather data
Weather_data = dt.fread("/usr/users/osatohanmwen/G2F_competition_2024/PhenoWeather_data_final.csv")
Weather_data = Weather_data.to_pandas()
Weather_data = Weather_data.drop(columns=['Hybrid','Field_Location','Year','Env','Yield_Mg_ha'])
Weather_data = Weather_data[Weather_data['Hybrid_Env'].isin(Meta_data['Hybrid_Env'])]
print(Weather_data.shape)

####Genomic data 
Geno_data = dt.fread("/usr/users/osatohanmwen/G2F_competition_2024/Genomic_dodev_matrix.csv")
Geno_data=Geno_data.to_pandas()

###Merge all data together

WC_meta_data = pd.merge(Meta_data, Weather_data, on='Hybrid_Env',how='inner')
print(WC_meta_data.shape)

WC_meta_geno_data = pd.merge(WC_meta_data, Geno_data, on='Hybrid',how='left')
print(WC_meta_geno_data.shape)

WC_meta_geno_data = WC_meta_geno_data[WC_meta_geno_data['Year'].isin([2019, 2020, 2021, 2022, 2023])].reset_index(drop=True)

print(WC_meta_geno_data.shape)

###for Test set
Meta_datatest = dt.fread("/usr/users/osatohanmwen/G2F_competition_2024/TestPhenometal_data_final.csv")
Meta_datatest = Meta_datatest.to_pandas()
print(Meta_data.shape)

#Weather data
Weather_datatest = dt.fread("/usr/users/osatohanmwen/G2F_competition_2024/TestPhenoweather_data_final.csv")
Weather_datatest = Weather_datatest.to_pandas()
Weather_datatest = Weather_datatest.drop(columns=['Hybrid','Field_Location','Year','Env','Yield_Mg_ha'])
Weather_datatest = Weather_datatest[Weather_datatest['Hybrid_Field'].isin(Meta_datatest['Hybrid_Field'])]
print(Weather_datatest.shape)


###Merge all data together
WC_meta_datatest = pd.merge(Meta_datatest, Weather_datatest, on='Hybrid_Field',how='inner')
print(WC_meta_datatest.shape)

WC_meta_geno_datatest = pd.merge(WC_meta_datatest, Geno_data, on='Hybrid',how='left')
print(WC_meta_geno_datatest.shape)

#features = np.concatenate([Full_data.iloc[:,1:-1].values,Full_data_1.iloc[:,1:-1].values], axis=1)
features_train = WC_meta_geno_data.drop(columns=['Hybrid','Year','Field_Location','Env','Hybrid_Env','Yield_Mg_ha'])
features_test = WC_meta_geno_datatest.drop(columns=['Hybrid','Year','Field_Location','Env','Hybrid_Field','Yield_Mg_ha'])

outcome = WC_meta_geno_data['Yield_Mg_ha'] 
years   = WC_meta_geno_data['Year']


# Check feature alignment
print("Training features:", features_train.columns)
print("Test features:", features_test.columns)

# Align columns in test data with training data
features_test = features_test.reindex(columns=features_train.columns, fill_value=0)

t_start = time.time()

XGBr_clf= xgb.XGBRegressor(random_state = 99)

# Bayesian optimization using an iterative Gaussian process 

params_lg = {

'n_estimators': (2000,7000, "log-uniform"), # No of trees# 220,520,620,
'learning_rate' : (0.005, 0.3,"log-uniform"), 

'max_depth':(2,15,"log-uniform") , # maximum depth to explore (meaning the longest path between the root node and the leaf node.'max_depth': [10,15],
'min_child_weight' : (1,20, "log-uniform"),
'subsample': (0.1, 1), 
'colsample_bytree' : (0.1, 1), 

'gamma' : (0.01, 0.5, "log-uniform"), 
'reg_alpha' : (10, 200, "log-uniform"), 
'reg_lambda' : (1, 10, "log-uniform"), 

'n_jobs' : [-1] ,
'seed' : [99]
}

# Define Leave-One-Year-Out Cross-Validation
cv = LeaveOneGroupOut()

Bayes_search = BayesSearchCV(
    XGBr_clf,params_lg,n_iter=20, 
    n_jobs = -1,cv=cv,scoring='neg_mean_squared_error',
    random_state=99,
    verbose=0)

print(Bayes_search.total_iterations)

Bayes_search.fit(features_train,outcome,groups=years)

print(Bayes_search.best_estimator_)

#update XGBM parameters usingb the best estimator which was obtained by using BayesSearchCV.

XGBr_clf_best = Bayes_search.best_estimator_

XGBr_clf_best.fit(features_train, outcome)

y_pred_grid = XGBr_clf_best.predict(features_test)

##observed and predicted dataframe
Data_OP = pd.DataFrame(
        {'Env' : WC_meta_geno_datatest['Env'],
         'Hybrid' : WC_meta_geno_datatest['Hybrid'],
         'Yield_Mg_ha': y_pred_grid
         }
    )

print(f"completed in {time.time() - t_start:.2f} seconds.")

print(Data_OP.info())
print(Data_OP.isnull().sum())  # Check for missing values
print(Data_OP.dtypes)  # Confirm expected data types

Data_OP.to_csv("G2F_comp_MWdomdev.csv",index=False)
