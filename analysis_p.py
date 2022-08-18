#!/usr/bin/env python
# coding: utf-8

# In[1]:


# personally made imports
import acquire_p

# typical imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# modeling methods
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.preprocessing import PolynomialFeatures

# working with dates
from datetime import datetime

# to evaluated performance using rmse
from sklearn.metrics import mean_squared_error
from math import sqrt 

# for tsa 
import statsmodels.api as sm

# holt's linear trend model. 
from statsmodels.tsa.api import Holt

#clean look
import warnings
warnings.filterwarnings("ignore")


# In[2]:


def visual(df):
    plt.figure(figsize=(16, 16))
    for i, col in enumerate(df.columns):
        plot_number = i + 1
        l= len(df.columns)
        plt.subplot(9,1,plot_number)
        sns.lineplot(x = df.index, y = df[col])
        plt.suptitle('---------------------20XX-------------------')
plt.tight_layout()


# In[3]:


lake = acquire_p.get_bilancino_data()


# In[4]:


lake = acquire_p.prepare(lake)


# In[5]:


example = lake[lake.index < '2006-01-01']


# In[6]:


example.columns


# In[7]:


example = example.drop(columns = ['Rainfall_S_Piero', 'Rainfall_Mangona', 'Rainfall_S_Agata',
       'Rainfall_Cavallina', 'Flow_Rate'])


# In[8]:


example = example.resample('M').mean()


# In[28]:


lake.columns


# In[29]:


def spearmanr(df):
    spear = []
    cat_col = ['Rainfall_S_Piero', 'Rainfall_Mangona', 'Rainfall_S_Agata',
       'Rainfall_Cavallina', 'Rainfall_Le_Croci', 'Temperature_Le_Croci',
       'Lake_Level', 'Flow_Rate']
    for col in cat_col:
        stats.spearmanr(df.Lake_Level, df[col])
        spear.append(stats.spearmanr(df.Lake_Level, df[col]))
    l = pd.DataFrame(spear, index = cat_col).round(4)
    return l


# In[30]:


def spearmanr_a(df):
    spear = []
    cat_col = ['Rainfall_S_Piero', 'Rainfall_Mangona', 'Rainfall_S_Agata',
       'Rainfall_Cavallina', 'Rainfall_Le_Croci', 'Temperature_Le_Croci',
       'Lake_Level', 'Flow_Rate', 'avg_rain']
    for col in cat_col:
        stats.spearmanr(df.Lake_Level, df[col])
        spear.append(stats.spearmanr(df.Lake_Level, df[col]))
    l = pd.DataFrame(spear, index = cat_col).round(4)
    return l


# # 3 Month Shift
# 3 months is appoximately 90 days

# In[27]:


def shift90(df):
    col_list = ['Rainfall_S_Piero', 'Rainfall_Mangona', 'Rainfall_S_Agata', 'Rainfall_Cavallina', 'Rainfall_Le_Croci', 'Temperature_Le_Croci']
    df[col_list] = df[col_list].shift(periods = 90, fill_value = 0)
    df = df[df.index >= '2005-04-01']
    return df


# # Model

# In[32]:


def evaluate(target_var, yhat_df, validate):
    '''
    This function will take the actual values of the target_var from validate, 
    and the predicted values stored in yhat_df, 
    and compute the rmse, rounding to 0 decimal places. 
    it will return the rmse. 
    '''
    rmse = round(sqrt(mean_squared_error(validate[target_var], yhat_df[target_var])), 4)
    return rmse


# In[34]:


def plot_and_eval(target_var, train, yhat_df, validate):
    '''
    This function takes in the target var name (string), and returns a plot
    of the values of train for that variable, validate, and the predicted values from yhat_df. 
    it will als lable the rmse. 
    '''
    plt.figure(figsize = (12,4))
    plt.plot(train[target_var], label='Train', linewidth=1)
    plt.plot(validate[target_var], label='Validate', linewidth=1)
    plt.plot(yhat_df[target_var])
    plt.title(target_var)
    rmse = evaluate(target_var, yhat_df, validate)
    print(rmse)
    plt.show()
    return rmse


# In[26]:


# function to store the rmse so that we can compare
def append_eval_df(model_type, target_var, yhat_df, validate):
    '''
    this function takes in as arguments the type of model run, and the name of the target variable. 
    It returns the eval_df with the rmse appended to it for that model and target_var. 
    '''
    rmse = evaluate(target_var, yhat_df, validate)
    d = {'model_type': [model_type],
        'rmse': [rmse]}
    d = pd.DataFrame(d)
    return eval_df.append(d, ignore_index = True)


# In[35]:


eval_df = pd.DataFrame(columns=['model_type', 'rmse'])
eval_df


# In[ ]:





# In[ ]:


def polynomial_transform(x_tr_data, y_tr_data, x_val_data, y_val_data):
    # make the polynomial features to get a new set of features
    pf = PolynomialFeatures(degree=2)

    # fit and transform X_train_scaled ONLY training gets fit, even for learning transformation!!!
    x_tr_data_deg2 = pf.fit_transform(x_tr_data)

    # transform X_validate_scaled & X_test_scaled
    x_val_data_deg2 = pf.transform(x_val_data)

    # create the model object
    lm2 = LinearRegression(normalize=True)

    # fit the model to our training data
    lm2.fit(x_tr_data_deg2, y_tr_data)

    # predict train
    y_tr_data_deg2 = lm2.predict(x_tr_data_deg2)

    # evaluate: rmse
    rmse_train = mean_squared_error(y_tr_data, y_tr_data_deg2)**(1/2)

    # predict validate
    y_val_data_deg2 = lm2.predict(x_val_data_deg2)

    # evaluate: rmse
    rmse_validate = mean_squared_error(y_val_data, y_val_data_deg2)**(1/2)

    print("RMSE for Polynomial Model, degrees=2\nTraining/In-Sample: ", rmse_train.round(2), 
          "\nValidation/Out-of-Sample: ", rmse_validate.round(2))

    return y_tr_data_deg2, y_val_data_deg2 


# In[ ]:





# In[ ]:


def glm(x_tr_data, y_tr_data, x_val_data, y_val_data):

    # create the model object
    lm = TweedieRegressor(power=1, alpha=0)

    # fit the model ONLY to our training data.  

    lm.fit(x_tr_data, y_tr_data)

    # predict train
    y_tr_predict_glm = lm.predict(x_tr_data)

    # evaluate: rmse
    rmse_train = mean_squared_error(y_tr_data, y_tr_predict_glm)**(1/2)

    # predict validate
    y_val_predict_glm = lm.predict(x_val_data)

    # evaluate: rmse
    rmse_validate = mean_squared_error(y_val_data, y_val_predict_glm)**(1/2)

    print("RMSE for OLS using GLM\nTraining/In-Sample: ", rmse_train.round(2), 
          "\nValidation/Out-of-Sample: ", rmse_validate.round(2))
    return y_tr_predict_glm, y_val_predict_glm


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # Test

# In[ ]:


def evaluate_test(target_var, yhat_df, test):
    '''
    This function will take the actual values of the target_var from validate, 
    and the predicted values stored in yhat_df, 
    and compute the rmse, rounding to 0 decimal places. 
    it will return the rmse. 
    '''
    rmse = round(sqrt(mean_squared_error(test[target_var], yhat_df[target_var])), 4)
    return rmse


# In[ ]:


def plot_and_eval_test(target_var, train, validate, test, yhat_df):
    '''
    This function takes in the target var name (string), and returns a plot
    of the values of train for that variable, validate, and the predicted values from yhat_df. 
    it will als lable the rmse. 
    '''
    plt.figure(figsize = (12,4))
    plt.plot(train[target_var], label='Train', linewidth=1)
    plt.plot(validate[target_var], label='Validate', linewidth=1)
    plt.plot(test[target_var], label='Test', linewidth=1)
    plt.plot(yhat_df[target_var])
    plt.title(target_var)
    rmse = evaluate(target_var, yhat_df, test)
    print(rmse)
    plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




