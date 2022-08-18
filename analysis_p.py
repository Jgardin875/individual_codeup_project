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


# In[ ]:





# # Stats Testing

# # I have all continuous variables so I will use Spearman or Pearson
# # Spearman is better with monotonic data so I choose that one

# In[9]:


cat_col = lake.columns


# In[10]:


stat_norm = []


# In[11]:


for col in cat_col:
    stats.spearmanr(lake.Lake_Level, lake[col])
    stat_norm.append(stats.spearmanr(lake.Lake_Level, lake[col]))


# In[12]:


stat_norm = pd.DataFrame(stat_norm, index = cat_col).round(4)


# In[13]:


stat_norm


# # 3 Month Shift
# 3 months is appoximately 90 days

# In[14]:


lake_s = acquire_p.get_bilancino_data()


# In[15]:


lake_s = acquire_p.prepare(lake_s)


# In[16]:


lake_s[['Rainfall_S_Piero', 'Rainfall_Mangona', 'Rainfall_S_Agata',
       'Rainfall_Cavallina', 'Rainfall_Le_Croci', 'Temperature_Le_Croci']] = lake_s[['Rainfall_S_Piero', 'Rainfall_Mangona', 'Rainfall_S_Agata',
       'Rainfall_Cavallina', 'Rainfall_Le_Croci', 'Temperature_Le_Croci']].shift(periods = 90, fill_value=0)


# In[17]:


lake_s.resample('M').sum()


# In[18]:


lake.resample('M').sum()


# In[19]:


lake_s.head(91)


# In[20]:


lake_s = lake_s[lake_s.index >= '2005-04-01']


# In[21]:


cat_col = lake_s.columns


# In[22]:


stat_3m_s = []


# In[23]:


for col in cat_col:
    stats.spearmanr(lake_s.Lake_Level, lake_s[col])
    stat_3m_s.append(stats.spearmanr(lake_s.Lake_Level, lake_s[col]))


# In[24]:


stat_3m_s = pd.DataFrame(stat_3m_s, index = cat_col, columns = ['corr_3m_s', 'p_3m_s']).round(4)


# In[25]:


stat_3m_s


# # Compare

# In[26]:


stat_sum = stat_norm.join(stat_3m_s)


# In[27]:


stat_sum


# In[28]:


stat_sum_cor = stat_sum[['correlation', 'corr_3m_s']]


# In[29]:


stat_sum_cor = stat_sum_cor.abs()


# In[30]:


stat_sum_cor = stat_sum_cor.drop(index = ['Flow_Rate', 'Lake_Level'])


# In[31]:


#spear_cor = stat_sum_cor.plot(kind = 'barh', title = 'Strenght of Correlation to Lake Level')


# In[32]:


#plt.savefig('spear_cor.png')


# In[33]:


stat_sum_p = stat_sum[['pvalue', 'p_3m_s']]


# In[34]:


stat_sum_p = stat_sum_p.drop(index = ['Flow_Rate', 'Lake_Level'])


# In[35]:


#spear_p = stat_sum_p.plot(kind = 'barh', title = 'Statisitcal Likelihood of Random Correlation')


# In[36]:


#plt.savefig('spear_p.png')


# In[37]:


mod_col = lake_s.drop(columns = ['Rainfall_S_Piero', 'Rainfall_Mangona', 'Rainfall_S_Agata',
       'Rainfall_Cavallina', 'Rainfall_Le_Croci'])


# In[38]:


mod_col = mod_col.resample('M').mean()


# In[39]:


train = mod_col[mod_col.index < '2014-01-01']
validate = mod_col[(mod_col.index >= '2014-01-01') & (mod_col.index < '2018-01-01')]
test = mod_col[mod_col.index >= '2018-01-01']


# In[40]:


trainx = train.drop(columns = 'Lake_Level')
trainy = train.Lake_Level

validatex = validate.drop(columns = 'Lake_Level')
validatey = validate.Lake_Level

testx = test.drop(columns = 'Lake_Level')
testy = test.Lake_Level


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[41]:


def evaluate(target_var):
    '''
    This function will take the actual values of the target_var from validate, 
    and the predicted values stored in yhat_df, 
    and compute the rmse, rounding to 0 decimal places. 
    it will return the rmse. 
    '''
    rmse = round(sqrt(mean_squared_error(validate[target_var], yhat_df[target_var])), 4)
    return rmse


# In[42]:


def plot_and_eval(target_var):
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
    rmse = evaluate(target_var)
    print(rmse)
    plt.show()


# In[43]:


# function to store the rmse so that we can compare
def append_eval_df(model_type, target_var):
    '''
    this function takes in as arguments the type of model run, and the name of the target variable. 
    It returns the eval_df with the rmse appended to it for that model and target_var. 
    '''
    rmse = evaluate(target_var)
    d = {'model_type': [model_type],
        'rmse': [rmse]}
    d = pd.DataFrame(d)
    return eval_df.append(d, ignore_index = True)


# In[ ]:





# In[ ]:




