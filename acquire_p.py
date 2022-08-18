#!/usr/bin/env python
# coding: utf-8

# In[1]:


# personally made imports


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


print('get_bilancino_data()')
print('prepare(df)')


# In[3]:


import os

def get_bilancino_data():
    filename = "Lake_Bilancino.csv"
    
    # if file is available locally, read it
    if os.path.isfile(filename):
        return pd.read_csv(filename)


# In[10]:


def prepare(df):
    df.Date = pd.to_datetime(df.Date, infer_datetime_format= True)
    df.set_index('Date', inplace = True)
    df = df[df.index > '2004-12-31']
    df = df.sort_index()
    return df


# In[ ]:





# In[12]:


lake = get_bilancino_data()


# In[13]:


lake.shape


# In[14]:


lake = prepare(lake)


# In[15]:


lake.head(15)


# In[9]:


lake.shape


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




