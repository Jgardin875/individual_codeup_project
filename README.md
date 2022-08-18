# Individual_Codeup_Project


# Data Dictionary

Rainfall_S_Piero    - rainfall in mm
Rainfall_Mangona    - rainfall in mm
Rainfall_S_Agata    - rainfall in mm
Rainfall_Cavallina  - rainfall in mm 
Rainfall_Le_Croci   - rainfall in mm
Temperature_Le_Croci - temperature in 'C
Lake_Level - level in meters (m)
Flow_Rate - flow rate in cubic meters per second (mc/s)
avg_rain - avg of Mangona, S_Agata, Le_Croci rainfall


# Steps to reproduce

You will need a kaggle account.
https://www.kaggle.com/competitions/acea-water-prediction

Clone my repo (including the acquire_p.py)

Libraries used are pandas, matplotlib, seaborn, numpy, sklearn, datetime, math, statsmodels, and scipy.stats.


# Project Goals

Forcast water levels in Lake Bilancino to aid environmental services in determining how to handle daily consumptions efficiently.

# Project Desciption

As climate change impacts weather patterns, water may become increasingly in short supply. Learning how to predict and effecitvely manage water supplies, especially fresh water supplies, is crucial for city development and human survival. 

# Initial Testing and Hypothesis
1. Rain affects water levels in the lake
2. Temperature affects water levels in the lake
3. Flow rate out of the lake affects water levels in the lake

# Report Findings 
1. Rainfall Mangona, S_Agata, and Le_Croci were the rainfalls that had the most effect on water level. There is about a three month delay between rainfalls and changes in water level

2. Temperature had an effect on water level. There is about a three month delay between temperature changes and changes in water level.

3. Flow rate did affect water level. No significant delay in flow rate and water level.


# ------------- was the most effective model. 
# ------------- was the most effective feature engineering. 



#  Recommendations and Future Work

- combine model with models that predict changes in temperature and rainfall to get a more accurate

- regulate water usage to minimize use of limited water resources

- invest in salt-water conversion to fresh water

# Detailed Project Plan

### Mon afternoon
acquire data - download from kaggle.com (pd.read_csv)
clean data - dropped two years of missing data
convert dates to datetime format and set as index

### Tues
explore data
    univariant exploration
    mulitvariant exploration
    statistical exploration

### Wed
feature engineering
    -3 month shift
    -address sparse matrix
        -combine similar rainfall features
split data train, validate, test
split data between dependent and independent variables to minimize target leakage
prep readme/report with markdowns

### Thurs
models
    baseline
    compare models
    run final on test
final touches





















