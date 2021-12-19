# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 11:45:40 2021

@author: aditi
"""

'''
================================================================================================
PREDICTING THE PRICE OF PRE-OWNED CARS

Storm motors is an e-commerce company who act as mediators between parties interested in
selling and buying pre-owned cars.

For the year 2015-2016, they have recorded data about the seller and car including 
Specific details
condition of car
seller details
registration details
web advertisement details
make and model information
price

Storm Motors wishes to develop an algorithm to predict the price of the cars based on various
attributes associated with the car.
=================================================================================================





===================================================================================================
variable description
size : 50000 * 19

dateCrawled (date) - date when the ad first craeled, all field values are taken from this date
name (string) - string consists of car name, brand, model, etc.
seller (string) - nature of seller (private, commercial)
offerType (string) - whether the car is on offer or has the buyer requested for an offer
price (int) - price on the ad to sell the car
abtest (string) - twi versions of ad
vehicleTest (string) - types of cars
yearOfRegistration (int) - year in which was first registered
gearbox (string) - type of gearbox (manual or automatic)
powerPS (int) - power of the car (HP)
model (string) - model type of the car
kilometer (int) - no of kilometers the car has travelled
monthOfRegistration (int) - month of registration
fuelType (string) - types of fuel
brand (string) - make of car
notRepairedDamage (string) - status of repair for damages, if yes, damages have not been
                             rectified; if no, damages were taken care of
dateCreated (date) - date at which the ad at storm motor was created
postalCode (int) - postal code of seller
lastSeen (date) - when the crawler saw this ad last online
========================================================================================






============================================================================================
The variables can be grouped into different buckets on the information

Specification order : gearbox, power, fuelType
condition of car : notRepairDamaged, kilometer
seller details : seller, postalCode
Regsitration details : yearOfRegistration, monthOfRegistration
Make and Model : brand, model, vehicleType
Advertisement details : dateCrawled, name, abtest, dateCreated, lastSeen, offterType
======================================================================================================





==============================================================================================

Problem conceptualization :
    predict price of the cars based on various attributes
Problem characterization :
    function approximation


Apriori knowledge :
    dependent variable - numerical
    independent variables - numerical + categorical


flow chart :
    data set (50,000 * 19)
    function approximation model to predict the price of the car
==============================================================================================



==============================================================================================
Solution conceptualization :
    identify if data is clean
    look for missing values
    identify variables influencing price and look for relationships among variabless
        correlation, box plots, scatter plots, etc.
    identify outliers
        central tendency measures, dispersion measures, box plots, histograms, etc.
    identify if categories with meagre frequencies can be combined
    filter data based on logical checks
        price, year of registration, power
    reduced numer of data

Method identification :
    Linear regression
    Random forest
    
Realization of solution :
    assumption checks using regression diagnostics
    evaluation performance metrics
    if assumptions are satisfied and solutions are acceptable then model is good
    if performance metrics are not reasonable then a single model is not able to capture the
    variation in price as a whole
    in such cases, it would be better to subset data and build separate models
==============================================================================================
'''

import pandas as pd
import numpy as np
import seaborn as sns
import os

# setting dimensions for plot

sns.set(rc = {'figure.figsize' : (11.7,8.27)})

os.chdir("C:\\Aditi\\MSc\\Sem 4")
cars_data = pd.read_csv('cars_sampled.csv')
cars_data

cars = cars_data.copy()

cars.info()
cars.describe()
pd.set_option('display.float_format', lambda x : '%.3f' % x)

#to display maximum set of columns
pd.set_option('display.max_columns', 500)
cars.describe()

# dropping unwanted columns
col = ['name', 'dateCrawled', 'dateCreated', 'postalCode', 'lastSeen']
cars = cars.drop(columns=col, axis=1)

# removing duplicate records

cars.drop_duplicates(keep='first', inplace= True)


# ================================ DATA CLEANING =========================

#no of missing values in each columns

cars.isnull().sum()



# Variable yearOfRegistration

yearwise_count = cars['yearOfRegistration'].value_counts().sort_index()
yearwise_count

sum(cars['yearOfRegistration'] > 2018)
sum(cars['yearOfRegistration'] < 1950)

sns.regplot(x = 'yearOfRegistration', y='price', scatter=True, fit_reg=False, data=cars)


# working range - 1950 and 2018

#variable price
price_counts = cars['price'].value_counts().sort_index()
price_counts
sns.distplot(cars['price'])
cars['price'].describe()
sns.boxplot(y=cars['price'])
sum(cars['price']>150000)
sum(cars['price'] < 100)

#working range 100 and 150000

# variable powerPS

power_count = cars['powerPS'].value_counts().sort_index()
power_count
sns.distplot(cars['powerPS'])
cars['powerPS'].describe()
sns.boxplot(y=cars['powerPS'])
sns.regplot(x='powerPS', y='price', scatter=True, fit_reg=False, data=cars)

sum(cars['powerPS'] > 500)
sum(cars['powerPS'] <10)
# working range - 10 and 500



#=============================
#Working range of data
#=============================

#working range of data

cars = cars[
    (cars.yearOfRegistration <= 2018)
    & (cars.yearOfRegistration >= 1950)
    & (cars.price >= 100)
    & (cars.price <= 150000)
    & (cars.powerPS >= 10)
    & (cars.powerPS <= 500)]

cars
# ~6700 records are dropped


# further to simplify - variable reduction
# combining yearOfRegistration and monthOfRegistration

cars['monthOfRegistration'] /= 12
cars

#creating new variable Age by adding yearOfRegistration and monthOfRegistration

cars['Age'] = (2018 - cars['yearOfRegistration']) + cars['monthOfRegistration']
cars['Age'] = round(cars['Age'],2)
cars.Age.describe()


# dropping yearOfRegistration and monthOfRegistration

cars = cars.drop(columns=['yearOfRegistration', 'monthOfRegistration'], axis=1)


# Visualising parameters

# Age

sns.distplot(cars['Age'])
sns.boxplot(y=cars['Age'])

# price
sns.distplot(cars['price'])
sns.boxplot(y=cars['price'])

#powerPS
sns.distplot(cars['powerPS'])
sns.boxplot(y=cars['powerPS'])


#visualizing parameters after narrowing working ranges

#age vs price

sns.regplot(x='Age', y='price', scatter=True, fit_reg=False,  data=cars)



#powerPS vs price

sns.regplot(x = 'powerPS', y='price', scatter=True, fit_reg=False, data=cars)


# seller
cars['seller'].value_counts()
pd.crosstab(cars['seller'], columns='count', normalize=True)
sns.countplot(x='seller',data=cars)
#fewer cars ahve 'commercial' => insignificant


# variable offerType
cars['offerType'].value_counts()
sns.countplot(x='offerType', data=cars)

#all cars have 'offer' => insignificant

#abtest

cars['abtest'].value_counts()
sns.countplot(x='abtest', data=cars)

sns.boxplot(x='abtest', y='price', data=cars)
# for every price value there is almost 50-50 distribution
# does not affect price => insignificant


# vehicleType

cars['vehicleType'].value_counts()
pd.crosstab(index=cars['vehicleType'], columns='count', normalize=True)*100
sns.countplot(x='vehicleType', data=cars)
sns.boxplot(x='vehicleType', y='price', data=cars)

#for different categories of vehicleType, the price ranges are different
#this affects price => retain

#gearbox
cars['gearbox'].value_counts()
pd.crosstab(index=cars['gearbox'], columns='count', normalize=True)*100
sns.countplot(x='gearbox', data=cars)
sns.boxplot(x='gearbox', y='price', data=cars)

#gearbox affects price => retain

# model
cars['model'].value_counts()
pd.crosstab(index=cars['model'], columns='count', normalize=True)*100
sns.countplot(x='model', data=cars)
sns.boxplot(x='model', y='price', data=cars)


# kilometer
cars['kilometer'].value_counts().sort_index()
pd.crosstab(index=cars['kilometer'], columns='count', normalize=True)*100
sns.countplot(x='kilometer', data=cars)
sns.boxplot(x='kilometer', y='price', data=cars)
cars.kilometer.describe()

#kilometer affects price => retain


#fuelType
cars['fuelType'].value_counts().sort_index()
pd.crosstab(index=cars['fuelType'], columns='count', normalize=True)*100
sns.countplot(x='fuelType', data=cars)
sns.boxplot(x='fuelType', y='price', data=cars)

#fuelType affects price => retain



#brand
cars['brand'].value_counts().sort_index()
pd.crosstab(index=cars['brand'], columns='count', normalize=True)*100
sns.countplot(x='brand', data=cars)
sns.boxplot(x='brand', y='price', data=cars)

#affects price => retain



#notRepairedDamage
cars['notRepairedDamage'].value_counts().sort_index()
pd.crosstab(index=cars['notRepairedDamage'], columns='count', normalize=True)*100
sns.countplot(x='notRepairedDamage', data=cars)
sns.boxplot(x='notRepairedDamage', y='price', data=cars)

#affects price



#removing insignificant variables

col = ['seller', 'offerType', 'abtest']
cars = cars.drop(columns = col, axis=1)
cars_copy = cars.copy()



#===================================================================
#correlation
# ======================================================================

cars_select1 = cars.select_dtypes(exclude=[object])
correlation = cars_select1.corr()
round(correlation, 3)
cars_select1.corr().loc[ : , 'price'].abs().sort_values(ascending=False)[1:]




'''
We are going to build a Linear Regression and Rnadom forest model on two sets
of data

1. data obtained by omitting rows with any missing value
2. data obtained by imputing the missing values
'''
#===============================================
# omitting missing values
# ===============================================


cars_omit = cars.dropna(axis=0)


#categorical to dummy

cars_omit = pd.get_dummies(cars_omit, drop_first=True)


#libraries

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


#===============================================
# model building with omitted data
# ===============================================


x1 = cars_omit.drop(['price'], axis='columns', inplace=False)
y1 = cars_omit['price']

#plotting the variable price
prices = pd.DataFrame({"1. Before": y1, "2. After":np.log(y1)})
prices.hist()

#transforming price as a logarithmic value

y1 = np.log(y1)

x_train, x_test, y_train, y_test = train_test_split(x1, y1, test_size=0.3, random_state=3)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)




#===============================================
# baseline model for omitted data
# ===============================================


'''
we are making a base model by using test data mean value
this is to set a benchmark and to compare with our regression model
'''

# mean of test data

base_pred = np.mean(y_test)
print(base_pred)

#repeating same value till length of test data
base_pred = np.repeat(base_pred, len(y_test))

#finding RMSE

base_root_mean_square_error = np.sqrt(mean_squared_error(y_test, base_pred))
base_root_mean_square_error


#=================================
#Linear Regression
#==================================


lgr = LinearRegression(fit_intercept=True)
model_lr1 = lgr.fit(x_train, y_train)

pred_lr1 = lgr.predict(x_test)

#mse and rmse
lr_mse1 = mean_squared_error(y_test, pred_lr1)
lr_rmse1 = np.sqrt(lr_mse1)
print(lr_rmse1)

# r squared

r2_test_lr1 = model_lr1.score(x_test, y_test)
r2_train_lr1 = model_lr1.score(x_train, y_train)
print(r2_test_lr1, r2_train_lr1)

#reg diagnostics
#residual plot analysis - diff btw actual - pred

residuals1 = y_test - pred_lr1
sns.regplot(x=pred_lr1, y=residuals1, scatter=True, fit_reg=False, data=cars)
residuals1.describe()


#=================================
#random forests with omitted data
#==================================

# model parameters

rf = RandomForestRegressor(n_estimators=100, max_features='auto', max_depth=100, 
                           min_samples_split=10, min_samples_leaf=4, random_state=1 )


model_rf1 = rf.fit(x_train, y_train)
pred_rf1 = rf.predict(x_test)

rf_mse1 = mean_squared_error(y_test, pred_rf1)
rf_rmse1 = np.sqrt(rf_mse1)
rf_rmse1


r2_test_rf1 = model_rf1.score(x_test, y_test)
r2_train_rf1 = model_rf1.score(x_train, y_train)

print(r2_test_rf1, r2_train_rf1)



#=================================
#model building with imputed data
#==================================


cars_imputed = cars.apply(lambda x:x.fillna(x.median()) if x.dtypes=='float'\
                          else x.fillna(x.value_counts().index[0]))

    
cars_imputed.isnull().sum()


cars_imputed = pd.get_dummies(cars_imputed, drop_first=True)




x2 = cars_imputed.drop(['price'], axis='columns', inplace=False)
y2 = cars_imputed['price']


prices = pd.DataFrame({'1. Before':y2, '2. After':np.log(y2)})

prices.hist()

y2 = np.log(y2)


x_train1, x_test1, y_train1, y_test1 = train_test_split(x2, y2, test_size=0.3, random_state=3)





base_pred = np.mean(y_test1)
base_pred

base_pred = np.repeat(base_pred, len(y_test1))

base_root_mean_square_error_imputed = np.sqrt(mean_squared_error(y_test1, base_pred))

base_root_mean_square_error_imputed



# LINEAR REGRESSION

lgr2 = LinearRegression(fit_intercept=(True))
model_lgr2 = lgr2.fit(x_train1, y_train1)

pred_lgr2 = lgr2.predict(x_test1)


#mse rmse

lgr_mse2 = mean_squared_error(y_test1, pred_lgr2)
lgr_rmse2 = np.sqrt(lgr_mse2)
print(lgr_rmse2)


# random forest

rf2 = RandomForestRegressor(n_estimators=100, max_features='auto', max_depth=100, 
                           min_samples_split=10, min_samples_leaf=4, random_state=1 )


model_rf2 = rf2.fit(x_train1, y_train1)
pred_rf2 = rf2.predict(x_test1)

rf_mse2 = mean_squared_error(y_test1, pred_rf2)
rf_rmse2 = np.sqrt(rf_mse2)
rf_rmse2


r2_test_rf2 = model_rf2.score(x_test1, y_test1)
r2_train_rf2 = model_rf2.score(x_train1, y_train1)

print(r2_test_rf2, r2_train_rf2)

