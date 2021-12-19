# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 11:40:00 2021

@author: aditi
"""

'''
===============================================================================================
Subsidy Inc. delivers subsidies to individuals based on their income
Accurate income data is one of the hardest peice of data to obtain across the world
Subsidy Inc. has obtained a large data set of authenticated data on individual income,
demographic parameters, and a few financial parameters.
Subsidy Inc. wishes to :
    Develop an income classifier system for individuals



The objective is to :
    Simplify the data system by reducing the number of variables to be studied, without 
    sacrificing too much of accuracy. Such a system would help Subsidy Inc. in planning
    subsidy outlay, monitoring and preventing misuse.
================================================================================================
'''

'''
==============================================================================================
1. age  - age of the individual
 
2. JobType - working status of person, whoch sector does he work in

3. EdType - The level of education

4. maritalstatus - The marital status of the individual

5. occupation - The type of work the individual does

6. relationship - relationship of individual to his/her household

7. race - the individual's race

8. gender - the individual's gender

9. capitalgain - the capital gains of the individual (from selling an asset such as stock or
                                                   bond for more than the purchase price)

10. capitalloss - teh capital losses of the individual (for selling an asset for less than the 
                                                    original purchase price)

11. hoursperweek - the no of hours he/she works per week

12. nativecountry - the native country of the individual

13. SalStat - the outcome variable indicating a person's salary status
==============================================================================================

'''



'''
==============================================================================================

Problem conceptualization :
    Develop an income classifier for individuals with reduced no.of variables
    Problem characterization - Classification
    
Apriori Known :
    Dependent variable - categorical (binary)
    
    Independent variables - numerical + categorical
    

Data set (31,978 * 13)
Numerical - 4, Categorical - 9
        |
Classification technique (supervised)
        |
Classifier model to find income bracket
==============================================================================================


'''



'''
==============================================================================================

Solution Conceptualization :
    Identify if data is clean
    
    Look for missing values
    
    Identify variables influencing salary status and look for possible relationships
    between variables
    (correlation, chi-square test, box plots, scatter plots, etc)
    
    Identify if categories can be combined
    
    Build a model with reduced number of variables to classify the individual's salary
    status to plan subsidy outlay, monitor and prevent misuse
==============================================================================================
    
    
'''


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

os.chdir("C:\\Aditi\\MSc\\Sem 4")
data_income = pd.read_csv("income.csv")

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

data = data_income.copy()

"""
==============================================================================================

#Exploratory data analysis :

# 1. Getting to know data
# 2. Data Preprocessing (Missing values)
# 3. Cross tables and data visualization
==============================================================================================

"""

# =====================================================
#Getting to know the data
# ========================================================
# *** To check varibales' data type

print(data.info())
data.isnull().sum() #No missing data


#**** Summary of numerical variables
summary_num = data.describe()
print(summary_num)

# Summary of categorical variables
summary_cat = data.describe(include="O")
print(summary_cat)

# Frequency of each category
data.columns
data['JobType'].value_counts()
data['EdType'].value_counts()
data['maritalstatus'].value_counts()
data['occupation'].value_counts()
data['relationship'].value_counts()
data['race'].value_counts()
data['gender'].value_counts()
data['nativecountry'].value_counts()
data['SalStat'].value_counts()


# Checking for unique classes
print(np.unique(data['JobType']))
print(np.unique(data['occupation']))
# there exists ' ?' instead of nan

'''
Go back and read the data by including "na_values=[' ?]"
'''

data = pd.read_csv('income.csv', na_values=[' ?'])
data.info() 
data.isnull().sum()

missing = data[data.isnull().any(axis=1)]
#axis=1 => to consider at least one column value is missing


'''
Points to note :
1. Missing values in Jobtypes  = 1809
2. Missing values in occupation = 1816
3. There are 1809 rows where two specific columns i.e. occupation & jobtype have missing
values.
4. (1816-1809) = 7 => you stull have occupation unfilled for these 7 rows.
Since, jobtype is Never worked
'''

 data2 = data.dropna(axis=0)
#got rid of all the rows with missing values


#relationship between independent variables
correlation= data2.corr()
print(correlation)


# Cross tables and data visualisation

#extracting the column names
data2.columns

#Gender proportion table

gender = pd.crosstab(index= data2['gender'], columns= 'count', normalize=True)
gender

#Gender vs salary status :

gender_salstat = pd.crosstab(index= data2['gender'], columns=data2['SalStat'],
                             margins=True, normalize='index')
print(gender_salstat)


############ Frequency distribution of salary status
SalStat = sns.countplot(data2['SalStat'])
print(SalStat)

salstat = pd.crosstab(index=data2['SalStat'], columns='count', normalize=True)
salstat

''' 75% of people's salary status is <=50,000
    25% of people's salary status is >50,000'''
    


############## Histogram of Age

sns.distplot(data2['age'], bins=10, kde=False)
# People with age 20-45 age are high in frequency

########## Boxplot - Age vs Salary status

sns.boxplot('SalStat', 'age', data=data2)
data2.groupby('SalStat')['age'].median()

#people with 35-50 age are more likely to earn > 50000 salary
#people with 25-35 age are more likely to earn <= 50000 salary


################ EDA
#jobtype vs salary status

sns.countplot(x='JobType',hue='SalStat', data=data2)

pd.crosstab(index=data2['JobType'], columns=data2['SalStat'], normalize='index')
round((pd.crosstab(index=data2['JobType'], columns=data2['SalStat'], normalize='index')), 3)*100


# education vs salary status

sns.countplot(x='EdType', hue='SalStat', data=data2)

round((pd.crosstab(index=data2['EdType'], columns=data2['SalStat'], normalize='index')), 3)*100


#Occupation vs salary status
sns.countplot(x='occupation', hue='SalStat', data=data2)

round((pd.crosstab(index=data2['occupation'], columns=data2['SalStat'], normalize='index')), 3)*100



#Capital gain

sns.distplot(data2['capitalgain'], kde=False, bins=10)
pd.crosstab(index=data2['capitalgain'], columns='count')

sns.distplot(data2['capitalloss'], kde=False, bins=10)
pd.crosstab(index=data2['capitalloss'], columns='count')


# Hours per week vs salary

sns.boxplot('SalStat', 'hoursperweek', data=data2)


# ======================================================================
# LOGISTIC REGRESSION
# =====================================================================

data2['SalStat'].value_counts()

#Reindexing the salary status names to 0,1
data2['SalStat'] = data2['SalStat'].map({' less than or equal to 50,000':0, ' greater than 50,000':1})
data2

#converting categorical variables to dummy variables
new_data = pd.get_dummies(data2, drop_first=True)

# storing the column names

columns_list = list(new_data.columns)
print(columns_list)

#separating the input names from the data

features_names = list(set(columns_list) - set(['SalStat']))
print(features_names)

#storing the output values in y
y = new_data['SalStat'].values
print(y)
type(y)
type(features)

# storing the values from input features
x = new_data[features_names].values
print(x)


# Splitting the data into train and test
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3, random_state=0)

#make an instance of the model
logistic = LogisticRegression()

#Fitting the values for x and y
logistic.fit(train_x, train_y)
logistic.coef_
logistic.intercept_


#Predicting from test data
prediction = logistic.predict(test_x)
prediction

#confusion matrix

confusion_matrix = confusion_matrix(test_y, prediction)
print(confusion_matrix)

# Accuracy
accuracy = accuracy_score(test_y, prediction)
accuracy

#printing the misclassified values from prediction
print('Misclassified sample : %d' % (test_y != prediction).sum())


# ========== REMOVING INSIGNIFICANT VARIABLES
data
data2
new_data


print(data2['SalStat'])

cols = ['gender', 'nativecountry', 'race','JobType']
new_data = data2.drop(cols, axis=1)
new_data

new_data = pd.get_dummies(new_data, drop_first=True)

# storing the column names
columns_list = list(new_data.columns)
columns_list

# separating the input names from data
feature_names = list(set(columns_list) - set(['SalStat']))
feature_names
new_data[feature_names] #whole dataframe with feature columns

#storing feature values in x
x = new_data[feature_names].values
x

# storing output values in y
y = new_data['SalStat'].values 
y


# splitting the data into train and test
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3, random_state=0)

logistic = LogisticRegression()
logistic.fit(train_x, train_y)

pred = logistic.predict(test_x)

accuracy = accuracy_score(test_y, pred)
accuracy


'''
=================KNN===========================
'''
#importing the library of KNN

from sklearn.neighbors import KNeighborsClassifier

KNN = KNeighborsClassifier(n_neighbors = 5)
KNN.fit(train_x, train_y)

pred= KNN.predict(test_x)
pred

accuracy = accuracy_score(test_y, pred)
accuracy

conf_mat = confusion_matrix(test_y, pred)
conf_mat

print('\t','Predicted values')
print('original values','\n', conf_mat)


print('Misclassified samples : %d' % (test_y != pred).sum())


'''
Effect of K value on classifier
'''
misclassified_sample = []
for i in range(1,20):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(train_x, train_y)
    pred_i = knn.predict(test_x)
    misclassified_sample.append((test_y != pred_i).sum())

print(misclassified_sample)
