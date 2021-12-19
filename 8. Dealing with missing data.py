# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 11:04:26 2021

@author: aditi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

os.chdir('C:\\Aditi\\MSc\\Sem 4')

cars = pd.read_csv("Toyota.txt")
cars.dropna(axis=0, inplace=True)


#checking the number of missing values
cars.isnull().sum()
cars.isna().sum()

#subsetting the rows that have one or more missing values

missing = cars[cars.isnull().any(axis=1)]

# any(axis=1) will give all the rows where atleast one column value is missing

missing

d = cars.describe()

print(d)

#calculate mean
cars['Age'].mean()


#filling na values with mean of age column
#inplace = True to replace in the existing dataframe itself

cars['Age'].fillna(cars['Age'].mean(), inplace=True)

cars['KM'].median()

cars['KM'].fillna(cars['KM'].median(), inplace=True)

cars['HP'].mean()

cars['HP'].fillna(cars['HP'].mean(), inplace=True)




cars['FuelType'].value_counts()
cars['FuelType'].value_counts().index[0]

cars['FuelType'].fillna(cars['FuelType'].value_counts().index[0], inplace=True)


cars['FuelType'].mode()[0]

cars['MetColor'].mode()[0]

cars['MetColor'].fillna(cars['MetColor'].mode()[0], inplace=True)

cars['MetColor'].value_counts().index[0]

'''
To fill the NA/NaN values in both numerical and categorical variables at one stretch
'''
#lambda x is defining a function x inside a function
cars = cars.apply(lambda x:x.fillna(x.mean()) if x.dtype=='float' else x.fillna(x.value_counts().index[0]))
cars = cars.apply(lambda x:x.fillna(x.mean()) if x.dtype=='float' else x.fillna(x.mode()[0]))
