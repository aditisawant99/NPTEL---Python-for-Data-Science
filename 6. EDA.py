# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 20:20:16 2021

@author: aditi
"""

import os
import pandas as pd
import numpy as np

os.chdir('C:\\Aditi\\MSc\\Sem 4')
cars = pd.read_csv('Toyota.txt')
cars


cars2 = cars.copy(deep=True)

'''
#Frequency Tables

pandas.crosstab() - simple cross-tabulation of one, two (or more) factors
'''
pd.crosstab(index = cars2['FuelType'], columns='count', dropna=True)

'''
Two-way tables
To look at the frequency distribution of gearbox(Automatic) types with respect to different fuel types
'''

pd.crosstab(index=cars2['Automatic'], columns=cars2['FuelType'], dropna=True)

'''
Joint probability - likelihood of two independent events happening at the same time
'''

pd.crosstab(index = cars2['Automatic'],
            columns = cars2["FuelType"],
            normalize=True,
            dropna = True)

'''
Marginal probability - probability of the occurrence of the single event
'''
pd.crosstab(index = cars2['Automatic'],
            columns = cars2["FuelType"],
            normalize=True,
            margins=True,
            dropna = True)


'''
Conditional Probability - probability of an event (A) given that another event (B) has
already occurred

Given the type of gear box, prob of different fuel type
'''
pd.crosstab(index = cars2['Automatic'],
            columns = cars2["FuelType"],
            normalize= 'index',
            margins=True,
            dropna = True)

pd.crosstab(index = cars2['Automatic'],
            columns = cars2["FuelType"],
            normalize= 'columns',
            margins=True,
            dropna = True)


'''
Correlation
'''
numdata = cars2.select_dtypes(exclude=[object])
corr_mat=numdata.corr()
corr_mat
