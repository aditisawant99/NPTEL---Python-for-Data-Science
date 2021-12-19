# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 21:15:05 2021

@author: aditi
"""

'''
READING DATA

.csv
.xlsx
.txt
'''
import os
import pandas as pd

#csv
os.chdir("C:\\Aditi\\MSc\\Sem 3\\SC-V")
data = pd.read_csv('Breast_cancer_data.csv')
data

'''
Junk values can be converted to missing values by passing them as a list to the
parameter 'na_vales'
for eg.
data = pd.read_csv('iris.txt', na_values=['??','###'])
'''


#excel
os.chdir("C:\\Aditi\\MSc\\Sem 3\\SC-V")
data = pd.read_excel('Breast_cancer_data.xlsx', sheet_name= 'Breast_cancer_data')
data

#text
os.chdir("C:\\Aditi\\MSc\\Sem 3\\SC-V\\P3")
data = pd.read_table('iris.txt', sep=',')
data

'''
all columns read and stored in a single column of dataframe
In order to avoid this, provide a delimiter to the parameters 'sep' or 'delimiter'
'''




'''
PANDAS
'''

import os
import pandas as pd
import numpy as np

os.chdir('C:\\Aditi\\MSc\\Sem 4')
cars = pd.read_csv('Toyota.txt')
cars


'''
Copy of original data

Shallow copy -
It only creates a new variable that shares the reference of the original
object
Any changes made to a copy of object will be reflected in the original object as well
'''
samp = cars.copy(deep=False)
#or
samp = cars
'''
Deep copy -
A copy of object is copied in other object with no reference to the original
Any changes made to a copy of object will not be reflected in the original object
'''
samp = cars.copy(deep=True)
samp


'''
Attributes of Data
Dataframe.index
To get the index (row labels) of the dataframe
'''

cars.index

cars.columns #column names

cars.size  #total no.of elements from df

cars.shape #rows and columns number

cars.memory_usage() #in terms of bytes

cars.ndim #no.of axes / array dimensions = 2 (row and col)


'''
Indexing and Slicing
'.' and '[]'
'''

cars.head(6) #first 6 rows, default = 5

cars.tail(5) #last 5 rows

cars.at[4,'FuelType'] # label-based scalar lookups

cars.iat[5,6] # integer-based lookups

'''
to access a group of rows and columns by label(s) .loc[] can be used
'''
cars.loc[ : , ('FuelType','HP')]



'''
Datatypes

Character types
'''

cars.dtypes #data type of each column

cars.get_dtype_counts() #counts of unique data types in the dataframe

'''
selecting data based on data types

select_dtypes() - returns a subset of the columns from dataframe based on the column
dtypes
'''
cars.select_dtypes(exclude=[object])
cars


cars.info() #concise summary of dataframe
cars.info()

'''
Unique elements in a column
'''

np.unique(cars['KM'])
np.unique(cars['HP'])


'''
converting variable's data types

astype() - used to explicitly convert data types from one to another
'''

cars['MetColor'] = cars['MetColor'].astype('object')
cars['Automatic'] = cars['Automatic'].astype('object')


cars['FuelType'].nbytes

cars['FuelType'] = cars['FuelType'].astype('category')

cars.info()

np.unique(cars['Doors'])
cars['Doors'].replace('three', 3, inplace=True)


'''
to detect missing values

df.isnull().sum()
'''
cars = pd.DataFrame(cars)
cars.isnull().sum()
