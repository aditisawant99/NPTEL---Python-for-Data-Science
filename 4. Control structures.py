# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 19:27:14 2021

@author: aditi
"""

'''
If-else-then

Execute certain commands only when certain condition(s) is (are) satisfied
(if - then - else)

Execute certain commands repeatedly and use a certain logic to stop the iteration
(for, while loops)
'''


'''
If, If else and If-elif-else are a family of constructs where:
    
A condition is first checked, if it is satisfied then operations are performed
If the condition is not satisfied, code exit construct or moves on to other options
'''

'''
For loop
Execute certain commands repeatedly and use a certain logic to stop the iteration
(for loop)

While loop
Used when a set of commands are to be executed depending on a specific condition
'''

import os
import pandas as pd
import numpy as np

os.chdir('C:\\Aditi\\MSc\\Sem 4')
cars = pd.read_csv('Toyota.txt')
cars

'''
we will create 3 bins from 'Price' variable using if else and for loops
the binned values will be stored as class in a new column, 'Price Class'
'''

#create a column at 10th position with blank values

cars.insert(10, "Price_Class", "")
cars

#FOR LOOP

for i in range(0, len(cars['Price']),1):
    if (cars['Price'][i] <= 8450):
        cars['Price_Class'][i] = "Low"
    elif ((cars['Price'][i] > 11950)):
        cars['Price_Class'][i] = "High"
    else:
        cars['Price_Class'][i] = "Medium"
        

cars


#WHILE LOOP

i = 0 
while i < len(cars['Price']):
    if (cars['Price'][i] <= 8450):
        cars['Price_Class'][i] = "Low"
    elif ((cars['Price'][i] > 11950)):
        cars['Price_Class'][i] = "High"
    else:
        cars['Price_Class'][i] = "Medium"
    i+=1
    

cars

'''
series.value_counts() - returns series containing count of unique values
'''
cars['Price_Class'].value_counts()
# no of vlaues under each category low medium and high

cars['Price_Class'] = cars['Price_Class'].astype('category')
cars.info()



''' 
def function_name(parameters):
    statements
'''

'''
Converting the Age variable from months to years by defining a function
'''
cars.insert(11, 'Age_Converted', 0)
cars

def month_to_year(val):
   val_conv = val/12
   return val_conv

cars['Age_Converted'] = month_to_year(cars['Age'])
cars

cars['Age_Converted'] = round(cars['Age_Converted'], 1)        
cars


'''
Functions in python takes multiple input objects but returns only one object as output
'''


cars.insert(12, 'km_per_month',0)

def converter(val1, val2):
    val_converted = val1/12
    ratio = val2/val1
    return [val_converted, ratio]

#output is returned in the form of a list

'''
Here, Age and KM columns are input to the function
The outputs are assigned to 'Age_Converted' amd 'km_per_month'
'''

cars["Age_Converted"], cars["km_per_month"] = converter(cars['Age'], cars['KM'])
cars

cars['Age_Converted'] = round(cars['Age_Converted'], 1)        
cars['km_per_month'] = round(cars['km_per_month'], 2)        
cars