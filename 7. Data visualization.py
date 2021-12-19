# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 12:59:27 2021

@author: aditi
"""

'''
matplotlib
pandas visualization
seaborn
ggplot
plotly
'''

'''
Scatterplot
(correlation plot)
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

os.chdir('C:\\Aditi\\MSc\\Sem 4')

cars = pd.read_csv("Toyota.txt")
cars.dropna(axis=0, inplace=True)

plt.scatter(cars['Age'], cars['Price'], c='red')
plt.title('Scatter plot of Price vs Age of the cars')
plt.xlabel('Age (months)')
plt.ylabel('Price (Euros)')
plt.show()




'''
Histogram
frequency distribution of numerical variables
ranges and bins
'''

plt.hist(cars['KM'], color='green', edgecolor='white', bins=5)
plt.title('histogram of KM')
plt.xlabel('Kilometer')
plt.ylabel('Frequency')
plt.show()

'''
Bar plot
Frequency of Categorical data
'''
cars['FuelType'].value_counts()

counts = [1264, 155, 17]
fuel = ('Petrol', 'Diesel', 'CNG')
index = np.arange(0, len(fuel), 1)
index

plt.bar(index, counts, color=['red', 'blue', 'cyan'])
plt.title('Bar plot of fuel types')
plt.xlabel('fuel types')
plt.ylabel('frequency')
plt.xticks(index, fuel, rotation=90)
plt.show()



import seaborn as sns


'''
scatter plot
'''

sns.set(style='darkgrid')
sns.regplot(x=cars['Age'], y=cars['Price'], fit_reg=False, marker='*') #regression plot

# scatter plot of price vs age by fueltype

sns.lmplot(x='Age', y='Price', data=cars, fit_reg=False, hue='FuelType', legend=True,
           palette = 'Set1')


'''
Histogram
'''

sns.distplot(cars['Age'], kde=False, bins=5)


'''
Bar plot
'''

sns.countplot(x='FuelType', data=cars)

#Grouped bar plot of FuelType and Automatic

sns.countplot(x="FuelType", data=cars, hue='Automatic')


'''
Box and whiskers plot - numerical variable
'''
sns.boxplot(y=cars['Price'])


#box plot for numerical vs categorical variable
#price of the cars for various fuel types

sns.boxplot(x = cars['FuelType'], y=cars['Price'])

#Grouped box and whiskers plot

sns.boxplot(x='FuelType', y = 'Price', data=cars, hue='Automatic' )


'''
Box and whiskers plot and histogram
'''

f, (ax_box, ax_hist) = plt.subplots(2, gridspec_kw = {"height_ratios":(0.15,0.85)})

sns.boxplot(cars['Price'], ax=ax_box)

sns.distplot(cars['Price'], ax=ax_hist, kde=False)


'''
Pairwise plots
'''

sns.pairplot(cars, kind='scatter', hue='FuelType')
plt.show()
