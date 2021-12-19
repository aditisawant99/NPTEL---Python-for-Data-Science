# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 13:03:27 2021

@author: aditi
"""

''' Setting working directory 

Method 1 : icon - drag and drop
Method 2 : os
Method 3 : cd '''

#Import os to set up the working directory
import os

#Setting the working directory

os.chdir('C:/Users/aditi')

cd C:/Users/aditi

''' File creation

Method 1 : icon
Method 2 : File
Method 3 : Ctrl + n '''


''' Variable creation '''

a = 11
b = a*10
print(a,b)

''' selecting multiple lines : 
    select lines that have to be commented and press ctrl + 1 '''

''' clearing an overpopulated console :
    %clear --> in console
    ctrl + l
    
removing/ deleting variable(s) : '''

del a

#%reset in consolw and type y after the prompt to delete all variables

''' Basic libraries in python
Numpy - Numerical python
Pandas - dataframe python
Matplotlib - visualization
Sklearn - machine learning
'''

import numpy
content = dir(numpy)
print(content)

''' Naming conventions 
Camel : ageEmp
Snake : age_emp
Pascal : AgeEmp
'''

'''Assigning values to multiple variables '''
 physics, chem, math = 89,90,75


'''data types 
boolean
integer
complex
float
string
'''

'''Identifying object data type'''
type(physics)

''' Verifying object data type'''
type(physics) is int

'''coerce object to new data type'''

phy = float(physics)
type(phy)


''' Types of operators
Arithmetic
Assignment
Relational or comparison
logical
bitwise

+ addition
- subtraction
* multiplication
/ division
% remainder
** exponent

PEDMAS
Parenthesis ()
Exponent **
Division /
Multiplication *
Addition +
Subtraction -


ASSIGNMENT OPERATORS
= assign values from right side operands to left side operand
+=
-=
*=
/=

RELATIONAL OPERATORS
returns boolean

<
>
<=
>=
==
!=

LOGICAL OPERATORS
operands are conditional statements and returns boolean value
designed to work with scalars or boolean values

or  (one of the statements is satisfied --> true)
and  (all of the statements is satisfied --> true)
not 

eg: (x>y) or (x<y) --> make sure to enclose the statements in parenthesis


BITWISE 

when operands are integers
integers are treated as a string of binary digits
operates bit by bit

Precedence of operators:
    ()
    **
    /
    *
    +,-
    &
    |
    ==, !=, >, >=, <, <=
    not
    and
    or

'''

''' LISTS
different data types
[]

'''

employee_list = [[1,2,3],['Ram','Preeti','Satish','John'], 4]

#modifying lists
employee_list[2] = 5
print(employee_list)

employee_list[1][3]='Karan'

'''
append() - adds an object at the end of the list
'''
employee_list[0].append(4)
employee_list[0].append(5)
employee_list[1].append('Suraj')
employee_list.append(6)

employee_list[0].append([6,7])

'''
pop() - displays the object that is being removed from the list at the 
specified index number
'''

employee_list[0].pop()
employee_list.pop()

'''
insert() - adds an object at the given position in a list
'''
employee_list[0].insert(0,6) #position 0, object 6

'''
del - removes the object at the specified index number
'''
del employee_list[0][0]

'''
remove() - removes the first matching object from a list
'''
employee_list[1].remove('Ram')
employee_list[0].remove(5)

del employee_list


'''
TUPLES
()
immutable
'''

emp = ('p001', 'john', 35, 40000)
print(emp)

#extract
emp[0]

#slicing

emp[0:4]


english = (99,98,89,97,95)
len(emp)
min(english)
max(english)

comb = emp + english
comb


'''
DICTIONARY
hash-table data structure
{}
key value pairs
'''

fuel = {'petrol':1, 'diesel':2, 'cng':3}

fuel['petrol']
fuel.keys()
fuel.values()
fuel.items()

#to add a key value pair
fuel['electric'] = 4
fuel

#to adda key value pair
fuel.update({'iii':5})

#to modify an existing value
fuel['cng'] = 5

# del - removes the key value pair

del fuel['iii']

#clear
fuel.clear()



'''
SETS
a collection of distinct objects
do not hold duplicate items
stores elements in no particular order
created using {}
'''

age = {56,78,89,54,43,56}
age #unique items only

'''
add() - adds element(s) to the existing set at any position 
'''
empname = {'a','b','c','e','f','f'}
empname

empname.add('g')

'''
discard() - removes matching objects from an existing set
'''
empname.discard('g')

'''
clear() - removes all the elements from an existing set
'''
empname.clear()


'''
Set operations
'''
jr_ds = {'R','Python','Tableau'}
sr_ds = {'R','Python','scala','Java','Tableau'}

jr_ds
sr_ds

'''
union() - returns all elements belonging to both set A and B
'''
union = jr_ds.union(sr_ds)
union

'''
intersection() - returns elements common to set A and B
'''
inter = jr_ds.intersection(sr_ds)
inter

'''
set difference
difference() - returns elements belonging to A but not B
'''

diff = jr_ds.difference(sr_ds)
diff

'''
symmetric_difference() - returns elements not common to both sets
1 - (AnB)
'''
sym_diff = jr_ds.symmetric_difference(sr_ds)
sym_diff

