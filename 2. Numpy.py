# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 10:33:51 2021

@author: aditi
"""
'''
NUMPY
ARRAYS
MATRIX
LINEAR ALEGBRA
'''

'''
NUMPY

Mathematical and logical operations on arrays
Fourier transforms
linear algebra operations
random number generation
'''

import numpy as np
x = np.array([2,3,4,5])
type(x)
x
print(x)

'''
Numpy can handle different categorical entities
'''
x = np.array([2,3,'n',5])
print(x)

#All elements are coerced to same data type


'''
numpy.linspace() - returns equally spaced numbers within the given range based on
the sample number

np.linspace(start, stop, num, dtype, retstep)
start - start of the interval range
stop - end of the interval range
num - number of samples to be generated
dtype - type of output array
retstep - return the sample, step value
'''
b = np.linspace( start = 1, stop = 5, num=10, endpoint=True, retstep = True )
b 


'''
np.arange() - returns equally spaced numbers within the given range based on step size

np.arange(start, stop, step)
'''

np.arange(1, 10, 2)

'''
numpy.ones() = returns an array of given shape and type filled with ones

np.ones(shape, dtype)
'''

np.ones(shape=(3, 4))


'''
np.zeros() - returns an array of given shape and type filled with zeros

np.zeros(shape,dtype)
'''

np.zeros((3,4), dtype=int)

'''
numpy.random.rand()
'''
np.random.rand(5,2)


'''
numpy.logspace() - returns equally spaced numbers based on log scale

np.logspace(start, stop, num, endpoint, base, dtype)
'''

np.logspace(1, 10, num=5, endpoint=True, base=10)



'''
speed
'''

x=range(1000)
timeit sum(x)

y=np.array(x)
timeit np.sum(y)

'''
storage space

getsizeof() - returns the size of the object in bytes
itemsize - returns the size of one element of a numpy array

size of the list can be found by multiplying the size of an individual element 
with the number of elements in the list

size of an array can be found by multiplying the size of an individual element 
with the number of elements in the array
'''

import sys
sys.getsizeof(1) * len(x)

y.itemsize * y.size



'''
Reshaping an array

reshape() - recasts an array to new shape without changing its data
'''
np.arange(1,10)
grid = np.arange(1, 10).reshape(3,3)
grid


a = np.array([[1,2,3],[4,5,6],[7,8,9]])
a


'''
shape() - returns dimensions of an array
'''

grid.shape


'''add, multiply, subtract, divide, remainder '''

a = np.array([[1,2,3],[4,5,6],[7,8,9]])
a

b = np.arange(start=11, stop=20).reshape(3,3)
b

np.add(a,b)
np.multiply(a,b)
np.subtract(a,b)
np.divide(a,b)
np.remainder(a,b)


'''accessing components'''
a[0,1]
a[1:3] # 2nd and 3rd row
a[ :, 0] #all rows, 1st column
a[0] #1st row
a[0, :] #1st row


suba = a[0:2 , 0:2]
suba

suba[0,0] = 12
a
 '''modifying the subset will update the original array as well'''
 
 '''
 np.transpose() - permute the dimensions of array
 '''
 
 np.transpose(a)
 
 '''
 append() - adds values at the end of the array
 
 np.append(array, axis)
 '''
 
a_row = np.append(a,[[10,11,14]], 0)
a_row


new_col = np.array([21,22,23]).reshape(3,1)
new_col
a_col = np.append(a,new_col,1)
a_col


'''
insert() - adds values at a given position and axis in an array

np.insert(array, obj, values, axis)

array - input array
obj - index position
values - array of values to be inserted
axis - axis along which values should be inserted
'''

a_ins = np.insert(a,1,[13,15,16], axis=0)

'''
delete() - removes values at a given position and axis in an array

np.delete(array, obj, axis)
'''
a_del = np.delete(a_ins, 2, axis=0)
a_del








'''
MATRIX

matrix() - returns a matrix from an array type object or string of data

np.matrix(data)
'''

import numpy as np

a = np.matrix("1,2,3,4; 4,5,6,7; 7,8,9,10")
a

'''
shape() - returns number of rows and columns from a matrix
'''
a.shape #shape of matrix
a.shape[0] #no. of rows in matrix
a.shape[1] #no.of columns in matrix


'''
size() - returns the number of elements in the matrix
'''
a.size


'''
insert() - adds a value at a given position and axis ina matrix

np.insert(matrix, obj, values, axis)
'''

col_new = np.matrix("2, 3, 4")
col_new

a = np.insert(a, 0, col_new, axis=1) #column inserted at 1st position or 0th index
a


row_new = np.matrix("4,5,6,7,9")
a = np.insert(a, 0, row_new, axis=0)
a

a[1,1] = -3
a


a[1, :] #extracting 2nd row
a[1]


a[ : , 2] #extracting 3rd column

a[1,2]



'''
matrix operations
'''

c = np.matrix(np.arange(0,20)).reshape(5,4)
d = np.matrix(np.arange(20,40)).reshape(5,4)

c
d

np.add(c,d)
np.subtract(c,d)
np.dot(c,d) #performs matrix multiplication between two matrices

#number of columns in matrix c (4) should be equal to number of rows in matrix d (5)

d = np.transpose(d)
np.dot(c,d)

np.matmul(c,d)
c@d


#element wise multi
d = np.matrix(np.arange(20,40)).reshape(5,4)
np.multiply(c,d)

np.divide(c,d)







'''
LINEAR ALGEBRA


determinant of matrix (square matrix)

np.linalg.det() - returns determinant of the matrix

'''
x = np.matrix("4,5,16,7; 2,-3,2,3; 3,4,5,6; 4,7,8,9")
x

det_mat = np.linalg.det(x)
det_mat

'''
rank - the number of linearly independent rows and linearly independent columns

np.linalg.matrix_rank() - returns rank of the matrix
'''

rank_mat = np.linalg.matrix_rank(x)
rank_mat
# 4 linearly independent rows


'''
inverse of matrix = A^(-1) = 1/|A| adjoint A

np.linalg.inv() - returns the multiplicative inverse of a matrix
'''

a = np.matrix("3,1,2; 3,2,5; 6,7,8")
np.linalg.inv(a)

b = np.matrix("2,1,2; 1,0,1; 3,1,3")
inv = np.linalg.inv(b)


#if determinant is 0, inverse does not exist

det = np.linalg.det(b)
det


'''
System of linear equations

3x + y + 2z = 2
3x +2y + 5z =-1
6x +7y + 8z = 3

Ax = b

(3 1 2)      (x)             (2)
( 3 4 5)     (y)     =       (-1)
( 6 7 8)     (z)             (3)

x = A^(-1) b

np.linalg.solve() - return the solution to the system Ax = b

'''

A = np.matrix("3,1,2; 3,2,5; 6,7,8") 
A

b = np.matrix("2,-1,3").transpose()
b
sol_linear = np.linalg.solve(A,b)
sol_linear




