# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 22:18:43 2019

@author: TEJ
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool



---------------    INTRODUCTION TO PYTHON

         MATPLOTLIB
         
- Matplot is a python library that help us to plot data.
- The easiest and basic plots are line, scatter and histogram plots.

- Line plot is better when x axis is time.
- Scatter is better when there is correlation between two variables
- Histogram is better when we need to see distribution of numerical data.
- Customization: Colors,labels,thickness of line, title, opacity, grid, figsize, ticks of axis and linestyle

- just example (speed and Defence are two feactures) (replace speed and defence to other feactures)

# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line
data.Speed.plot(kind = 'line', color = 'g',label = 'Speed',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')
data.Defense.plot(color = 'r',label = 'Defense',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')
plt.legend(loc='upper right')     # legend = puts label into plot
plt.xlabel('x axis')              # label = name of label
plt.ylabel('y axis')
plt.title('Line Plot')            # title = title of plot
plt.show()


# Scatter Plot (attack and defence two feactures) (replace atttack and defence to other feactures)
# x = attack, y = defense
data.plot(kind='scatter', x='Attack', y='Defense',alpha = 0.5,color = 'red')
plt.xlabel('Attack')              # label = name of label
plt.ylabel('Defence')
plt.title('Attack Defense Scatter Plot')            # title = title of plot


# Histogram (speed is just a feacture)(we can replace speed by other feacture)
# bins = number of bar in figure
data.Speed.plot(kind = 'hist',bins = 50,figsize = (12,12))
plt.show()

################################################
 
                           #  FUNCTIONS

tuble: sequence of immutable python objects. 
cant modify values 
tuble uses paranthesis like tuble = (1,2,3) 
unpack tuble into several variables like a,b,c = tuble


def tuble_ex():
#    """ return defined t tuble"""
    t = (1,2,3)
    return t
a,b,c = tuble_ex()
print(a,b,c)


What we need to know about scope:

global: defined main body in script
local: defined in a function
built in scope: names in predefined built in scope module such as print, len 


x = 2
def f():
    x = 3
    return x
print(x)      # x = 2 global scope
print(f())    # x = 3 local scope


x = 5
def f():
    y = 2*x        # there is no local scope x
    return y
print(f())         # it uses global scope x
# First local scopesearched, then global scope searched,
   # if two of them cannot be found lastly built in scope searched.


 #   NESTED FUNCTION¶

- function inside function.
- There is a LEGB rule that is search local scope, enclosing function, global and built in scopes,
   respectively.


#nested function
def square():
    """ return square of value """
    def add():
        """ add two local variable """
        x = 2
        y = 3
        z = x + y
        return z
    return add()**2
print(square())    


#   DEFAULT and FLEXIBLE ARGUMENTS


Default argument example: 
def f(a, b=1):
  """ b = 1 is default argument"""
Flexible argument example: 
def f(*args):
 """ *args can be one or more"""

def f(** kwargs)
 """ **kwargs is a dictionary"""


lets write some code to practice

# default arguments
def f(a, b = 1, c = 2):
    y = a + b + c
    return y
print(f(5))
# what if we want to change default arguments
print(f(5,4,3))


##    LAMBDA FUNCTION

square = lambda x: x**2     # where x is name of argument
print(square(4))
tot = lambda x,y,z: x+y+z   # where x,y,z are names of arguments
print(tot(1,2,3))



#    ANONYMOUS FUNCTİON

- Like lambda function but it can take more than one arguments.
- map(func,seq) : applies a function to all the items in a list

number_list = [1,2,3]
y = map(lambda x:x**2,number_list)
print(list(y))


#  ITERATORS

iterable is an object that can return an iterator
iterable: an object with an associated iter() method 
example: list, strings and dictionaries
iterator: produces next value with next() method

# iteration example
name = "ronaldo"
it = iter(name)
print(next(it))    # print next iteration
print(*it)         # print remaining iteration


#    zip(): zip lists

# zip example
list1 = [1,2,3,4]
list2 = [5,6,7,8]
z = zip(list1,list2)
print(z)
z_list = list(z)
print(z_list)



un_zip = zip(*z_list)
un_list1,un_list2 = list(un_zip) # unzip returns tuble
print(un_list1)
print(un_list2)
print(type(un_list2))



#   LIST COMPREHENSİON
One of the most important topic of this kernel 
We use list comprehension for data analysis often. 
list comprehension: collapse for loops for building lists into a single line 
Ex: num1 = [1,2,3] and we want to make it num2 = [2,3,4]. This can be done with for loop. However it is unnecessarily long. We can make it one line code that is list comprehension.

# Example of list comprehension
num1 = [1,2,3]
num2 = [i + 1 for i in num1 ]
print(num2)




[i + 1 for i in num1 ]: list of comprehension 
i +1: list comprehension syntax 
for i in num1: for loop syntax 
i: iterator 
num1: iterable object

# Conditionals on iterable
num1 = [5,10,15]
num2 = [i**2 if i == 10 else i-5 if i < 7 else i+5 for i in num1]
print(num2)

######################################################################

list=['c','df','da','df']
df=pd.DataFrame(list)
print(df)

df['z']=1
print(df)



























