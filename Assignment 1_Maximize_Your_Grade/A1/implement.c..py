#!/usr/bin/env python
# coding: utf-8

# In[25]:


import numpy as np
from itertools import combinations

def EnumerateAll(mlist, m, n):
    ''' Enumerate all the n-tuple from mlist.
        where mlist contains m numbers.
        We assume m >= n.
    ''' 

    # this is just for demo purpose.
    # write your own code for question (3) here.
    tmp=list(combinations(mlist,n))
    tmp_arr=[]
    for i in tmp:
        tmp_arr.append([int(y) for y in i])
    return tmp_arr


def SolveLP(A, b, G):
    '''Solve the linear programming problem
        Max G(x)
        st. Ax <= b
             x >= 0
    '''
    # step 0: initialization
    maxg = 0;

    # step 1a: enumuate all combinations
    [m, n] = A.shape
    lst = EnumerateAll(np.arange(m), m, n)

    # step 1b: compute all the intersection points
    points = [];
    for idx in lst:
        Ai = A[idx, :]
        bi = b[idx]
        xi = np.linalg.solve(Ai, bi)

        # step 2: check the feasibility of the itersection point
        feasible = 1
        for i in range(m):
            if np.dot(A[i,:], xi) < b[i]:  # violate a constraints
                feasible = 0
        if feasible == 1:            # only add the feasible point
            points.append(xi)

    # step 3: evaluate the G function for all intersection points
    values = []
    for ptx in points:
        values.append(np.dot(G[0:n], ptx))

    # step 4: find the point with the largest value as the result
    maxg = max(values)
    maxidx = values.index(maxg)
    x = points[maxidx]

    return x, maxg

#-------------------------------#
# main program starts from here.#
#-------------------------------#
# Put all the coefficients of the constrains into a matrix A and a vector b

A = np.array([[1,3,2,1],[-1,-2,4,7],[1,1,2,-1], [3,2,-1,2], [1,4,-2,-1]])
b = np.array([150,30,70,100,90])
G = np.array([1,1.2,1.5,3])

# solve this problem
[x, maxg] = SolveLP(A, b, G)
print(x)
print(maxg)


# %%