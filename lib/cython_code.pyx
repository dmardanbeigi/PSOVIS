'''
Created on Dec 25, 2017

@author: Diako
'''

import math
import numpy as np
from cython.parallel import prange
# cimport numpy as np
# np.import_array()


## getting the coordinates a line from a to b passes through.
# @cython.boundscheck(False)
cpdef list Transfer(list P1,list P2,list piv,list R,int x_or_y):
    """Brensenham line algorithm"""
     
     
    cdef int N #x, y, x2, y2, dx, sx,sy,dy,d,steep,i
    cdef list out# coords
     
    out=[]
    N=len(P1)
    for i in range(0,N):
        if (x_or_y==0):
            out.append(-(np.dot([P1[i]-piv[0],P2[i]-piv[1]] , np.matrix(R).T)[0,0]+ piv[0]))
        else:
            out.append(-(np.dot([P1[i]-piv[0],P2[i]-piv[1]] , np.matrix(R).T)[0,1]+ piv[1]))
    return out



