#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 14:57:43 2016

@author: iki
"""
from __future__ import division
import numpy as np
import scipy.io
import urllib2


#Download Dataset
url = "http://ufldl.stanford.edu/housenumbers/train_32x32.mat"

file_name = url.split('/')[-1]
u = urllib2.urlopen(url)
f = open(file_name, 'wb')
meta = u.info()
file_size = int(meta.getheaders("Content-Length")[0])
print "Downloading: %s Bytes: %s" % (file_name, file_size)

file_size_dl = 0
block_sz = 8192
while True:
    buffer = u.read(block_sz)
    if not buffer:
        break

    file_size_dl += len(buffer)
    f.write(buffer)
    status = r"%10d  [%3.2f%%]" % (file_size_dl, file_size_dl * 100. / file_size)
    status = status + chr(8)*(len(status)+1)
    print status,

f.close()



#Load dataset
mat = scipy.io.loadmat('train_32x32.mat')
mat = mat['X']
b,h,d,n = mat.shape


#Convert all RGB-Images to greyscale
greyImg = np.zeros(shape =(n, b*h))

def rgb2gray(rgb):
    
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
    

for i in range(n):
    #Convert to greyscale
    img = rgb2gray(mat[:,:,:,i])
    
    #Convert to array
    img = img.reshape(1,1024)
    
    #2D Dataset    
    greyImg[i,:] = img



scipy.io.savemat('manipulated_StreetView.mat', mdict={'train': greyImg})
        
