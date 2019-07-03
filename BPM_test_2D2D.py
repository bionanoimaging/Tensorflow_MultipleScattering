#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 17:05:50 2019

@author: bene
"""

# test BPM class
from src import BPM as bpm
import tensorflow as tf
import numpy as np
import NanoImagingPack as nip
import InverseModelling as im

'''TEST SCRIPT for the BPM algorithm
B.Diederich, 26.06.2019'''

# define some parameters
mysize = (20,128,128) # Z;XY
mypixelsize = (.65/4, .65/4, .65/4)
# create a pseudo input field
myinputfield = nip.ones((mysize[1],mysize[2]))+0j # plane wave
TF_A_input = tf.constant(myinputfield)

# load some object slice 
myobjslice = .1*nip.extract(nip.readim(), ROIsize=mysize)+0j
TF_obj_input = tf.constant(myobjslice)

# create the BPM object with default parameters
myBPM = bpm.BPM(mysize = mysize, pixelsize=mypixelsize) 

myBPM.printvar()
myBPM.propagate(TF_A_input=myinputfield, TF_obj_input = TF_obj_input, proptype = '2D_2D')

# visualize the kernel
myBPM.visKernel()
# compute the result
myres = myBPM.compute()

print('Display the result')
nip.view(np.abs(myres))


