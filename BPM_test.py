#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 17:05:50 2019

@author: bene
"""

# test BPM class
import BPM as bpm
import tensorflow as tf
import numpy as np
import NanoImagingPack as nip
import InverseModelling as im

# create a pseudo input field
myinputfield = np.ones((128,128))+0j
TF_A_input = tf.constant(myinputfield)


myBPM = bpm.BPM()
myBPM.printvar()
myBPM.propagate(TF_A_input=myinputfield, TF_obj_input = None, proptype = '2D_2D')

# visualize the kernel
myBPM.visKernel()
# compute the result
myres = myBPM.compute()

print('Display the result')
nip.view(np.angle(myres))