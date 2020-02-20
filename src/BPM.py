#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 17:46:26 2019

@author: bene
"""

import NanoImagingPack as nip
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import InverseModelling as im
#import InverseModelling as im 

class BPM(object):
    def __init__(self, lambda_0 = .65, n_embb = 1., mysize = ((1,128,128)), pixelsize = None, z_start = None, z_end = None):
        
        #TODO: We should take care of padding the object
        #TODO: Creating the input beam inside this class is best practice
        #TODO: Take care of the refrcoseffective
        
        ''' Class for creating a propagation step á la BPM
        lambda_0 = wavelength in background
        n_0 = background RI
        mysize = integer number of pixels
        pixelsize = pixelsize in same measures as wavelength
        order of pixels is Z,X,Y'''
        
        print('Initializing the BPM class')
        
        self.lambda_0 = lambda_0
        # refractive index immersion and embedding
        self.n_embb = n_embb
        self.lambda_m = self.lambda_0/self.n_embb; # wavelength in the medium


        if pixelsize is None :
            # assume lambda/4 sampling
            self.dz = self.lambda_m/4
            self.dx = self.lambda_m/4
            self.dy = self.lambda_m/4

        else:                    
            if pixelsize[0] is None:
                self.dz = self.lambda_m/4
            else:
                self.dz = pixelsize[0]
            
            if pixelsize[1] is None:
                self.dx = self.lambda_m/4
            else:
                self.dx = pixelsize[1]
           
            if pixelsize[2] is None:
                self.dy = self.lambda_m/4
            else:
                self.dy = pixelsize[2]

        # define size
        self.mysize = mysize
        
        if len(mysize)==3:
            # we have only one illumination direction to propagate 
            self.Nx = mysize[1]
            self.Ny = mysize[2]
            self.Nz = mysize[0]
            self.Nillu = 1
            print('We propagate one illumination mode')
            
        elif len(mysize)==4:
            # we have only one illumination direction to propagate 
            self.Nx = mysize[1]
            self.Ny = mysize[2]
            self.Nz = mysize[0]
            self.Nillu = mysize[-1]
            print('We propagate '+str(self.Nillu)+ ' illumination modes')
        
        # define some values
        self.my_n = self.n_embb+np.zeros(self.mysize)
        self.input_field = 0j + np.ones((self.Nx, self.Ny, self.Nillu))
        
        # compute the distances we want stuff to be propagated 
        if z_start is None:
            z_start = 0.
        if z_end is None:
            z_end = self.dz*self.Nz
            
        self.z_start = z_start
        self.z_end = z_end
        if self.z_end == self.dz:
            self.dz_steps = self.dz
        else:
            self.dz_steps = np.linspace(self.z_start, self.z_end, self.Nz)
        

        
    def compute_generic_propagator(self):
        ''' Forward propagator for 2D and 3D
        (Ewald sphere based) DO NOT USE NORMALIZED COORDINATES HERE
        Basically following the Fresnel Kernel '''
        
        # compute the frequency grid
        self.kxysqr= (nip.abssqr(nip.xx((self.mysize[1], self.mysize[2]), freq='ftfreq') / self.dx) + 
                      nip.abssqr(nip.yy((self.mysize[1], self.mysize[2]), freq='ftfreq') / self.dy)) + 0j
        self.k0=1/self.lambda_m
        self.kzsqr= nip.abssqr(self.k0) - self.kxysqr
        self.kz=np.sqrt(self.kzsqr)
        self.kz[self.kzsqr < 0]=0 # get rid of evanescent components
        self.dphi = 2*np.pi*self.kz# propagator for one slice
        self.dphi *= (self.dphi > 0) 
        self.Allprop = 1j * np.expand_dims(self.dphi,-1) * self.dz_steps
        self.Allprop = np.exp(np.transpose(self.Allprop,(-1,0,1)))

    def compute_2D_propagator(self):
        #self.A_input = self.intensityweights *np.exp((2*np.pi*1j) *
        #    (self.kxcoord * tf_helper.repmat4d(tf_helper.xx((self.mysize[1], self.mysize[2])), self.Nc) 
        #   + self.kycoord * tf_helper.repmat4d(tf_helper.yy((self.mysize[1], self.mysize[2])), self.Nc))) # Corresponds to a plane wave under many oblique illumination angles - bfxfun
         
        # effect of the illumination angle is kep as zero for now
        self.RefrCos = 1.
        print('Beware that we do not take care of the inclination angle yet!!')
        
        # compute the generic propagator
        self.compute_generic_propagator()

        # Precalculate the oblique effect on OPD to speed it up
        self.RefrEffect = 1j * self.k0 * self.RefrCos

        
    def propagate(self, TF_A_input = None, TF_obj_input = None, proptype = '2D_2D'):
        '''
        This is the generic propagator 
        
        
        TF_A_input - the n-dimensional electrical input field 
        TF_obj_input - the n-dimensional refractive index distribution 
        proptype - Want to map 2D->2D, 2D->3D
        '''
        
        if TF_A_input is None:
            TF_A_input = tf.constant(self.input_field)
            print('We use the default input field - which is a plane wave!')
            
            
        self.compute_2D_propagator()
        if proptype=='2D_2D':
            # only propagate a 2D slice to a 2D slice at certain distance
            self.TF_A_output = self.__propagate2D2D(TF_A_input, TF_obj_input)
        elif proptype=='2D_3D':
            # propagate a 2D slice to a 3D volume

            self.TF_A_output = self.__propagate2D3D(TF_A_input)
        elif proptype=='MultipleScaterring':
            # propagate a 2D slice to a 3D volume
            self.TF_A_output = self.__propagate2D3D(TF_A_input) 
            
        return self.TF_A_output

    def __propagate2D2D(self, TF_A_input, TF_obj_input):
        ''' This propagates the inputfield to a full 3D stack'''
        
        # Porting numpy to Tensorflow
        # Define slice-wise propagator (i.e. Fresnel kernel)
        self.TF_Allprop = tf.cast(tf.complex(np.real(np.squeeze(self.Allprop)),np.imag(np.squeeze(self.Allprop))), dtype=tf.complex64)
        self.TF_RefrEffect = tf.cast(self.RefrEffect, tf.complex64)
        self.TF_obj_input = tf.cast(TF_obj_input, tf.complex64)
        
        # This corresponds to the input illumination modes
        is_not_tf = True
        if is_not_tf:
           TF_A_input = tf.cast(tf.complex(np.real(TF_A_input),np.imag(TF_A_input)), dtype=tf.complex64)
        self.TF_A_input =  TF_A_input 
        #self.TF_RefrEffect = tf.constant(self.RefrEffect, dtype=tf.complex64)

        # Split Step Fourier Method
        with tf.name_scope('Fwd_Propagate'):
            if TF_obj_input is not None:
                with tf.name_scope('Refract'):
                    # beware the "i" is in TF_RefrEffect already!
                    self.TF_f = tf.exp(self.TF_RefrEffect*self.TF_obj_input)
                    self.TF_A_prop = self.TF_A_input * self.TF_f  # refraction step
            else:
                self.TF_A_prop = self.TF_A_input                                                 
            with tf.name_scope('Propagate'):
                self.TF_A_output = im.ift2d(im.ft2d(tf.expand_dims(self.TF_A_prop,0))* self.TF_Allprop) # diffraction step

        return self.TF_A_output
    
    
    def visKernel(self):
        ''' THis function visualizes the Propagation-Kernel'''
        #plt.subplot(121), plt.title('Kernel (Magn.)'), plt.imshow(np.abs(self.myprop))
        #plt.subplot(122), plt.title('Kernel (Angle.)'), plt.imshow(np.angle(self.myprop))
        nip.view(np.abs(self.Allprop))
        nip.view(np.angle(self.Allprop))
        
    def visInputfield(self):
        ''' THis function visualizes the INput Field'''
        nip.view(self.dphi)

        
    def compute(self):
        ''' this is a helper function to evaluate the result from the BPM class 
        it computes the TF graph into a numpy object'''
        print('Open A TF session object')
        sess = tf.Session()
        self.NP_A_output = sess.run(self.TF_A_output)
        return self.NP_A_output 
        
    def printvar(self):
        ''' PRint all variables - only for debugging puposes'''
        try:
            print('The following parameters were set: ')
            print('lambda_0: '+str(self.lambda_0))
            print('lambda_m: '+str(self.lambda_m))         
            print('dx: '+str(self.dx))
            print('dy: '+str(self.dy))
            print('dz: '+str(self.dz))
            print('Nx: '+str(self.Nx))
            print('Ny: '+str(self.Ny))
            print('Nz: '+str(self.Nz))
            print('z_start '+str(self.z_start))
            print('z_end: '+str(self.z_end))
            print('n_embb: '+str(self.n_embb))
            print('dz_steps: ' +str(self.dz_steps))
        except:
            print('Did you initialize the class correclty?')


    def __propagate2D3D(self, TF_A_input, TF_obj_input):
        ''' This propagates the inputfield by one step'''
        
        # Porting numpy to Tensorflow
        # Define slice-wise propagator (i.e. Fresnel kernel)
        self.TF_Allprop = tf.cast(tf.complex(np.real(np.squeeze(self.myprop)),np.imag(np.squeeze(self.myprop))), dtype=tf.complex64)
        self.TF_RefrEffect = tf.cast(self.RefrEffect, tf.complex64)
        self.TF_obj_input = tf.cast(TF_obj_input, tf.complex64)
        
        # This corresponds to the input illumination modes
        is_not_tf = True
        if is_not_tf:
           TF_A_input = tf.cast(tf.complex(np.real(TF_A_input),np.imag(TF_A_input)), dtype=tf.complex64)
        self.TF_A_input =  TF_A_input 
        #self.TF_RefrEffect = tf.constant(self.RefrEffect, dtype=tf.complex64)


        with tf.name_scope('Refract'):
            # beware the "i" is in TF_RefrEffect already!
            if(self.is_padding):
                tf_paddings = tf.constant([[self.mysize_old[1]//2, self.mysize_old[1]//2], [self.mysize_old[2]//2, self.mysize_old[2]//2]])
                TF_real = tf.pad(TF_real_3D[-pz,:,:], tf_paddings, mode='CONSTANT', name='TF_obj_real_pad')
                TF_imag = tf.pad(TF_imag_3D[-pz,:,:], tf_paddings, mode='CONSTANT', name='TF_obj_imag_pad')
            else:
                TF_real = (TF_real_3D[-pz,:,:])
                TF_imag = (TF_imag_3D[-pz,:,:])
                

            self.TF_f = tf.exp(self.TF_RefrEffect*tf.complex(TF_real, TF_imag))
            self.TF_A_prop = self.TF_A_prop * self.TF_f  # refraction step
        with tf.name_scope('Propagate'):
            self.TF_A_prop = tf.ifft2d(tf.fft2d(self.TF_A_prop) * self.TF_Allprop) # diffraction step
            if(is_debug): self.TF_A_prop = tf.Print(self.TF_A_prop, [], 'Performing Slice Propagation')     

            for pz in range(0, self.mysize[0]):
            # Split Step Fourier Method
                with tf.name_scope('Fwd_Propagate'):
                    if TF_obj_input is not None:
                        with tf.name_scope('Refract'):
                            # beware the "i" is in TF_RefrEffect already!
                            self.TF_f = tf.exp(self.TF_RefrEffect*self.TF_obj_input)
                            self.TF_A_prop = self.TF_A_input * self.TF_f  # refraction step
                    else:
                        self.TF_A_prop = self.TF_A_input                                                 
                    with tf.name_scope('Propagate'):
                        self.TF_A_output = im.ift(im.ft(self.TF_A_prop) * self.TF_Allprop) # diffraction step
    
        return self.TF_A_output

    '''
    def __propagate3D(self, TF_A_input):
         This propagates the inputfield by one step''
        
        ## propagate the field through the entire object for all angles simultaneously
        #self.A_prop = np.transpose(self.A_input,[3, 0, 1, 2])  # ??????? what the hack is happening with transpose?!
        ''
        
        
        ' Porting numpy to Tensorflow '
        # Define slice-wise propagator (i.e. Fresnel kernel)
        self.TF_Allprop = tf.cast(tf.complex(np.real(np.squeeze(self.myprop)),np.imag(np.squeeze(self.myprop))), dtype=tf.complex64)

        # A propagator for all slices (2D->3D)
        self.TF_myAllSlicePropagator = tf.cast(tf.complex(np.real(self.myAllSlicePropagator), np.imag(self.myAllSlicePropagator)), tf.complex64)

        
        # This corresponds to the input illumination modes
        is_not_tf = True
        if is_not_tf:
           TF_A_input = tf.cast(tf.complex(np.real(self.TF_A_input),np.imag(self.TF_A_input)), dtype=tf.complex64)
        self.TF_A_input =  TF_A_input 
        #self.TF_RefrEffect = tf.constant(self.RefrEffect, dtype=tf.complex64)

        # simulate multiple scattering through object
        with tf.name_scope('Fwd_Propagate'):
            with tf.name_scope('Refract'):
                # beware the "i" is in TF_RefrEffect already!
                if(self.is_padding):
                    tf_paddings = tf.constant([[self.mysize_old[1]//2, self.mysize_old[1]//2], [self.mysize_old[2]//2, self.mysize_old[2]//2]])
                    TF_real = tf.pad(TF_real_3D[-pz,:,:], tf_paddings, mode='CONSTANT', name='TF_obj_real_pad')
                    TF_imag = tf.pad(TF_imag_3D[-pz,:,:], tf_paddings, mode='CONSTANT', name='TF_obj_imag_pad')
                else:
                    TF_real = (TF_real_3D[-pz,:,:])
                    TF_imag = (TF_imag_3D[-pz,:,:])
                    

                self.TF_f = tf.exp(self.TF_RefrEffect*tf.complex(TF_real, TF_imag))
                self.TF_A_prop = self.TF_A_prop * self.TF_f  # refraction step
            with tf.name_scope('Propagate'):
                self.TF_A_prop = tf.ifft2d(tf.fft2d(self.TF_A_prop) * self.TF_Allprop) # diffraction step
                if(is_debug): self.TF_A_prop = tf.Print(self.TF_A_prop, [], 'Performing Slice Propagation')     


 

            with tf.name_scope('Fwd_Propagate'):
            #print('---------ATTENTION: We are inverting the RI!')
            for pz in range(0, self.mysize[0]):
                with tf.name_scope('Refract'):
                    # beware the "i" is in TF_RefrEffect already!
                    if(self.is_padding):
                        tf_paddings = tf.constant([[self.mysize_old[1]//2, self.mysize_old[1]//2], [self.mysize_old[2]//2, self.mysize_old[2]//2]])
                        TF_real = tf.pad(TF_real_3D[-pz,:,:], tf_paddings, mode='CONSTANT', name='TF_obj_real_pad')
                        TF_imag = tf.pad(TF_imag_3D[-pz,:,:], tf_paddings, mode='CONSTANT', name='TF_obj_imag_pad')
                    else:
                        TF_real = (TF_real_3D[-pz,:,:])
                        TF_imag = (TF_imag_3D[-pz,:,:])
                        

                    self.TF_f = tf.exp(self.TF_RefrEffect*tf.complex(TF_real, TF_imag))
                    self.TF_A_prop = self.TF_A_prop * self.TF_f  # refraction step
                with tf.name_scope('Propagate'):
                    self.TF_A_prop = tf.ifft2d(tf.fft2d(self.TF_A_prop) * self.TF_Allprop) # diffraction step
                    if(is_debug): self.TF_A_prop = tf.Print(self.TF_A_prop, [], 'Performing Slice Propagation')     

     '''
     
