#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 Julien Flamant
#
# Distributed under terms of the CeCILL Free Software Licence Agreement

'''
Module for the LTI filtering of bivariate signals.
'''

import numpy as np
import quaternion

from . import qfft

class Filter(object):
    def __init__(self, N, dt=1.0):

        self.f = np.fft.fftfreq(N, d=dt)
        self.N = N
        self.dt = dt
    
    # def plot(self):
    #     # ''' Displays the spectral response of the filter for different input types'''
    #     # fig, ax = plt.subplots()
        
    #     # N = len(self.f)
    #     # ax.plot(self.f, randn(N))


class HermitianFilter(Filter):
    def __init__(self, N, K, eta, mu, dt=1.0):
        # initialize Filter
        Filter.__init__(self, N, dt=dt)

        # several tests to ensure proper feed
        for param in [K, eta, mu]:
            if np.size(param) != 1 and np.size(param) != N:
                raise ValueError('Parameters should be either scalar or of size N')
        
       
        if np.size(K) == 1:
            Kvec = np.ones(N)*K
        else:
            Kvec = K
        if np.size(eta) == 1:
            etavec = np.ones(N)*eta
        else:
            etavec = eta
        if np.size(mu) ==1:
            muvec = np.ones(N)*mu/np.abs(mu)
        else:
            muvec = np.zeros(N, dtype='quaternion')
            muvec[np.abs(mu) > 0] = mu[np.abs(mu) > 0]/np.abs(mu)[np.abs(mu) > 0]
        # ensure symmetry relations
        qi = quaternion.x
        
        Kvec[N//2 +1:] = Kvec[1:N//2][::-1] # K(-v) = K(v)
        etavec[N//2 +1:] = etavec[1:N//2][::-1] # eta(-v) = eta(v)
        
        # mu(-v) = conj_i(mu(v))
        muvec[N//2 + 1:] = -qi*np.conj(muvec[1:N//2][::-1])*qi
        muvec[0] = .5*(muvec[1] + muvec[-1])
        muvec[N//2] = .5*(muvec[N//2+1] + muvec[N//2-1])
        
        
        # save
        self.K = Kvec
        self.eta = etavec
        self.mu = muvec
        
    def output(self, x):
        ''' returns the output of the filter given an input signal x'''
        
        if np.size(x) != self.N:
            raise ValueError('Size of input array should be the same as the constructed filter')

        X = qfft.Qfft(x)
        
        qj = quaternion.y
        
        Y = self.K*(X - self.eta*(self.mu*X)*qj)
        y = qfft.iQfft(Y)
        
        return y
        
class UnitaryFilter(Filter):
    def __init__(self, N, mu, alpha, phi, dt=1.0):
        # initialize Filter
        Filter.__init__(self, N, dt=dt)
        
        # several tests to ensure proper feed
        for param in [mu, alpha, phi]:
            if np.size(param) != 1 and np.size(param) != N:
                raise ValueError('Parameters should be either scalar or of size N')

        if np.size(mu) ==1:
            muvec = np.ones(N)*mu/np.abs(mu)
        else:
            muvec = np.zeros(N, dtype='quaternion')
            muvec[np.abs(mu) > 0] = mu[np.abs(mu) > 0]/np.abs(mu)[np.abs(mu) > 0]
        
        if np.size(alpha) == 1:
            alphavec = np.ones(N)*alpha
        else:
            alphavec = alpha
            
        if np.size(phi) == 1:
            phivec = np.ones(N)*phi
        else:
            phivec = phi
        # ensure symmetry relations
        qi = quaternion.x
        
        alphavec[N//2 +1:] = alphavec[1:N//2][::-1] # alpha(-v) = alpha(v)
        phivec[N//2 +1:] = -phivec[1:N//2][::-1] # phi(-v) = -phi(v)
        phivec[0] = 0
        phivec[N//2] = 0

        # mu(-v) = invol_i(mu(v))
        muvec[N//2 + 1:] = -(qi*muvec[1:N//2][::-1])*qi
        muvec[0] = .5*(muvec[1] + muvec[-1])
        muvec[N//2] = .5*(muvec[N//2+1] + muvec[N//2-1])
        
        
        # save
        self.alpha = alphavec
        self.phi = phivec
        self.mu = muvec
        
    def output(self, x):
        ''' returns the output of the filter given an input signal x'''
        
        if np.size(x) != self.N:
            raise ValueError('Size of input array should be the same as the constructed filter')

        X = qfft.Qfft(x)
        
        qj = quaternion.y
        
        Y = (np.exp(self.mu*self.alpha/2)*X)*np.exp(qj*self.phi)
        y = qfft.iQfft(Y)
        
        return y
        