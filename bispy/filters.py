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
    '''
    Hermitian filter for bivariate signals.
    The Hermitian filtering relations reads in the QFT spectral domain:
        Y(nu) = K(nu)*[X(nu) - eta(nu)*mu(nu)*X(nu)*qj]
    where K is the homogeneous gain of the filter, eta is the polarizing power
    and mu the axis of the filter.

    Parameters
    ----------

    N : int
        length of the filter

    K : array_type or float
        homogeneous gain array (should be of size N). If K is a float, then a
        constant gain is assumed throughout frequencies.

    eta : array_type or float
        polarizing power array (should be of size N). If eta is a float, then a
        constant polarizing is assumed throughout frequencies.

    mu : array_type (quaternion) or quaternion
        diattenuation axis quaternion array (should be of size N and of dtype quaternion).

    dt : float (optional)
        time sampling step (default 1)

    Attributes
    ----------

    N : int
        length of the filter

    f : array_type
        sampled frequencies

    dt : float
        time sampling step (default 1)

    K, eta, mu : array_types
        filter parameters
    '''

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
        ''' returns the output of the filter given an input signal x

        '''

        if np.size(x) != self.N:
            raise ValueError('Size of input array should be the same as the constructed filter')

        X = qfft.Qfft(x)

        qj = quaternion.y

        Y = self.K*(X - self.eta*(self.mu*X)*qj)
        y = qfft.iQfft(Y)

        return y

class UnitaryFilter(Filter):
    '''
    Unitary filter for bivariate signals.
    The Unitary filtering relation reads in the QFT spectral domain:
        Y(nu) = exp(mu(nu)*alpha(nu) / 2)*X(nu)exp(1j*phi(nu))
    where phi is phase delay of the filter, mu its axis and alpha is the
    birefringence angle.

    Parameters
    ----------

    N : int
        length of the filter

    mu : array_type (quaternion)
        birefringence axis quaternion array (should be of size N and of dtype quaternion).

    alpha : array_type
        birefringence angle array (should be of size N). If alpha is a float, then alpha is assumed constant throughout frequencies.

    phi : array_type or float
        phase delay array (should be of size N). If phi is a float, then a
        constant phase delay is assumed throughout frequencies.

    dt : float (optional)
        time sampling step (default 1)

    Attributes
    ----------

    N : int
        length of the filter

    f : array_type
        sampled frequencies

    dt : float
        time sampling step (default 1)

    mu, alpha, phi : array_types
        filter parameters
    '''
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
