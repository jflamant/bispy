#!/usr/bin/env python
#-*- coding: utf-8 -*-


"""
This module contains QFT routines

"""
import numpy as np
from .utils import sympSplit, sympSynth

__all__ = ['Qfft', 'iQfft', 'Qfftshift', 'iQfftshift', 'Qfftfreq']


# QFTTs functions


def Qfft(x, **kwargs):
    ''' Performs QFT using 2 ffts.

    Parameters
    ----------
    x : array_type

    Returns
    -------
    X : array_type
    '''
    x_1, x_2 = sympSplit(np.ascontiguousarray(x))  # ascontiguous may be needed

    X_1 = np.fft.fft(x_1, **kwargs)
    X_2 = np.fft.fft(x_2, **kwargs)

    X = sympSynth(X_1, X_2)

    return X


def iQfft(X, **kwargs):
    ''' Performs inverse QFT.

    Parameters
    ----------
    X : array_type

    Returns
    -------
    x : array_type
    '''

    X_1, X_2 = sympSplit(X)

    x_1 = np.fft.ifft(X_1, **kwargs)
    x_2 = np.fft.ifft(X_2, **kwargs)

    x = sympSynth(x_1, x_2)

    return x


# Qfft manipulations

def Qfftshift(X):
    ''' Shifts the QFT array

    Parameters
    ----------
    X : array_type

    Returns
    -------
    Xshifted : array_type

    '''

    return np.fft.fftshift(X)


def iQfftshift(X):
    ''' Unshifts the QFFT array

    Parameters
    ----------
    X : array_type

    Returns
    -------
    Xunshifted : array_type
    '''

    return np.fft.ifftshift(X)


def Qfftfreq(N, dt=1.0):
    ''' Return the sampled frequencies, from time spacing dt.

    See numpy.fft.fftfreq for further reference.

    Parameters
    ----------
    N : int
        length of the signal
    dt : float, optional
        time sampling step. Default 1.0

    Returns
    -------
    f : array_type
        sampled frequencies
    '''

    return np.fft.fftfreq(N, d=dt)
