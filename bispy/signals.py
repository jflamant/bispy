#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module for the synthesis of prototype bivariate signals
"""
import numpy as np
import quaternion

from .utils import euler2quat, sympSynth, sympSplit
from .filters import HermitianFilter
from .spectral import quaternionPSD
from .timefrequency import _hembedding


def _resolve_rng(rng):
    ''' Source of random numbers for the `rng` arguments.

    None uses numpy's global random state (``np.random``), as in earlier versions,
    so results seeded with ``np.random.seed`` do not change. A RandomState is used
    as is. Anything else (int seed, SeedSequence, Generator) goes through
    ``np.random.default_rng``.
    '''
    if rng is None:
        return np.random
    if isinstance(rng, np.random.RandomState):
        return rng
    return np.random.default_rng(rng)

class stationaryBivariate(HermitianFilter):
    '''
    Simulates realizations of a stationary Gaussian random bivariate sequence with specficied quaternion PSD.

    The simulation method relies on spectral synthesis and is approximate.
    The quality of the approximation increases with the size N (aka the number of frequency bins).


    Parameters
    ----------
    targetPSD: quaternionPSD object
        target PSD of the signal to sample from

    Attributes
    ----------
    simulation : array_type
        array of size (M, N) where M is the number of independent realizations of the signal and N is the length of the simulated sequence.

    '''

    def __init__(self, targetPSD):
        # check input is a quaternionPSD object
        if isinstance(targetPSD, quaternionPSD) is False:
            raise ValueError("target PSD should be a quaternionPSD object")

        N = targetPSD.N
        dt = targetPSD.dt

        # filter parameters
        mu = targetPSD.mu

        eta = np.zeros_like(targetPSD.Phi)
        valid = targetPSD.Phi > 0
        eta[valid] = (1-np.sqrt(1-targetPSD.Phi[valid]**2))/targetPSD.Phi[valid]
        K = np.sqrt(targetPSD.S0/(1+eta**2))

        super(stationaryBivariate, self).__init__(N, K, eta, mu, dt=dt)

        self.simulation = None

    def simulate(self, M, rng=None):
        '''
        Simulate realizations of the stationary Gaussian random bivariate signal with specified quaternion PSD.

        Parameters
        ----------
        M : int
            number of independent realizations to simulate

        rng : None, int, numpy.random.Generator or numpy.random.RandomState, optional
            source of randomness, see `bivariatewhiteNoise`. A seed initializes
            one generator shared by the M realizations.

        '''
        if rng is not None:
            rng = _resolve_rng(rng)  # one generator for all realizations
        self.simulation = np.zeros((M, self.N), dtype='quaternion')
        for m in range(M):
            w = bivariatewhiteNoise(self.N, 1, rng=rng)
            self.simulation[m, :] =  self.output(w)




def bivariateAMFM(a, theta, chi, phi, Hembedding=True, complexOutput=False):

    ''' Construct a bivariate AM-FM model with specified parameters.

     The output x[n] is constructed as::

        x[n] = a[n] * np.exp(i * theta[n]) * np.exp(-k * chi[n]) * np.exp(j * phi[n])

    Parameters
    ----------
    a, theta, chi, phi : array_type
        These are instantaneous geometrical and phase parameters.

    Hembedding : bool, optional
        If `True`, returns the H-embedding signal of x, otherwise returns x
        (1, i)-complex (as a quaternion array). Default is `True`.

    complexOutput : bool, optional
        If `True`, output is a complex numpy array. Otherwise output is a
        quaternion numpy array. Default is `False`.

    Returns
    -------
    x : array_type

    See also
    --------
    euler2quat

    '''

    # N = np.size(phi)

    # if np.size(theta) != N or np.size(chi) != N or np.size(phi) != N:
    #     raise ValueError('All parameters should have same length!')

    x = euler2quat(a, theta, chi, phi)

    if Hembedding is True:
        return x
    else:
        x1, x2 = sympSplit(x)
        if complexOutput is True:
            return x1.real + 1j * x2.real
        else:
            return sympSynth(x1.real, x2.real)


def bivariatewhiteNoise(N, S0, P=0, theta=0, complexOutput=False, Hembedding=False, rng=None):

    ''' Generates a bivariate white noise with prescribed polarization
        properties using the Unpolarized/Polarized part decomposition.

        Parameters
        ----------
        N : int
            length of the signal
        S0 : float
            white noise power
        P : float, optional
            degree of polarization, must be 0 <= P <= 1. Default is 0
        theta : float, optional
            angle of linear polarization. Default is 0
        complexOutput: bool, optional
            If `True`, output is a complex numpy array. Otherwise output is a
            quaternion numpy array. Default is `False`.
        Hembedding : bool, optional
            If `True`, output is the H-embedding of the noise (quaternion
            analytic signal), as a quaternion numpy array. Cannot be combined
            with `complexOutput`. Default is `False`.
        rng : None, int, numpy.random.Generator or numpy.random.RandomState, optional
            Source of randomness. With `None` (default), numpy's global random
            state is used, so that ``np.random.seed`` gives the same results as
            in earlier versions. An int seeds a new ``numpy.random.Generator``.

        Returns
        -------
        w : array_type
            bivariate white noise signal
    '''
    # check value of P
    if (0 <= P <= 1) is False:
        raise ValueError('Degree of polarization P must be between 0 and 1 !')
    if complexOutput and Hembedding:
        raise ValueError('complexOutput and Hembedding cannot both be True')

    rng = _resolve_rng(rng)

    #  unpolarized part
    wu = 1 / np.sqrt(2) * (rng.standard_normal(N) + 1j * rng.standard_normal(N))

    #  polarized part
    wp = rng.standard_normal(N)

    #  use of the UP decomposition to construct the output
    w = (S0)**0.5 * (np.sqrt(1 - P) * wu + np.sqrt(P) *
     np.exp(1j * theta) * wp)

    if complexOutput is True:
        return w
    wq = sympSynth(w.real, w.imag)
    if Hembedding is True:
        return _hembedding(wq)
    return wq
