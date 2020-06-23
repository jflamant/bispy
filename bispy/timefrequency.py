#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 Julien Flamant
#
# Distributed under terms of the CeCILL Free Software Licence Agreement

'''
Module for the time-frequency analysis of bivariate signals.
'''
# import modules and packages
import numpy as np
import quaternion

import matplotlib.pyplot as plt
import matplotlib.colors as col
from mpl_toolkits.mplot3d import Axes3D  # required for 3D plot
from matplotlib.collections import LineCollection
import scipy.signal as sg

from . import qfft
from . import utils


class Hembedding(object):
    ''' H-embedding class. Computes quaternion embedding of complex-valued
    signals.

    Parameters
    ----------
    q : array_type
        quaternion input signal

    Attributes
    ----------
    signal : array_type
        original input signal

    Hembedding : array_type
        Quaternion-embedding of the input signal

    a : array_type
        instantaneous amplitude

    theta : array_type
        instantaneous orientation

    chi : array_type
        instantaneous ellipticity

    phi : array_type
        instantaneous phase
    '''

    def __init__(self, q):

        if q.dtype != 'quaternion':
            raise ValueError('array should be of quaternion type')

        self.signal = q

        # compute H-extension of the signal
        N = np.size(q, 0)
        Q = qfft.Qfft(q)

        # filter frequencies
        h = np.zeros((N, 4))
        h[0, 0] = 1
        h[N // 2, 0] = 1
        h[1: N // 2, 0] = 2
        hq = quaternion.as_quat_array(h)

        self.Hembedding = qfft.iQfft(Q * hq)

        # Compute Euler angle form
        a, theta, chi, phi = utils.quat2euler(self.Hembedding)

        self.a = a
        self.theta = theta
        self.chi = chi
        self.phi = phi

    # TODO: routine for plotting the extracted parameters?

class TFPrepresentation(object):

    def __init__(self, x, **kwargs):
        """
        Base time-frequency-polarization representation object.
        This is a low-level function, not meant to be used directly.
        """
        # check dimension of input array
        if x.ndim > 1:
            x = x.ravel()
        # check dtype of signal x and convert if necessary
        if x.dtype != 'quaternion':
            x = utils.sympSynth(x.real, x.imag)
        self.x = x
        N = x.shape[0]

        # timestamps of the signal x
        t = kwargs.get('t')
        if t is None:
            t = np.arange(N)
        self.t = t

        # # number of frequency bins
        # NFFT = kwargs.get('NFFT')
        # if NFFT is None:
        #     NFFT = nextpow2(N)
        # elif NFFT < 0:
        #     raise ValueError('Nfft should be greater than 0.')
        # else:
        #     NFFT = nextpow2(NFFT)
        # self.NFFT = NFFT
        # # sampled frequencies
        # self.f = np.fft.fftfreq(NFFT) / (t[1] - t[0])

        # sampled instants (spacing)
        # spacing = kwargs.get('spacing')
        # if spacing is None:
        #     spacing = 1
        #
        # self.sampled_index = np.arange(0, N, spacing)
        # self.sampled_time = t[::spacing]


        # init representation
        self.tfpr = None

        # init Stokes parameters
        self.S0 = None
        self.S1 = None
        self.S2 = None
        self.S3 = None

        self.S1n = None
        self.S2n = None
        self.S3n = None

    def normalizeStokes(self, tol=0.01):
        ''' Re-compute normalized Stokes parameters with a different normalization.

        Parameters
        ----------
        tol : float
            tolerance parameter. Default is 0.01.
        '''
        if self.S0 is not None:
            if np.sum(np.abs(self.S0)) > 0:
                self.S1n, self.S2n, self.S3n = utils.normalizeStokes(self.S0, self.S1, self.S2, self.S3, tol=tol)


    def plotSignal(self, kind='2D'):
        '''
        Plot the bivariate signal x.

        Parameters
        ----------
        kind : string, '2D' or '3D'
            type of plot. See `utils.visual`.
        '''
        if kind == '2D':
            fig, ax = utils.visual.plot2D(self.t, self.x)
        elif kind == '3D':
            fig, ax = utils.visual.plot3D(self.t, self.x)
        return fig, ax

    def _plotStokes(self, t, f, S0_cmap='viridis', s_cmap='coolwarm', single_sided=True, affine=False):
        ''' Time-frequency plot of time-frequency energy map (S0) and time-frequency polarization parameters (normalized Stokes parameters S1n, S2n, S3n)

        Parameters
        ----------
        t : array_type
            sampled times array
        f : array_type
            frequencies array (assuming unshifted)
        S0_cmap : colormap (sequential)
            to use for S0 time-frequency distribution
        s_cmap : colormap (diverging)
            to use for normalized Stokes time-frequency distribution

        Returns
        -------
        fig, ax : figure and axis handles
            may be needed to tweak the plot
        '''
        if affine:
            # size of plot
            A = np.random.rand(1, 4)
            w, h = plt.figaspect(A)
            labelsize= 20

            fig, ax = plt.subplots(ncols=4, figsize=(w, h), sharey=True, gridspec_kw = {'width_ratios':[1, 1, 1, 1]})

            im0 = ax[0].pcolormesh(t, f, self.S0, cmap=S0_cmap)
            im1 = ax[1].pcolormesh(t, f, self.S1n, cmap=s_cmap, vmin=-1, vmax=+1)
            im2 = ax[2].pcolormesh(t, f, self.S2n, cmap=s_cmap, vmin=-1, vmax=+1)
            im3 = ax[3].pcolormesh(t, f, self.S3n, cmap=s_cmap, vmin=-1, vmax=+1)

            # adjust figure
            cbarax1 = fig.add_axes([0.96, 0.12, 0.01, 0.8])
            cbar1 = fig.colorbar(im1, cax=cbarax1, orientation='vertical', ticks=[-1, 0, 1])
            #cbar1.ax.set_xticklabels([-1, 0, 1])
            #cbar1.ax.xaxis.set_ticks_position('top')

            label =[r'$S_0$', r'$s_1$', r'$s_2$', r'$s_3$']
            for i, axis in enumerate(ax):
                axis.set_xlabel('Time')
                axis.set_aspect(1./axis.get_data_ratio())
                axis.set_adjustable('box')
                axis.set_title(label[i], y = 0.85, size=labelsize)
                axis.set_xlim(self.t.min(), self.t.max())

            # set ylabls
            ax[0].set_ylabel('Frequency')
            fig.subplots_adjust(left=0.05, right=0.95, wspace=0.05, top=0.92, bottom=0.12)
            return fig, ax
        else:

            f = np.fft.fftshift(f)
            #tt, ff = np.meshgrid(t, np.fft.fftshift(f))
            # size of plot
            A = np.random.rand(1, 4)
            w, h = plt.figaspect(A)
            labelsize= 20

            fig, ax = plt.subplots(ncols=4, figsize=(w, h), sharey=True, gridspec_kw = {'width_ratios':[1, 1, 1, 1]})

            im0 = ax[0].imshow(np.fft.fftshift(self.S0, 0), cmap=S0_cmap, extent=[t.min(), t.max(), f.min(), f.max()], origin='lower')
            im1 = ax[1].imshow(np.fft.fftshift(self.S1n, axes=0), cmap=s_cmap, vmin=-1, vmax=+1, extent=[t.min(), t.max(), f.min(), f.max()], origin='lower')
            im2 = ax[2].imshow(np.fft.fftshift(self.S2n, axes=0), cmap=s_cmap, vmin=-1, vmax=+1, extent=[t.min(), t.max(), f.min(), f.max()], origin='lower')
            im3 = ax[3].imshow(np.fft.fftshift(self.S3n, axes=0), cmap=s_cmap, vmin=-1, vmax=+1, extent=[t.min(), t.max(), f.min(), f.max()], origin='lower')

            if single_sided is True:
                ax[0].set_ylim(0, f.max())
            # adjust figure
            cbarax1 = fig.add_axes([0.96, 0.12, 0.01, 0.8])
            cbar1 = fig.colorbar(im1, cax=cbarax1, orientation='vertical', ticks=[-1, 0, 1])
            #cbar1.ax.set_xticklabels([-1, 0, 1])
            #cbar1.ax.xaxis.set_ticks_position('top')

            label =[r'$S_0$', r'$s_1$', r'$s_2$', r'$s_3$']
            for i, axis in enumerate(ax):
                axis.set_xlabel('Time')
                axis.set_aspect(1./axis.get_data_ratio())
                axis.set_adjustable('box')
                axis.set_title(label[i], y = 0.85, size=labelsize)
                axis.set_xlim(self.t.min(), self.t.max())

            # set ylabls
            ax[0].set_ylabel('Frequency')
            fig.subplots_adjust(left=0.05, right=0.95, wspace=0.05, top=0.92, bottom=0.12)
            return fig, ax


class QSTFT(TFPrepresentation):
    ''' Compute the Quaternion-Short Term Fourier Transform for bivariate
    signals taken as (1, i)-quaternion valued signals.

    Parameters
    ----------
    x : array_type
        input signal array

    t : array_type (optional)
        time samples array. Default is t = np.arange(x.shape[0])

    Attributes
    ----------
    t : array_type
        time samples array

    x : array_type
        input signal array

    params : dict
        parameters used for the computation of the Q-STFT.

    sampled_time : array_type
        sampled times instants

    f : array_type
        sampled frequencies

    tfpr : array_type
        Q-STFT coefficients array

    S0, S1, S2, S3 : array_type
        Time-frequency Stokes parameters, non-normalized [w.r.t. S0]

    S1n, S2n, S3n : array_type
        normalized time-frequency Stokes parameters [w.r.t. S0] using the
        tolerance factor `tol`. See `utils.normalizeStokes`.

    ridges : list
        List of ridges index and values extracted from the time-frequency
        energy density S0. Requires call of `extractRidges` for ridges to
        be added.
    '''

    def __init__(self, x, t=None):
        # init main base object
        super(QSTFT, self).__init__(x=x, t=t)
        # init frequencies, sampled times and params directly
        self.f = None
        self.sampled_times = None
        self.params = None

        #init ridges
        self.ridges = []

    def compute(self, window='hamming', nperseg=128, noverlap=None, nfft=None,
    boundary='zeros', tol=0.01, ridges=False):
        '''
        Compute the Q-STFT of the signal x.

        It takes advantages of the scipy.signal.stft function for greater
        flexibility.

        Parameters
        ----------
        window, nperseg, noverlap, nfft, boundary : stft parameters
            See `scipy.signal.stft`
        tol : float, optional
            tolerance factor used in normalization of Stokes parameters.
            Default to 0.01
        ridges: bool, optional
            If True, compute also the ridges of the transform.
            Default to `False`. Ridges can be later computed using
            `extractRidges()`.
        '''
        # parameters
        self.params = dict(fs = 1./(self.t[1]-self.t[0]),
                    window=window,
                    nperseg=nperseg,
                    noverlap=noverlap,
                    nfft=nfft,
                    boundary=boundary,
                    return_onesided=False,
                    detrend=False,
                    padded=True)

        # split x = x_1 + i x_2
        x1, x2 = utils.sympSplit(self.x)

        # Compute the Q-STFT using scipy.signal.stft on x1, x2
        f, sampled_times, temp1 = sg.stft(x1, **self.params )
        _, _ , temp2 = sg.stft(x2, **self.params)

        # update Attributes
        self.f = f
        self.sampled_times = sampled_times
        # recombine
        self.tfpr = utils.sympSynth(temp1, temp2)

        # old version
        # sizewindow = np.size(self.window, 0)
        # # check size of window is odd
        # if sizewindow % 2 == 0:
        #     raise ValueError('Window size must me odd.')
        #
        # Lh = (sizewindow - 1) // 2  # half size index
        # N = self.x.shape[0]
        # print('Computing Q-STFT coefficients')
        # temp = np.zeros_like(self.tfpr)
        # for ti, ts in enumerate(self.sampled_index):
        #
        #     taumin = - min([round(self.NFFT / 2) - 1, Lh, ts])
        #     taumax = min([round(self.NFFT / 2) - 1, Lh, N - ts - 1])
        #     tau = np.arange(taumin, taumax + 1)
        #     indices = ((self.NFFT + tau) % self.NFFT).astype(int)
        #
        #     windowInd = self.window[(Lh + tau).astype(int)]
        #     windowIndq = utils.sympSynth(np.conj(windowInd) / np.linalg.norm(windowInd), 0)
        #
        #     temp[indices, ti] = self.x[(ts + tau).astype(int)] * windowIndq
        #
        # temp = qfft.Qfft(temp, axis=0)
        #self.tfpr = temp

        # Compute the Time-Frequency Stokes parameters S0, S1, S2, S3
        print('Computing Time-Frequency Stokes parameters')

        self.S0 = np.norm(self.tfpr)  # already squared norm with this definition

        # compute the j-involution + conjugation
        qjq = utils.StokesNorm(self.tfpr)
        qjq_float = quaternion.as_float_array(qjq)

        self.S1 = qjq_float[..., 2]
        self.S2 = qjq_float[..., 3]
        self.S3 = qjq_float[..., 1]

        # normalized Stokes parameters
        self.S1n, self.S2n, self.S3n = utils.normalizeStokes(self.S0, self.S1, self.S2, self.S3, tol=tol)

        if ridges is True:
            self.extractRidges()

    def inverse(self, mask=None):
        '''Compute inverse Q-STFT

            Parameters
            ----------
            mask: array_type
                mask applied to Q-STFT coefficients prior to inversion.
                If mask=None, no mask is employed.
        '''

        # construct dict for inversion
        inversion_dict =  dict(fs = self.params['fs'],
                            window=self.params['window'],
                            nperseg=self.params['nperseg'],
                            noverlap=self.params['noverlap'],
                            nfft=self.params['nfft'],
                            boundary=self.params['boundary'],
                            input_onesided=False)

        if mask is None:
            mask = np.ones(self.S0.shape, dtype=bool)

        tfp1, tfp2 = utils.sympSplit(self.tfpr*mask)
        t, x1 = sg.istft(tfp1, **inversion_dict)
        __, x2 = sg.istft(tfp2, **inversion_dict)

        xr = utils.sympSynth(x1, x2)
        return t, xr

    def extractRidges(self, parThresh=4, parMinD=3):
        ''' Extracts ridges from the time-frequency energy density S0.

        Parameters
        ----------
        parThresh : float, optional
            Controls the threshold at which local maxima of S0 are accepted or
            rejected. Larger values of `parThresh`increase the number of
            eligible points.

        parMinD : float, optional
            Ridge smoothness parameter. Controls at which maximal distance
            can be located two eligible same ridge points. The smaller
            `parMinD`is the smoother ridges are.

        Returns
        -------
        ridges : list
            list of detected ridges
        '''
        nfft = self.params['nfft']
        # Extract ridges
        print('Extracting ridges')
        self.ridges = _extractRidges(self.S0[:nfft//2, :], parThresh, parMinD)

    def plotRidges(self, quivertdecim=10):
        ''' Plot S0, the orientation and ellipticity recovered from the
        ridges in time-frequency domain

        If ridges are not extracted yet, it runs `extractRidges` method first.

        Parameters
        ----------
        quivertdecim : int, optional
            time-decimation index (allows faster and cleaner visualization of
            orientation vector field)

        Returns
        -------
        fig, ax : figure and axis handles
            may be needed to tweak the plot
        '''

        #  default colormaps
        cmap_S0 = 'Greys'
        cmap_theta = 'hsv'
        cmap_chi = 'coolwarm'

        # check whether ridges have been computed

        if len(self.ridges) == 0:
            print('No ridges detected, computing ridges.')
            self.extractRidges()

        # create ridge mask
        maskRidge = np.zeros(self.S0.shape, dtype=bool)
        for r in self.ridges:
            maskRidge[r[0], r[1]] = True

        # Compute orientation and ellipticity values

        S1mask = np.ma.masked_where(maskRidge == False, self.S1n)
        S2mask = np.ma.masked_where(maskRidge == False, self.S2n)
        S3mask = np.ma.masked_where(maskRidge == False, self.S3n)

        theta = .5 * np.arctan2(S2mask, S1mask)
        ori = np.exp(1j * theta)

        chi = 0.5 * np.arcsin(S3mask)

        N = np.size(self.t)

        # prepare meshgrid
        tt, ff = np.meshgrid(self.sampled_times, np.fft.fftshift(self.f))
        # size of plot
        A = np.random.rand(1, 3)
        w, h = plt.figaspect(A)
        labelsize= 20

        fig, ax = plt.subplots(ncols=3, figsize=(w, h), sharey=True, gridspec_kw = {'width_ratios':[1, 1, 1]})

        #im0 = ax[0].imshow(np.fft.fftshift(self.S0, 0), cmap=cmap_S0,origin='lower', extent=[self.sampled_times.min(), self.sampled_times.max(), 0, self.f.max()], aspect='auto')
        im0 = ax[0].imshow(np.fft.fftshift(self.S0, 0), cmap=cmap_S0, extent=[self.sampled_times.min(), self.sampled_times.max(), self.f.min(), self.f.max()], origin='lower')

        im1 = ax[1].quiver(self.sampled_times[::quivertdecim], self.f, np.real(ori[:, ::quivertdecim]), (np.imag(ori[:, ::quivertdecim])), theta[:, ::quivertdecim], clim=[-np.pi/2, np.pi/2], cmap=cmap_theta, headaxislength=0,headlength=0.001, pivot='middle',width=0.005, scale=15)

        for r in self.ridges:
            points = np.array([self.sampled_times[r[1]], self.f[r[0]]]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            lc = LineCollection(segments, cmap=plt.get_cmap(cmap_chi),
                                norm=plt.Normalize(-np.pi / 4, np.pi / 4))
            lc.set_array(chi[(r[0], r[1])])
            lc.set_linewidth(5)
            im2 = ax[2].add_collection(lc)

        #im2 = ax[2].imshow(chi, vmin=-np.pi/4, vmax=np.pi/4,  interpolation='none', origin='lower', aspect='auto', cmap='coolwarm', extent=[self.t.min(), self.t.max(), 0, self.f[N/2-1]])

        # adjust figure
        fig.subplots_adjust(left=0.05, top=0.8, right=0.99, wspace=0.05)

        for i, axis in enumerate(ax):
            axis.set_xlabel('Time')
            axis.set_ylim([0, self.f.max()])
            axis.set_xlim(self.t.min(), self.t.max())
            axis.set_aspect(1./axis.get_data_ratio())
            axis.set_adjustable('box')


        cbarax0 = fig.add_axes([0.09, 0.83, 0.224, 0.03])
        cbar0 = fig.colorbar(im0, cax=cbarax0, orientation='horizontal', ticks=[0, np.max(self.S0)])
        cbar0.ax.set_xticklabels(['', ''])
        cbar0.ax.xaxis.set_ticks_position('top')

        cbarax1 = fig.add_axes([0.185+0.224, 0.83, 0.224, 0.03])
        cbar1 = fig.colorbar(im1, cax=cbarax1, orientation='horizontal', ticks=[-np.pi/2, 0, np.pi/2])
        cbar1.ax.set_xticklabels([r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$'])
        cbar1.ax.xaxis.set_ticks_position('top')

        cbarax2 = fig.add_axes([0.725, 0.83, 0.224, 0.03])
        cbar2 = fig.colorbar(im2, cax=cbarax2, ticks=[-np.pi/4, 0, np.pi/4], orientation='horizontal')
        cbar2.ax.set_xticklabels([r'$-\frac{\pi}{4}$', r'$0$', r'$\frac{\pi}{4}$'])
        cbar2.ax.xaxis.set_ticks_position('top')


        ax[0].set_ylabel('Frequency [Hz]')
        ax[0].set_title('Time-Frequency energy density', y=1.14)
        ax[1].set_title('Instantaneous orientation', y=1.14)
        ax[2].set_title('Instantaneous ellipticity', y=1.14)

        return fig, ax

    def plotStokes(self, S0_cmap='viridis', s_cmap='coolwarm', single_sided=True):
        ''' Time-frequency plot of time-frequency energy map (S0) and time-frequency polarization parameters (normalized Stokes parameters S1n, S2n, S3n)

        Parameters
        ----------
        S0_cmap : colormap (sequential)
            to use for S0 time-frequency distribution
        s_cmap : colormap (diverging)
            to use for normalized Stokes time-frequency distribution

        Returns
        -------
        fig, ax : figure and axis handles
            may be needed to tweak the plot
        '''

        return self._plotStokes(self.sampled_times, self.f, S0_cmap=S0_cmap, s_cmap=s_cmap, single_sided=single_sided)


#!---- Quaternion Continuous Wavelet Transform --------------------------!#
class QCWT(TFPrepresentation):
    def __init__(self, x, t=None):
        ''' Compute the Quaternion-Continuous Wavelet Transform for bivariate
        signals taken as (1, i)-quaternion valued signals.

        Parameters
        ----------
        x : array_type
            input signal array

        t : array_type (optional)
            time samples array. Default is t = np.arange(x.shape[0])

        Attributes
        ----------
        t : array_type
            time samples array

        x : array_type
            input signal array

        params : dict
            parameters used for the computation of the Q-CWT.

        sampled_frequencies : array_type
            sampled frequencies

        tfpr : array_type
            Q-STFT coefficients array

        S0, S1, S2, S3 : array_type
            Time-frequency Stokes parameters, non-normalized [w.r.t. S0]

        S1n, S2n, S3n : array_type
            normalized time-frequency Stokes parameters [w.r.t. S0] using the
            tolerance factor `tol`. See `utils.normalizeStokes`.

        ridges : list
            List of ridges index and values extracted from the time-frequency
            energy density S0. Requires call of `extractRidges` for ridges to
            be added.
        '''

        # init main base object
        super(QCWT, self).__init__(x=x, t=t)
        # init frequencies, scales and params directly
        self.sampled_frequencies = None
        self.params = None

        #init ridges
        self.ridges = []

    def _getWavelet(self, Nscales, **waveletParams):
        # construct wavelet array (len(x), Nscales)
        N = self.x.shape[0]
        W = np.zeros((Nscales, N), dtype='quaternion')
        Fs = 1./(self.t[1]-self.t[0])
        f = np.fft.fftfreq(N, d=1./Fs)

        wType = waveletParams['type']

        if wType not in ['Morse', 'Morlet']:
            raise ValueError("Unknown value for wavelet type %s, must be one of: "
            "{'Morse', 'Morlet'}" % wType)

        if wType == 'Morse':

            if waveletParams.get('beta') is None:
                beta = 3
            else:
                beta = waveletParams['beta']

            if waveletParams.get('gamma') is None:
                gamma = 1
            else:
                gamma = waveletParams['gamma']
            if waveletParams.get('norm') is None:
                mode = 'bandpass'
            else:
                mode = waveletParams['norm']

            fc = (beta/gamma)**(1./gamma) # central frequency of Morse wavelets

            for fi, fsampled in enumerate(self.sampled_frequencies):
                fnorm = f[:N//2]*fc/(fsampled)
                temp = fnorm**beta*np.exp(-fnorm**gamma)
                norm = self._getNormalization(wType, beta, gamma, mode=mode, Fs=Fs, fc=fc, fsampled=fsampled)
                W[fi, :N//2] = utils.sympSynth(temp*norm, 0)

        elif wType == 'Morlet':
            if waveletParams.get('eta') is None:
                eta = 2*np.pi
            else:
                eta = waveletParams.get('eta')

            for fi, fsampled in enumerate(self.sampled_frequencies):
                s = eta /(2*np.pi*fsampled) # scale

                prefactor = (np.pi)**(-1/4)/(1+np.exp(-eta**2)-2*np.exp(-3/4*eta**2))**1/2*np.sqrt(s)

                temp = prefactor*(np.exp(-0.5*(s*2*np.pi*f - eta)**2) - np.exp(-0.5*((s*2*np.pi*f)**2 + eta**2)))
                W[fi, :] = utils.sympSynth(temp, 0)

        return W


    def _getNormalization(self, wType, beta=None, gamma=None, Fs=None, fc=None, fsampled=None, mode='bandpass'):
        if wType == 'Morse':
            '''
            See e.g.,
            Lilly, Jonathan M., and Sofia C. Olhede. 2009. “Higher-Order Properties of Analytic Wavelets.” IEEE Transactions on Signal Processing 57 (1): 146–60. doi:10.1109/TSP.2008.2007607.

            Olhede, Sofia C., and Andrew T. Walden. 2002. “Generalized Morse Wavelets.” IEEE Transactions on Signal Processing 50 (11): 2661–70. doi:10.1109/TSP.2002.804066.
            '''
            import scipy.special as sp

            if mode == 'bandpass':
                a = 2*(np.exp(1)*gamma/beta)**(beta/gamma)
            elif mode == 'energy':
                r = (2*beta+1)/gamma
                a = (gamma*(2**r)/sp.gamma(r))**(0.5)*np.sqrt(Fs*fc/(fsampled))
        return a

    def compute(self, fmin, fmax, waveletParams, Nscales=50, tol=0.01, ridges=False):
        ''' Compute the Q-CWT of x using a specified wavelet.

            Parameters
            ----------
            fmin, fmax : float
                min and max frequencies
            waveletParams : dict
                dictionary containing wavelet features. Currently 2 types,
                'Morlet' and 'Morse' are supported.
            Nscales : int
                number of scales to analyze. Controls the size of the
                sampled_frequencies array.
            tol : float, optional
                tolerance factor used in normalization of Stokes parameters.
                Default to 0.01
            ridges: bool, optional
                If True, compute also the ridges of the transform.
                Default to `False`. Ridges can be later computed using
                `extractRidges()`.

        '''

        #deine sampled frequencies from fmin and fmax
        Fs = self.t[1]-self.t[0]
        N = self.x.shape[0]
        self.sampled_frequencies = np.logspace(np.log10(fmin), np.log10(fmax), Nscales) # note to self: allow other choices as well?

        W =self._getWavelet(Nscales=Nscales, **waveletParams)

        X = qfft.Qfft(self.x)  # Precompute the QFT of signal sarray
        temp = np.zeros((Nscales, N), dtype='quaternion')
        for k in range(Nscales):
            temp[k, :] = qfft.iQfft(X * W[k, :])

        self.tfpr = temp
        self.params = dict(fmin=fmin, fmax=fmax, wavelets = W)

        # Compute the Time-Frequency Stokes parameters S0, S1, S2, S3
        print('Computing Time-Frequency Stokes parameters')

        self.S0 = np.norm(self.tfpr)  # already squared norm with this definition

        # compute the j-involution + conjugation
        qjq = utils.StokesNorm(self.tfpr)
        qjq_float = quaternion.as_float_array(qjq)

        self.S1 = qjq_float[..., 2]
        self.S2 = qjq_float[..., 3]
        self.S3 = qjq_float[..., 1]

        # normalized Stokes parameters
        self.S1n, self.S2n, self.S3n = utils.normalizeStokes(self.S0, self.S1, self.S2, self.S3, tol=tol)

        if ridges is True:
            self.extractRidges()

    def extractRidges(self, parThresh=4, parMinD=3):
        ''' Extracts ridges from the time-scale energy density S0.

        Parameters
        ----------
        parThresh : float, optional
            Controls the threshold at which local maxima of S0 are accepted or
            rejected. Larger values of `parThresh`increase the number of
            eligible points.

        parMinD : float, optional
            Ridge smoothness parameter. Controls at which maximal distance
            can be located two eligible same ridge points. The smaller
            `parMinD`is the smoother ridges are.

        Returns
        -------
        ridges : list
            list of detected ridges
        '''

        print('Extracting ridges')
        self.ridges = _extractRidges(self.S0, parThresh, parMinD)


    def plotStokes(self, S0_cmap='viridis', s_cmap='coolwarm'):
        ''' Time-frequency plot of time-frequency energy map (S0) and time-frequency polarization parameters (normalized Stokes parameters S1n, S2n, S3n)

        Parameters
        ----------
        S0_cmap : colormap (sequential)
            to use for S0 time-frequency distribution
        s_cmap : colormap (diverging)
            to use for normalized Stokes time-frequency distribution

        Returns
        -------
        fig, ax : figure and axis handles
            may be needed to tweak the plot
        '''

        return self._plotStokes(self.t, self.sampled_frequencies, S0_cmap=S0_cmap, s_cmap=s_cmap, affine=True)

    def plotRidges(self, quivertdecim=10):

        ''' Plot S0, and the orientation and ellipticity recovered from the
        ridges in time-scale domain

        If ridges are not extracted yet, it runs `extractRidges` method first.

        Parameters
        ----------
        quivertdecim : int, optional
            time-decimation index (allows faster and cleaner visualization of
            orientation vector field)

        Returns
        -------
        fig, ax : figure and axis handles
            may be needed to tweak the plot
        '''
        #  default colormaps
        cmap_S0 = 'Greys'
        cmap_theta = 'hsv'
        cmap_chi = 'coolwarm'

        # check whether ridges have been computed

        if len(self.ridges) == 0:
            print('No ridges detected, computing ridges.')
            self.extractRidges()

        # create ridge mask
        maskRidge = np.zeros(self.S0.shape, dtype=bool)
        for r in self.ridges:
            maskRidge[r[0], r[1]] = True

        # Compute orientation and ellipticity values

        S1mask = np.ma.masked_where(maskRidge == False, self.S1n)
        S2mask = np.ma.masked_where(maskRidge == False, self.S2n)
        S3mask = np.ma.masked_where(maskRidge == False, self.S3n)

        theta = .5*np.arctan2(S2mask, S1mask)
        ori = np.exp(1j * theta)

        chi = 0.5 * np.arcsin(S3mask)

        N = np.size(self.t)
        fig, ax = plt.subplots(ncols=3, figsize=(12, 5), sharey=True)
        im0 = ax[0].pcolormesh(self.t, self.sampled_frequencies, self.S0, cmap=cmap_S0)

        im1 = ax[1].quiver(self.t[::quivertdecim], self.sampled_frequencies, np.real(ori[:, ::quivertdecim]), (np.imag(ori[:, ::quivertdecim])), theta[:, ::quivertdecim], clim=[-np.pi/2, np.pi/2], cmap=cmap_theta, headaxislength=0,headlength=0.001, pivot='middle',width=0.005, scale=15)

        for r in self.ridges:
            points = np.array([self.t[r[1]], self.sampled_frequencies[r[0]]]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            lc = LineCollection(segments, cmap=plt.get_cmap(cmap_chi),
                                norm=plt.Normalize(-np.pi / 4, np.pi / 4))
            lc.set_array(chi[(r[0], r[1])])
            lc.set_linewidth(3)
            im2 = ax[2].add_collection(lc)

        #im2 = ax[2].imshow(chi, vmin=-np.pi/4, vmax=np.pi/4,  interpolation='none', origin='lower', aspect='auto', cmap='coolwarm', extent=[self.t.min(), self.t.max(), 0, self.f[N/2-1]])

        # adjust figure
        fig.subplots_adjust(left=0.05, top=0.8, right=0.99, wspace=0.05)

        cbarax0 = fig.add_axes([0.05, 0.83, 0.303, 0.03])
        cbar0 = fig.colorbar(im0, cax=cbarax0, orientation='horizontal', ticks=[0, np.max(self.S0)])
        cbar0.ax.set_xticklabels(['', ''])
        cbar0.ax.xaxis.set_ticks_position('top')

        cbarax1 = fig.add_axes([0.369, 0.83, 0.303, 0.03])
        cbar1 = fig.colorbar(im1, cax=cbarax1, orientation='horizontal', ticks=[-np.pi/2, 0, np.pi/2])
        cbar1.ax.set_xticklabels([r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$'])
        cbar1.ax.xaxis.set_ticks_position('top')

        cbarax2 = fig.add_axes([0.686, 0.83, 0.303, 0.03])
        cbar2 = fig.colorbar(im2, cax=cbarax2, ticks=[-np.pi/4, 0, np.pi/4], orientation='horizontal')
        cbar2.ax.set_xticklabels([r'$-\frac{\pi}{4}$', r'$0$', r'$\frac{\pi}{4}$'])
        cbar2.ax.xaxis.set_ticks_position('top')

        for i, axis in enumerate(ax):
            axis.set_xlim([self.t.min(), self.t.max()])
            axis.set_ylim([self.sampled_frequencies.min(), self.sampled_frequencies.max()])

        ax[0].set_xlabel('Time [s]')
        ax[0].set_ylabel('Frequency [Hz]')

        ax[0].set_title('Time-frequency energy density', y=1.14)
        ax[1].set_title('Instantaneous orientation', y=1.14)
        ax[2].set_title('Instantaneous ellipticity', y=1.14)

        return fig, ax

#
# Low-level functions
#


def _extractRidges(density, parThresh, parMinD):

    A, B = density.shape  # A: len of frequency axis, B len of time axis

    # find all local maximas

    locMax = np.zeros((A, B), dtype=bool)
    thresh = np.max(density) / parThresh
    for ind in range(B):

        detectmax = sg.argrelextrema(density[:, ind], np.greater)[0]
        ismaxOK = density[detectmax, ind] > thresh
        locMax[detectmax, ind] = True * ismaxOK

    # chain the ridges
    ridges = []

    currentRidget = []
    currentRidgef = []

    while np.any(locMax):

        freqMask, timeMask = np.where(locMax)

        currentRidget.append(timeMask[0])
        currentRidgef.append(freqMask[0])

        locMax[freqMask[0], timeMask[0]] = False

        freqMask, timeMask = np.where(locMax)

        FLAG = False  # Avoid undifined FLAG if condition is false.
        if len(timeMask) > 1:
            FLAG = True
        while FLAG:

            distances = np.sqrt((timeMask-currentRidget[-1])**2 + (freqMask-currentRidgef[-1])**2)

            minD = np.where(distances == distances.min())[0][0]
            if (distances[minD] < parMinD) and (len(timeMask) > 1):
                currentRidget.append(timeMask[minD])
                currentRidgef.append(freqMask[minD])
                locMax[freqMask[minD], timeMask[minD]] = False
                freqMask, timeMask = np.where(locMax)
            else:
                FLAG = False
                if len(timeMask) == 1:
                    currentRidget.append(timeMask[0])
                    currentRidgef.append(freqMask[0])
                ridges.append((currentRidgef, currentRidget))
                currentRidget = []
                currentRidgef = []
                print('Ridge added')

    print(str(len(ridges)) + ' ridges were recovered.')

    return ridges


def log2(x):
    return np.log(x) / np.log(2)


def nextpow2(i):
    n = 1
    while n < i: n *= 2
    return int(n)
