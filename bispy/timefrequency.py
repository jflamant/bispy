# This file contains Time-Frequency tools for bivariate data. Each tool is
# coded as an object, with particular methods associated with it.

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


class QSTFT(object):
    ''' Compute the Quaternion-Short Term Fourier Transform for bivariate
    signals taken as (1, i)-quaternion valued signals.

    Parameters
    ----------
    t : array_type
        time samples array

    x : array_type
        input signal array (has to be of quaternion dtype)
    
    window : array_type
        window to use. Length of the window has to be odd. Windows can be
        generated using `utils.windows` methods

    spacing : int
        time index spacing between successive points

    Nfft : int, optional
        number of frequency bins. If 0, use size(x). Default is 0.

    tol : float, optional
        tolerance factor used in Stokes parameters normalization. Default
        is 1.0

    Attributes
    ----------
    t : array_type
        time samples array

    signal : array_type
        input signal array

    window : array_type
        window used

    spacing : int
        time index spacing between successive points

    Nfft : int
        number of frequency bins

    f : array_type
        sampled frequencies array

    QSTFT : array_type
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

    def __init__(self, t, x, window, spacing, Nfft=0, tol=0.01):

        if x.dtype != 'quaternion':
            raise ValueError('signal array should be of quaternion type.')

        # Store the signal and parameters
        self.t = t
        self.signal = x
        self.window = window
        self.spacing = spacing

        N = np.size(x, 0)

        if Nfft == 0:
            Nfft = nextpow2(N)
        elif Nfft < 0:
            raise ValueError('Nfft should be greater than 0.')
        else:
            Nfft = nextpow2(Nfft)

        self.Nfft = int(Nfft)
        self.f = np.fft.fftfreq(self.Nfft) / (t[1] - t[0])

        # Compute the Q-STFT
        sizewindow = np.size(window, 0)
        # check size of window is odd
        if sizewindow % 2 == 0:
            raise ValueError('Window size must me odd.')

        Lh = (sizewindow - 1) / 2  # half size index
        sampled_time = np.arange(0, N, spacing)
        P = np.size(sampled_time)
        S = np.zeros((Nfft, P), dtype='quaternion')

        print('Computing Q-STFT coefficients')
        for ti, ts in enumerate(sampled_time):

            taumin = - min([round(Nfft / 2) - 1, Lh, ts])
            taumax = min([round(Nfft / 2) - 1, Lh, N - ts - 1])
            tau = np.arange(taumin, taumax + 1)
            indices = ((Nfft + tau) % Nfft).astype(int)

            windowInd = window[(Lh + tau).astype(int)]
            windowIndq = utils.sympSynth(np.conj(windowInd) / np.linalg.norm(windowInd), 0)

            S[indices, ti] = x[(ts + tau).astype(int)] * windowIndq

            S[:, ti] = qfft.Qfft(S[:, ti])

        self.QSTFT = S

        # Compute the Time-Frequency Stokes parameters S0, S1, S2, S3
        print('Computing Time-Frequency Stokes parameters')

        self.S0 = np.norm(S)  # already squared norm with this definition

        # compute the j-involution + conjugation
        qqj = utils.StokesNorm(S)
        qqj_float = quaternion.as_float_array(qqj)

        self.S1 = qqj_float[..., 0]
        self.S2 = qqj_float[..., 1]
        self.S3 = - qqj_float[..., 3]

        # normalized Stokes parameters

        self.S1n, self.S2n, self.S3n = utils.normalizeStokes(self.S0, self.S1, self.S2, self.S3, tol=tol)

        # init ridges

        self.ridges = []

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

        # Extract ridges
        print('Extracting ridges')
        self.ridges = _extractRidges(self.S0[:self.Nfft // 2, :], parThresh, parMinD)


    def plotStokes(self, cmocean=False):
        ''' Plots S1n, S2n, S3n (normalized Stokes parameters) in time-frequency domain

        Parameters
        ----------
        cmocean : bool, optional
            activate use of cmocean colormaps

        Returns
        -------
        fig, ax : figure and axis handles
            may be needed to tweak the plot
        '''
        # check whether cmocean colormaps should be used
        if cmocean is True:
            import cmocean
            cmap = cmocean.cm.balance
        else:
            cmap = 'coolwarm'

        N = np.size(self.t)
        fig, ax = plt.subplots(ncols=3, figsize=(12, 5), sharey=True)

        im0 = ax[0].imshow(self.S1n[:self.Nfft // 2, :], origin='lower', interpolation='none', aspect='auto',cmap=cmap, extent=[self.t.min(), self.t.max(), 0, self.f[self.Nfft // 2-1]], vmin=-1, vmax=+1)

        im1 = ax[1].imshow(self.S2n[:self.Nfft // 2, :], origin='lower', interpolation='none', aspect='auto',cmap=cmap, extent=[self.t.min(), self.t.max(), 0, self.f[self.Nfft // 2-1]], vmin=-1, vmax=+1)
        im2 = ax[2].imshow(self.S3n[:self.Nfft // 2, :], origin='lower', interpolation='none', aspect='auto',cmap=cmap, extent=[self.t.min(), self.t.max(), 0, self.f[self.Nfft // 2-1]], vmin=-1, vmax=+1)

        # adjust figure
        fig.subplots_adjust(left=0.05, top=0.8, right=0.99, wspace=0.05)

        cbarax1 = fig.add_axes([0.369, 0.83, 0.303, 0.03])
        cbar1 = fig.colorbar(im1, cax=cbarax1, orientation='horizontal', ticks=[-1, 0, 1])
        cbar1.ax.set_xticklabels([-1, 0, 1])
        cbar1.ax.xaxis.set_ticks_position('top')

        ax[1].set_xlim([self.t.min(), self.t.max()])
        ax[2].set_xlim([self.t.min(), self.t.max()])

        ax[0].set_xlabel('Time [s]')

        ax[0].set_ylabel('Frequency [Hz]')

        ax[0].set_title(r'$S_1$', y=0.9, size=20)
        ax[1].set_title(r'$S_2$', y=0.9, size=20)
        ax[2].set_title(r'$S_3$', y=0.9, size=20)

        return fig, ax

    def plotRidges(self, quivertdecim=10, cmocean=False):
        ''' Plot S0, and the orientation and ellipticity recovered from the
        ridges in time-frequency domain

        If ridges are not extracted yet, it runs `extractRidges` method first.

        Parameters
        ----------
        quivertdecim : int, optional
            time-decimation index (allows faster and cleaner visualization of
            orientation vector field)
        cmocean : bool, optional
            activate use of cmocean colormaps

        Returns
        -------
        fig, ax : figure and axis handles
            may be needed to tweak the plot
        '''

        # check whether cmocean colormaps should be used
        if cmocean is True:
            import cmocean
            cmap_S0 = cmocean.cm.gray_r
            cmap_theta = cmocean.cm.phase
            cmap_chi = cmocean.cm.balance
        else:
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
        fig, ax = plt.subplots(ncols=3, figsize=(12, 5), sharey=True)
        im0 = ax[0].imshow(self.S0[:self.Nfft // 2, :], interpolation='none', origin='lower', aspect='auto',cmap=cmap_S0, extent=[self.t.min(), self.t.max(), 0, self.f[self.Nfft // 2-1]])


        im1 = ax[1].quiver(self.t[::int(self.spacing)][::quivertdecim], self.f[:int(self.Nfft / 2)], np.real(ori[:self.Nfft // 2, ::quivertdecim]), (np.imag(ori[:self.Nfft // 2, ::quivertdecim])), theta[:self.Nfft // 2, ::quivertdecim], clim=[-np.pi/2, np.pi/2], cmap=cmap_theta, headaxislength=0,headlength=0.001, pivot='middle',width=0.005, scale=15)

        for r in self.ridges:
            points = np.array([self.t[::self.spacing][r[1]], self.f[r[0]]]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            lc = LineCollection(segments, cmap=plt.get_cmap(cmap_chi),
                                norm=plt.Normalize(-np.pi / 4, np.pi / 4))
            lc.set_array(chi[(r[0], r[1])])
            lc.set_linewidth(5)
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

        ax[1].set_xlim([self.t.min(), self.t.max()])
        ax[2].set_xlim([self.t.min(), self.t.max()])

        ax[0].set_xlabel('Time [s]')

        ax[0].set_ylabel('Frequency [Hz]')

        ax[0].set_title('Time-Frequency energy density', y=1.14)

        ax[1].set_title('Instantaneous orientation', y=1.14)
        ax[2].set_title('Instantaneous ellipticity', y=1.14)

        return fig, ax




class QCWT(object):
    ''' Compute the Quaternion-Continuous Wavelet Transform for bivariate
    signals taken as (1, i)-quaternion valued signals.

    Parameters
    ----------
    t : array_type
        time samples array

    x : array_type
        input signal array (has to be of quaternion dtype)

    wave : string, optional
        wavelet to use. Default is `Morlet`. For now, only wavelet available.

    eta : float, optional
        central frequency of the mother wavelet. Default is 5.

    Nscale : int, optional
        number of scales to span. If 0, Nscale = N where N is the length of the
        signal. Default is 0.

    tol : float, optional
        tolerance factor used in Stokes parameters normalization. Default
        is 1.0

    Attributes
    ----------
    t : array_type
        time samples array

    signal : array_type
        input signal array

    eta : float
        central frequency of the mother wavelet

    Fs : float
        sampling frequency

    s : array_type
        sampled scales array

    f : array_type
        sampled frequencies array

    QCWT : array_type
        Q-CWT coefficients array

    S0, S1, S2, S3 : array_type
        Time-scale Stokes parameters, non-normalized [w.r.t. S0]

    S1n, S2n, S3n : array_type
        normalized time-scale Stokes parameters [w.r.t. S0] using the
        tolerance factor `tol`. See `utils.normalizeStokes`.

    ridges : list
        List of ridges index and values extracted from the time-scale
        energy density S0. Requires call of `extractRidges` for ridges to
        be added.
    '''

    def __init__(self, t, x, wave='Morlet', eta=5, Nscale=0, tol=0.01):
        ''' smin, smax are defined wrt :
        -- smin given by ratio of signal size and support of wavelet (8 for morlet)
        -- smax as eta/Fs
        '''
        # Store the signal and parameters
        self.t = t
        self.signal = x
        self.eta = eta
        self.Fs = 1 / (t[1] - t[0])

        if x.dtype != 'quaternion':
            raise ValueError('signal array should be of quaternion type.')

        if Nscale == 0: # if nothing is specified
            Nscale = np.size(x, 0)

        N = np.size(x, 0)

        # Define s values
        smax = N/self.Fs/8
        smin = eta/(np.pi*self.Fs)
        self.s = np.logspace(np.log(smax)/np.log(2), np.log(smin)/np.log(2), Nscale, base=2)  # ordering from low freqencie to large
        self.f = eta / (2*np.pi*self.s)

        # compute QCWT

        qfftx = qfft.Qfft(x)  # Precompute the QFT of signal sarray
        omega = 2 * np.pi * np.fft.fftfreq(N) * self.Fs
        S = np.zeros((Nscale, N), dtype='quaternion')

        print('Computing Q-CWT')
        for kscale in range(Nscale):

            if wave == 'Morlet':
                s = self.s[kscale]
                prefactor = (np.pi)**(-1/4)/(1+np.exp(-eta**2)-2*np.exp(-3/4*eta**2))**1/2*np.sqrt(s)

                Psi = prefactor*(np.exp(-0.5*(s*omega - eta)**2) - np.exp(-0.5*((s*omega)**2 + eta**2)))
            else:
                raise ValueError('Wavelet type not recognized. Only Morlet is implemented for now.')

            Psiq = utils.sympSynth(Psi, 0)

            S[kscale, :] = qfft.iQfft(qfftx * Psiq)

        self.QCWT = S

        # Compute the Time-Scale Stokes parameters S0, S1, S2, S3
        print('Computing Time-Scale Stokes parameters')
        self.S0 = np.norm(S)  # squared normed in this defitions

        # compute the j-involution + conjugation
        qqj = utils.StokesNorm(S)
        qqj_float = quaternion.as_float_array(qqj)

        self.S1 = qqj_float[..., 0]
        self.S2 = qqj_float[..., 1]
        self.S3 = - qqj_float[..., 3]

        # normalized Stokes parameters

        self.S1n, self.S2n, self.S3n = utils.normalizeStokes(self.S0, self.S1, self.S2, self.S3, tol=tol)

        # init ridges

        self.ridges = []

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


    def plotStokes(self, cmocean=False):
        ''' Plots S1n, S2n, S3n (normalized Stokes parameters) in time-scale domain

        Parameters
        ----------
        cmocean : bool, optional
            activate use of cmocean colormaps

        Returns
        -------
        fig, ax : figure and axis handles
            may be needed to tweak the plot
        '''

        # check whether cmocean colormaps should be used
        if cmocean is True:
            import cmocean
            cmap = cmocean.cm.balance
        else:
            cmap = 'coolwarm'

        N = np.size(self.t)
        fig, ax = plt.subplots(ncols=3, figsize=(12, 5), sharey=True)

        im0 = ax[0].imshow(self.S1n, origin='lower', interpolation='none', aspect='auto',cmap=cmap, extent=[self.t.min(), self.t.max(), -log2(self.s[0]), -log2(self.s[-1])], vmin=-1, vmax=+1)

        im1 = ax[1].imshow(self.S2n, origin='lower', interpolation='none', aspect='auto',cmap=cmap, extent=[self.t.min(), self.t.max(), -log2(self.s[0]), -log2(self.s[-1])], vmin=-1, vmax=+1)
        im2 = ax[2].imshow(self.S3n, origin='lower', interpolation='none', aspect='auto',cmap=cmap, extent=[self.t.min(), self.t.max(), -log2(self.s[0]), -log2(self.s[-1])], vmin=-1, vmax=+1)

        # adjust figure
        fig.subplots_adjust(left=0.05, top=0.8, right=0.99, wspace=0.05)

        cbarax1 = fig.add_axes([0.369, 0.83, 0.303, 0.03])
        cbar1 = fig.colorbar(im1, cax=cbarax1, orientation='horizontal', ticks=[-1, 0, 1])
        cbar1.ax.set_xticklabels([-1, 0, 1])
        cbar1.ax.xaxis.set_ticks_position('top')

        ax[1].set_xlim([self.t.min(), self.t.max()])
        ax[2].set_xlim([self.t.min(), self.t.max()])

        ax[0].set_xlabel('Time [s]')

        ax[0].set_ylabel(r'$-\log_2(s)$')

        ax[0].set_title(r'$S_1$', y=0.9, size=20)
        ax[1].set_title(r'$S_2$', y=0.9, size=20)
        ax[2].set_title(r'$S_3$', y=0.9, size=20)

        return fig, ax

    def plotRidges(self, quivertdecim=10, cmocean=False):

        ''' Plot S0, and the orientation and ellipticity recovered from the
        ridges in time-scale domain

        If ridges are not extracted yet, it runs `extractRidges` method first.

        Parameters
        ----------
        quivertdecim : int, optional
            time-decimation index (allows faster and cleaner visualization of
            orientation vector field)
        cmocean : bool, optional
            activate use of cmocean colormaps

        Returns
        -------
        fig, ax : figure and axis handles
            may be needed to tweak the plot
        '''
        # check whether cmocean colormaps should be used
        if cmocean is True:
            import cmocean
            cmap_S0 = cmocean.cm.gray_r
            cmap_theta = cmocean.cm.phase
            cmap_chi = cmocean.cm.balance
        else:
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
        im0 = ax[0].imshow(self.S0, interpolation='none', origin='lower', aspect='auto',cmap=cmap_S0, extent=[self.t.min(), self.t.max(), -log2(self.s[0]), -log2(self.s[-1])])


        im1 = ax[1].quiver(self.t[::quivertdecim], -log2(self.s), np.real(ori[:, ::quivertdecim]), (np.imag(ori[:, ::quivertdecim])), theta[:, ::quivertdecim], clim=[-np.pi/2, np.pi/2], cmap=cmap_theta, headaxislength=0,headlength=0.001, pivot='middle',width=0.005, scale=15)

        for r in self.ridges:
            points = np.array([self.t[r[1]], -log2(self.s[r[0]])]).T.reshape(-1, 1, 2)
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

        ax[1].set_xlim([self.t.min(), self.t.max()])
        ax[2].set_xlim([self.t.min(), self.t.max()])

        ax[0].set_xlabel('Time [s]')
        ax[0].set_ylim([-log2(self.s[0]), -log2(self.s[-1])])
        ax[0].set_ylabel(r'$-\log_2(s)$')

        ax[0].set_title('Time-scale energy density', y=1.14)
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
    return n
