#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This program contains utility tools.
"""

import numpy as np
import quaternion

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # required for 3D plot
import matplotlib.gridspec as gridspec # required for 2D plot

__all__ = ['sympSplit', 'sympSynth', 'StokesNorm', 'normalizeStokes',
    'Stokes2geo', 'geo2Stokes', 'quat2euler', 'euler2quat']


def sympSplit(q):

    '''Splits a quaternion array into two complex arrays.

    The decomposition reads::

        q = q_1 + i q_2

    where q_1, q_2 are complex (1, 1j) numpy arrays

    Parameters
    ----------
    q : quaternion numpy array

    Returns
    -------
    q_1, q_2 : complex numpy arrays

    See also
    --------
    sympSynth

    Examples
    --------
    >>> q
    array([[quaternion(0.3, 0.47, -0.86, -0.42),
            quaternion(0.24, -1.07, -2.11, 0.37),
            quaternion(-0.24, -1.36, -1.14, 1.69)],
           [quaternion(0.4, -0.61, 0.04, -0.03),
            quaternion(-1.58, -1.69, -1.18, -1.02),
            quaternion(0.78, -1.06, -1.05, -0.62)]], dtype=quaternion)
    >>> q_1, q_2 = sympSplit(q)
    >>> q_1
    array([[ 0.30-0.86j,  0.24-2.11j, -0.24-1.14j],
       [ 0.40+0.04j, -1.58-1.18j,  0.78-1.05j]])
    >>> q_2
    array([[ 0.47-0.42j, -1.07+0.37j, -1.36+1.69j],
       [-0.61-0.03j, -1.69-1.02j, -1.06-0.62j]])
    '''

    if q.dtype != 'quaternion':
        raise ValueError('array should be of quaternion type')

    qfloat = quaternion.as_float_array(q)
    q_1 = qfloat[..., 0] + 1j * qfloat[..., 2]
    q_2 = qfloat[..., 1] + 1j * qfloat[..., 3]

    return q_1, q_2


def sympSynth(q_1, q_2):

    '''Constructs a quaternion array from two complex arrays.

    The decomposition reads::

        q = q_1 + i q_2

    where q_1, q_2 are complex (1, 1j) numpy arrays

    Parameters
    ----------
    q_1, q_2 : complex numpy arrays

    Returns
    -------
    q : quaternion numpy array

    See also
    --------
    sympSplit

    Examples
    --------
    >>> q_1
    array([[ 0.30-0.86j,  0.24-2.11j, -0.24-1.14j],
       [ 0.40+0.04j, -1.58-1.18j,  0.78-1.05j]])
    >>> q_2
    array([[ 0.47-0.42j, -1.07+0.37j, -1.36+1.69j],
       [-0.61-0.03j, -1.69-1.02j, -1.06-0.62j]])
    >>> sympSynth(q_1 q_2)
    array([[quaternion(0.3, 0.47, -0.86, -0.42),
            quaternion(0.24, -1.07, -2.11, 0.37),
            quaternion(-0.24, -1.36, -1.14, 1.69)],
           [quaternion(0.4, -0.61, 0.04, -0.03),
            quaternion(-1.58, -1.69, -1.18, -1.02),
            quaternion(0.78, -1.06, -1.05, -0.62)]], dtype=quaternion)

    '''
    # construct correct dimension of float array (shape(q_1), 4)
    dimArray = list(q_1.shape)
    dimArray.append(4)

    qfloat = np.zeros(tuple(dimArray))

    qfloat[..., 0] = np.real(q_1)
    qfloat[..., 1] = np.real(q_2)
    qfloat[..., 2] = np.imag(q_1)
    qfloat[..., 3] = np.imag(q_2)

    return quaternion.as_quat_array(qfloat)


''' Stokes related functions, and geometric parameters extraction '''


def StokesNorm(q):
    ''' Return the Stokes-Poincaré norm of a quaternion.

    The Stokes-Poincaré norm is defined by::

        StokesNorm(q) = -q*j*np.conj(q)

    with j = quaternion(0, 0, 1, 0).

    Parameters
    ----------
    q : quaternion numpy array

    Returns
    -------
    q*j*np.conj(q) : Stokes-Poincaré norm of q

    See also
    --------
    quat2euler

    '''

    if q.dtype != 'quaternion':
        raise ValueError('array should be of quaternion type')

    # compute j-product
    jq = quaternion.y * np.conj(q)

    return q * jq


def normalizeStokes(S0, S1, S2, S3, tol=0.0):
    ''' Normalize Stokes parameters S1, S2, S3  by S0.

    Normalization can be performed using a soft thresholding-like method, if
    regularization is needed::

        Si = Si/(S0 + tol*np.max(S0))

    where i = 1, 2, 3 and `tol` is the tolerance factor. This function assumes
    that the maximum value of S0 has a significance for the whole indices of
    the Si arrays.

    Parameters
    ----------
    S0, S1, S2, S3 : array_type
    tol : float, optional

    Returns
    -------
    S1n, S2n, S3n : array_type

    See also
    --------
    quat2euler
    '''

    epsilon = tol * np.max(S0)  # soft thresholding

    S1n = S1 / (S0 + epsilon)
    S2n = S2 / (S0 + epsilon)
    S3n = S3 / (S0 + epsilon)

    return S1n, S2n, S3n


def Stokes2geo(S0, S1, S2, S3, tol=0.0):
    ''' Return geometric parameters from Stokes parameters.

    It returns the decomposition in a, theta, chi and degree of polarization
    Phi.

    Parameters
    ----------
    S0, S1, S2, S3 : array_type
    tol : float, optional

    Returns
    -------

    a, theta, chi, Phi : array_type

    See also
    --------
    quat2euler
    normalizeStokes
    geo2Stokes

    '''

    # normalize
    S1n, S2n, S3n = normalizeStokes(S0, S1, S2, S3, tol=tol)
    Phi = np.sqrt(S1n**2 + S2n**2 + S3n**2)

    # estimate geometrical paramaters

    a = np.sqrt(Phi*S0)
    theta = 0.5 * np.arctan2(S2n, S1n)
    chi = 0.5 * np.arcsin(S3n/Phi)

    return a, theta, chi, Phi


def geo2Stokes(a, theta, chi, Phi=1):
    '''
    Compute Stokes parameters from geometric parameters.

    Parameters
    ----------
    a, theta, chi : array_type
    Phi : array_type, optional

    Returns
    -------
    S0, S1, S2, S3 : array_type

    See also
    --------
    quat2euler
    Stokes2geo

    '''

    S0 = np.abs(a)**2
    S1 = np.abs(a)**2 * Phi * np.cos(2 * theta) * np.cos(2 * chi)
    S2 = np.abs(a)**2 * Phi * np.sin(2 * theta) * np.cos(2 * chi)
    S3 = np.abs(a)**2 * Phi * np.sin(2 * chi)

    return S0, S1, S2, S3


def quat2euler(q):
    '''Euler polar form of a quaternion array.

    The decomposition reads::

        q = a * np.exp(i * theta) * np.exp(-k * chi) * np.exp(j * phi)

    with a > 0, -pi/2 < theta < pi/2, -pi/4 < chi < pi/4 and -pi < phi < pi .

    Parameters
    ----------
    q : quaternion numpy array

    Returns
    -------
    a, theta, chi, phi : array_type

    See also
    --------
    euler2quat

    '''

    S0 = np.norm(q)  # squared modulus

    qjq = StokesNorm(q)
    qjq_float = quaternion.as_float_array(qjq)
    S1 = qjq_float[..., 2]
    S2 = qjq_float[..., 3]
    S3 = qjq_float[..., 1]

    a, theta, chi, Phi = Stokes2geo(S0, S1, S2, S3)

    qi = quaternion.x
    qk = quaternion.z

    prefactor = a * np.exp(qi * theta) * np.exp(-qk * chi)
    expjphi = quaternion.as_float_array(prefactor**(-1) * q)

    expjphi_cplx = expjphi[..., 0] + 1j * expjphi[..., 2]

    phi = np.angle(expjphi_cplx)

    return a, theta, chi, phi


def euler2quat(a, theta, chi, phi):
    ''' Quaternion from Euler polar form.

    The decomposition reads::

        q = a * np.exp(i * theta) * np.exp(-k * chi) * np.exp(j * phi)

    with a > 0, -pi/2 < theta < pi/2, -pi/4 < chi < pi/4 and -pi < phi < pi .

    Parameters
    ----------
    a, theta, chi, phi : array_type

    Returns
    -------
    q : quaternion numpy array

    See also
    --------
    quat2euler

    '''

    qi = quaternion.x
    qj = quaternion.y
    qk = quaternion.z

    q = a * np.exp(qi * theta) * np.exp(-qk * chi) * np.exp(qj * phi)

    return q


''' Windows related functions '''


class windows(object):

    ''' Windows functions static methods.

    These window functions are provided for convenience, and are meant to be
    used with the QSTFT class.
    '''

    def __init__(self):
        pass

    @staticmethod
    def rectangle(N):
        ''' Rectangle window'''
        window = np.ones(N)
        return window

    @staticmethod
    def hamming(N):
        ''' Hamming window'''
        window = 0.54 - 0.46 * np.cos(2.0 * np.pi * np.arange(1, N + 1) / (N + 1))
        return window

    @staticmethod
    def hanning(N):
        ''' Hanning window'''
        window = 0.50 - 0.50 * np.cos(2.0 * np.pi * np.arange(1,
            N + 1) / (N + 1))
        return window

    @staticmethod
    def gaussian(N, sigma=0.005):
        '''Gaussian window'''
        if sigma > 0.5:
            raise ValueError('Sigma must be smaller than 0.5')
        else:
            window = np.exp(np.log(sigma) * np.linspace(-1, 1, N)**2)

        return window

def polarizationEllipse(theta, chi, a=1, N=128):

    '''Returns the trace of the polarization ellipse given its orientation and ellipticity.

    Parameters
    ----------
    theta : float
        Orientation of the ellipse, must be between -pi/2 and pi/2

    chi : float
        Ellipticity. It defines the shape of the ellipse, must be between -pi/4 and pi/4

    a : float, optional
        Scale parameter. Default is 1.

    N : int, optional
        Length of the complex trace. Default is 128.

    Returns
    -------

    phi : array_type
        Curvilinear absciss of the polarization ellipe

    ell : array_type
        Complex trace of the polarization ellipse.
    '''

    phi = np.linspace(0, 2*np.pi, N)

    ell = a*np.exp(1j*theta)*(np.cos(chi)*np.cos(phi)+1j*np.sin(chi)*np.sin(phi))

    return phi, ell


class visual(object):
    '''
    Static methods for visualization of bivariate signals.
    '''

    def __init__(self):
        pass

    @staticmethod
    def plot2D(t, q, labels=['u(t)', 'v(t)']):
        ''' 2D plot of a bivariate signal.

        Plots the 2D trace, and time evolution of each component.

        Parameters
        ----------
        t, q : array_type
            time and signal arrays (signal array may be either complex or quaternion type)
        labels : [label1, label2]
            list of labels to display.

        Returns
        -------
        fig, ax : figure and axis handles
        '''

        fig = plt.figure(figsize=(10, 4))

        N = np.size(t)
        q1, q2 = sympSplit(q)

        gs = gridspec.GridSpec(2, 5)
        gs.update(hspace=0.1, wspace=0.1, bottom=0.18, left=0.09, top=0.95, right=0.94)


        ax1 = plt.subplot(gs[0, 2:])
        ax2 = plt.subplot(gs[1, 2:])
        ax3 = plt.subplot(gs[:, :2])

        # ax1
        ax1.spines['top'].set_visible(False)
        ax1.spines['left'].set_visible(False)
        ax1.spines['bottom'].set_visible(False)
        ax1.set_xticks([])
        ax1.yaxis.set_ticks_position('right')
        ax1.spines['right'].set_position(('outward', 10))
        #ax2
        ax2.spines['top'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax2.yaxis.set_ticks_position('right')
        ax2.spines['right'].set_position(('outward', 10))
        ax2.spines['bottom'].set_position(('outward', 10))

        #ax3
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        ax3.spines['left'].set_position(('outward', 10))
        ax3.spines['bottom'].set_position(('outward', 10))

        # plots
        ax1.plot(t, q1.real)
        ax2.plot(t, q2.real)
        ax3.plot(q1.real, q2.real)

        ax3.set_aspect('equal', 'box')
        # get limits
        lims = ax3.get_xlim() + ax3.get_ylim()
        li = np.max(np.abs(lims))
        #set lims
        for ax in [ax1, ax2, ax3]:
            ax.set_ylim([-li, li])
        ax3.set_xlim([-li, li])
        ax3.set_xlabel(labels[0])
        ax3.set_ylabel(labels[1])

        ax1.set_title(labels[0])
        ax2.set_title(labels[1])
        ax2.set_xlabel('time')
        return fig, [ax1, ax2, ax3]

    @staticmethod
    def plot3D(t, q):
        ''' 3D plot of a bivariate signal

        Parameters
        ----------
        t, q : array_type
            time and signal arrays (signal array may be either complex or quaternion type)

        Returns
        -------
        fig, ax : figure and axis handles

        '''

        if q.dtype == 'quaternion':
            u, v = sympSplit(q)
            x = u.real + 1j * v.real
        else:
            x = q  # complex array

        if len(q.shape) > 2:
            raise ValueError('Data should be a vector to be 3D plotted')

        fig = plt.figure()
        ax_sig = fig.add_subplot(projection = '3d')
        # ax_sig
        ax_sig.plot(t, np.real(x), np.imag(x), color='k')

        tmin = ax_sig.get_xlim3d()[0]
        tmax = ax_sig.get_xlim3d()[1]
        xmin = min(ax_sig.get_ylim3d()[0], ax_sig.get_zlim3d()[0])
        xmax = max(ax_sig.get_ylim3d()[1], ax_sig.get_zlim3d()[1])
        ymin = min(ax_sig.get_ylim3d()[0], ax_sig.get_zlim3d()[0])
        ymax = max(ax_sig.get_ylim3d()[1], ax_sig.get_zlim3d()[1])

        # surfaces

        # complex plane
        xx_c, yy_c = np.meshgrid(np.linspace(xmin, xmax), np.linspace(ymin, ymax))
        #ax_sig.plot_surface(-.05*(tmin+tmax), xx_c, yy_c,  alpha=0.05, color='gray', rstride = 100, cstride=100)
        ax_sig.plot(x.real, x.imag, -.05*(tmin+tmax), zdir='x', color='gray')
        ax_sig.set_xlim([-.05*(tmin+tmax), tmax])

        # real proj
        xx_r, yy_r = np.meshgrid(np.linspace(tmin, tmax), np.linspace(xmin, xmax))
        #ax_sig.plot_surface(xx_r, yy_r, 1.05*ymin, alpha=0.05, color='gray', rstride = 100, cstride=100)
        ax_sig.plot(t, x.real, ymin*1.05, zdir='z', color='gray')
        ax_sig.set_zlim([1.05*ymin, ymax])

        #imaginary proj
        xx_i, yy_i = np.meshgrid(np.linspace(tmin, tmax), np.linspace(ymin, ymax))
        #ax_sig.plot_surface(xx_i, 1.05*xmax, yy_i,  alpha=0.05, color='gray',rstride = 100, cstride=100)
        ax_sig.plot(t, x.imag, 1.05*xmax, zdir='y', color='gray')
        ax_sig.set_ylim([xmin, 1.05*xmax])

        # replot to avoid 'overlays'
        ax_sig.plot(t, np.real(x), np.imag(x), color='k')
        #proj3d.persp_transformation = _orthogonal_proj
        fig.show()
        return fig, ax_sig


# workaround orthographic projection (deprecated)
# from mpl_toolkits.mplot3d import proj3d

# def _orthogonal_proj(zfront, zback):
#     a = (zfront+zback)/(zfront-zback)
#     b = -2*(zfront*zback)/(zfront-zback)
#     # -0.0001 added for numerical stability as suggested in:
#     # http://stackoverflow.com/questions/23840756
#     return np.array([[1,0,0,0],
#                         [0,1,0,0],
#                         [0,0,a,b],
#                         [0,0,-0.0001,zback]])
