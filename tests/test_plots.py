import numpy as np
import matplotlib.pyplot as plt
import quaternion  # load the quaternion module
import bispy as bsp



def _create_data():
    N = 1024 # length of the signal

    # linear chirps constants
    a = 250*np.pi
    b = 50*np.pi
    c = 150*np.pi

    # time vector
    t = np.linspace(0, 1, N)

    # first chirp
    theta1 = np.pi/4 # constant orientation
    chi1 = np.pi/6-t # reversing ellipticity
    phi1 = b*t+a*t**2 # linear chirp

    # second chirp
    theta2 = np.pi/4*10*t # rotating orientation
    chi2 = 0 # constant null ellipticity
    phi2 = c*t+a*t**2 # linear chirp

    # common amplitude -- simply a window
    env = bsp.utils.windows.hanning(N)


    # define chirps x1 and x2
    x1 = bsp.signals.bivariateAMFM(env, theta1, chi1, phi1)
    x2 = bsp.signals.bivariateAMFM(env, theta2, chi2, phi2)

    # sum it
    x = x1 + x2

    return t, x

def test_plot2D():

    t, x = _create_data()
    fig, ax = bsp.utils.visual.plot2D(t, x)

def test_plot3D():
    t, x = _create_data()
    fig, ax = bsp.utils.visual.plot3D(t, x)
