import numpy as np
import matplotlib.pyplot as plt
import quaternion  # load the quaternion module
import bispy as bsp

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

# plot 2D and 3D
fig, ax = bsp.utils.visual.plot2D(t, x)
fig, ax = bsp.utils.visual.plot3D(t, x)

S = bsp.timefrequency.QSTFT(x, t)

S.compute(window='hamming', nperseg=101, noverlap=100, nfft=N)
fig, ax = S.plotStokes()

S.extractRidges()

fig, ax = S.plotRidges(quivertdecim=30)

N = 1024 # length of the signal

# hyperbolic chirps parameters
alpha = 15*np.pi
beta = 5*np.pi
tup = 0.8 # set blow-up time value

t = np.linspace(0, 1, N) # time vector

# chirp 1 parameters
theta1 = -np.pi/3 # constant orientation
chi1 = np.pi/6 # constant ellipticity
phi1 = alpha/(.8-t) # hyperbolic chirp

# chirp 2 parameters
theta2 = 5*t # rotating orientation
chi2 = -np.pi/10 # constant ellipticity
phi2 = beta/(.8-t) # hyperbolic chirp

# envelope
env = np.zeros(N)
Nmin = int(0.1*N) # minimum value of N such that x is nonzero
Nmax = int(0.75*N) # maximum value of N such that x is nonzero

env[Nmin:Nmax] = bsp.utils.windows.hanning(Nmax-Nmin)

x1  = bsp.signals.bivariateAMFM(env, theta1, chi1, phi1)
x2  = bsp.signals.bivariateAMFM(env, theta2, chi2, phi2)

x = x1 + x2

fig, ax = bsp.utils.visual.plot2D(t, x)

waveletParams = dict(type='Morse', beta=12, gamma=3)
S = bsp.timefrequency.QCWT(x, t)

fmin = 0.01
fmax = 400
S.compute(fmin, fmax, waveletParams, N)

fig, ax = S.plotStokes()


S.extractRidges()

fig, ax = S.plotRidges(quivertdecim=40)