import numpy as np
import matplotlib.pyplot as plt
import math

x_des = np.arctan(np.linspace(0, 5, 350)-2.5)+2
y_des = np.linspace(0, 5, 350)

t = np.arange(0, 350)


def reference_velocity(x, y):
    n  = x.shape[0]
    dt = 5 / n
    xdot = np.zeros(n)
    ydot = np.zeros(n)
    for i in range(1, n):
        xdot[i] = (x[i] - x[i-1]) / dt
        ydot[i] = (y[i] - y[i-1]) / dt
    return xdot, ydot


def quadrotor_trajectory_calculator(k, x_des, y_des):
    n = x_des.shape[0]
    xdot_des, ydot_des = reference_velocity(x_des, y_des)

    x    = np.zeros(n)
    y    = np.zeros(n)
    xdot = np.zeros(xdot_des.shape[0])
    ydot = np.zeros(ydot_des.shape[0])

    x[0] = x_des[0]
    y[0] = y_des[0]
    # xdot[0]=xdot_des[0]
    # ydot[0]=ydot_des[0]

    # fix thrust c = 1
    c    = 1
    dt   = 5 / n

    for i in range(1, n):

        phi     = np.rad2deg(k[0] * (x[i-1] - x_des[i-1]) + k[1] * (xdot[i-1] - xdot_des[i-1]))
        theta   = np.rad2deg(k[0] * (y[i-1] - y_des[i-1]) + k[1] * (ydot[i-1] - ydot_des[i-1]))

        xddot   = c * (np.cos(phi) * np.sin(theta))
        yddot   = c * (np.sin(theta) * np.sin(phi))

        xdot[i] = xdot[i-1] + xddot * dt
        ydot[i] = ydot[i-1] + yddot * dt
        x[i]    = x[i-1] + xdot[i] * dt
        y[i]    = y[i-1] + ydot[i] * dt
    return x


k = np.array([[0.2], [0.5]])
x = quadrotor_trajectory_calculator(k, x_des, y_des)
plt.plot(t, x_des, c='r', label='desired')
plt.plot(t, x, c='b', label='actual')
# This has no relationship with variance. Sometimes the variance is small when the trajectory is very flat.
# But this has huge difference with the desired trajectory.
plt.legend()
plt.show()
