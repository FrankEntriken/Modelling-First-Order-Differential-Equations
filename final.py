import pandas as pd
import numpy as np
import numba as nb
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


@nb.jit
def dxdt(y, z):
    """Computes the equation for x prime"""
    dx = -y - z
    return dx


@nb.jit
def dydt(x, y, a):
    """Computes the equation for y prime"""
    dy = x + a * y
    return dy


@nb.jit
def dzdt(x, z, b, c):
    """Computes the equation for z prime"""
    dz = b + z * (x - c)
    return dz


@nb.jit 
def solve_odes(c, T=500, dt=0.001, a=0.2, b=0.2):
    """Solves the first order differential equations using Runge-Kutta 4th Method"""
    t = np.arange(0, T, dt)
    x = np.zeros_like(t)
    y = np.zeros_like(t)
    z = np.zeros_like(t)

    for i in range(1, len(t)):
        x1 = dt * dxdt(y[i - 1], z[i - 1])
        y1 = dt * dydt(x[i - 1], y[i - 1], a)
        z1 = dt * dzdt(x[i - 1], z[i - 1], b, c)

        x2 = dt * dxdt(y[i - 1] + (y1 / 2), z[i - 1] + (z1 / 2))
        y2 = dt * dydt(x[i - 1] + (x1 / 2), y[i - 1] + (y1 / 2), a)
        z2 = dt * dzdt(x[i - 1] + (x1 / 2), z[i - 1] + (z1 / 2), b, c)

        x3 = dt * dxdt(y[i - 1] + (y2 / 2), z[i - 1] + (z2 / 2))
        y3 = dt * dydt(x[i - 1] + (x2 / 2), y[i - 1] + (y2 / 2), a)
        z3 = dt * dzdt(x[i - 1] + (x2 / 2), z[i - 1] + (z2 / 2), b, c)

        x4 = dt * dxdt(y[i - 1] + (y3), z[i - 1] + (z3))
        y4 = dt * dydt(x[i - 1] + (x3), y[i - 1] + (y3), a)
        z4 = dt * dzdt(x[i - 1] + (x3), z[i - 1] + (z3), b, c)

        x[i] = x[i - 1] + (x1 + 2 * x2 + 2 * x3 + x4) / 6
        y[i] = y[i - 1] + (y1 + 2 * y2 + 2 * y3 + y4) / 6
        z[i] = z[i - 1] + (z1 + 2 * z2 + 2 * z3 + z4) / 6

    return pd.DataFrame({"t": t, "x": x, "y": y, "z": z})

#TIME PLOTS
@nb.jit
def plotx(sol, T=500):
    """Plots x as a function of time"""
    t = sol["t"]
    x = sol["x"]

    plot = plt.figure(figsize=(12, 8))
    plt.plot(t, x, color="b")
    plt.title("x vs t")
    plt.xlabel("t")
    plt.ylabel("x")
    plt.xlim(0, T)
    plt.ylim(-12, 12)
    plt.legend
    plt.grid(True)
    plt.show()


@nb.jit
def ploty(sol, T=500):
    """Plots y as a function of time"""
    t = sol["t"]
    y = sol["y"]

    plot = plt.figure(figsize=(12, 8))
    plt.plot(t, y, color="b")
    plt.title("y vs t")
    plt.xlabel("t")
    plt.ylabel("y")
    plt.xlim(0, T)
    plt.ylim(-12, 12)
    plt.legend
    plt.grid(True)
    plt.show()


@nb.jit
def plotz(sol, T=500):
    """Plots z as a function of time"""
    t = sol["t"]
    z = sol["z"]

    plot = plt.figure(figsize=(12, 8))
    plt.plot(t, z, color="b")
    plt.title("z vs t")
    plt.xlabel("t")
    plt.ylabel("z")
    plt.xlim(0, T)
    plt.ylim(-12, 12)
    plt.legend
    plt.grid(True)
    plt.show()

#2D PLOTS
@nb.jit
def plotxy(sol, S=100):
    """Plots x and y in a 2D plot"""
    dt = 0.001
    N = int(S/dt)
    x = sol["x"][N:]
    y = sol["y"][N:]

    plot = plt.figure(figsize=(12, 8))
    plt.plot(x, y, color="b")
    plt.title("x vs y")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim(-12, 12)
    plt.ylim(-12, 12)
    plt.legend
    plt.grid(True)
    plt.show()

@nb.jit
def plotyz(sol, S=100):
    """Plots x and z in a 2D plot"""
    dt = 0.001
    N = int(S/dt)
    y = sol["y"][N:]
    z = sol["z"][N:]

    plot = plt.figure(figsize=(12, 8))
    plt.plot(y, z, color="b")
    plt.title("y vs z")
    plt.xlabel("y")
    plt.ylabel("z")
    plt.xlim(-12, 12)
    plt.ylim(-12, 12)
    plt.legend
    plt.grid(True)
    plt.show()

@nb.jit
def plotxz(sol, S=100):
    """Plots x and z in a 2D plot"""
    dt = 0.001
    N = int(S/dt)
    x = sol["x"][N:]
    z = sol["z"][N:]

    plot = plt.figure(figsize=(12, 8))
    plt.plot(x, z, color="b")
    plt.title("x vs z")
    plt.xlabel("x")
    plt.ylabel("z")
    plt.xlim(-12, 12)
    plt.ylim(-12, 12)
    plt.legend
    plt.grid(True)
    plt.show()

#3D PLOT
@nb.jit
def plotxyz(sol, S=100):
    """Plots x and y and z in a 3D plot"""
    dt = 0.001
    N = int(S/dt)
    x = sol["x"][N:]
    y = sol["y"][N:]
    z = sol["z"][N:]

    plot = plt.figure(figsize=(12, 8))
    ax = plot.gca(projection="3d")
    ax.plot(x, y, z, color="b")
    plt.title("x vs y vs z")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_xlim(-12, 12)
    ax.set_ylim(-12, 12)
    ax.set_zlim(0, 25)
    ax.legend
    ax.grid(True)
    plt.show()

#FIND MAX
@nb.jit
def findmaxima(x, S=100):
    """Finds the local maxima from the array of x values"""
    m = []
    for i in range(1+S, len(x)-S):

        if (x[i] > x[i-1] and x[i] > x[i+1]):
            m.append(x[i])

    return np.asarray(m)

#SCATTER
@nb.jit
def scatter(dc = 0.01):
    """Plots a scatter of all local maxima of x(t) as a function of c"""
    cs = np.arange(2, 6, dc)

    plot = plt.figure(figsize=(12, 8))
    plt.title("scatterplot of local maxima of x")
    plt.xlabel("c")
    plt.ylabel("local maxima")
    plt.xlim(2, 6)
    plt.ylim(3, 12)
    plt.grid(True)

    for c in cs:
        sol = solve_odes(c)
        x = sol["x"]
        ms = findmaxima(x)
        cy = np.zeros_like(ms)
        for i in range(1, len(cy)):
            cy[i] = c
        plt.scatter(cy, ms)

    plt.show()
