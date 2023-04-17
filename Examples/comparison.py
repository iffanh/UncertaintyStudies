import matplotlib.pyplot as plt
import numpy as np

def pn1(x, u):
    return (u**3 - u**2 + 3*u/5 - 1/7) + (-45*u**2/30 + 45*u/35 - 45/126)*(x**2-1/3)

def pn2(x, u):
    return 0.555556*(u - 3/5)**3 + 0.4444444*u**3 + (1.666667*(u - 3/5)**3 - 1.666667*u**3)*(x**2-1/3)

def pn3(x, u):
    return 1.333333*(u - 1/4)**3 - 0.3333333*u**3 + (4*(u - 1/4)**3 - 4*u**3)*(x**2-1/3)

def f(x, u):
    return (u-x**2)**3

for i, u in enumerate(np.linspace(-1, 1, 10)):
    xs = np.linspace(-1.0, 1.0, 100)
    plt.figure()
    plt.plot(xs, f(xs, u), label="Original")
    plt.plot(xs, pn1(xs, u), label="Approx 1")
    plt.plot(xs, pn2(xs, u), label="Approx 2")
    plt.plot(xs, pn3(xs, u), label="Approx 2")
    plt.legend()
    plt.savefig(f"./Examples/comparison_{i}.png")