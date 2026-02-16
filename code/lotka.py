import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def lotka_simulation(alpha=2.0, beta=0.1, gamma=1.5, delta=0.075, x0=40, y0=9,
                     t_end=15, num_points=300):

    alpha = 2.0
    beta = 0.1
    gamma = 1.5
    delta = 0.075

    def f_lotkavolterra(X, t):
        x, y = X
        dxdt = alpha * x - beta * x * y
        dydt = -gamma * y + delta * x * y
        dXdt = [dxdt, dydt]
        return dXdt

    x0 = 40
    y0 = 9
    X0 = [x0, y0]
    t = np.linspace(0, 15, num_points)
    X = odeint(f_lotkavolterra, X0, t)

    return t, X

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", nargs='?', default=None,
                        help="If provided, save the plot to this file instead of displaying it.")
    parser.add_argument('--alpha', type=float, default=2.0, help="Prey growth rate")
    args = parser.parse_args()

    t, X = lotka_simulation(alpha=args.alpha)

    fig = plt.figure(figsize=(4, 3))
    ax = fig.add_subplot(111)
    ax.plot(t, X[:, 0], 'b', label='Prey (x)')
    ax.plot(t, X[:, 1], 'r', label='Predator (y)')
    ax.set_xlabel('Time')
    ax.set_ylabel('Population')
    ax.legend()
    plt.tight_layout()

    if args.filename is None:
        plt.show()
    else:
        fig.savefig(args.filename)
