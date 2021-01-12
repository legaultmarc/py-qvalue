import matplotlib.pyplot as plt
import numpy as np

from scipy.interpolate import UnivariateSpline


def plot_pi0(l, pi0s, spline):
    plt.scatter(l, pi0s, s=0.5)
    plt.plot(l, spline(l), label="cubic spline", color="black")
    plt.xlabel("$\lambda$")
    plt.ylabel("$\hat{\pi}_0(\lambda)$")
    plt.legend()
    plt.show()


def pi_0(ps, l=0.5):
    return np.sum(ps > l) / (ps.shape[0] * (1 - l))


def qvalue(ps, do_plot_pi0=False):
    # Implementation of qvalue as described in Storey, Tibshirani (2003) PNAS
    # https://www.pnas.org/content/pnas/100/16/9440.full.pdf

    idx = np.argsort(ps)
    ps = ps[idx]
    m = ps.shape[0]

    # Same range as original in paper.
    l = np.arange(0.01, 0.96, 0.01)
    pi0s = np.empty(l.shape[0])

    # Calculate pi hat for different values of lambda.
    for i, cur_l in enumerate(l):
        pi0s[i] = pi_0(ps, cur_l)

    # Fit spline.
    spline = UnivariateSpline(x=l, y=pi0s, k=3)

    if do_plot_pi0:
        plot_pi0(l, pi0s, spline)

    # Predicted pi0 when lambda -> 1
    pi0 = spline(1)

    qs = np.empty(m, dtype=float)
    qs[-1] = pi0 * ps[-1]

    for i in reversed(range(m - 1)):
        qs[i] = np.minimum(
            pi0 * m * ps[i] / (i + 1),
            qs[i + 1]
        )

    return qs[np.argsort(idx)]
