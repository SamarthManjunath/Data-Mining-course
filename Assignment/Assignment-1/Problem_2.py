#Name: Samarth Manjunath
#UTA ID: 1001522809

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from distutils.version import LooseVersion
from scipy.stats import norm
from sklearn.neighbors import KernelDensity
if LooseVersion(matplotlib.__version__) >= '2.1':
    dp = {'density': True}
else:
    dp = {'normed': True}

np.random.seed(1)
#here the value of N has been declared
N = 20
X = np.concatenate((np.random.normal(0, 1, int(0.3 * N)),
                    np.random.normal(5, 1, int(0.7 * N))))[:, np.newaxis]

#plot the graph as per the requirements
xpt = np.linspace(-5, 10, 1000)[:, np.newaxis]
bins = np.linspace(-5, 10, 10)

#Graph is plotted as per requirements
fig, ax = plt.subplots(2, 2, sharex=True, sharey=True)
fig.subplots_adjust(hspace=0.05, wspace=0.05)

ax[0, 0].hist(X[:, 0], bins=bins, fc='#AAAAFF', **dp)
ax[0, 0].text(-3.5, 0.31, "Required Histogram")

ax[0, 1].hist(X[:, 0], bins=bins + 0.75, fc='#AAAAFF', **dp)
ax[0, 1].text(-3.5, 0.31, "Histogram, bins shifted")

#kernel density is calculated here.
kde = KernelDensity(kernel='gaussian', bandwidth=0.75).fit(X)
log_dens = kde.score_samples(xpt)
ax[1, 1].fill(xpt[:, 0], np.exp(log_dens), fc='#AAAAFF')
ax[1, 1].text(-3.5, 0.31, "Plotted Gaussian Kernel Density")

#for loop to plot the graphs based on ravel()
for axi in ax.ravel():
    axi.plot(X[:, 0], np.full(X.shape[0], -0.01), '+k')
    axi.set_xlim(-4, 9)
    axi.set_ylim(-0.02, 0.34)

#For loop for normalized density
for axi in ax[:, 0]:
    axi.set_ylabel('Required Normalized Density')

for axi in ax[1, :]:
    axi.set_xlabel('x')

xpt = np.linspace(-6, 6, 1000)[:, None]
X_src = np.zeros((1, 1))

fig, ax = plt.subplots(2, 3, sharex=True, sharey=True)
fig.subplots_adjust(left=0.05, right=0.95, hspace=0.05, wspace=0.05)


def format_func(x, loc):
    if x == 0:
        return '0'
    elif x == 1:
        return 'h'
    elif x == -1:
        return '-h'
    else:
        return '%ih' % x

#for loop to calculate gaussian density
for i, kernel in enumerate(['gaussian']):
    axi = ax.ravel()[i]
    log_dens = KernelDensity(kernel=kernel).fit(X_src).score_samples(xpt)
    axi.fill(xpt[:, 0], np.exp(log_dens), '-k', fc='#AAAAFF')
    axi.text(-2.6, 0.95, kernel)

    axi.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
    axi.xaxis.set_major_locator(plt.MultipleLocator(1))
    axi.yaxis.set_major_locator(plt.NullLocator())

    axi.set_ylim(0, 1.05)
    axi.set_xlim(-2.9, 2.9)

ax[0, 1].set_title('Kernels which are available')#available kernels is calculated here.
N = 500 #N is 500 here and changes has to be made for different values of N
np.random.seed(1)
X = np.concatenate((np.random.normal(0, 1, int(0.3 * N)),
                    np.random.normal(5, 1, int(0.7 * N))))[:, np.newaxis]

xpt = np.linspace(-5, 10, 1000)[:, np.newaxis]

true_dens = (0.3 * norm(0, 1).pdf(xpt[:, 0])
             + 0.7 * norm(5, 1).pdf(xpt[:, 0]))

#graphs are plotted
fig, ax = plt.subplots()
ax.fill(xpt[:, 0], true_dens, fc='blue', alpha=0.2,
        label='input distribution')

#gaussian densities are calculated
for kernel in ['gaussian']:
    kde = KernelDensity(kernel=kernel, bandwidth=0.5).fit(X)#change the bandwidth values here
    log_dens = kde.score_samples(xpt)
    ax.plot(xpt[:, 0], np.exp(log_dens), '-',
            label="kernel = '{0}'".format(kernel))

ax.text(6, 0.38, "N={0} points".format(N))

ax.legend(loc='upper left')
ax.plot(X[:, 0], -0.005 - 0.01 * np.random.random(X.shape[0]), '+k')

ax.set_xlim(-4, 9)
ax.set_ylim(-0.02, 0.4)
#plot is shown here.
plt.show()