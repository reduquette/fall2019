import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from scipy.stats import norm

sns.set_style('white')
sns.set_context('talk')
np.random.seed(123)


def mcmc_mh(data,
            mu_initial=1,
            num_samples=10,
            proposal_width=1,
            mu_prior_mu=0,
            mu_prior_sd=1,
            plot=False,
            one_axis=False):
    # initial guess
    mu_current = mu_initial
    posterior = []
    if plot:
        fig = plt.figure(figsize=(4, 8))
        tick_spacing = (1, .1)  #(x,y)
    for i in range(num_samples):
        # proposed updated guess -- drawn from a normal around mu_current
        mu_proposal = norm(mu_current, proposal_width).rvs()
        # is proposal a better guess than mu?
        likelihood_current = norm(mu_current, 1).pdf(data).prod()
        likelihood_proposal = norm(mu_proposal, 1).pdf(data).prod()

        # not sure where mu_prior_mu comes from
        prior_current = norm(mu_prior_mu, mu_prior_sd).pdf(mu_current)
        prior_proposal = norm(mu_prior_mu, mu_prior_sd).pdf(mu_proposal)

        p_current = likelihood_current * prior_current
        p_proposal = likelihood_proposal * prior_proposal

        p_accept = p_proposal / p_current

        accept = np.random.rand() < p_accept

        if accept:
            # Update position
            mu_current = mu_proposal
        posterior.append(mu_current)
        if plot:
            if not one_axis or i == 0:
                if one_axis:
                    ax = plt.gca()
                else:
                    ax = plt.subplot(
                        num_samples, 1, i + 1, label="Iteration" + str(i + 1))
                    ax.set_title("Iteration" + str(i + 1))
                ax.xaxis.set_major_locator(
                    ticker.MultipleLocator(tick_spacing[0]))
                ax.yaxis.set_major_locator(
                    ticker.MultipleLocator(tick_spacing[1]))
                sns.distplot(data, ax=ax)
            x = np.linspace(-5, 5, 100)
            y = norm(loc=mu_proposal, scale=1).pdf(x)
            plt.plot(x, y, label="Iteration" + str(i))

    if plot:
        plt.tight_layout()
        if one_axis:
            plt.legend()
        plt.savefig("yikes.png")
        plt.show()
    return posterior


# ax = plt.subplot()
# sns.distplot(data, kde=False, ax=ax)
# _ = ax.set(
#     title='Histogram of observed data', xlabel='x', ylabel='# observations')
# plt.show()


def main():
    data = np.random.randn(20)
    posterior = mcmc_mh(data, num_samples=1000)
    fig = plt.figure()
    ax = plt.subplot(2, 1, 1)
    sns.distplot(posterior[:], ax=ax, label='estimated posterior')
    ax.set(xlabel='mu', ylabel='belief')
    ax.legend()
    ax2 = plt.subplot(2, 1, 2)
    ax2.plot(np.linspace(1, 1000, 1000), posterior)
    plt.show()


main()
