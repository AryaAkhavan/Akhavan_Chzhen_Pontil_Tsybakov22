"""Summary
"""
import numpy as np
from helpers import softmax, mirror_projection, norm_dual


class BlackBox:
    def __init__(self, estimator, objective,
                 online=False, verbose=1, sigma=0,
                 noise_family='Gaussian'):
        """Black box with mirror descent optimizer

        Parameters
        ----------
        estimator : Oracle type
            Gradient estimator with estimate method
        objective : function
            returns value
        online : bool, optional
            Unused for now, but for future when we will make online exps
        verbose : int, optional
            Level of verbose
        """

        self.estimator = estimator
        self.objective = objective
        self.verbose = verbose
        self.online = online
        self.sigma = sigma
        self.noisy = False
        if sigma > 0:
            self.noisy = True
        self.noise_family = noise_family


    def noise(self, t):
        if self.noise_family == 'Gaussian':
            return self.sigma * np.random.randn(1)
        if self.noise_family == 'Bernoulli':
            return self.sigma * (2 * np.random.binomial(1, .5) - 1)
        else:
            return 0


    def eval(self, x, t, for_report=False):
        """Method that queries the black box function

        Parameters
        ----------
        x : ndarray
            current point
        t : int
            iteration number, for online extension

        Returns
        -------
        float
            query output
        """
        if for_report:
            if self.online:
                return self.objective(x, t)
            else:
                return self.objective(x)

        if self.online:
            return self.objective(x, t) + self.noise(t)
        else:
            return self.objective(x) + self.noise(t)


    def set_step_size(self, radius, grad_norm_sum):
        if grad_norm_sum == 0:
            return 1
        return min(1, radius / np.sqrt(grad_norm_sum))


    def optimize(self, max_iter,
                 constr_type, radius=1., norm_str_conv=2):
        """Optimize implements mirror descent

        Parameters
        ----------
        max_iter : int
            number of iteration of MD
        constr_type : mixed
            type of constraints which determines mirror projection
        radius : float, optional
            radius measured in prox function
        norm_str_conv : int, optional
            strong convexity norm

        Returns
        -------
        list
            values of function along projection
        """

        dim = self.estimator.dim
        grad_norm_sum = 0
        cumul_grad = np.zeros(dim)
        report = []
        x_final_unnormalized = np.zeros(dim)
        for t in range(max_iter):
            x_new = mirror_projection(cumul_grad, constr_type)
            x_final_unnormalized += x_new
            if self.verbose >= 2:
                print(f"Iteration number: {t+1}\n",
                      f"Estimate vector: {x_final_unnormalized/(t+1)}")
            report.append(self.eval(x_final_unnormalized/(t+1), t+1, True))
            if self.verbose >= 2:
                print(f"Current: {x_final_unnormalized/(t+1)}")
            grad, norm_dual = self.estimator.estimate(x_new, t+1,
                                                      self.eval, self.noisy)
            grad_norm_sum += norm_dual
            eta = self.set_step_size(radius, grad_norm_sum)
            cumul_grad -= eta * grad
        if self.verbose >= 3:
            print(f"Estimate: {x_final_unnormalized / max_iter}")
        return report
