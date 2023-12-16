#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Analytic Acquisition Functions that evaluate the posterior without performing
Monte-Carlo sampling.
"""

from __future__ import annotations

import math

from abc import ABC

from contextlib import nullcontext
from copy import deepcopy

from typing import Dict, Optional, Tuple, Union

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.objective import PosteriorTransform
from botorch.exceptions import UnsupportedError
from botorch.models.gp_regression import FixedNoiseGP
from botorch.models.gpytorch import GPyTorchModel
from botorch.models.model import Model
from botorch.utils.constants import get_constants_like
from botorch.utils.probability import MVNXPB
from botorch.utils.probability.utils import (
    log_ndtr as log_Phi,
    log_phi,
    log_prob_normal_in,
    ndtr as Phi,
    phi,
)
from botorch.utils.safe_math import log1mexp, logmeanexp
from botorch.utils.transforms import convert_to_target_pre_hook, t_batch_mode_transform
from torch import Tensor
from torch.nn.functional import pad

_sqrt_2pi = math.sqrt(2 * math.pi)
# the following two numbers are needed for _log_ei_helper
_neg_inv_sqrt2 = -(2**-0.5)
_log_sqrt_pi_div_2 = math.log(math.pi / 2) / 2


from botorch.acquisition import AnalyticAcquisitionFunction
from botorch.models.model import Model
from typing import Any, Optional, Union
from torch import Tensor
from botorch.acquisition.objective import PosteriorTransform
from botorch.utils.transforms import convert_to_target_pre_hook, t_batch_mode_transform
# from botorch.acquisition import *
from abc import ABC
from typing import Dict, Optional, Tuple, Union
import botorch.acquisition as temp
import botorch.acquisition 
from botorch.utils.probability.utils import log_phi, log_ndtr,ndtr as Phi, phi

# def _gamma(
#     mean: Tensor, sigma: Tensor, fstar: Tensor, maximize: bool
# ) -> Tensor:
#     """Returns `u = (mean - best_f) / sigma`, -u if maximize == True."""
#     u = (fstar - mean) / sigma
#     return u if maximize else -u



class AnalyticAcquisitionFunction(AcquisitionFunction, ABC):
    r"""
    Base class for analytic acquisition functions.

    :meta private:
    """

    def __init__(
        self,
        model: Model,
        posterior_transform: Optional[PosteriorTransform] = None,
        **kwargs,
    ) -> None:
        r"""Base constructor for analytic acquisition functions.

        Args:
            model: A fitted single-outcome model.
            posterior_transform: A PosteriorTransform. If using a multi-output model,
                a PosteriorTransform that transforms the multi-output posterior into a
                single-output posterior is required.
        """
        super().__init__(model=model)
        posterior_transform = self._deprecate_acqf_objective(
            posterior_transform=posterior_transform,
            objective=kwargs.get("objective"),
        )
        if posterior_transform is None:
            if model.num_outputs != 1:
                raise UnsupportedError(
                    "Must specify a posterior transform when using a "
                    "multi-output model."
                )
        else:
            if not isinstance(posterior_transform, PosteriorTransform):
                raise UnsupportedError(
                    "AnalyticAcquisitionFunctions only support PosteriorTransforms."
                )
        self.posterior_transform = posterior_transform

    def set_X_pending(self, X_pending: Optional[Tensor] = None) -> None:
        raise UnsupportedError(
            "Analytic acquisition functions do not account for X_pending yet."
        )

    def _mean_and_sigma(
        self, X: Tensor, compute_sigma: bool = True, min_var: float = 1e-12
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Computes the first and second moments of the model posterior.

        Args:
            X: `batch_shape x q x d`-dim Tensor of model inputs.
            compute_sigma: Boolean indicating whether or not to compute the second
                moment (default: True).
            min_var: The minimum value the variance is clamped too. Should be positive.

        Returns:
            A tuple of tensors containing the first and second moments of the model
            posterior. Removes the last two dimensions if they have size one. Only
            returns a single tensor of means if compute_sigma is True.
        """
        self.to(device=X.device)  # ensures buffers / parameters are on the same device
        posterior = self.model.posterior(
            X=X, posterior_transform=self.posterior_transform
        )
        mean = posterior.mean.squeeze(-2).squeeze(-1)  # removing redundant dimensions
        if not compute_sigma:
            return mean, None
        sigma = posterior.variance.clamp_min(min_var).sqrt().view(mean.shape)
        return mean, sigma


# --------------- Helper functions for analytic acquisition functions. ---------------


def _scaled_improvement(
    mean: Tensor, sigma: Tensor, best_f: Tensor, maximize: bool
) -> Tensor:
    """Returns `u = (mean - best_f) / sigma`, -u if maximize == True."""
    u = (mean - best_f) / sigma
    return u if maximize else -u


def _gamma(
    mean: Tensor, sigma: Tensor, fstar: Tensor, maximize: bool
) -> Tensor:
    """Returns `u = (mean - best_f) / sigma`, -u if maximize == True."""
    u = (fstar - mean) / sigma
    return u if maximize else -u


def _ei_helper(u: Tensor) -> Tensor:
    """Computes phi(u) + u * Phi(u), where phi and Phi are the standard normal
    pdf and cdf, respectively. This is used to compute Expected Improvement.
    """
    return phi(u) + u * Phi(u)



from torch.nn.functional import softplus
import math

from typing import Callable, Tuple, Union

def cauchy(x: Tensor) -> Tensor:
    """Computes a Lorentzian, i.e. an un-normalized Cauchy density function."""
    return 1 / (1 + x.square())

def fatplus(x: Tensor, tau: Union[float, Tensor] = 0.5) -> Tensor:
    """Computes a fat-tailed approximation to `ReLU(x) = max(x, 0)` by linearly
    combining a regular softplus function and the density function of a Cauchy
    distribution. The coefficient `alpha` of the Cauchy density is chosen to guarantee
    monotonicity and convexity.

    Args:
        x: A Tensor on whose values to compute the smoothed function.
        tau: Temperature parameter controlling the smoothness of the approximation.

    Returns:
        A Tensor of values of the fat-tailed softplus.
    """

    def _fatplus(x: Tensor) -> Tensor:
        alpha = 1e-1  # guarantees monotonicity and convexity (TODO: ref + Lemma 4)
        return softplus(x) + alpha * cauchy(x)

    return tau * _fatplus(x / tau)



class ExpectedImprovement(AnalyticAcquisitionFunction):
    r"""Single-outcome Expected Improvement (analytic).

    Computes classic Expected Improvement over the current best observed value,
    using the analytic formula for a Normal posterior distribution. Unlike the
    MC-based acquisition functions, this relies on the posterior at single test
    point being Gaussian (and require the posterior to implement `mean` and
    `variance` properties). Only supports the case of `q=1`. The model must be
    single-outcome.

    `EI(x) = E(max(f(x) - best_f, 0)),`

    where the expectation is taken over the value of stochastic function `f` at `x`.

    Example:
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> EI = ExpectedImprovement(model, best_f=0.2)
        >>> ei = EI(test_X)

    NOTE: It is *strongly* recommended to use LogExpectedImprovement instead of regular
    EI, because it solves the vanishing gradient problem by taking special care of
    numerical computations and can lead to substantially improved BO performance.
    """

    def __init__(
        self,
        model: Model,
        best_f: Union[float, Tensor],
        posterior_transform: Optional[PosteriorTransform] = None,
        maximize: bool = True,
        **kwargs,
    ):
        r"""Single-outcome Expected Improvement (analytic).

        Args:
            model: A fitted single-outcome model.
            best_f: Either a scalar or a `b`-dim Tensor (batch mode) representing
                the best function value observed so far (assumed noiseless).
            posterior_transform: A PosteriorTransform. If using a multi-output model,
                a PosteriorTransform that transforms the multi-output posterior into a
                single-output posterior is required.
            maximize: If True, consider the problem a maximization problem.
        """
        super().__init__(model=model, posterior_transform=posterior_transform, **kwargs)
        self.register_buffer("best_f", torch.as_tensor(best_f))
        self.maximize = maximize

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:

        r"""Evaluate Expected Improvement on the candidate set X.

        Args:
            X: A `(b1 x ... bk) x 1 x d`-dim batched tensor of `d`-dim design points.
                Expected Improvement is computed for each point individually,
                i.e., what is considered are the marginal posteriors, not the
                joint.

        Returns:
            A `(b1 x ... bk)`-dim tensor of Expected Improvement values at the
            given design points `X`.
        """
        mean, sigma = self._mean_and_sigma(X)
        u = _scaled_improvement(mean, sigma, self.best_f, self.maximize)
        return sigma * _ei_helper(u)

  

class TruncatedExpectedImprovement(AnalyticAcquisitionFunction):
   

    def __init__(
        self,
        model: Model,
        best_f: Union[float, Tensor],
        fstar: Union[float, Tensor],
        posterior_transform: Optional[PosteriorTransform] = None,
        maximize: bool = True,
        **kwargs,
    ):
        
        super().__init__(model=model, posterior_transform=posterior_transform, **kwargs)
        self.register_buffer("best_f", torch.as_tensor(best_f))
        self.register_buffer("fstar", torch.as_tensor(fstar))
        self.maximize = maximize

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
       
        mean, sigma = self._mean_and_sigma(X)
        u1 = _scaled_improvement(mean, sigma, self.best_f, self.maximize)
        u2 = _scaled_improvement(mean, sigma, self.fstar, self.maximize)

        return (sigma * _ei_helper(u1)) -(sigma * _ei_helper(u2))
    

 
    
class MES_KnownOptimum(AnalyticAcquisitionFunction):
   
    def __init__(
        self,
        model: Model,
        #best_f: Union[float, Tensor],
        fstar: Union[float, Tensor],
        posterior_transform: Optional[PosteriorTransform] = None,
        maximize: bool = True,
        **kwargs,
    ):
        
        
        super().__init__(model=model, posterior_transform=posterior_transform, **kwargs)
        #self.register_buffer("best_f", torch.as_tensor(best_f))
        self.register_buffer("fstar", torch.as_tensor(fstar))
        self.maximize = maximize

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
       
        mean, sigma = self._mean_and_sigma(X)
        gamma = _gamma(mean, sigma, self.fstar, self.maximize)
        
        # use fat softplus to make the value smaller than -15 be -15
        max_dis = 20.
        gamma = fatplus(gamma+max_dis,tau=0.2)-torch.as_tensor(max_dis) 
        gamma = - (fatplus(-gamma+max_dis,tau=0.2)-torch.as_tensor(max_dis))

        res = (gamma*phi(gamma))/(2*Phi(gamma)) - log_ndtr(gamma)  #  torch.log(Phi(gamma))
       
        return res
    

class Fstar_pdf(AnalyticAcquisitionFunction):
   
    def __init__(
        self,
        model: Model,
        #best_f: Union[float, Tensor],
        fstar: Union[float, Tensor],
        posterior_transform: Optional[PosteriorTransform] = None,
        maximize: bool = True,
        **kwargs,
    ):
        
        
        super().__init__(model=model, posterior_transform=posterior_transform, **kwargs)
        #self.register_buffer("best_f", torch.as_tensor(best_f))
        self.register_buffer("fstar", torch.as_tensor(fstar))
        self.maximize = maximize

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
       
        mean, sigma = self._mean_and_sigma(X)
        gamma = _gamma(mean, sigma, self.fstar, self.maximize)
        
        # use fat softplus to make the value smaller than -15 be -15
        max_dis = 20.
        gamma = fatplus(gamma+max_dis,tau=0.2)-torch.as_tensor(max_dis) 
        gamma = - (fatplus(-gamma+max_dis,tau=0.2)-torch.as_tensor(max_dis))
        
        res = phi(gamma)  
       
        return res
    
    


class Fstar_pdf_GradientEnhanced(AnalyticAcquisitionFunction):
   
    def __init__(
        self,
        model: Model,
        #best_f: Union[float, Tensor],
        fstar: Union[float, Tensor],
        posterior_transform: Optional[PosteriorTransform] = None,
        maximize: bool = True,
        **kwargs,
    ):
        
        
        super().__init__(model=model, posterior_transform=posterior_transform, **kwargs)
        #self.register_buffer("best_f", torch.as_tensor(best_f))
        self.register_buffer("fstar", torch.as_tensor(fstar))
        self.maximize = maximize

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
       
        mean, sigma = self._mean_and_sigma(X)
        gamma = _gamma(mean, sigma, self.fstar, self.maximize)
        
        
        # use fat softplus to make the value smaller than -15 be -15
        max_dis = 20.
        gamma = fatplus(gamma+max_dis,tau=0.2)-torch.as_tensor(max_dis) 
        gamma = - (fatplus(-gamma+max_dis,tau=0.2)-torch.as_tensor(max_dis))
        
        part1 = log_phi(gamma)
        
        # part 2 calculation
        D = X.shape[-1]
        mean_d, variance_d = self.model.posterior_derivative(X)
        
        logpdf_total = torch.zeros(X.shape[0])
        
        for d in range(D):
            mean_list = mean_d[:,d]
            sigma_list = torch.sqrt(variance_d[:,d,d])
            
            u = (torch.as_tensor(0.)-mean_list)/sigma_list
            
            
            # use fat softplus to make the value smaller than -15 be -15
            max_dis = 20.
            u = fatplus(u+max_dis,tau=0.2)-torch.as_tensor(max_dis) 
            u = - (fatplus(-u+max_dis,tau=0.2)-torch.as_tensor(max_dis))

            
            pdf_temp = log_phi(u)  #log_phi(u)  phi(u)
            
            logpdf_total += pdf_temp
            
            # print(pdf_temp)
            
        #print('log pdf is: ', pdf_total)
        # print('part 2: ',logpdf_total)
        # print(logpdf_total.shape)
        
        part2 = logpdf_total
       
        return part1  + part2
    
    
    
class Fstar_pdf_GradientEnhanced_fantasy(AnalyticAcquisitionFunction):
   
    def __init__(
        self,
        model: Model,
        #best_f: Union[float, Tensor],
        fstar: Union[float, Tensor],
        posterior_transform: Optional[PosteriorTransform] = None,
        maximize: bool = True,
        **kwargs,
    ):
        
        
        super().__init__(model=model, posterior_transform=posterior_transform, **kwargs)
        #self.register_buffer("best_f", torch.as_tensor(best_f))
        self.register_buffer("fstar", torch.as_tensor(fstar))
        self.maximize = maximize

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
       
        mean, sigma = self._mean_and_sigma(X)
        gamma = _gamma(mean, sigma, self.fstar, self.maximize)
        
        
        # use fat softplus to make the value smaller than -15 be -15
        max_dis = 20.
        gamma = fatplus(gamma+max_dis,tau=0.2)-torch.as_tensor(max_dis) 
        gamma = - (fatplus(-gamma+max_dis,tau=0.2)-torch.as_tensor(max_dis))
        
        part1 = log_phi(gamma)
        
        # part 2 calculation
        logpdf_total = torch.zeros(X.shape[0])
        D = X.shape[-1]
        
        
        for ii in range(X.shape[0]):
            
            x_temp = X[ii]
            
            logpdf_temp = 0.
            
            x_temp = x_temp.reshape(1,1,D)
            
            model_temp = self.model.get_fantasy_model(x_temp.reshape(-1,D), torch.tensor([self.fstar.item()]).reshape(-1,1))
            model_temp.N = self.model.N+1
            model_temp.train_targets = model_temp.train_targets.reshape(model_temp.N)
            
            mean_d, variance_d = model_temp.posterior_derivative(x_temp.reshape(-1,1,D))
        
       
            for d in range(D):
                mean_list = mean_d[:,d]
                sigma_list = torch.sqrt(variance_d[:,d,d])
                
                u = (torch.as_tensor(0.)-mean_list)/sigma_list
                
                
                # use fat softplus to make the value smaller than -15 be -15
                max_dis = 20.
                u = fatplus(u+max_dis,tau=0.2)-torch.as_tensor(max_dis) 
                u = - (fatplus(-u+max_dis,tau=0.2)-torch.as_tensor(max_dis))

                
                pdf_temp = log_phi(u)  #log_phi(u)  phi(u)
                
                logpdf_temp += pdf_temp.item()
                
            logpdf_total[ii] = logpdf_temp
            
            # print(pdf_temp)
            
        #print('log pdf is: ', pdf_total)
        # print('part 2: ',logpdf_total)
        # print(logpdf_total.shape)
        
        part2 = logpdf_total
       
        return part1  + part2
    
    
    
    

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
def new_np_acq(X,dim,acq_function):
    
    X = torch.tensor(X).reshape(-1,1,dim)
    res = acq_function(X).detach().numpy()
    
    return res
    

def My_acquisition_opt(acq_function,dim): #bound should an array of size dim*2
    
  bounds=np.array([0.,1.]*dim).reshape(-1,2)
  
  dim = bounds.shape[0]
  opts ={'maxiter':50*dim,'maxfun':50*dim,'disp': False}

  restart_num = 3*dim
  X_candidate = []
  AF_candidate = []

  for i in range(restart_num):
    init_X = np.random.uniform(bounds[:, 0], bounds[:, 1],size=(30*dim, dim))
    value_holder = new_np_acq(init_X,dim,acq_function)   #acq_function(torch.tensor(init_X).reshape(-1,1,dim))
    
    value_holder = value_holder
      
    x0=init_X[np.argmax(value_holder)]

    res = minimize(lambda x: -new_np_acq(X=x,dim=dim,acq_function=acq_function),x0,
                                  bounds=bounds,method="L-BFGS-B",options=opts) #L-BFGS-B  nelder-mead(better for rough function) Powell

    X_temp =  res.x  
    AF_temp = new_np_acq(X_temp,dim,acq_function) 
    
    X_candidate.append(X_temp)
    AF_candidate.append(AF_temp)

  X_next = X_candidate[np.argmax(AF_candidate)]

  return X_next
    
    