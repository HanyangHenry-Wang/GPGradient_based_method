import torch
import gpytorch
import botorch
from gpytorch.constraints.constraints import Interval

class ExactGPSEModel(gpytorch.models.ExactGP, botorch.models.gpytorch.GPyTorchModel):
    """An exact Gaussian process (GP) model with a squared exponential (SE) kernel.

    ExactGP: The base class of gpytorch for any Gaussian process latent function to be
        used in conjunction with exact inference.
    GPyTorchModel: The easiest way to use a GPyTorch model in BoTorch.
        This adds all the api calls that botorch expects in its various modules.

    Attributes:
        train_x: (N x D) The training features X.
        train_y: (N x 1) The training targets y.
        lengthscale_constraint: Constraint for lengthscale of SE-kernel, gpytorch.constraints.
        lengthscale_hyperprior: Hyperprior for lengthscale of SE-kernel, gpytorch.priors.
        outputscale_constraint: Constraint for outputscale of SE-kernel, gpytorch.constraints.
        outputscale_hyperprior: Hyperprior for outputscale of SE-kernel, gpytorch.priors.
        noise_constraint: Constraint for noise, gpytorch.constraints.
        noise_hyperprior: Hyperprior for noise, gpytorch.priors.
        ard_num_dims: Set this if you want a separate lengthscale for each input dimension.
            Should be D if train_x is a N x D matrix.
        prior_mean: Value for constant mean.
    """

    _num_outputs = 1  # To inform GPyTorchModel API.

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        lengthscale_constraint=None,
        lengthscale_hyperprior=None,
        outputscale_constraint=None,
        outputscale_hyperprior=None,
        noise_constraint=  Interval(10**(-6),2*10**(-6)), #Interval(10**(-6),2*10**(-6)), #None, 
        noise_hyperprior=None,
        ard_num_dims=None,
        prior_mean=0,
    ):
        """Inits GP model with data and a Gaussian likelihood."""
        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=noise_constraint, noise_prior=noise_hyperprior
        )
        if train_y is not None:
            train_y = train_y.squeeze(-1)
        super(ExactGPSEModel, self).__init__(train_x, train_y, likelihood)

        self.mean_module = gpytorch.means.ZeroMean()  #gpytorch.means.ConstantMean()
        if prior_mean != 0:
            self.mean_module.initialize(constant=prior_mean)
            self.mean_module.constant.requires_grad = False

        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                ard_num_dims=ard_num_dims,
                lengthscale_prior=lengthscale_hyperprior,
                lengthscale_constraint=lengthscale_constraint,
            ),
            outputscale_prior=outputscale_hyperprior,
            outputscale_constraint=outputscale_constraint,
        )
        # Initialize lengthscale and outputscale to mean of priors.
        if lengthscale_hyperprior is not None:
            self.covar_module.base_kernel.lengthscale = lengthscale_hyperprior.mean
        if outputscale_hyperprior is not None:
            self.covar_module.outputscale = outputscale_hyperprior.mean

    def forward(self, x):
        """Compute the prior latent distribution on a given input.

        Typically, this will involve a mean and kernel function. The result must be a
        MultivariateNormal. Calling this model will return the posterior of the latent
        Gaussian process when conditioned on the training data. The output will be a
        MultivariateNormal.

        Args:
            x: (n x D) The test points.

        Returns:
            A MultivariateNormal.
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    



class DerivativeExactGPSEModel(ExactGPSEModel):
    """Derivative of the ExactGPSEModel w.r.t. the test points x.

    Since differentiation is a linear operator this is again a Gaussian process.

    Attributes:
        D: Dimension of train_x-/input-data.
        normalize: Optional normalization function for policy parameterization.
        unnormalize: Optional unnormalization function for policy
            parameterization.
        N_max: Maximum number of training samples (train_x, N) for model inference.
        lengthscale_constraint: Constraint for lengthscale of SE-kernel, gpytorch.constraints.
        lengthscale_hyperprior: Hyperprior for lengthscale of SE-kernel, gpytorch.priors.
        outputscale_constraint: Constraint for outputscale of SE-kernel, gpytorch.constraints.
        outputscale_hyperprior: Hyperprior for outputscale of SE-kernel, gpytorch.priors.
        noise_constraint: Constraint for noise, gpytorch.constraints.
        noise_hyperprior: Hyperprior for noise, gpytorch.priors.
        ard_num_dims: Set this if you want a separate lengthscale for each input dimension.
            Should be D if train_x is a N x D matrix.
        prior_mean: Value for constant mean.
    """

    def __init__(
        self,
        D: int,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        normalize=None,
        unnormalize=None,
        N_max=None,
        lengthscale_constraint=None,
        lengthscale_hyperprior=None,
        outputscale_constraint=None,
        outputscale_hyperprior=None,
        noise_constraint= Interval(10**(-6),2*10**(-6)), #Interval(10**(-6),2*10**(-6)), #None, 
        noise_hyperprior=None,
        ard_num_dims=None,
        prior_mean=0.0,
    ):
        """Inits GP model with data and a Gaussian likelihood."""
        # train_x_init, train_y_init = (
        #     torch.empty(0, D),
        #     torch.empty(0),
        # )
        super(DerivativeExactGPSEModel, self).__init__(
            train_x,
            train_y,
            lengthscale_constraint,
            lengthscale_hyperprior,
            outputscale_constraint,
            outputscale_hyperprior,
            noise_constraint,
            noise_hyperprior,
            ard_num_dims,
            prior_mean,
        )

        self.N_max = N_max
        self.D = D
        self.N = self.train_inputs[0].shape[0] #    self.N = train_x.shape[0]
        
        # self.train_xs = train_x_init
        # self.train_ys = train_y_init
        if normalize is None:
            normalize = lambda params: params
        self.normalize = normalize
        if unnormalize is None:
            unnormalize = lambda params: params
        self.unnormalize = unnormalize


    # def get_L_lower(self):
    #     """Get Cholesky decomposition L, where L is a lower triangular matrix.

    #     Returns:
    #         Cholesky decomposition L.
    #     """
    #     return (
    #         self.prediction_strategy.lik_train_train_covar.root_decomposition()
    #         .root.evaluate()
    #         .detach()
    #     )

    def get_KXX_inv(self):
        """Get the inverse matrix of K(X,X).

        Returns:
            The inverse of K(X,X).
        """
        L_inv_upper = self.prediction_strategy.covar_cache.detach()
        return L_inv_upper @ L_inv_upper.transpose(0, 1)

    # def get_KXX_inv_old(self):
    #     """Get the inverse matrix of K(X,X).

    #     Not as efficient as get_KXX_inv.

    #     Returns:
    #         The inverse of K(X,X).
    #     """
    #     X = self.train_inputs[0]
    #     sigma_n = self.likelihood.noise_covar.noise.detach()
    #     return torch.inverse(
    #         self.covar_module(X).evaluate() + torch.eye(X.shape[0]) * sigma_n
    #     )

    def _get_KxX_dx(self, x):
        """Computes the analytic derivative of the kernel K(x,X) w.r.t. x.

        Args:
            x: (n x D) Test points.

        Returns:
            (n x D) The derivative of K(x,X) w.r.t. x.
        """
        X = self.train_inputs[0]
        n = x.shape[0]
        K_xX = self.covar_module(x, X).evaluate()
        lengthscale = self.covar_module.base_kernel.lengthscale.detach()
        return (
            -torch.eye(self.D, device=x.device)
            / lengthscale ** 2
            @ (
                (x.view(n, 1, self.D) - X.view(1, self.N, self.D))
                * K_xX.view(n, self.N, 1)
            ).transpose(1, 2)
        )

    def _get_Kxx_dx2(self):
        """Computes the analytic second derivative of the kernel K(x,x) w.r.t. x.

        Args:
            x: (n x D) Test points.

        Returns:
            (n x D x D) The second derivative of K(x,x) w.r.t. x.
        """
        lengthscale = self.covar_module.base_kernel.lengthscale.detach()
        sigma_f = self.covar_module.outputscale.detach()
        return (
            torch.eye(self.D, device=lengthscale.device) / lengthscale ** 2
        ) * sigma_f

    def posterior_derivative(self, x):
        """Computes the posterior of the derivative of the GP w.r.t. the given test
        points x.

        Args:
            x: (n x D) Test points.

        Returns:
            A GPyTorchPosterior.
        """
        if self.prediction_strategy is None:
            self.posterior(x)  # Call this to update prediction strategy of GPyTorch.
        K_xX_dx = self._get_KxX_dx(x)
        mean_d = K_xX_dx @ self.get_KXX_inv() @ self.train_targets
        variance_d = (
            self._get_Kxx_dx2() - K_xX_dx @ self.get_KXX_inv() @ K_xX_dx.transpose(1, 2)
        )
        variance_d = variance_d.clamp_min(1e-9)

        return mean_d, variance_d
    
    


class DerivativeExactGPSEModel_2(ExactGPSEModel): #this one is friendly to get_fansty 
    
    """Derivative of the ExactGPSEModel w.r.t. the test points x.

    Since differentiation is a linear operator this is again a Gaussian process.

    Attributes:
        D: Dimension of train_x-/input-data.
        normalize: Optional normalization function for policy parameterization.
        unnormalize: Optional unnormalization function for policy
            parameterization.
        N_max: Maximum number of training samples (train_x, N) for model inference.
        lengthscale_constraint: Constraint for lengthscale of SE-kernel, gpytorch.constraints.
        lengthscale_hyperprior: Hyperprior for lengthscale of SE-kernel, gpytorch.priors.
        outputscale_constraint: Constraint for outputscale of SE-kernel, gpytorch.constraints.
        outputscale_hyperprior: Hyperprior for outputscale of SE-kernel, gpytorch.priors.
        noise_constraint: Constraint for noise, gpytorch.constraints.
        noise_hyperprior: Hyperprior for noise, gpytorch.priors.
        ard_num_dims: Set this if you want a separate lengthscale for each input dimension.
            Should be D if train_x is a N x D matrix.
        prior_mean: Value for constant mean.
    """

    def __init__(
        self,
        D: int,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        normalize=None,
        unnormalize=None,
        N_max=None,
        lengthscale_constraint=None,
        lengthscale_hyperprior=None,
        outputscale_constraint=None,
        outputscale_hyperprior=None,
        noise_constraint= Interval(10**(-6),2*10**(-6)), #Interval(10**(-6),2*10**(-6)), #None, 
        noise_hyperprior=None,
        ard_num_dims=None,
        prior_mean=0.0,
    ):
        """Inits GP model with data and a Gaussian likelihood."""
        # train_x_init, train_y_init = (
        #     torch.empty(0, D),
        #     torch.empty(0),
        # )
        super(DerivativeExactGPSEModel_2, self).__init__(
            train_x,
            train_y,
            lengthscale_constraint,
            lengthscale_hyperprior,
            outputscale_constraint,
            outputscale_hyperprior,
            noise_constraint,
            noise_hyperprior,
            ard_num_dims,
            prior_mean,
        )

        self.N_max = N_max
        self.D = D
        self.N = self.train_inputs[0].shape[0] #    self.N = train_x.shape[0]
        
        # self.train_xs = train_x_init
        # self.train_ys = train_y_init
        if normalize is None:
            normalize = lambda params: params
        self.normalize = normalize
        if unnormalize is None:
            unnormalize = lambda params: params
        self.unnormalize = unnormalize




    def get_KXX_inv(self,index=0):
        """Get the inverse matrix of K(X,X).

        Returns:
            The inverse of K(X,X).
        """
        L_inv_upper = self.prediction_strategy.covar_cache.detach()
        L_inv_upper = L_inv_upper.reshape(-1,self.N,self.N)
        L_inv_upper = L_inv_upper[index]
        
        return L_inv_upper @ L_inv_upper.transpose(0, 1)



    def _get_KxX_dx(self, x,index=0):
        """Computes the analytic derivative of the kernel K(x,X) w.r.t. x.

        Args:
            x: (n x D) Test points.

        Returns:
            (n x D) The derivative of K(x,X) w.r.t. x.
        """
        X = self.train_inputs[0]
        X = X.reshape(-1,self.N,self.D)
        X = X[index]
        
        n = x.shape[0]
        K_xX = self.covar_module(x, X).evaluate()
        lengthscale = self.covar_module.base_kernel.lengthscale.detach()
        return (
            -torch.eye(self.D, device=x.device)
            / lengthscale ** 2
            @ (
                (x.view(n, 1, self.D) - X.view(1, self.N, self.D))
                * K_xX.view(n, self.N, 1)
            ).transpose(1, 2)
        )

    def _get_Kxx_dx2(self):
        """Computes the analytic second derivative of the kernel K(x,x) w.r.t. x.

        Args:
            x: (n x D) Test points.

        Returns:
            (n x D x D) The second derivative of K(x,x) w.r.t. x.
        """
        lengthscale = self.covar_module.base_kernel.lengthscale.detach()
        sigma_f = self.covar_module.outputscale.detach()
        return (
            torch.eye(self.D, device=lengthscale.device) / lengthscale ** 2
        ) * sigma_f

    def posterior_derivative(self, x,index=0):
        """Computes the posterior of the derivative of the GP w.r.t. the given test
        points x.

        Args:
            x: (n x D) Test points.

        Returns:
            A GPyTorchPosterior.
        """
        
        train_targets = self.train_targets.reshape(-1,self.N)
        
        if self.prediction_strategy is None:
            self.posterior(x)  # Call this to update prediction strategy of GPyTorch.
        K_xX_dx = self._get_KxX_dx(x,index)
        mean_d = K_xX_dx @ self.get_KXX_inv(index) @ train_targets[index]
        variance_d = (
            self._get_Kxx_dx2() - K_xX_dx @ self.get_KXX_inv(index) @ K_xX_dx.transpose(1, 2)
        )
        variance_d = variance_d #.clamp_min(1e-9)

        return mean_d, variance_d

