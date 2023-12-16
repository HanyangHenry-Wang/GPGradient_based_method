import torch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.test_functions import Ackley,Beale,Branin,Rosenbrock,SixHumpCamel,Hartmann,Powell,DixonPrice,Levy,StyblinskiTang,Griewank
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
from botorch.optim import optimize_acqf
from botorch.utils.transforms import unnormalize,normalize
from torch.quasirandom import SobolEngine


from model import DerivativeExactGPSEModel
from acquisition import (MES_KnownOptimum,TruncatedExpectedImprovement,ExpectedImprovement,Fstar_pdf,
                            Fstar_pdf_GradientEnhanced,Fstar_pdf_GradientEnhanced_fantasy)
from acquisition import  My_acquisition_opt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double

torch.set_default_dtype(dtype)



def get_initial_points(dim, n_pts, seed=0):
    sobol = SobolEngine(dimension=dim, scramble=True, seed=seed)
    X_init = sobol.draw(n=n_pts).to(dtype=dtype, device=device)
    return X_init




def experiment_running(N,iter_num,function,fstar,acquisition):
    
    bounds=function.bounds.to(device)
    dim = bounds.shape[1]
    standard_bounds=torch.tensor([0.,1.]*dim).reshape(-1,2).T.to(device)
    
    
    record = []

    for exp in range(N):

        print(exp)
        torch.manual_seed(exp)

        train_x_standard = get_initial_points(dim,4*dim,exp).to(device)
        train_x = unnormalize(train_x_standard, bounds).reshape(-1,dim)
        train_obj = function(train_x).unsqueeze(-1)

        best_value = train_obj.max().item()
        best_value_holder = [best_value]

        for i in range (iter_num):

            train_x_standard = normalize(train_x, bounds).to(device)
            train_obj_standard = (train_obj - train_obj.mean()) / train_obj.std()
            fstar_standard = (fstar - train_obj.mean()) / train_obj.std()
            
            torch.manual_seed(exp+iter_num)
            model = DerivativeExactGPSEModel(dim,train_x_standard, train_obj_standard).to(device)    

            mll = ExactMarginalLogLikelihood(model.likelihood, model) .to(device)

            torch.manual_seed(exp+iter_num)
            
            try:
                fit_gpytorch_mll(mll)
            except:
                pass
            
            
            torch.manual_seed(exp+iter_num)
            
            if acquisition == 'ExpectedImprovement':
                AF = ExpectedImprovement(model=model, best_f=train_obj_standard.max().item()) .to(device)
            elif acquisition == 'TruncatedExpectedImprovement':
                AF = TruncatedExpectedImprovement(model=model, best_f=train_obj_standard.max().item(),fstar=fstar_standard) .to(device)
            elif acquisition == 'MES_KnownOptimum':
                AF = MES_KnownOptimum(model=model, fstar=fstar_standard) .to(device)
            elif acquisition == 'Fstar_pdf':
                AF = Fstar_pdf(model=model, fstar=fstar_standard) .to(device)
            elif acquisition == 'Fstar_pdf_GradientEnhanced':
                AF = Fstar_pdf_GradientEnhanced(model=model, fstar=fstar_standard) .to(device)
            elif acquisition == 'Fstar_pdf_GradientEnhanced_fantasy':
                AF = Fstar_pdf_GradientEnhanced_fantasy(model=model, fstar=fstar_standard) .to(device)


            new_point_analytic, _ = optimize_acqf(
                acq_function=AF,
                bounds=standard_bounds .to(device),
                q=1,
                num_restarts=3*dim,
                raw_samples=30*dim,
                options={},
            )
            
            # np.random.seed(exp+i)
            # new_point_analytic = My_acquisition_opt(AF,dim)
            # new_point_analytic = torch.tensor(new_point_analytic).reshape(-1,dim)

            next_x = unnormalize(new_point_analytic, bounds).reshape(-1,dim)
            new_obj = function(next_x).unsqueeze(-1) .to(device)


            train_x = torch.cat((train_x, next_x))
            train_obj = torch.cat((train_obj, new_obj))

            best_value = train_obj.max().item()
            best_value_holder.append(best_value)

            print(best_value_holder[-1])

        best_value_holder = np.array(best_value_holder)
        record.append(best_value_holder)
        
    return record