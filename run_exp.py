from utilitis import experiment_running
import botorch
import numpy as np
import torch
from botorch.test_functions import Ackley,Beale,Branin,Rosenbrock,SixHumpCamel,Hartmann,Powell,DixonPrice,Levy,StyblinskiTang,Griewank


import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger('lengthscale').disabled = True
logging.getLogger('variance').disabled = True
logging.getLogger('psi').disabled = True


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double


function_information = []




temp={}
temp['name']='Branin2D' 
temp['function'] = Branin(negate=True)
temp['fstar'] =  -0.397887 
function_information.append(temp)

temp={}
temp['name']='Beale2D' 
temp['function'] = Beale(negate=True)
temp['fstar'] =  0. 
function_information.append(temp)

# temp={}
# temp['name']='SixHumpCamel2D' 
# temp['function'] = SixHumpCamel(negate=True)
# temp['fstar'] =  1.0317
# function_information.append(temp)

# temp={}
# temp['name']='Rosenbrock3D' 
# temp['function'] = Rosenbrock(dim=3,negate=True)
# temp['fstar'] =  0.
# function_information.append(temp)

# temp={}
# temp['name']='StyblinskiTang4D' 
# temp['function'] = StyblinskiTang(dim=4,negate=True)
# temp['fstar'] = 4*39.16599
# function_information.append(temp)



for information in function_information:

    fun = information['function']
    fstar = information['fstar']
    dim = fun.dim
    
    if dim <=3:
        iter_num = 50
        N = 12
        
    elif dim<=5:
        iter_num = 100
        N = 100
    else:
        iter_num = 150
        N = 20
        
    # res = experiment_running(N,iter_num,fun,fstar,'ExpectedImprovement')
    # np.savetxt('exp_res/'+information['name']+'_ExpectedImprovement', res, delimiter=',')
    
    # res = experiment_running(N,iter_num,fun,fstar,'TruncatedExpectedImprovement')
    # np.savetxt('exp_res/'+information['name']+'_TruncatedExpectedImprovement', res, delimiter=',')
    
    # res = experiment_running(N,iter_num,fun,fstar,'MES_KnownOptimum')
    # np.savetxt('exp_res/'+information['name']+'_MES_KnownOptimum', res, delimiter=',')
    
    # res = experiment_running(N,iter_num,fun,fstar,'Fstar_pdf')
    # np.savetxt('exp_res/'+information['name']+'_Fstar_pdf', res, delimiter=',')
    
    # res = experiment_running(N,iter_num,fun,fstar,'Fstar_pdf_GradientEnhanced')
    # np.savetxt('exp_res/'+information['name']+'_Fstar_pdf_GradientEnhanced', res, delimiter=',')
    
    res = experiment_running(N,iter_num,fun,fstar,'Fstar_pdf_GradientEnhanced_fantasy')
    np.savetxt('exp_res/'+information['name']+'_Fstar_pdf_GradientEnhanced_fantasy', res, delimiter=',')
    
