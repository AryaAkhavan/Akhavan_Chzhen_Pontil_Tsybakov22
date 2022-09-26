# <Numerical experiments: A gradient estimator via L1-randomization for online zero-order optimization with two point feedback>

## Project description

We implement two algorithms, namely *l_1 randomization* and *l_2 randomization* for optimizing a convex and Lipschitz function. The code provides a comparison between the performance of the above algorithms, which are detailed in the paper. The experiment shown in Section 7 of the paper is analyzed using the test function *SumFuncL1Test*, implemented in **objevives.py**.

## Installation

1. Python 3.9.12
2. Joblib (see https://pypi.org/project/joblib/#downloads)

## Usage

All parameters can be initialized in **experimetns.py**:

*dim*: The dimension of the objective function

*max_iter*: Maximum number of iterations

*sample*: The final plot is the average over *sample* number of trials

*constr_type*: The type of the constraint set, can be assigned as:

   1. 'simplex': If the constraint set is simplex 
   2. 'euclid_ball': If the constraint set is the unit sphere 
   3. 'pos': If the constraint set is the portion of the unit sphere that contains the vector with non-negative entries

 *objective*: The objective function. All objective functions are implemented in **objevives.py**
 
 *norm_str_conv*: Set *q* if the objective function is a Lipschitz function with respect to *l_q*-norm
 
 *sigma*:
 
   1. Under Assumption 2, set *0*
   2. Under Assumption 3, set the upper bound for the second moment of the noise 
        
*noise_family*: It is the distribution of noise that can be set as:

   1. 'Gaussian': If the noise is a standard Gaussian random variable
   2. ''Bernoulli': If the noise is a Bernoulli random variable


# Bibentry
```bib
@article{Akhavan_Chzhen_Pontil_Tsybakov22,
      title={A gradient estimator via L1-randomization for online zero-order optimization with two point feedback}, 
      author={A. Akhavan and E. Chzhen and M. Pontil and A. Tsybakov},
      year={2022},
      journal={NeurIPS22}
}

