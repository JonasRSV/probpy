WIP
---

End result is hopefully a PGM module built with numpy



Documentation
---

- [Core](#core)
  - [MCMC](#MCMC)
    - [Metropolis](#metropolis)
    - [Metropolis-Hastings](#metropolis-hastings)
  - [Integration](#integration)
    - [Uniform Importance Sampling](#uniform-importance-sampling)
  - [Density Estimation](#density-estimation)
    - [Non-parametric UCKD](#uckd)
    - [Non-parametric RCKD](#rckd)
  - [Distributions](#distributions)
    - [Normal](#normal)
    - [Multivariate Normal](#multivariate-normal)
    - [Uniform](#uniform)
    - [Multivariate Uniform](#multivariate-uniform)
    - [Bernoulli](#bernoulli)
    - [Categorical](#categorical)
    - [Dirichlet](#dirichlet)
    - [Beta](#beta)
    - [Exponential](#exponential)
    - [Binomial](#binomial)
    - [Multinomial](#multinomial)





## Core 

Central functionality

### MCMC

Sampling algorithms

#### Metropolis

As long as: p(x) * Z < q(x) * M, this will work, although it might be slow. 

```python3
from probpy.distributions import normal
from progpy.mcmc import metropolis

pdf = lambda x: normal.p(x, 0, 1) + normal.p(x, 6, 3) + normal.p(x, -6, 0.5)
samples = metropolis(size=100000, pdf=pdf, proposal=normal.freeze(mu=0, sigma=10), M=30.0)
```

<p align="center">
  <img width=600px heigth=300px src="images/metropolis.png" />
</p>


#### Metropolis-Hastings

```python3
from probpy.distributions import normal
from progpy.mcmc import metropolis_hastings, fast_metropolis_hastings

pdf = lambda x: normal.p(x, 0, 1) + normal.p(x, 6, 3) + normal.p(x, -6, 0.5)
samples = metropolis_hastings(size=50000, pdf=pdf, proposal=normal.freeze(sigma=0.5), initial=-5) # This one takes generic proposal
samples = fast_metropolis_hastings(size=50000, pdf=pdf, initial=-5.0, energy=1.0) # This one is much faster
```

<p align="center">
  <img width=600px heigth=300px src="images/metropolis-hastings.png" />
</p>


### Integration

Statistical integration algorithms 

#### Uniform Importance Sampling

```python3
from probpy.distributions import normal
from probpy.integration import uniform_importance_sampling

f = lambda x: -np.square(x) + 3
result = uniform_importance_sampling(size=10000, 
                                     function=f, 
                                     domain=(-2, 2), 
                                     proposal=normal.freeze(mu=0, sigma=2))

```
> Wolfram alpha gives 6.66666..

<p align="center">
  <img width=600px heigth=300px src="images/uniform_importance_sampling.png" />
</p>

Multivariate example

```python3
from probpy.distributions import multivariate_normal
from probpy.integration import uniform_importance_sampling

f = lambda x: -np.square(x[:, 0]) + np.square(x[:, 1])

lower_bound = np.array([0, 0])
upper_bound = np.array([4, 2])

results = uniform_importance_sampling(size=100000,
                                      function=f,
                                      domain=(lower_bound, upper_bound),
                                      proposal=multivariate_normal.freeze(mu=np.zeros(2), sigma=np.eye(2) * 2)))
```
> Wolfram alpha gives -32

<p align="center">
  <img width=600px heigth=300px src="images/uniform_importance_sampling_multivariate.png" />
</p>

### Density Estimation


#### UCKD

> Un-normalised Convolution Kernel Density

```python3
from probpy.distributions import normal
from probpy.density import UCKD
from probpy.mcmc import fast_metropolis_hastings

def distribution(x):
    return 0.3333 * normal.p(x, -2, 1) + 0.3333 * normal.p(x, 2, 0.2) + 0.3333 * normal.p(x, 4, 0.2)


samples = fast_metropolis_hastings(50000, distribution, initial=0.0, energy=1.0) # Create some sample to fit

density = UCKD(variance=5.0)
density.fit(samples)

lb, ub = -6, 6
n = 2000

x = np.linspace(lb, ub, n)
y = density.p(x)
y = y / (y.sum() / (n / (ub - lb))) # To make nice plot we renormalize

```

<p align="center">
  <img width=600px heigth=300px src="images/uckd.png" />
</p>

#### RCKD

> Renormalized Convolution Kernel Density (Much slower than UCKD)

```python3
from probpy.distributions import normal
from probpy.density import RCKD
from probpy.mcmc import fast_metropolis_hastings

def distribution(x):
    return 0.3333 * normal.p(x, -2, 1) + 0.3333 * normal.p(x, 2, 0.2) + 0.3333 * normal.p(x, 4, 0.2)


samples = fast_metropolis_hastings(50000, distribution, initial=0.0, energy=1.0) # Create some sample to fit

density = RCKD(variance=5.0, error=1)
density.fit(samples)

lb, ub = -6, 6
n = 2000

x = np.linspace(lb, ub, n)
y = density.p(x) # No need to renormalize here
```

<p align="center">
  <img width=600px heigth=300px src="images/rckd.png" />
</p>



### Distributions

PDFs / PMFs and sampling functions (the sampling mostly just uses numpy.random.xx)


#### Normal

```python3
from probpy.distributions import normal

# Sampling
samples = 10000
mu, sigma = 0, 0.1
n = normal.sample(mu, sigma, samples)


# PDF
x = np.linspace(-4, 4, 100)
n = normal.p(x, mu, sigma)
```

<p align="center">
  <img width=600px heigth=300px src="images/normal.png" />
</p>


#### Multivariate normal


```python3
from probpy.distributions import multivariate_normal

# Sampling
samples = 10000
mu = np.zeros(2)
mu[0] = 1
sigma = np.eye(2)
sigma[0, 0] = 0.3
sigma[0, 1], sigma[1, 0] = 0.5, 0.5

n = multivariate_normal.sample(mu, sigma, samples)

# PDF
X = ...
P = multivariate_normal.p(X, mu, sigma)
```

<p align="center">
  <img width=600px heigth=300px src="images/multi_normal.png" />
</p>

#### Uniform

```python3
from probpy.distributions import uniform

#Sampling
samples = 10000
a, b = -2, 3
n = uniform.sample(a, b, samples)

# PDF
x = np.linspace(-4, 4, 100)
n = uniform.p(x, a, b)
```

<p align="center">
  <img width=600px heigth=300px src="images/uniform.png" />
</p>

#### Multivariate uniform

```python3
from probpy.distributions import multivariate_uniform

#Sampling
samples = 10000
a = np.array([-2, -1])
b = np.array([2, 3])
n = multivariate_uniform.sample(a, b, samples)


# PDF
X = ...
P = multivariate_uniform.p(X, a, b)
```

<p align="center">
  <img width=600px heigth=300px src="images/multi_uniform.png" />
</p>

#### Bernoulli

```python3
from probpy.distributions import bernoulli

#Sampling
samples = 10000
p = 0.7
n = bernoulli.sample(p, samples)

# PDF
bernoulli.p(p, 0.0)
```

<p align="center">
  <img width=600px heigth=300px src="images/bernoulli.png" />
</p>

#### Categorical

```python3
from probpy.distributions import categorical

#Sampling
samples = 10000
p = np.array([0.3, 0.6, 0.1])
n = categorical.sample(p, samples)

# Onehot categories
categorical.one_hot(n, 3)

# PDF
c = 2
categorical.p(c, p)

```

<p align="center">
  <img width=600px heigth=300px src="images/categorical.png" />
</p>

#### Dirichlet

```python3
from probpy.distributions import dirichlet

#Sampling
samples = 10000
alpha = np.array([2.0, 3.0])
n = dirichlet.sample(alpha, samples)


x = ...
p = dirichlet.p(x, alpha)

```

> Note that plot is a projection onto 1D 

<p align="center">
  <img width=600px heigth=300px src="images/dirichlet.png" />
</p>

#### Beta

```python3
from probpy.distributions import beta

#Sampling
samples = 10000
a, b = np.array(10.0), np.array(30.0)
n = beta.sample(a, b, samples)


x = np.linspace(0.01, 0.99, 100)
y = beta.p(x, a, b)

```

<p align="center">
  <img width=600px heigth=300px src="images/beta.png" />
</p>


#### Exponential

```python3
from probpy.distributions import exponential

#Sampling
lam = 2
samples = 10000
n = exponential.sample(lam, samples)


# PDF
x = np.linspace(0.0, 5, 100)
y = exponential.p(x, lam)

```

<p align="center">
  <img width=600px heigth=300px src="images/exponential.png" />
</p>

#### Binomial

```python3
from probpy.distributions import binomial

#Sampling
n, p = 20, 0.2
samples = 10000
_n = binomial.sample(n, p, samples)


# PDF
x = np.arange(0, 20)
y = binomial.p(x, n, p)

```

<p align="center">
  <img width=600px heigth=300px src="images/binomial.png" />
</p>

#### Multinomial

```python3
from probpy.distributions import multinomial

#Sampling
n, p = 20, np.array([0.3, 0.5, 0.2])
samples = 10000
_n = multinomial.sample(n, p, samples)

# PDF
x = np.array([[i, j, k] for i in range(n + 1) for j in range(n + 1) for k in range(n + 1) if i + j + k == n])
y = multinomial.p(x, n, p)

```

<p align="center">
  <img width=600px heigth=300px src="images/multinomial.png" />
</p>
