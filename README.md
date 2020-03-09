WIP
---

End result is hopefully a PGM module built with numpy



Documentation
---

- [Core](#core)
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





## Core 

Central functionality


### Distributions

PDFs / PMFs and sampling functions


#### Normal

```python3
from probpy.distributions import normal

# Sampling
samples = 10000
mu, sigma = 0, 0.1
n = normal.sample(mu, sigma, samples)


# PDF
x = np.linspace(-4, 4, 100)
n = normal.p(mu, sigma, x)
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
P = multivariate_normal.p(mu, sigma, X)
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
n = uniform.p(a, b, x)
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
P = multivariate_uniform.p(a, b, X)
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
categorical.p(p, c)

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
p = dirichlet.p(alpha, x)

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
y = beta.p(a, b, x)

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
y = exponential.p(lam, x)

```

<p align="center">
  <img width=600px heigth=300px src="images/exponential.png" />
</p>
