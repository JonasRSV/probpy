![Probpy](images/logo.png)
---

[Documentation](https://jonasrsv.github.io/probpy)

A Probability Library 

Try it out
---

With docker installed run the command and browse the notebooks

```bash
sh demo_env.sh
```

Installation
---

TODO
```bash
...
```

Introduction
---

The fundamental building block is the RandomVariable

Random Variables can be created from any [distribution](https://jonasrsv.github.io/probpy/build/html/distributions.html) using the "med" function. "med" is Swedish for with. All random variables has a sampling function and a pdf/pmf function.
```python
import probpy as pp

rv = pp.normal.med(mu=0.0, sigma=1.0)

samples = rv.sample(size=5)
density = rv.p(samples)

print(samples)
print(density)
```

```bash
[ 0.56202855 -0.63491592 -1.11501445  0.10756346  1.40117568]
[0.34065788 0.32611746 0.21425955 0.39664108 0.14948112]
```

***Conditional Distributions***

Variables can be created with partial arguments to represent conditional distributions.
```python
import probpy as pp
rv = pp.normal.med(sigma=1.0)

print(rv.sample(0.0, size=5))
print(rv.sample(4.0, size=5))
```

```bash
[ 0.24050695  0.19103947  1.01564618 -0.37190388 -0.04080893]
[5.24490748 3.72506806 3.59844073 4.71898881 2.47418571]
```


These variables can be used in many functions in this library. Things ranging from estimating ***parameter posteriors***, ***predictive posteriors***, ***integration***, ***MCMC*** finding **modes** and more to come. 

check out the [notebooks](https://github.com/JonasRSV/probpy/tree/master/notebooks) for some usage examples. if you have docker installed just run

```bash
sh demo_env.sh
```

