# Analytical potentials 

This note describes how to analytically express potentials for one dimensional input distributions modified by translation and scaling. Using the calculation for uniform $\mathcal{U}[0, 1]$ and standard normal $\mathcal{N}(0, 1)$ with the Matérn kernel with smoothness parameter $5/2$, the following extensions allows us to apply them to any uniform or normal distribution. 


Let us define the potential of the measure $\mu$ w.r.t. a stationary kernel $k_\theta$ of scale parameter $\theta$ (also called correlation length) 
$$
P_{k_\theta, \mu}(x) := \int k_\theta(x, x') d \mu(x').
$$

## Invariance by tranlation
Let $T$ be the a real number, one can define $\mu + T$ as the pushforward measure of $\mu$ by the translation $t \mapsto t + T$. Then, using the pushforward measure definition and the kernel's stationarity: 
$$
\begin{align*}
P_{k_\theta, \mu + T}(x + T) &= \int k_\theta(x+T, x') d (\mu + T)(x')\\
        &= \int k_\theta(x+T, x') d\mu(x' - T)\\
        &= \int k_\theta(x+T, t+T) d\mu(t)\\
        &= \int k_\theta(x, t) d\mu(t)\\
        &= P_{k_\theta, \mu}(x).
\end{align*}
$$

## Rescaling

Let $\lambda$ be a positive real number, one can define $\lambda \mu$ as the pushforward measure of $\mu$ by the scaling function $t \mapsto \lambda t$. Since $k_\theta$ is a stationary kernel of scale $\theta$, there is a function $f$ such that for any real numbers $x, x'$, $k_\theta(x, x') = f\left(\frac{x - x'}{\theta}\right)$. 

$$
\begin{align*}
P_{k_\theta, \lambda \mu}(\lambda x) &= \int k_\theta(\lambda x, x') d (\lambda \mu)(x')\\
&= \frac 1 \lambda  \int k_\theta(\lambda x, x') d\mu \left(\frac{x'}{\lambda} \right)\\
&= \frac 1 \lambda  \int k_\theta(\lambda x, \lambda t) d\mu (t)\\

&= \frac 1 \lambda  \int f\left(\frac  {\lambda x - \lambda t}{\theta} \right) d\mu (t)\\
&= \frac 1 \lambda  \int f\left(\frac  {x - t}{\frac \theta \lambda} \right) d\mu (t)\\
&= \frac 1 \lambda  \int k_{\frac \theta \lambda}(x, t) d\mu (t)\\
&= \frac 1 \lambda P_{k_{\frac \theta \lambda}, \mu}(x)
\end{align*}
$$

