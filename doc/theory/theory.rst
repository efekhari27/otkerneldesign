Principle of kernel-based sampling methods
==========================================

This section presents kernel-based sampling methods developped in ``otkerneldesign`` for design of experiments, numerical integration and quantization.

**Introduction.**

Let us denote by :math:`\vect{X}_n = \left\{\vect{x}^{(1)},\ldots, \vect{x}^{(n)}\right\} \in \cD_\vect{X} \subset \Rset^p` 
the :math:`n`-sample of input realizations, also called input "design of experiments" (DoE) or simply design. 

From a computer experiment point of view, let us consider a costly function :math:`g:\cD_\vect{X} \rightarrow \Rset`, 
a first goal of the following DoE methods is to explore the input space in a space-filling way (e.g., to build a generic machine learning model of :math:`g`). 
However, this work will exploit these methods with a specific purpose in mind: to numerically integrate `g` 
against the probability density function :math:`f_{\vect{X}}` which relates to a central tendency estimation of an output 
random variable :math:`Y=g(\vect{X})`, resulting from an uncertainty propagation step.

Sampling methods based on the notion of discrepancy between distributions in a kernel-based 
functional space were used to approximate integrals. More precisely, one can mention the use 
of the distance called the *maximum mean discrepancy* (MMD) as a core ingredient of advanced sampling 
methods such as the *Support points* by (Mak & Joseph, 2018) and the *Kernel herding* by (Chen & al., 2010). 
Manipulating the MMD is convenient since its expression is simple ; it depends on an arbitrarily chosen kernel.  
Let us setup the introduction of the Kernel herding and Support points methods by briefly defining a few mathematical concepts. 

**Reproducing kernel Hilbert space.**

Assuming that :math:`k` is a symmetric and positive definite function :math:`k: \cD_\vect{X} \times \cD_\vect{X} \rightarrow \Rset`, 
latter called a "reproducing kernel" or simply a "kernel". A *reproducing kernel Hilbert space* (RKHS) is an inner product 
space :math:`\cH(k)` of functions :math:`g:\cD_\vect{X} \rightarrow \Rset` with the following properties:

* :math:`k(\cdot, \vect{x}) \in \cH(k), \quad \forall \vect{x} \in \cD_\vect{X}.`
* :math:`\langle g, k(\cdot, \vect{x}) \rangle_{\cH(k)} = g(\vect{x}), \quad \forall \vect{x} \in \cD_\vect{X}, \forall g \in \cH(k)`

Note that for a defined reproducing kernel, a unique RKHS exists and vice versa (see `C.Oates, 2021 <https://arxiv.org/pdf/2109.06075.pdf>`_ ).

**Potential.**

For any target distribution :math:`\mu`, its *potential* (also called "kernel mean embedding") associated with the kernel :math:`k` is defined as: 

.. math::
    :name: potential
    
    P_{\mu}(\vect{x}) := \int_{\cD_\vect{X}} k(\vect{x}, \vect{x}') \di \mu(\vect{x}').

Then, the potential of a discrete distribution :math:`\zeta_n = \frac1n \sum_{i=1}^{n} \delta(\vect{x}^{(i)})` 
(uniform mixture of Dirac distributions at the design points :math:`\vect{X}_n`) associated with the kernel :math:`k` can be expressed as:

.. math::
    :name: design_potential
    
    P_{\zeta_n}(\vect{x}) = \frac1n \sum_{i=1}^{n} k(\vect{x}, \vect{x}^{(i)}).

Close potentials can be interpreted to mean that the design :math:`\vect{X}_n` adequately quantizes :math:`\mu`

**Maximum mean discrepancy.**

A metric of discrepancy and quadrature error is offered by the MMD. 
This distance between two distributions :math:`\mu` and :math:`\zeta` is given by the 
maximal quadrature error committed for any function within the unit ball of an RKHS:

.. math::
    :name: mmd
    
    \mathrm{MMD}_k(\mu, \zeta) := 
    \sup_{\lVert g \lVert_{\cH(k)} \leq 1}
            \left | \int_{\cD_{\vect{X}}} g(\vect{x}) \di \mu(\vect{x}) - \int_{\cD_{\vect{X}}} g(\vect{x}) \di \zeta(\vect{x}) \right|.

Using the Cauchy-Schwartz inequality, one can demonstrate that 
:math:`\mathrm{MMD}_k(\mu, \zeta) = \left\lVert P_{\mu}(\vect{x}) - P_{\zeta}(\vect{x}) \right\lVert_{\cH(k)}` 
(see `C.Oates, 2021 <https://arxiv.org/pdf/2109.06075.pdf>`_ ). 
Moreover, a kernel :math:`k` is called characteristic if :math:`\mathrm{MMD}_k(\mu, \zeta) = 0` is equivalent to :math:`\mu = \zeta`.

Kernel herding
--------------
In this section we introduce the Kernel herding (KH) (Chen & al., 2010), a sampling method which intends to 
minimize a squared MMD by adding points iteratively. Considering a design :math:`\vect{X}_n` and its corresponding 
discrete distribution :math:`\zeta_n= \frac{1}{n} \sum_{i=1}^{n} \delta(\vect{x}^{(i)})`, a KH iteration can be written as 
an optimization over the point :math:`\vect{x}^{(n+1)} \in \cD_{\vect{X}}` of the following criterion:

.. math::
    :name: kh_criterion

    \vect{x}^{(n+1)} \in \argmin_{\vect{x} \in \mathcal{S}} \left(P_{\zeta_n}(\vect{x}) - P_{\mu}(\vect{x})\right)

considering a kernel :math:`k` and a given set :math:`\cS\subseteq\cD_{\vect{X}}` of candidate points 
(e.g., a fairly dense finite subset with size :math:`N \gg n` that emulates the target distribution). 
This compact criterion derives from the expression of a descent algorithm with respect to :math:`\vect{x}_{n+1}` 
(see (Pronzato & Zhigljavsky, 2020) for the full proof). 

In practice, :math:`P_{\mu}(\vect{x})` can be expressed analytically in the specific cases of input distribution and kernel 
(e.g., for independent uniform or normal inputs and a Matérn :math:`5/2` kernel (Fekhari & al., 2021)), making the computation very fast. 
Alternatively, the potential can be evaluated on an empirical measure :math:`\mu_N`, substituting :math:`\mu`, 
formed by a dense and large-size sample of :math:`\mu` (e.g., the candidate set :math:`\mathcal{S}`). 
:math:`P_{\mu}(\vect{x})` is then approached by :math:`P_{\mu_N}(\vect{x}) = (1/N)\, \sum_{j=1}^N k(\vect{x}, \vect{x}'^{(j)})`, 
which can be injected in :ref:`(4) <kh_criterion>` to solve the following optimization:

.. math::
  :name: kh_estimation

    \vect{x}^{(n+1)} \in \argmin_{\vect{x}\in\mathcal{S}} \left( \frac{1}{n} \sum_{i=1}^{n} k(\vect{x},\vect{x}^{(i)}) 
    - \frac{1}{N} \sum_{j=1}^N k(\vect{x},\vect{x}'^{(j)}) \right) \,.

When no observation is available, which is the common situation at the design stage, 
the kernel hyperparameters (e.g., correlation lengths) have to be set to heuristic values. 
MMD minimization is quite versatile and was explored in details by (Teymur & al., 2021)
or (Pronzato & Zhigljavsky, 2020), however the method is very sensitive to the kernel chosen and its tuning. 
Support points is a closely related method with a more rigid mathematical structure but interesting performances.

Greedy support points
---------------------
Support points (SP) (Mak & Joseph, 2018) are such that their associated empirical distribution 
:math:`\zeta_n` has minimum energy distance with respect to a target distribution :math:`\mu`. 
This criterion can be seen as a particular case of the MMD for the characteristic "energy-distance" kernel given by: 

.. math::
  :name: energy_kernel

  k_E(\vect{x},\vect{x}') = \frac{1}{2}\, \left(\| \vect{x} \| + \| \vect{x}' \| - \| \vect{x}-\vect{x}' \|\right)\,.

Compared to more heuristic methods for solving quantization problems, Support points
benefit from the theoretical guarantees of MMD minimization in terms of convergence of :math:`\zeta_n` to :math:`\mu` as :math:`n\to\infty`. 

At first sight, this optimization problem seems intractable, although (Mak & Joseph, 2018) propose to 
rewrite the function as a difference of convex functions in :math:`\vect{X}_n`, which yields a difference-of-convex program. 
To simplify the algorithm and keep an iterative design, a different approach will be used here. 
At iteration :math:`n+1`, the algorithm solves greedily the MMD minimization between :math:`\zeta_n` and :math:`\mu` for the candidate set :math:`\mathcal{S}`:

.. math::
    :name: greedy_criterion

    \vect{x}^{(n+1)} \in \argmin_{\vect{x}\in\mathcal{S}} \Bigg( \frac{1}{N} \sum_{j=1}^N \|\vect{x}-\vect{x}'^{(j)}\| 
    - \frac{1}{n+1} \sum_{i=1}^{n} \|\vect{x}-\vect{x}^{(i)}\| \Bigg) \,.

For this criterion, one can notice that it is almost identical to the KH one in :ref:`(4) <kh_criterion>` when 
taking as kernel the energy-distance kernel given in :ref:`(6) <energy_kernel>`.
These two iterative methods were exploited in (Fekhari & al., 2021) to study new ways to construct 
a validation set for machine learning models by conveniently selecting a test set for a better model performance estimation.

References
----------
- Chen, Y., M. Welling, & A. Smola (2010). Super-samples from kernel herding. In Proceedings of the Twenty-Sixth
  Conference on Uncertainty in Artificial Intelligence, pp. 109 – 116. `PDF <https://arxiv.org/pdf/1203.3472.pdf>`_
- Mak, S. & V. R. Joseph (2018). Support points. The Annals of Statistics 46, 2562 – 2592. `PDF <https://projecteuclid.org/journals/annals-of-statistics/volume-46/issue-6A/Support-points/10.1214/17-AOS1629.full>`_
- Fekhari, E., B. Iooss, J. Mure, L. Pronzato, & M. Rendas (2022). Model predictivity assessment: incremental
  test-set selection and accuracy evaluation. preprint. `PDF <https://hal.archives-ouvertes.fr/hal-03523695/document>`_
- Briol, F.-X., C. Oates, M. Girolami, M. Osborne, & D. Sejdinovic (2019). Probabilistic Integration: A Role in
  Statistical Computation? Statistical Science 34, 1 – 22. `PDF <https://projecteuclid.org/journals/statistical-science/volume-34/issue-1/Rejoinder-Probabilistic-Integration-A-Role-in-Statistical-Computation/10.1214/18-STS683.full>`_
- Pronzato, L. & A. Zhigljavsky (2020). Bayesian quadrature and energy minimization for space-filling design.
  SIAM/ASA Journal on Uncertainty Quantification 8, 959 – 1011 `PDF <https://hal.archives-ouvertes.fr/hal-01864076v3/document>`_
- Huszár, F. & D. Duvenaud (2012). Optimally-Weighted Herding is Bayesian Quadrature. In Proceedings of the
  Twenty-Eighth Conference on Uncertainty in Artificial Intelligence, pp. 377 – 386. `PDF <https://arxiv.org/pdf/1204.1664.pdf>`_
- Teymur, O., J. Gorham, M. Riabiz, & C. Oates (2021). Optimal quantisation of probability measures using 
  maximum mean discrepancy. In International Conference on Artificial Intelligence and Statistics, pp. 1027 – 1035. `PDF <http://proceedings.mlr.press/v130/teymur21a/teymur21a.pdf>`_

