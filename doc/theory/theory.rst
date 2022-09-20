Principle of kernel-based sampling methods
==========================================

Let us denote by :math:`\\vect{X}_n = \\left\\{\\vect{x}^{(1)},\\ldots, \\vect{x}^{(n)}\\right\\} \\in \\cD_\\vect{X} \\subset \\Rset^p` 
the :math:`n`-sample of input realizations, also called input ``design of experiments'' (DoE) or simply design. 
Considering a costly function :math:`g:\\cD_\\vect{X} \\rightarrow \\Rset`, a first goal of the following DoE methods is 
to explore the input space in a space-filling way (e.g., to build a generic metamodel of :math:`g`). 
However, this work will exploit these methods with a specific purpose in mind: to numerically integrate $g$ 
against the probability density function :math:`f_{\\vect{X}}` which relates to a central tendency estimation of an output 
random variable :math:`Y=g(\\vect{X})`, resulting from an uncertainty propagation step.

Recently, other sampling methods based on the notions of discrepancy between distributions in a kernel-based 
functional space were used to approximate integrals. More precisely, one can mention the use 
of the distance called the *maximum mean discrepancy* (MMD) as a core ingredient of advanced sampling 
methods such as the *Support points* by (Mak & Joseph, 2018) and the *Kernel herding* by (Chen & al., 2010). 
The MMD is convenient to manipulate since it can simply be expressed using the underlying kernel arbitrarily arbitrary chosen. 
Let us setup the introduction of the Kernel herding and Support points methods by briefly defining a few mathematical concepts. 

References
----------
- Chen, Y., M. Welling, & A. Smola (2010). Super-samples from kernel herding. In Proceedings of the Twenty-Sixth
  Conference on Uncertainty in Artificial Intelligence, pp. 109 – 116.
- Mak, S. & V. R. Joseph (2018). Support points. The Annals of Statistics 46, 2562 – 2592.
- Fekhari, E., B. Iooss, J. Mure, L. Pronzato, & M. Rendas (2022). Model predictivity assessment: incremental
  test-set selection and accuracy evaluation. preprint.
- Briol, F.-X., C. Oates, M. Girolami, M. Osborne, & D. Sejdinovic (2019). Probabilistic Integration: A Role in
  Statistical Computation? Statistical Science 34, 1 – 22.
- Huszár, F. & D. Duvenaud (2012). Optimally-Weighted Herding is Bayesian Quadrature. In Proceedings of the
  Twenty-Eighth Conference on Uncertainty in Artificial Intelligence, pp. 377 – 386.
- Teymur, O., J. Gorham, M. Riabiz, & C. Oates (2021). Optimal quantisation of probability measures using 
  maximum mean discrepancy. In International Conference on Artificial Intelligence and Statistics, pp. 1027 – 1035.

