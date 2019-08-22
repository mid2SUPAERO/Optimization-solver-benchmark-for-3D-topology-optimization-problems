# GCMMA OC
A self-contained C++ implementation of MMA and GCMMA
# OC and GCMMA
*A self-contained C++ implementation of MMA and GCMMA.*

[![Build Status](https://travis-ci.com/jdumas/mma.svg?token=euzAY1sxC114E8ufzcZx&branch=master)](https://travis-ci.com/jdumas/mma)

This repository contains single-file C++ implementations of GCMMA and OC using MPI and PETSc, as described in [1,2,3,5].
The code in this repository is based on an original code by Niels Aage and Bendsøe , using the subproblem solver described in [4,5].


## Other Projects

- [TopOpt_in_PETSc](https://github.com/topopt/TopOpt_in_PETSc): Multi-processor implementation of MMA using MPI and PETSc.
- [NLopt](https://nlopt.readthedocs.io/en/latest/): Open-source library for nonlinear optimization with various bindings in different languages. Contains a different implementation of GCMMA.

## References

1. Svanberg, K. (1987). “The Method of Moving Asymptotes—a New Method for Structural Optimization”. International Journal for Numerical Methods in Engineering, 24(2), 359–373. doi : 10.1002/nme.1620240207.
2. Svanberg, K. (2002). “A Class of Globally Convergent Optimization Methods Based on Conservative Convex Separable Approximations”. SIAM Journal on Optimization, 12(2), 555–573. doi : 10.1137/s1052623499362822.
3. Svanberg, K. (2007). MMA and GCMMA - Two Methods for Nonlinear Optimization. Technical report. eprint: https://people.kth.se/~krille/mmagcmma.pdf
4. Aage, N., & Lazarov, B. S. (2013). Parallel framework for topology optimization using the method of moving asymptotes. Structural and Multidisciplinary Optimization, 47(4), 493–505. https://doi.org/10.1007/s00158-012-0869-2
5. M.P Bendsøe. Optimal shape design as a material distribution problem.
Structural Optimization, 1 :192–202, 1995
