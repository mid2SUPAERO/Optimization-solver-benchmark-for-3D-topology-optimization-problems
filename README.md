# Optimization-solver-benchmark-for-3D-topology-optimization-problems
#
  
The purpose of this internship is to benchmark different optimization solvers when applied to various ﬁnite element based structural topology optimization problems. An extensive and representative library of minimum compliance minimum compliance, and minimum volume problem instances for different sizes is developed for this benchmarking. The problems are based on a material interpolation scheme combined with a PDE ﬁlter. 
Different optimization solvers including Optimality Criteria (OC), the Method of Moving Asymptotes (MMA) and its globally convergent version GCMMA, the interior point solvers in IPOPT and the sequential quadratic programming method in NLOPT, are benchmarked on the library using performance proﬁles and data profiles.  Whenever possible the methods are applied to nested formulations of the problem.




En français :
Le but du stage est d'évaluer et comparer les performances de plusieurs solveurs d'optimisation (OC, MMA, GCMMA, Optimiseur dans IPOPT,NLOPT), quand appliqués à divers problèmes d'optimisation topologique de structures mécaniques sous contraintes dans le cadre de l'élasticité linéaire. Une bibliothèque complète, représentative d'une classe de problèmes de compliance minimale et de minimum volume d'une structure élastique pour différentes tailles d'éléments finis (E.F) est devéloppée pour cette analyse comparative.  Quand c'est possible les solveurs sont comparés quand appliqués à des problèmes aux formulations Nested.
