#ifndef __MMA__
#define __MMA__

#include <petsc.h>

class OC{
public:

  // Construct using defaults subproblem penalization
  OC(PetscInt m);

  // Destructor
  ~OC();

  // Set and solve a subproblem: return new xval
  PetscErrorCode Update(Vec xval, Vec dfdx, PetscScalar *gx, Vec *dgdx, Vec xmin, Vec xmax, PetscScalar volfrac);

  // Sets outer movelimits on all primal design variables
  // This is often requires to prevent the solver from oscilating
  PetscErrorCode SetOuterMovelimit(PetscScalar Xmin, PetscScalar Xmax, PetscScalar movelim, Vec x, Vec xmin, Vec xmax);

  // Inf norm on diff between two vectors: SHOULD NOT BE HERE - USE BASIC PETSc!!!!!
  PetscScalar DesignChange(Vec x, Vec xold);

private:

  PetscInt m; // Number of constraints

  // Global iteration counter
  PetscInt k;

  // Math helpers
  PetscScalar Min(PetscScalar d1, PetscScalar d2);
  PetscScalar Max(PetscScalar d1, PetscScalar d2);
  PetscInt Min(PetscInt d1, PetscInt d2);
  PetscInt Max(PetscInt d1, PetscInt d2);
  PetscScalar Abs(PetscScalar d1);

};


#endif
