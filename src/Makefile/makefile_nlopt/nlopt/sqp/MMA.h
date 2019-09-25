#ifndef __MMA__
#define __MMA__

#include <petsc.h>
#include <nlopt.hpp>
#include <math.h>
#include <stdlib.h>
#include <string.h>


static const int c__0 = 0;
static const int c__1 = 1;
static const int c__2 = 2;

typedef struct {
    double t, f0, h1, h2, h3, h4;
    int n1, n2, n3;
    double t0, gs;
    double tol;
    int line;
    double alpha;
    int iexact;
    int incons, ireset, itermx;
    double *x0;
} slsqpb_state;

#define SS(var) state->var = var
#define SAVE_STATE \
     SS(t); SS(f0); SS(h1); SS(h2); SS(h3); SS(h4);	\
     SS(n1); SS(n2); SS(n3); \
     SS(t0); SS(gs); \
     SS(tol); \
     SS(line); \
     SS(alpha); \
     SS(iexact); \
     SS(incons); SS(ireset); SS(itermx)

#define RS(var) var = state->var
#define RESTORE_STATE \
     RS(t); RS(f0); RS(h1); RS(h2); RS(h3); RS(h4);	\
     RS(n1); RS(n2); RS(n3); \
     RS(t0); RS(gs); \
     RS(tol); \
     RS(line); \
     RS(alpha); \
     RS(iexact); \
     RS(incons); RS(ireset); RS(itermx)

class OC{
public:

  // Construct using defaults subproblem penalization
  OC(PetscInt m);

  // Destructor
  ~OC();
  

  // Set and solve a subproblem: return new xval
  PetscErrorCode Update(Vec xval, PetscScalar fx, Vec dfdx, PetscScalar *gx, Vec *dgdx, Vec xmin, Vec xmax,
				double *c, double * grad, double * cgrad, double *xcur, double *w, int *jw,int mpi, int pi,
				int mpi1,int ni, double fcur,double acc, int iter, int mode, int lew_w, int len_jw, 
				slsqpb_state state);

  // Sets outer movelimits on all primal design variables
  // This is often requires to prevent the solver from oscilating
  PetscErrorCode SetOuterMovelimit(PetscScalar Xmin, PetscScalar Xmax, PetscScalar movelim, Vec x, Vec xmin, Vec xmax);

  // Inf norm on diff between two vectors: SHOULD NOT BE HERE - USE BASIC PETSc!!!!!
  PetscScalar DesignChange(Vec x, Vec xold);
  double d_sign(double a, double s) { return s < 0 ? -a : a; }
  void length_work(int *LEN_W, int *LEN_JW, int M, int MEQ, int N);
  
private:

/**************************************************************************/
  int nlopt_isinf(double x) {
     	return fabs(x) >= HUGE_VAL * 0.99
	#ifdef HAVE_ISINF
	  	|| isinf(x)
	#endif
	  	;
	}

  void h12_(const int *mode, int *lpivot, int *l1, 
		 int *m, double *u, const int *iue, double *up, 
		double *c__, const int *ice, const int *icv, const int *ncv);
  void nnls_(double *a, int *mda, int *m, int *
	n, double *b, double *x, double *rnorm, double *w, 
		double *z__, int *indx, int *mode);
  void ldp_(double *g, int *mg, int *m, int *n, 
	double *h__, double *x, double *xnorm, double *w, 
	int *indx, int *mode);
  void lsi_(double *e, double *f, double *g, 
	double *h__, int *le, int *me, int *lg, int *mg, 
	int *n, double *x, double *xnorm, double *w, int *
	jw, int *mode);
  void hfti_(double *a, int *mda, int *m, int *
	n, double *b, int *mdb, const int *nb, double *tau, int 
	*krank, double *rnorm, double *h__, double *g, int *ip);
  void lsei_(double *c__, double *d__, double *e, 
	double *f, double *g, double *h__, int *lc, int *
	mc, int *le, int *me, int *lg, int *mg, int *n, 
	double *x, double *xnrm, double *w, int *jw, int *mode);
   void lsq_(int *m, int *meq, int *n, int *nl, 
	int *la, double *l, double *g, double *a, double *
	b, const double *xl, const double *xu, double *x, double *y, 
	double *w, int *jw, int *mode);
  void ldl_(int *n, double *a, double *z__, 
	double *sigma, double *w);
  void slsqpb_(int *m, int *meq, int *la, int *
		    n, double *x, const double *xl, const double *xu, double *f, 
		    double *c__, double *g, double *a, double *acc, 
		    int *iter, int *mode, double *r__, double *l, 
		    double *x0, double *mu, double *s, double *u, 
		    double *v, double *w, int *iw, slsqpb_state *state);
  void slsqp(int *m, int *meq, int *la, int *n,
		  double *x, const double *xl, const double *xu, double *f, 
		  double *c__, double *g, double *a, double *acc, 
		  int *iter, int *mode, double *w, int *l_w__, int *
		jw, int *l_jw__, slsqpb_state *state);
  
  

  



/**************************************************************************/

  PetscInt m; // Number of constraints

  // Global iteration counter
  PetscInt k;
  
 
 
  // Math helpers
  PetscScalar Min(PetscScalar d1, PetscScalar d2);
  PetscScalar Max(PetscScalar d1, PetscScalar d2);
  PetscInt Min(PetscInt d1, PetscInt d2);
  PetscInt Max(PetscInt d1, PetscInt d2);
  PetscScalar Abs(PetscScalar d1);

/********************************* BLAS1 routines *************************/
  void dcopy___(int *n_, const double *dx, int incx, double *dy, int incy);	
  void daxpy_sl__(int *n_, const double *da_, const double *dx, int incx, double *dy, int incy);
  double ddot_sl__(int *n_, double *dx, int incx, double *dy, int incy);
  double dnrm2___(int *n_, double *dx, int incx);
  void dsrot_(int n, double *dx, int incx, double *dy, int incy, double *c__, double *s_);
  void dsrotg_(double *da, double *db, double *c, double *s);
  void dscal_sl__(int *n_, const double *da, double *dx, int incx);
/**************************************************************************/
};


#endif
