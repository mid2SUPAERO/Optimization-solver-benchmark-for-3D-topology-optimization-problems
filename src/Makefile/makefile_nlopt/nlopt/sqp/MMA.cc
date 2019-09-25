// -*- coding: utf-8; tab-width: 4; indent-tabs-mode: t; c-basic-offset: 4 -*-

#include <MMA.h>
#include <iostream>
#include <math.h>

using namespace std;

// Constructor
OC::OC(PetscInt mm){
	/*
	 * inputs
	 * mm  : Number of constraints
	 */
	m = mm;
	k = 0;  /* bi-section counter, we dont restart */

	/* OC is only valid only for a signle equality constraint equation */
	if (m != 1)
		PetscPrintf(PETSC_COMM_WORLD,"OC only valid for a single constraint");
}

// Set and solve a subproblem: return new xval
PetscErrorCode OC::Update(Vec xval, PetscScalar fx, Vec dfdx, PetscScalar *gx, Vec *dgdx,
				Vec xmin, Vec xmax,double *c, double *grad, double * cgrad, double *xcur, double *w, int *jw,
				int mpi, int pi,int mpi1,int ni, double fcur,double acc, int iter, int mode, 
				int len_w, int len_jw, slsqpb_state state){
	/*
	 * inputs
	 * xval : Design variables (vector)
	 * dfdx : Sensitivities of objective (vector)
	 * gx   : Array with constraint values (array)
	 * dgdx : Sensitivities of constraints (array of of vectors)
	 * xmin : Lower limit of design variables (vector)
	 * xmax : Upper limit of design variables (vector)
	 *
	 * output
	 * xval : Updated design variables
	 */

	PetscErrorCode ierr=0;
	PetscInt       nglob, nloc, nloc1, nloc2, nloc3, nloc4; /* vector size */
	PetscScalar    *pxval,*pxold,*pdfdx,*pdgdx;  /* pointers to local data */
	double    *pxmin,*pxmax;  /* pointers to local data */
	Vec            xold;
	


	/* OC is only valid only for a signle equality constraint equation */
	if (m != 1){
		PetscPrintf(PETSC_COMM_WORLD,"OC only valid for a single constraint");
		ierr = -1;CHKERRQ(ierr);
	}

	/* copy xval to xold */
	ierr = VecDuplicate(xval, &xold); CHKERRQ(ierr);
	ierr = VecCopy(xval, xold); CHKERRQ(ierr);

	/* get number of local entries */
	ierr = VecGetLocalSize(xval,&nloc);CHKERRQ(ierr);
	ierr = VecGetLocalSize(dfdx,&nloc1);CHKERRQ(ierr);
	ierr = VecGetLocalSize(dgdx[0],&nloc2);CHKERRQ(ierr);
	ierr = VecGetLocalSize(xmin,&nloc3);CHKERRQ(ierr);
	ierr = VecGetLocalSize(xmax,&nloc4);CHKERRQ(ierr);

	/* check that lenghts match */
	if ( nloc != nloc1 && nloc != nloc2 && nloc != nloc3 && nloc != nloc4 )
		ierr = -1; CHKERRQ(ierr);

	/* get global number of global entries aka number of elements */
	ierr = VecGetSize(xval, &nglob);

	/* get pointers to local data that will not be changed within loop */
	ierr = VecGetArray(xold,&pxold);CHKERRQ(ierr);
	ierr = VecGetArray(dfdx,&pdfdx);CHKERRQ(ierr);
	ierr = VecGetArray(dgdx[0],&pdgdx);CHKERRQ(ierr);
	ierr = VecGetArray(xmin,&pxmin);CHKERRQ(ierr);
	ierr = VecGetArray(xmax,&pxmax);CHKERRQ(ierr);

	
	pxval = xcur;
	fcur  = fx;
	gx    = c;
	pdfdx = grad;
	pdgdx = cgrad;
	
	
	slsqp(&mpi, &pi, &mpi1, &ni,
		pxval, pxmin, pxmax, &fcur,
		gx, pdfdx, pdgdx,
		&acc, &iter, &mode,
		w, &len_w, jw, &len_jw,
		&state);
	
	
	
	
	/* restore local data that has been modified for collective operations */
	ierr = VecRestoreArray(xval,&pxval);CHKERRQ(ierr);


	/* clean up */
	
	VecRestoreArray(xold,&pxold);
	VecRestoreArray(dfdx,&pdfdx);
	VecRestoreArray(dgdx[0],&pdgdx);
	VecRestoreArray(xmin,&pxmin);
	VecRestoreArray(xmax,&pxmax);
	VecDestroy(&xold);

	

	return ierr;
}

/* this function sets vectors xmin, xmax to be used in Update due from vector x,
 * scalars Xmax, Xmin, movelim */
PetscErrorCode OC::SetOuterMovelimit(PetscScalar Xmin, PetscScalar Xmax,
									 PetscScalar movlim, Vec x, Vec xmin, Vec xmax){

	PetscErrorCode ierr=0;

	PetscScalar *xv,*xmiv,*xmav;
	PetscInt nloc;
	VecGetLocalSize(x,&nloc);
	VecGetArray(x,&xv);
	VecGetArray(xmin,&xmiv);
	VecGetArray(xmax,&xmav);
	for (PetscInt i=0;i<nloc;i++){
		xmav[i] = Min(Xmax, xv[i] + movlim);
		xmiv[i] = Max(Xmin, xv[i] - movlim);
	}
	VecRestoreArray(x,&xv);
	VecRestoreArray(xmin,&xmiv);
	VecRestoreArray(xmax,&xmav);
	return ierr;
}

PetscScalar OC::DesignChange(Vec x, Vec xold){


	PetscScalar *xv, *xo;
	PetscInt nloc;
	VecGetLocalSize(x,&nloc);
	VecGetArray(x,&xv);
	VecGetArray(xold,&xo);
	PetscScalar ch = 0.0;
	for (PetscInt i=0;i<nloc;i++){
		ch = PetscMax(ch,PetscAbsReal(xv[i]-xo[i]));
		xo[i] = xv[i];
	}
	PetscScalar tmp;
	MPI_Allreduce(&ch, &tmp, 1,MPIU_SCALAR, MPI_MAX,PETSC_COMM_WORLD );
	ch = tmp;
	VecRestoreArray(x,&xv);
	VecRestoreArray(xold,&xo);

	return(ch);

}

OC::~OC(){

}



// PRIVATE METHODS

void OC::h12_(const int *mode, int *lpivot, int *l1, 
		 int *m, double *u, const int *iue, double *up, 
		double *c__, const int *ice, const int *icv, const int *ncv){
	
	/* Initialized data */

    const double one = 1.;

    /* System generated locals */
    int u_dim1, u_offset, i__1, i__2;
    double d__1;

    /* Local variables */
    double b;
    int i__, j, i2, i3, i4;
    double cl, sm;
    int incr;
    double clinv;
    /* Parameter adjustments */
    u_dim1 = *iue;
    u_offset = 1 + u_dim1;
    u -= u_offset;
    --c__;

    /* Function Body */
    if (0 >= *lpivot || *lpivot >= *l1 || *l1 > *m) {
	goto L80;
    }
    cl = (d__1 = u[*lpivot * u_dim1 + 1], fabs(d__1));
    if (*mode == 2) {
	goto L30;
    }
/*     ****** CONSTRUCT THE TRANSFORMATION ****** */
    i__1 = *m;
    for (j = *l1; j <= i__1; ++j) {
	sm = (d__1 = u[j * u_dim1 + 1], fabs(d__1));
/* L10: */
	cl = MAX(sm,cl);
    }
    if (cl <= 0.0) {
	goto L80;
    }
    clinv = one / cl;
/* Computing 2nd power */
    d__1 = u[*lpivot * u_dim1 + 1] * clinv;
    sm = d__1 * d__1;
    i__1 = *m;
    for (j = *l1; j <= i__1; ++j) {
/* L20: */
/* Computing 2nd power */
	d__1 = u[j * u_dim1 + 1] * clinv;
	sm += d__1 * d__1;
    }
    cl *= sqrt(sm);
    if (u[*lpivot * u_dim1 + 1] > 0.0) {
	cl = -cl;
    }
    *up = u[*lpivot * u_dim1 + 1] - cl;
    u[*lpivot * u_dim1 + 1] = cl;
    goto L40;
/*     ****** APPLY THE TRANSFORMATION  I+U*(U**T)/B  TO C ****** */
L30:
    if (cl <= 0.0) {
	goto L80;
    }
L40:
    if (*ncv <= 0) {
	goto L80;
    }
    b = *up * u[*lpivot * u_dim1 + 1];
    if (b >= 0.0) {
	goto L80;
    }
    b = one / b;
    i2 = 1 - *icv + *ice * (*lpivot - 1);
    incr = *ice * (*l1 - *lpivot);
    i__1 = *ncv;
    for (j = 1; j <= i__1; ++j) {
	i2 += *icv;
	i3 = i2 + incr;
	i4 = i3;
	sm = c__[i2] * *up;
	i__2 = *m;
	for (i__ = *l1; i__ <= i__2; ++i__) {
	    sm += c__[i3] * u[i__ * u_dim1 + 1];
/* L50: */
	    i3 += *ice;
	}
	if (sm == 0.0) {
	    goto L70;
	}
	sm *= b;
	c__[i2] += sm * *up;
	i__2 = *m;
	for (i__ = *l1; i__ <= i__2; ++i__) {
	    c__[i4] += sm * u[i__ * u_dim1 + 1];
/* L60: */
	    i4 += *ice;
	}
L70:
	;
    }
L80:
    return;
} /* h12_ */

void OC::nnls_(double *a, int *mda, int *m, int *
	n, double *b, double *x, double *rnorm, double *w, 
		double *z__, int *indx, int *mode){

	/* Initialized data */

    const double one = 1.;
    const double factor = .01;

    /* System generated locals */
    int a_dim1, a_offset, i__1, i__2;
    double d__1;

    /* Local variables */
    double c__;
    int i__, j, k, l;
    double s, t;
    int ii, jj, ip, iz, jz;
    double up;
    int iz1, iz2, npp1, iter;
    double wmax, alpha, asave;
    int itmax, izmax, nsetp;
    double unorm;

    /* Parameter adjustments */
    --z__;
    --b;
    --indx;
    --w;
    --x;
    a_dim1 = *mda;
    a_offset = 1 + a_dim1;
    a -= a_offset;

    /* Function Body */
/*     revised          Dieter Kraft, March 1983 */
    *mode = 2;
    if (*m <= 0 || *n <= 0) {
	goto L290;
    }
    *mode = 1;
    iter = 0;
    itmax = *n * 3;
/* STEP ONE (INITIALIZE) */
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
/* L100: */
	indx[i__] = i__;
    }
    iz1 = 1;
    iz2 = *n;
    nsetp = 0;
    npp1 = 1;
    x[1] = 0.0;
    dcopy___(n, &x[1], 0, &x[1], 1);
/* STEP TWO (COMPUTE DUAL VARIABLES) */
/* .....ENTRY LOOP A */
L110:
    if (iz1 > iz2 || nsetp >= *m) {
	goto L280;
    }
    i__1 = iz2;
    for (iz = iz1; iz <= i__1; ++iz) {
	j = indx[iz];
/* L120: */
	i__2 = *m - nsetp;
	w[j] = ddot_sl__(&i__2, &a[npp1 + j * a_dim1], 1, &b[npp1], 1)
		;
    }
/* STEP THREE (TEST DUAL VARIABLES) */
L130:
    wmax = 0.0;
    i__2 = iz2;
    for (iz = iz1; iz <= i__2; ++iz) {
	j = indx[iz];
	if (w[j] <= wmax) {
	    goto L140;
	}
	wmax = w[j];
	izmax = iz;
L140:
	;
    }
/* .....EXIT LOOP A */
    if (wmax <= 0.0) {
	goto L280;
    }
    iz = izmax;
    j = indx[iz];
/* STEP FOUR (TEST INDX J FOR LINEAR DEPENDENCY) */
    asave = a[npp1 + j * a_dim1];
    i__2 = npp1 + 1;
    h12_(&c__1, &npp1, &i__2, m, &a[j * a_dim1 + 1], &c__1, &up, &z__[1], &
	    c__1, &c__1, &c__0);
    unorm = dnrm2___(&nsetp, &a[j * a_dim1 + 1], 1);
    t = factor * (d__1 = a[npp1 + j * a_dim1], fabs(d__1));
    d__1 = unorm + t;
    if (d__1 - unorm <= 0.0) {
	goto L150;
    }
    dcopy___(m, &b[1], 1, &z__[1], 1);
    i__2 = npp1 + 1;
    h12_(&c__2, &npp1, &i__2, m, &a[j * a_dim1 + 1], &c__1, &up, &z__[1], &
	    c__1, &c__1, &c__1);
    if (z__[npp1] / a[npp1 + j * a_dim1] > 0.0) {
	goto L160;
    }
L150:
    a[npp1 + j * a_dim1] = asave;
    w[j] = 0.0;
    goto L130;
/* STEP FIVE (ADD COLUMN) */
L160:
    dcopy___(m, &z__[1], 1, &b[1], 1);
    indx[iz] = indx[iz1];
    indx[iz1] = j;
    ++iz1;
    nsetp = npp1;
    ++npp1;
    i__2 = iz2;
    for (jz = iz1; jz <= i__2; ++jz) {
	jj = indx[jz];
/* L170: */
	h12_(&c__2, &nsetp, &npp1, m, &a[j * a_dim1 + 1], &c__1, &up, &a[jj * 
		a_dim1 + 1], &c__1, mda, &c__1);
    }
    k = MIN(npp1,*mda);
    w[j] = 0.0;
    i__2 = *m - nsetp;
    dcopy___(&i__2, &w[j], 0, &a[k + j * a_dim1], 1);
/* STEP SIX (SOLVE LEAST SQUARES SUB-PROBLEM) */
/* .....ENTRY LOOP B */
L180:
    for (ip = nsetp; ip >= 1; --ip) {
	if (ip == nsetp) {
	    goto L190;
	}
	d__1 = -z__[ip + 1];
	daxpy_sl__(&ip, &d__1, &a[jj * a_dim1 + 1], 1, &z__[1], 1);
L190:
	jj = indx[ip];
/* L200: */
	z__[ip] /= a[ip + jj * a_dim1];
    }
    ++iter;
    if (iter <= itmax) {
	goto L220;
    }
L210:
    *mode = 3;
    goto L280;
/* STEP SEVEN TO TEN (STEP LENGTH ALGORITHM) */
L220:
    alpha = one;
    jj = 0;
    i__2 = nsetp;
    for (ip = 1; ip <= i__2; ++ip) {
	if (z__[ip] > 0.0) {
	    goto L230;
	}
	l = indx[ip];
	t = -x[l] / (z__[ip] - x[l]);
	if (alpha < t) {
	    goto L230;
	}
	alpha = t;
	jj = ip;
L230:
	;
    }
    i__2 = nsetp;
    for (ip = 1; ip <= i__2; ++ip) {
	l = indx[ip];
/* L240: */
	x[l] = (one - alpha) * x[l] + alpha * z__[ip];
    }
/* .....EXIT LOOP B */
    if (jj == 0) {
	goto L110;
    }
/* STEP ELEVEN (DELETE COLUMN) */
    i__ = indx[jj];
L250:
    x[i__] = 0.0;
    ++jj;
    i__2 = nsetp;
    for (j = jj; j <= i__2; ++j) {
	ii = indx[j];
	indx[j - 1] = ii;
	dsrotg_(&a[j - 1 + ii * a_dim1], &a[j + ii * a_dim1], &c__, &s);
	t = a[j - 1 + ii * a_dim1];
	dsrot_(*n, &a[j - 1 + a_dim1], *mda, &a[j + a_dim1], *mda, &c__, &s);
	a[j - 1 + ii * a_dim1] = t;
	a[j + ii * a_dim1] = 0.0;
/* L260: */
	dsrot_(1, &b[j - 1], 1, &b[j], 1, &c__, &s);
    }
    npp1 = nsetp;
    --nsetp;
    --iz1;
    indx[iz1] = i__;
    if (nsetp <= 0) {
	goto L210;
    }
    i__2 = nsetp;
    for (jj = 1; jj <= i__2; ++jj) {
	i__ = indx[jj];
	if (x[i__] <= 0.0) {
	    goto L250;
	}
/* L270: */
    }
    dcopy___(m, &b[1], 1, &z__[1], 1);
    goto L180;
/* STEP TWELVE (SOLUTION) */
L280:
    k = MIN(npp1,*m);
    i__2 = *m - nsetp;
    *rnorm = dnrm2___(&i__2, &b[k], 1);
    if (npp1 > *m) {
	w[1] = 0.0;
	dcopy___(n, &w[1], 0, &w[1], 1);
    }
/* END OF SUBROUTINE NNLS */
L290:
    return;
} /* nnls_ */

void OC::ldp_(double *g, int *mg, int *m, int *n, 
	double *h__, double *x, double *xnorm, double *w, 
	int *indx, int *mode){

	/* Initialized data */

    const double one = 1.;

    /* System generated locals */
    int g_dim1, g_offset, i__1, i__2;
    double d__1;

    /* Local variables */
    int i__, j, n1, if__, iw, iy, iz;
    double fac;
    double rnorm;
    int iwdual;

    /* Parameter adjustments */
    --indx;
    --h__;
    --x;
    g_dim1 = *mg;
    g_offset = 1 + g_dim1;
    g -= g_offset;
    --w;

    /* Function Body */
    *mode = 2;
    if (*n <= 0) {
	goto L50;
    }
/*  STATE DUAL PROBLEM */
    *mode = 1;
    x[1] = 0.0;
    dcopy___(n, &x[1], 0, &x[1], 1);
    *xnorm = 0.0;
    if (*m == 0) {
	goto L50;
    }
    iw = 0;
    i__1 = *m;
    for (j = 1; j <= i__1; ++j) {
	i__2 = *n;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    ++iw;
/* L10: */
	    w[iw] = g[j + i__ * g_dim1];
	}
	++iw;
/* L20: */
	w[iw] = h__[j];
    }
    if__ = iw + 1;
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	++iw;
/* L30: */
	w[iw] = 0.0;
    }
    w[iw + 1] = one;
    n1 = *n + 1;
    iz = iw + 2;
    iy = iz + n1;
    iwdual = iy + *m;
/*  SOLVE DUAL PROBLEM */
    nnls_(&w[1], &n1, &n1, m, &w[if__], &w[iy], &rnorm, &w[iwdual], &w[iz], &
	    indx[1], mode);
    if (*mode != 1) {
	goto L50;
    }
    *mode = 4;
    if (rnorm <= 0.0) {
	goto L50;
    }
/*  COMPUTE SOLUTION OF PRIMAL PROBLEM */
    fac = one - ddot_sl__(m, &h__[1], 1, &w[iy], 1);
    d__1 = one + fac;
    if (d__1 - one <= 0.0) {
	goto L50;
    }
    *mode = 1;
    fac = one / fac;
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
/* L40: */
	x[j] = fac * ddot_sl__(m, &g[j * g_dim1 + 1], 1, &w[iy], 1);
    }
    *xnorm = dnrm2___(n, &x[1], 1);
/*  COMPUTE LAGRANGE MULTIPLIERS FOR PRIMAL PROBLEM */
    w[1] = 0.0;
    dcopy___(m, &w[1], 0, &w[1], 1);
    daxpy_sl__(m, &fac, &w[iy], 1, &w[1], 1);
/*  END OF SUBROUTINE LDP */
L50:
    return;
} /* ldp_ */


void OC::lsi_(double *e, double *f, double *g, 
	double *h__, int *le, int *me, int *lg, int *mg, 
	int *n, double *x, double *xnorm, double *w, int *
	jw, int *mode){

	/* Initialized data */

    const double epmach = 2.22e-16;
    const double one = 1.;

    /* System generated locals */
    int e_dim1, e_offset, g_dim1, g_offset, i__1, i__2, i__3;
    double d__1;

    /* Local variables */
    int i__, j;
    double t;

    /* Parameter adjustments */
    --f;
    --jw;
    --h__;
    --x;
    g_dim1 = *lg;
    g_offset = 1 + g_dim1;
    g -= g_offset;
    e_dim1 = *le;
    e_offset = 1 + e_dim1;
    e -= e_offset;
    --w;

    /* Function Body */
/*  QR-FACTORS OF E AND APPLICATION TO F */
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
/* Computing MIN */
	i__2 = i__ + 1;
	j = MIN(i__2,*n);
	i__2 = i__ + 1;
	i__3 = *n - i__;
	h12_(&c__1, &i__, &i__2, me, &e[i__ * e_dim1 + 1], &c__1, &t, &e[j * 
		e_dim1 + 1], &c__1, le, &i__3);
/* L10: */
	i__2 = i__ + 1;
	h12_(&c__2, &i__, &i__2, me, &e[i__ * e_dim1 + 1], &c__1, &t, &f[1], &
		c__1, &c__1, &c__1);
    }
/*  TRANSFORM G AND H TO GET LEAST DISTANCE PROBLEM */
    *mode = 5;
    i__2 = *mg;
    for (i__ = 1; i__ <= i__2; ++i__) {
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    if ((d__1 = e[j + j * e_dim1], fabs(d__1)) < epmach) {
		goto L50;
	    }
/* L20: */
	    i__3 = j - 1;
	    g[i__ + j * g_dim1] = (g[i__ + j * g_dim1] - ddot_sl__(&i__3, &g[
		    i__ + g_dim1], *lg, &e[j * e_dim1 + 1], 1)) / e[j + j *
		     e_dim1];
	}
/* L30: */
	h__[i__] -= ddot_sl__(n, &g[i__ + g_dim1], *lg, &f[1], 1);
    }
/*  SOLVE LEAST DISTANCE PROBLEM */
    ldp_(&g[g_offset], lg, mg, n, &h__[1], &x[1], xnorm, &w[1], &jw[1], mode);
    if (*mode != 1) {
	goto L50;
    }
/*  SOLUTION OF ORIGINAL PROBLEM */
    daxpy_sl__(n, &one, &f[1], 1, &x[1], 1);
    for (i__ = *n; i__ >= 1; --i__) {
/* Computing MIN */
	i__2 = i__ + 1;
	j = MIN(i__2,*n);
/* L40: */
	i__2 = *n - i__;
	x[i__] = (x[i__] - ddot_sl__(&i__2, &e[i__ + j * e_dim1], *le, &x[j], 1))
	     / e[i__ + i__ * e_dim1];
    }
/* Computing MIN */
    i__2 = *n + 1;
    j = MIN(i__2,*me);
    i__2 = *me - *n;
    t = dnrm2___(&i__2, &f[j], 1);
    *xnorm = sqrt(*xnorm * *xnorm + t * t);
/*  END OF SUBROUTINE LSI */
L50:
    return;
} /* lsi_ */

void OC::hfti_(double *a, int *mda, int *m, int *
	n, double *b, int *mdb, const int *nb, double *tau, int 
	*krank, double *rnorm, double *h__, double *g, int *
		ip){
	
	/* Initialized data */

    const double factor = .001;

    /* System generated locals */
    int a_dim1, a_offset, b_dim1, b_offset, i__1, i__2, i__3;
    double d__1;

    /* Local variables */
    int i__, j, k, l;
    int jb, kp1;
    double tmp, hmax;
    int lmax, ldiag;

    /* Parameter adjustments */
    --ip;
    --g;
    --h__;
    a_dim1 = *mda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --rnorm;
    b_dim1 = *mdb;
    b_offset = 1 + b_dim1;
    b -= b_offset;

    /* Function Body */
    k = 0;
    ldiag = MIN(*m,*n);
    if (ldiag <= 0) {
	goto L270;
    }
/*   COMPUTE LMAX */
    i__1 = ldiag;
    for (j = 1; j <= i__1; ++j) {
	if (j == 1) {
	    goto L20;
	}
	lmax = j;
	i__2 = *n;
	for (l = j; l <= i__2; ++l) {
/* Computing 2nd power */
	    d__1 = a[j - 1 + l * a_dim1];
	    h__[l] -= d__1 * d__1;
/* L10: */
	    if (h__[l] > h__[lmax]) {
		lmax = l;
	    }
	}
	d__1 = hmax + factor * h__[lmax];
	if (d__1 - hmax > 0.0) {
	    goto L50;
	}
L20:
	lmax = j;
	i__2 = *n;
	for (l = j; l <= i__2; ++l) {
	    h__[l] = 0.0;
	    i__3 = *m;
	    for (i__ = j; i__ <= i__3; ++i__) {
/* L30: */
/* Computing 2nd power */
		d__1 = a[i__ + l * a_dim1];
		h__[l] += d__1 * d__1;
	    }
/* L40: */
	    if (h__[l] > h__[lmax]) {
		lmax = l;
	    }
	}
	hmax = h__[lmax];
/*   COLUMN INTERCHANGES IF NEEDED */
L50:
	ip[j] = lmax;
	if (ip[j] == j) {
	    goto L70;
	}
	i__2 = *m;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    tmp = a[i__ + j * a_dim1];
	    a[i__ + j * a_dim1] = a[i__ + lmax * a_dim1];
/* L60: */
	    a[i__ + lmax * a_dim1] = tmp;
	}
	h__[lmax] = h__[j];
/*   J-TH TRANSFORMATION AND APPLICATION TO A AND B */
L70:
/* Computing MIN */
	i__2 = j + 1;
	i__ = MIN(i__2,*n);
	i__2 = j + 1;
	i__3 = *n - j;
	h12_(&c__1, &j, &i__2, m, &a[j * a_dim1 + 1], &c__1, &h__[j], &a[i__ *
		 a_dim1 + 1], &c__1, mda, &i__3);
/* L80: */
	i__2 = j + 1;
	h12_(&c__2, &j, &i__2, m, &a[j * a_dim1 + 1], &c__1, &h__[j], &b[
		b_offset], &c__1, mdb, nb);
    }
/*   DETERMINE PSEUDORANK */
    i__2 = ldiag;
    for (j = 1; j <= i__2; ++j) {
/* L90: */
	if ((d__1 = a[j + j * a_dim1], fabs(d__1)) <= *tau) {
	    goto L100;
	}
    }
    k = ldiag;
    goto L110;
L100:
    k = j - 1;
L110:
    kp1 = k + 1;
/*   NORM OF RESIDUALS */
    i__2 = *nb;
    for (jb = 1; jb <= i__2; ++jb) {
/* L130: */
	i__1 = *m - k;
	rnorm[jb] = dnrm2___(&i__1, &b[kp1 + jb * b_dim1], 1);
    }
    if (k > 0) {
	goto L160;
    }
    i__1 = *nb;
    for (jb = 1; jb <= i__1; ++jb) {
	i__2 = *n;
	for (i__ = 1; i__ <= i__2; ++i__) {
/* L150: */
	    b[i__ + jb * b_dim1] = 0.0;
	}
    }
    goto L270;
L160:
    if (k == *n) {
	goto L180;
    }
/*   HOUSEHOLDER DECOMPOSITION OF FIRST K ROWS */
    for (i__ = k; i__ >= 1; --i__) {
/* L170: */
	i__2 = i__ - 1;
	h12_(&c__1, &i__, &kp1, n, &a[i__ + a_dim1], mda, &g[i__], &a[
		a_offset], mda, &c__1, &i__2);
    }
L180:
    i__2 = *nb;
    for (jb = 1; jb <= i__2; ++jb) {
/*   SOLVE K*K TRIANGULAR SYSTEM */
	for (i__ = k; i__ >= 1; --i__) {
/* Computing MIN */
	    i__1 = i__ + 1;
	    j = MIN(i__1,*n);
/* L210: */
	    i__1 = k - i__;
	    b[i__ + jb * b_dim1] = (b[i__ + jb * b_dim1] - ddot_sl__(&i__1, &
		    a[i__ + j * a_dim1], *mda, &b[j + jb * b_dim1], 1)) / 
		    a[i__ + i__ * a_dim1];
	}
/*   COMPLETE SOLUTION VECTOR */
	if (k == *n) {
	    goto L240;
	}
	i__1 = *n;
	for (j = kp1; j <= i__1; ++j) {
/* L220: */
	    b[j + jb * b_dim1] = 0.0;
	}
	i__1 = k;
	for (i__ = 1; i__ <= i__1; ++i__) {
/* L230: */
	    h12_(&c__2, &i__, &kp1, n, &a[i__ + a_dim1], mda, &g[i__], &b[jb *
		     b_dim1 + 1], &c__1, mdb, &c__1);
	}
/*   REORDER SOLUTION ACCORDING TO PREVIOUS COLUMN INTERCHANGES */
L240:
	for (j = ldiag; j >= 1; --j) {
	    if (ip[j] == j) {
		goto L250;
	    }
	    l = ip[j];
	    tmp = b[l + jb * b_dim1];
	    b[l + jb * b_dim1] = b[j + jb * b_dim1];
	    b[j + jb * b_dim1] = tmp;
L250:
	    ;
	}
    }
L270:
    *krank = k;
} /* hfti_ */

void OC::lsei_(double *c__, double *d__, double *e, 
	double *f, double *g, double *h__, int *lc, int *
	mc, int *le, int *me, int *lg, int *mg, int *n, 
	double *x, double *xnrm, double *w, int *jw, int *mode){

	/* Initialized data */

    const double epmach = 2.22e-16;

    /* System generated locals */
    int c_dim1, c_offset, e_dim1, e_offset, g_dim1, g_offset, i__1, i__2, 
	    i__3;
    double d__1;

    /* Local variables */
    int i__, j, k, l;
    double t;
    int ie, if__, ig, iw, mc1;
    int krank;

    /* Parameter adjustments */
    --d__;
    --f;
    --h__;
    --x;
    g_dim1 = *lg;
    g_offset = 1 + g_dim1;
    g -= g_offset;
    e_dim1 = *le;
    e_offset = 1 + e_dim1;
    e -= e_offset;
    c_dim1 = *lc;
    c_offset = 1 + c_dim1;
    c__ -= c_offset;
    --w;
    --jw;

    /* Function Body */
    *mode = 2;
    if (*mc > *n) {
	goto L75;
    }
    l = *n - *mc;
    mc1 = *mc + 1;
    iw = (l + 1) * (*mg + 2) + (*mg << 1) + *mc;
    ie = iw + *mc + 1;
    if__ = ie + *me * l;
    ig = if__ + *me;
/*  TRIANGULARIZE C AND APPLY FACTORS TO E AND G */
    i__1 = *mc;
    for (i__ = 1; i__ <= i__1; ++i__) {
/* Computing MIN */
	i__2 = i__ + 1;
	j = MIN(i__2,*lc);
	i__2 = i__ + 1;
	i__3 = *mc - i__;
	h12_(&c__1, &i__, &i__2, n, &c__[i__ + c_dim1], lc, &w[iw + i__], &
		c__[j + c_dim1], lc, &c__1, &i__3);
	i__2 = i__ + 1;
	h12_(&c__2, &i__, &i__2, n, &c__[i__ + c_dim1], lc, &w[iw + i__], &e[
		e_offset], le, &c__1, me);
/* L10: */
	i__2 = i__ + 1;
	h12_(&c__2, &i__, &i__2, n, &c__[i__ + c_dim1], lc, &w[iw + i__], &g[
		g_offset], lg, &c__1, mg);
    }
/*  SOLVE C*X=D AND MODIFY F */
    *mode = 6;
    i__2 = *mc;
    for (i__ = 1; i__ <= i__2; ++i__) {
	if ((d__1 = c__[i__ + i__ * c_dim1], fabs(d__1)) < epmach) {
	    goto L75;
	}
	i__1 = i__ - 1;
	x[i__] = (d__[i__] - ddot_sl__(&i__1, &c__[i__ + c_dim1], *lc, &x[1], 1)) 
	     / c__[i__ + i__ * c_dim1];
/* L15: */
    }
    *mode = 1;
    w[mc1] = 0.0;
    i__2 = *mg; // BUGFIX for *mc == *n: changed from *mg - *mc, SGJ 2010
    dcopy___(&i__2, &w[mc1], 0, &w[mc1], 1);
    if (*mc == *n) {
	goto L50;
    }
    i__2 = *me;
    for (i__ = 1; i__ <= i__2; ++i__) {
/* L20: */
	w[if__ - 1 + i__] = f[i__] - ddot_sl__(mc, &e[i__ + e_dim1], *le, &x[1], 1);
    }
/*  STORE TRANSFORMED E & G */
    i__2 = *me;
    for (i__ = 1; i__ <= i__2; ++i__) {
/* L25: */
	dcopy___(&l, &e[i__ + mc1 * e_dim1], *le, &w[ie - 1 + i__], *me);
    }
    i__2 = *mg;
    for (i__ = 1; i__ <= i__2; ++i__) {
/* L30: */
	dcopy___(&l, &g[i__ + mc1 * g_dim1], *lg, &w[ig - 1 + i__], *mg);
    }
    if (*mg > 0) {
	goto L40;
    }
/*  SOLVE LS WITHOUT INEQUALITY CONSTRAINTS */
    *mode = 7;
    k = MAX(*le,*n);
    t = sqrt(epmach);
    hfti_(&w[ie], me, me, &l, &w[if__], &k, &c__1, &t, &krank, xnrm, &w[1], &
	    w[l + 1], &jw[1]);
    dcopy___(&l, &w[if__], 1, &x[mc1], 1);
    if (krank != l) {
	goto L75;
    }
    *mode = 1;
    goto L50;
/*  MODIFY H AND SOLVE INEQUALITY CONSTRAINED LS PROBLEM */
L40:
    i__2 = *mg;
    for (i__ = 1; i__ <= i__2; ++i__) {
/* L45: */
	h__[i__] -= ddot_sl__(mc, &g[i__ + g_dim1], *lg, &x[1], 1);
    }
    lsi_(&w[ie], &w[if__], &w[ig], &h__[1], me, me, mg, mg, &l, &x[mc1], xnrm,
	     &w[mc1], &jw[1], mode);
    if (*mc == 0) {
	goto L75;
    }
    t = dnrm2___(mc, &x[1], 1);
    *xnrm = sqrt(*xnrm * *xnrm + t * t);
    if (*mode != 1) {
	goto L75;
    }
/*  SOLUTION OF ORIGINAL PROBLEM AND LAGRANGE MULTIPLIERS */
L50:
    i__2 = *me;
    for (i__ = 1; i__ <= i__2; ++i__) {
/* L55: */
	f[i__] = ddot_sl__(n, &e[i__ + e_dim1], *le, &x[1], 1) - f[i__];
    }
    i__2 = *mc;
    for (i__ = 1; i__ <= i__2; ++i__) {
/* L60: */
	d__[i__] = ddot_sl__(me, &e[i__ * e_dim1 + 1], 1, &f[1], 1) - 
		ddot_sl__(mg, &g[i__ * g_dim1 + 1], 1, &w[mc1], 1);
    }
    for (i__ = *mc; i__ >= 1; --i__) {
/* L65: */
	i__2 = i__ + 1;
	h12_(&c__2, &i__, &i__2, n, &c__[i__ + c_dim1], lc, &w[iw + i__], &x[
		1], &c__1, &c__1, &c__1);
    }
    for (i__ = *mc; i__ >= 1; --i__) {
/* Computing MIN */
	i__2 = i__ + 1;
	j = MIN(i__2,*lc);
	i__2 = *mc - i__;
	w[i__] = (d__[i__] - ddot_sl__(&i__2, &c__[j + i__ * c_dim1], 1, &
		w[j], 1)) / c__[i__ + i__ * c_dim1];
/* L70: */
    }
/*  END OF SUBROUTINE LSEI */
L75:
    return;

}/* lsei_ */

void OC::lsq_(int *m, int *meq, int *n, int *nl, 
	int *la, double *l, double *g, double *a, double *
	b, const double *xl, const double *xu, double *x, double *y, 
	double *w, int *jw, int *mode){

	/* Initialized data */

    const double one = 1.;

    /* System generated locals */
    int a_dim1, a_offset, i__1, i__2;
    double d__1;

    /* Local variables */
    int i__, i1, i2, i3, i4, m1, n1, n2, n3, ic, id, ie, if__, ig, ih, il,
	     im, ip, iu, iw;
    double diag;
    int mineq;
    double xnorm;

    /* Parameter adjustments */
    --y;
    --x;
    --xu;
    --xl;
    --g;
    --l;
    --b;
    a_dim1 = *la;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --w;
    --jw;

    /* Function Body */
    n1 = *n + 1;
    mineq = *m - *meq;
    m1 = mineq + *n + *n;
/*  determine whether to solve problem */
/*  with inconsistent linerarization (n2=1) */
/*  or not (n2=0) */
    n2 = n1 * *n / 2 + 1;
    if (n2 == *nl) {
	n2 = 0;
    } else {
	n2 = 1;
    }
    n3 = *n - n2;
/*  RECOVER MATRIX E AND VECTOR F FROM L AND G */
    i2 = 1;
    i3 = 1;
    i4 = 1;
    ie = 1;
    if__ = *n * *n + 1;
    i__1 = n3;
    for (i__ = 1; i__ <= i__1; ++i__) {
	i1 = n1 - i__;
	diag = sqrt(l[i2]);
	w[i3] = 0.0;
	dcopy___(&i1, &w[i3], 0, &w[i3], 1);
	i__2 = i1 - n2;
	dcopy___(&i__2, &l[i2], 1, &w[i3], *n);
	i__2 = i1 - n2;
	dscal_sl__(&i__2, &diag, &w[i3], *n);
	w[i3] = diag;
	i__2 = i__ - 1;
	w[if__ - 1 + i__] = (g[i__] - ddot_sl__(&i__2, &w[i4], 1, &w[if__]
		, 1)) / diag;
	i2 = i2 + i1 - n2;
	i3 += n1;
	i4 += *n;
/* L10: */
    }
    if (n2 == 1) {
	w[i3] = l[*nl];
	w[i4] = 0.0;
	dcopy___(&n3, &w[i4], 0, &w[i4], 1);
	w[if__ - 1 + *n] = 0.0;
    }
    d__1 = -one;
    dscal_sl__(n, &d__1, &w[if__], 1);
    ic = if__ + *n;
    id = ic + *meq * *n;
    if (*meq > 0) {
/*  RECOVER MATRIX C FROM UPPER PART OF A */
	i__1 = *meq;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    dcopy___(n, &a[i__ + a_dim1], *la, &w[ic - 1 + i__], *meq);
/* L20: */
	}
/*  RECOVER VECTOR D FROM UPPER PART OF B */
	dcopy___(meq, &b[1], 1, &w[id], 1);
	d__1 = -one;
	dscal_sl__(meq, &d__1, &w[id], 1);
    }
    ig = id + *meq;
    if (mineq > 0) {
/*  RECOVER MATRIX G FROM LOWER PART OF A */
	i__1 = mineq;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    dcopy___(n, &a[*meq + i__ + a_dim1], *la, &w[ig - 1 + i__], m1);
/* L30: */
	}
    }
/*  AUGMENT MATRIX G BY +I AND -I */
    ip = ig + mineq;
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	w[ip - 1 + i__] = 0.0;
	dcopy___(n, &w[ip - 1 + i__], 0, &w[ip - 1 + i__], m1);
/* L40: */
    }
    i__1 = m1 + 1;
    /* SGJ, 2010: skip constraints for infinite bounds */
    for (i__ = 1; i__ <= *n; ++i__)
	 if (!nlopt_isinf(xl[i__])) w[(ip - i__1) + i__ * i__1] = +1.0;
    /* Old code: w[ip] = one; dcopy___(n, &w[ip], 0, &w[ip], i__1); */
    im = ip + *n;
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	w[im - 1 + i__] = 0.0;
	dcopy___(n, &w[im - 1 + i__], 0, &w[im - 1 + i__], m1);
/* L50: */
    }
    i__1 = m1 + 1;
    /* SGJ, 2010: skip constraints for infinite bounds */
    for (i__ = 1; i__ <= *n; ++i__)
	 if (!nlopt_isinf(xu[i__])) w[(im - i__1) + i__ * i__1] = -1.0;
    /* Old code: w[im] = -one;  dcopy___(n, &w[im], 0, &w[im], i__1); */
    ih = ig + m1 * *n;
    if (mineq > 0) {
/*  RECOVER H FROM LOWER PART OF B */
	dcopy___(&mineq, &b[*meq + 1], 1, &w[ih], 1);
	d__1 = -one;
	dscal_sl__(&mineq, &d__1, &w[ih], 1);
    }
/*  AUGMENT VECTOR H BY XL AND XU */
    il = ih + mineq;
    iu = il + *n;
    /* SGJ, 2010: skip constraints for infinite bounds */
    for (i__ = 1; i__ <= *n; ++i__) {
	 w[(il-1) + i__] = nlopt_isinf(xl[i__]) ? 0 : xl[i__];
	 w[(iu-1) + i__] = nlopt_isinf(xu[i__]) ? 0 : -xu[i__];
    }
    /* Old code: dcopy___(n, &xl[1], 1, &w[il], 1);
                 dcopy___(n, &xu[1], 1, &w[iu], 1);
		 d__1 = -one; dscal_sl__(n, &d__1, &w[iu], 1); */
    iw = iu + *n;
    i__1 = MAX(1,*meq);
    lsei_(&w[ic], &w[id], &w[ie], &w[if__], &w[ig], &w[ih], &i__1, meq, n, n, 
	    &m1, &m1, n, &x[1], &xnorm, &w[iw], &jw[1], mode);
    if (*mode == 1) {
/*   restore Lagrange multipliers */
	dcopy___(m, &w[iw], 1, &y[1], 1);
	dcopy___(&n3, &w[iw + *m], 1, &y[*m + 1], 1);
	dcopy___(&n3, &w[iw + *m + *n], 1, &y[*m + n3 + 1], 1);

	/* SGJ, 2010: make sure bound constraints are satisfied, since
	   roundoff error sometimes causes slight violations and
	   NLopt guarantees that bounds are strictly obeyed */
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	     if (x[i__] < xl[i__]) x[i__] = xl[i__];
	     else if (x[i__] > xu[i__]) x[i__] = xu[i__];
	}
    }
/*   END OF SUBROUTINE LSQ */
}/* lsq_ */

void OC::ldl_(int *n, double *a, double *z__, 
	double *sigma, double *w){

	/* Initialized data */

    const double one = 1.;
    const double four = 4.;
    const double epmach = 2.22e-16;

    /* System generated locals */
    int i__1, i__2;

    /* Local variables */
    int i__, j;
    double t, u, v;
    int ij;
    double tp, beta, gamma_, alpha, delta;

    /* Parameter adjustments */
    --w;
    --z__;
    --a;

    /* Function Body */
    if (*sigma == 0.0) {
	goto L280;
    }
    ij = 1;
    t = one / *sigma;
    if (*sigma > 0.0) {
	goto L220;
    }
/* PREPARE NEGATIVE UPDATE */
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
/* L150: */
	w[i__] = z__[i__];
    }
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	v = w[i__];
	t += v * v / a[ij];
	i__2 = *n;
	for (j = i__ + 1; j <= i__2; ++j) {
	    ++ij;
/* L160: */
	    w[j] -= v * a[ij];
	}
/* L170: */
	++ij;
    }
    if (t >= 0.0) {
	t = epmach / *sigma;
    }
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	j = *n + 1 - i__;
	ij -= i__;
	u = w[j];
	w[j] = t;
/* L210: */
	t -= u * u / a[ij];
    }
L220:
/* HERE UPDATING BEGINS */
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	v = z__[i__];
	delta = v / a[ij];
	if (*sigma < 0.0) {
	    tp = w[i__];
	}
	else /* if (*sigma > 0.0), since *sigma != 0 from above */ {
	    tp = t + delta * v;
	}
	alpha = tp / t;
	a[ij] = alpha * a[ij];
	if (i__ == *n) {
	    goto L280;
	}
	beta = delta / tp;
	if (alpha > four) {
	    goto L240;
	}
	i__2 = *n;
	for (j = i__ + 1; j <= i__2; ++j) {
	    ++ij;
	    z__[j] -= v * a[ij];
/* L230: */
	    a[ij] += beta * z__[j];
	}
	goto L260;
L240:
	gamma_ = t / tp;
	i__2 = *n;
	for (j = i__ + 1; j <= i__2; ++j) {
	    ++ij;
	    u = a[ij];
	    a[ij] = gamma_ * u + beta * z__[j];
/* L250: */
	    z__[j] -= v * u;
	}
L260:
	++ij;
/* L270: */
	t = tp;
    }
L280:
    return;
/* END OF LDL */
}/* ldl_ */


void OC::slsqpb_(int *m, int *meq, int *la, int *
		    n, double *x, const double *xl, const double *xu, double *f, 
		    double *c__, double *g, double *a, double *acc, 
		    int *iter, int *mode, double *r__, double *l, 
		    double *x0, double *mu, double *s, double *u, 
		    double *v, double *w, int *iw, 
			slsqpb_state *state){


	/* Initialized data */
    const double one = 1.;
    const double alfmin = .1;
    const double hun = 100.;
    const double ten = 10.;
    const double two = 2.;

    /* System generated locals */
    int a_dim1, a_offset, i__1, i__2;
    double d__1, d__2;

    /* Local variables */
    int i__, j, k;

    /* saved state from one call to the next;
    SGJ 2010: save/restore via state parameter, to make re-entrant. */
    double t, f0, h1, h2, h3, h4;
    int n1, n2, n3;
    double t0, gs;
    double tol;
    int line;
    double alpha;
    int iexact;
    int incons, ireset, itermx;
    RESTORE_STATE;


   /* Parameter adjustments */
    --mu;
    --c__;
    --v;
    --u;
    --s;
    --x0;
    --l;
    --r__;
    a_dim1 = *la;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --g;
    --xu;
    --xl;
    --x;
    --w;
    --iw;


   /* Function Body */
    if (*mode == -1) {
	goto L260;
    } else if (*mode == 0) {
	goto L100;
    } else {
	goto L220;
    }
L100:
    itermx = *iter;
    if (*acc >= 0.0) {
	iexact = 0;
    } else {
	iexact = 1;
    }
    *acc = fabs(*acc);
    tol = ten * *acc;
    *iter = 0;
    ireset = 0;
    n1 = *n + 1;
    n2 = n1 * *n / 2;
    n3 = n2 + 1;
    s[1] = 0.0;
    mu[1] = 0.0;
    dcopy___(n, &s[1], 0, &s[1], 1);
    dcopy___(m, &mu[1], 0, &mu[1], 1);
/*   RESET BFGS MATRIX */
L110:
    ++ireset;
    if (ireset > 5) {
	goto L255;
    }
    l[1] = 0.0;
    dcopy___(&n2, &l[1], 0, &l[1], 1);
    j = 1;
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	l[j] = one;
	j = j + n1 - i__;
/* L120: */
    }
/*   MAIN ITERATION : SEARCH DIRECTION, STEPLENGTH, LDL'-UPDATE */
L130:
    ++(*iter);
    *mode = 9;
    if (*iter > itermx && itermx > 0) { /* SGJ 2010: ignore if itermx <= 0 */
	goto L330;
    }
/*   SEARCH DIRECTION AS SOLUTION OF QP - SUBPROBLEM */
    dcopy___(n, &xl[1], 1, &u[1], 1);
    dcopy___(n, &xu[1], 1, &v[1], 1);
    d__1 = -one;
    daxpy_sl__(n, &d__1, &x[1], 1, &u[1], 1);
    d__1 = -one;
    daxpy_sl__(n, &d__1, &x[1], 1, &v[1], 1);
    h4 = one;
    lsq_(m, meq, n, &n3, la, &l[1], &g[1], &a[a_offset], &c__[1], &u[1], &v[1]
	    , &s[1], &r__[1], &w[1], &iw[1], mode);

/*   AUGMENTED PROBLEM FOR INCONSISTENT LINEARIZATION */
    if (*mode == 6) {
	if (*n == *meq) {
	    *mode = 4;
	}
    }
    if (*mode == 4) {
	i__1 = *m;
	for (j = 1; j <= i__1; ++j) {
	    if (j <= *meq) {
		a[j + n1 * a_dim1] = -c__[j];
	    } else {
/* Computing MAX */
		d__1 = -c__[j];
		a[j + n1 * a_dim1] = MAX(d__1,0.0);
	    }
/* L140: */
	}
	s[1] = 0.0;
	dcopy___(n, &s[1], 0, &s[1], 1);
	h3 = 0.0;
	g[n1] = 0.0;
	l[n3] = hun;
	s[n1] = one;
	u[n1] = 0.0;
	v[n1] = one;
	incons = 0;
L150:
	lsq_(m, meq, &n1, &n3, la, &l[1], &g[1], &a[a_offset], &c__[1], &u[1],
		 &v[1], &s[1], &r__[1], &w[1], &iw[1], mode);
	h4 = one - s[n1];
	if (*mode == 4) {
	    l[n3] = ten * l[n3];
	    ++incons;
	    if (incons > 5) {
		goto L330;
	    }
	    goto L150;
	} else if (*mode != 1) {
	    goto L330;
	}
    } else if (*mode != 1) {
	goto L330;
    }
/*   UPDATE MULTIPLIERS FOR L1-TEST */
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	v[i__] = g[i__] - ddot_sl__(m, &a[i__ * a_dim1 + 1], 1, &r__[1], 1);
/* L160: */
    }
    f0 = *f;
    dcopy___(n, &x[1], 1, &x0[1], 1);
    gs = ddot_sl__(n, &g[1], 1, &s[1], 1);
    h1 = fabs(gs);
    h2 = 0.0;
    i__1 = *m;
    for (j = 1; j <= i__1; ++j) {
	if (j <= *meq) {
	    h3 = c__[j];
	} else {
	    h3 = 0.0;
	}
/* Computing MAX */
	d__1 = -c__[j];
	h2 += MAX(d__1,h3);
	h3 = (d__1 = r__[j], fabs(d__1));
/* Computing MAX */
	d__1 = h3, d__2 = (mu[j] + h3) / two;
	mu[j] = MAX(d__1,d__2);
	h1 += h3 * (d__1 = c__[j], fabs(d__1));
/* L170: */
    }
/*   CHECK CONVERGENCE */
    *mode = 0;
    if (h1 < *acc && h2 < *acc) {
	goto L330;
    }
    h1 = 0.0;
    i__1 = *m;
    for (j = 1; j <= i__1; ++j) {
	if (j <= *meq) {
	    h3 = c__[j];
	} else {
	    h3 = 0.0;
	}
/* Computing MAX */
	d__1 = -c__[j];
	h1 += mu[j] * MAX(d__1,h3);
/* L180: */
    }
    t0 = *f + h1;
    h3 = gs - h1 * h4;
    *mode = 8;
    if (h3 >= 0.0) {
	goto L110;
    }
/*   LINE SEARCH WITH AN L1-TESTFUNCTION */
    line = 0;
    alpha = one;
    if (iexact == 1) {
	goto L210;
    }
/*   INEXACT LINESEARCH */
L190:
    ++line;
    h3 = alpha * h3;
    dscal_sl__(n, &alpha, &s[1], 1);
    dcopy___(n, &x0[1], 1, &x[1], 1);
    daxpy_sl__(n, &one, &s[1], 1, &x[1], 1);
    
    /* SGJ 2010: ensure roundoff doesn't push us past bound constraints */
    i__1 = *n; for (i__ = 1; i__ <= i__1; ++i__) {
	 if (x[i__] < xl[i__]) x[i__] = xl[i__];
	 else if (x[i__] > xu[i__]) x[i__] = xu[i__];
    }

    /* SGJ 2010: optimizing for the common case where the inexact line
       search succeeds in one step, use special mode = -2 here to
       eliminate a a subsequent unnecessary mode = -1 call, at the 
       expense of extra gradient evaluations when more than one inexact
       line-search step is required */
    *mode = line == 1 ? -2 : 1;
    goto L330;
L200:
    if (h1 <= h3 / ten || line > 10) {
	goto L240;
    }
/* Computing MAX */
    d__1 = h3 / (two * (h3 - h1));
    alpha = MAX(d__1,alfmin);
    goto L190;
/*   EXACT LINESEARCH */
L210:
#if 0
    /* SGJ: see comments by linmin_ above: if we want to do an exact linesearch
       (which usually we probably don't), we should call NLopt recursively */
    if (line != 3) {
	alpha = linmin_(&line, &alfmin, &one, &t, &tol);
	dcopy___(n, &x0[1], 1, &x[1], 1);
	daxpy_sl__(n, &alpha, &s[1], 1, &x[1], 1);
	*mode = 1;
	goto L330;
    }
#else
    *mode = 9 /* will yield nlopt_failure */; return;
#endif
    dscal_sl__(n, &alpha, &s[1], 1);
    goto L240;
/*   CALL FUNCTIONS AT CURRENT X */
L220:
    t = *f;
    i__1 = *m;
    for (j = 1; j <= i__1; ++j) {
	if (j <= *meq) {
	    h1 = c__[j];
	} else {
	    h1 = 0.0;
	}
/* Computing MAX */
	d__1 = -c__[j];
	t += mu[j] * MAX(d__1,h1);
/* L230: */
    }
    h1 = t - t0;
    switch (iexact + 1) {
	case 1:  goto L200;
	case 2:  goto L210;
    }
/*   CHECK CONVERGENCE */
L240:
    h3 = 0.0;
    i__1 = *m;
    for (j = 1; j <= i__1; ++j) {
	if (j <= *meq) {
	    h1 = c__[j];
	} else {
	    h1 = 0.0;
	}
/* Computing MAX */
	d__1 = -c__[j];
	h3 += MAX(d__1,h1);
/* L250: */
    }
    if (((d__1 = *f - f0, fabs(d__1)) < *acc || dnrm2___(n, &s[1], 1) < *
	    acc) && h3 < *acc) {
	*mode = 0;
    } else {
	*mode = -1;
    }
    goto L330;
/*   CHECK relaxed CONVERGENCE in case of positive directional derivative */
L255:
    if (((d__1 = *f - f0, fabs(d__1)) < tol || dnrm2___(n, &s[1], 1) < tol)
	     && h3 < tol) {
	*mode = 0;
    } else {
	*mode = 8;
    }
    goto L330;
/*   CALL JACOBIAN AT CURRENT X */
/*   UPDATE CHOLESKY-FACTORS OF HESSIAN MATRIX BY MODIFIED BFGS FORMULA */
L260:
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	u[i__] = g[i__] - ddot_sl__(m, &a[i__ * a_dim1 + 1], 1, &r__[1], 1) - v[i__];
/* L270: */
    }
/*   L'*S */
    k = 0;
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	h1 = 0.0;
	++k;
	i__2 = *n;
	for (j = i__ + 1; j <= i__2; ++j) {
	    ++k;
	    h1 += l[k] * s[j];
/* L280: */
	}
	v[i__] = s[i__] + h1;
/* L290: */
    }
/*   D*L'*S */
    k = 1;
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	v[i__] = l[k] * v[i__];
	k = k + n1 - i__;
/* L300: */
    }
/*   L*D*L'*S */
    for (i__ = *n; i__ >= 1; --i__) {
	h1 = 0.0;
	k = i__;
	i__1 = i__ - 1;
	for (j = 1; j <= i__1; ++j) {
	    h1 += l[k] * v[j];
	    k = k + *n - j;
/* L310: */
	}
	v[i__] += h1;
/* L320: */
    }
    h1 = ddot_sl__(n, &s[1], 1, &u[1], 1);
    h2 = ddot_sl__(n, &s[1], 1, &v[1], 1);
    h3 = h2 * .2;
    if (h1 < h3) {
	h4 = (h2 - h3) / (h2 - h1);
	h1 = h3;
	dscal_sl__(n, &h4, &u[1], 1);
	d__1 = one - h4;
	daxpy_sl__(n, &d__1, &v[1], 1, &u[1], 1);
    }
    d__1 = one / h1;
    ldl_(n, &l[1], &u[1], &d__1, &v[1]);
    d__1 = -one / h2;
    ldl_(n, &l[1], &v[1], &d__1, &u[1]);
/*   END OF MAIN ITERATION */
    goto L130;
/*   END OF SLSQPB */
L330:
    SAVE_STATE;
}/* slsqpb_ */

/* *********************************************************************** */
/*                              optimizer                               * */
/* *********************************************************************** */

void OC::slsqp(int *m, int *meq, int *la, int *n,
		  double *x, const double *xl, const double *xu, double *f, 
		  double *c__, double *g, double *a, double *acc, 
		  int *iter, int *mode, double *w, int *l_w__, int *
		jw, int *l_jw__, slsqpb_state *state){


     /* System generated locals */
     int a_dim1, a_offset, i__1, i__2;

    /* Local variables */
    int n1, il, im, ir, is, iu, iv, iw, ix, mineq;
	
    /* Parameter adjustments */
    --c__;
    a_dim1 = *la;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --g;
    --xu;
    --xl;
    --x;
    --w;
    --jw;


   /* Function Body */
    n1 = *n + 1;
    mineq = *m - *meq + n1 + n1;
    il = (n1 * 3 + *m) * (n1 + 1) + (n1 - *meq + 1) * (mineq + 2) + (mineq << 
	    1) + (n1 + mineq) * (n1 - *meq) + (*meq << 1) + n1 * *n / 2 + (*m 
	    << 1) + *n * 3 + (n1 << 2) + 1;
    /* Computing MAX */
    i__1 = mineq, i__2 = n1 - *meq;
    im = MAX(i__1,i__2);
    if (*l_w__ < il || *l_jw__ < im) {
	*mode = MAX(10,il) * 1000;
	*mode += MAX(10,im);
	return;
     }
    /*   PREPARE DATA FOR CALLING SQPBDY  -  INITIAL ADDRESSES IN W */
    im = 1;
    il = im + MAX(1,*m);
    il = im + *la;
    ix = il + n1 * *n / 2 + 1;
    ir = ix + *n;
    is = ir + *n + *n + MAX(1,*m);
    is = ir + *n + *n + *la;
    iu = is + n1;
    iv = iu + n1;
    iw = iv + n1;
    slsqpb_(m, meq, la, n, &x[1], &xl[1], &xu[1], f, &c__[1], &g[1], &a[
	    a_offset], acc, iter, mode, &w[ir], &w[il], &w[ix], &w[im], &w[is]
	    , &w[iu], &w[iv], &w[iw], &jw[1], state);
    state->x0 = &w[ix];
    return;

}/* slsqp_ */

void OC::length_work(int *LEN_W, int *LEN_JW, int M, int MEQ, int N){
	
	int N1 = N+1, MINEQ = M-MEQ+N1+N1;
     	*LEN_W = (3*N1+M)*(N1+1) 
	  	+(N1-MEQ+1)*(MINEQ+2) + 2*MINEQ
          	+(N1+MINEQ)*(N1-MEQ) + 2*MEQ + N1
          	+(N+1)*N/2 + 2*M + 3*N + 3*N1 + 1;
     	*LEN_JW = MINEQ;
}



PetscScalar OC::Min(PetscScalar d1, PetscScalar d2){
	return d1<d2 ? d1 : d2;
}

PetscScalar OC::Max(PetscScalar d1, PetscScalar d2){
	return d1>d2 ? d1 : d2;
}

PetscInt OC::Min(PetscInt d1, PetscInt d2){
	return d1<d2 ? d1 : d2;
}

PetscInt OC::Max(PetscInt d1, PetscInt d2){
	return d1>d2 ? d1 : d2;
}

PetscScalar OC::Abs(PetscScalar d1){
	return d1>0 ? d1 : -1.0*d1;
}

/********************************* BLAS1 routines *************************/

/* COPIES A VECTOR, X, TO A VECTOR, Y, with the given increments */
void OC::dcopy___(int *n_, const double *dx, int incx, 
		     double *dy, int incy)
{
     int i, n = *n_;
     
     if (n <= 0) return;
     if (incx == 1 && incy == 1)
	  memcpy(dy, dx, sizeof(double) * ((unsigned) n));
     else if (incx == 0 && incy == 1) {
	  double x = dx[0];
	  for (i = 0; i < n; ++i) dy[i] = x;
     }
     else {
	  for (i = 0; i < n; ++i) dy[i*incy] = dx[i*incx];
     }
} /* dcopy___ */

/* CONSTANT TIMES A VECTOR PLUS A VECTOR. */
void OC::daxpy_sl__(int *n_, const double *da_, const double *dx, 
		       int incx, double *dy, int incy)
{
     int n = *n_, i;  
     double da = *da_;

     if (n <= 0 || da == 0) return;
     for (i = 0; i < n; ++i) dy[i*incy] += da * dx[i*incx];
}

/* dot product dx dot dy. */
double OC::ddot_sl__(int *n_, double *dx, int incx, double *dy, int incy)
{
     int n = *n_, i;
     long double sum = 0;
     if (n <= 0) return 0;
     for (i = 0; i < n; ++i) sum += dx[i*incx] * dy[i*incy];
     return (double) sum;
}

/* compute the L2 norm of array DX of length N, stride INCX */
double OC::dnrm2___(int *n_, double *dx, int incx)
{
     int i, n = *n_;
     double xmax = 0, scale;
     long double sum = 0;
     for (i = 0; i < n; ++i) {
          double xabs = fabs(dx[incx*i]);
          if (xmax < xabs) xmax = xabs;
     }
     if (xmax == 0) return 0;
     scale = 1.0 / xmax;
     for (i = 0; i < n; ++i) {
          double xs = scale * dx[incx*i];
          sum += xs * xs;
     }
     return xmax * sqrt((double) sum);
}

/* apply Givens rotation */
void OC::dsrot_(int n, double *dx, int incx, 
		   double *dy, int incy, double *c__, double *s_)
{
     int i;
     double c = *c__, s = *s_;

     for (i = 0; i < n; ++i) {
	  double x = dx[incx*i], y = dy[incy*i];
	  dx[incx*i] = c * x + s * y;
	  dy[incy*i] = c * y - s * x;
     }
}

/* construct Givens rotation */
void OC::dsrotg_(double *da, double *db, double *c, double *s)
{
     double absa, absb, roe, scale;

     absa = fabs(*da); absb = fabs(*db);
     if (absa > absb) {
	  roe = *da;
	  scale = absa;
     }
     else {
	  roe = *db;
	  scale = absb;
     }

     if (scale != 0) {
	  double r, iscale = 1 / scale;
	  double tmpa = (*da) * iscale, tmpb = (*db) * iscale;
	  r = (roe < 0 ? -scale : scale) * sqrt((tmpa * tmpa) + (tmpb * tmpb)); 
	  *c = *da / r; *s = *db / r; 
	  *da = r;
	  if (*c != 0 && fabs(*c) <= *s) *db = 1 / *c;
	  else *db = *s;
     }
     else { 
	  *c = 1; 
	  *s = *da = *db = 0;
     }
}

/* scales vector X(n) by constant da */
void OC::dscal_sl__(int *n_, const double *da, double *dx, int incx)
{
     int i, n = *n_;
     double alpha = *da;
     for (i = 0; i < n; ++i) dx[i*incx] *= alpha;
}
/**************************************************************************/