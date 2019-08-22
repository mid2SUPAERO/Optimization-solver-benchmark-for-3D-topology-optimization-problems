// -*- coding: utf-8; tab-width: 4; indent-tabs-mode: t; c-basic-offset: 4 -*-

#include <MMA.h>
#include <iostream>
#include <math.h>

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
PetscErrorCode OC::Update(Vec xval, Vec dfdx, PetscScalar *gx, Vec *dgdx,
						  Vec xmin, Vec xmax, PetscScalar volfrac){
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
	PetscInt       nBi, nBiMax;  /* iteration counters */
	PetscInt       nglob, nloc, nloc1, nloc2, nloc3, nloc4; /* vector size */
	PetscScalar    l1, l2, lmid;  /* lagrangian multiplier */
	PetscScalar    vol, vol0, voltarget;  /* "volumes" */
	PetscScalar    *pxval,*pxold,*pdfdx,*pdgdx;  /* pointers to local data */
	PetscScalar    *pxmin,*pxmax;  /* pointers to local data */
	Vec            xold;

	/* hardcoded stuff that should have been defined by inputs to constructor: */
	nBiMax = 1000;  /* max number of bi-sections*/
	l1 = 0.0;  /* lower init guess */
	l2 = 1.0e5;  /* upper init guess */

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

	/* bi-section algorithm */
	nBi = 0;      /* zero local counter */
	while ((l2-l1)/(l1+l2) > 1.0e-8 && nBi < nBiMax) {
		nBi++;      /* local counter */
		k++;      /* global counter */

		lmid = 0.5*(l2+l1);

		/* get pointer to local data */
		ierr = VecGetArray(xval,&pxval);CHKERRQ(ierr);

		/* do point wise operation */
		for (PetscInt i=0;i<nloc;i++){
			/* compute new value assuming uniform cell size*/
			pxval[i] = pxold[i]*pow(-pdfdx[i]/(pdgdx[i]*lmid),0.5);

			/* constrain xval within limits */
			if ( pxval[i] < pxmin[i] )
                pxval[i] = pxmin[i];
			else if ( pxval[i] > pxmax[i] )
                pxval[i] = pxmax[i];
		}

		/* restore local data that has been modified for collective operations */
		ierr = VecRestoreArray(xval,&pxval);CHKERRQ(ierr);

		/* calculate new volume collectively */
		ierr = VecSum(xval, &vol); CHKERRQ(ierr); /* used "volume" assuming uniform cell size */

		/* bi-sect */
		if( vol/(volfrac*((PetscScalar)nglob)) > 1.0 ) {
			l1 = lmid;
		} else {
			l2 = lmid;
		}

	}

	/* clean up */
	VecRestoreArray(xold,&pxold);
	VecRestoreArray(dfdx,&pdfdx);
	VecRestoreArray(dgdx[0],&pdgdx);
	VecRestoreArray(xmin,&pxmin);
	VecRestoreArray(xmax,&pxmax);
	VecDestroy(&xold);

	/* print some diagnostics */
	PetscPrintf(PETSC_COMM_WORLD,
				"Optimality criteria: Number of bi-sections=%d\n",nBi);

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
