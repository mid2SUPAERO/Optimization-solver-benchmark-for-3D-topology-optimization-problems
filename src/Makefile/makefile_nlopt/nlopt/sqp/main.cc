// -*- coding: utf-8; tab-width: 4; indent-tabs-mode: t; c-basic-offset: 4 -*-

#include <petsc.h>
#include <TopOpt.h>
#include <LinearElasticity.h>
#include <MMA.h>
#include <Filter.h>
#include <MPIIO.h>
#include <mpi.h>

#include <iomanip>
#include <iostream>
#include <vector>

#include <nlopt.hpp>
using namespace std;

/*
Authors: Niels Aage, Erik Andreassen, Boyan Lazarov, August 2013

Disclaimer:                                                              
The authors reserves all rights but does not guaranty that the code is   
free from errors. Furthermore, we shall not be liable in any event     
caused by the use of the program.                                     
 */

static char help[] = "3D TopOpt using KSP-MG on PETSc's DMDA (structured grids) \n";



int main(int argc, char *argv[]){

	// Error code for debugging
	PetscErrorCode ierr;


	// Initialize PETSc / MPI and pass input arguments to PETSc
	PetscInitialize(&argc,&argv,PETSC_NULL,help);

	// STEP 1: THE OPTIMIZATION PARAMETERS, DATA AND MESH (!!! THE DMDA !!!)
	TopOpt *opt = new TopOpt();

	// STEP 2: THE PHYSICS
	LinearElasticity *physics = new LinearElasticity(opt);

	// STEP 3: THE FILTERING
	Filter *filter = new Filter(opt);
	
	// STEP 4: VISUALIZATION USING VTK
	MPIIO *output = new MPIIO(opt->da_nodes,3,"ux, uy, uz",2,"x, xPhys");

	// STEP 5: THE OPTIMIZER
	PetscInt itr=0;
	OC *oc = NULL;	
	oc = new OC(opt->m);

	// STEP 6: FILTER THE INITIAL DESIGN/RESTARTED DESIGN
	ierr = filter->FilterProject(opt); CHKERRQ(ierr);
	
	// STEP 7: OPTIMIZATION LOOP
	PetscScalar ch = 1.0;
	double t1,t2;


	slsqpb_state state = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,NULL};
	unsigned mtot = 1;
	unsigned ptot = 0;
	int mpi = (int) (mtot + ptot), pi = (int) ptot, ni = (int) opt->n, mpi1 = mpi > 0 ? mpi : 1;
	int len_w, len_jw, *jw;
	int mode = 0, prev_mode = 0;
	double acc = 0; /* we do our own convergence tests below */
	int iter = 0; /* tell sqsqp to ignore this check, since we check evaluation counts ourselves */
	unsigned max_cdim = 1;
	//oc->length_work(&len_w, &len_jw, mpi, pi, ni);
	double *work, *cgrad, *c, *grad, *w, fcur, *xcur, fprev, *xprev, *cgradtmp;
	int n = opt->n;
	#define U(n) ((unsigned) (n))
		work = (double *) malloc(sizeof(double) * (U(mpi1) * (n + 1) 
							+  U(mpi)
							+  n+1 + n + n + max_cdim*n
							+   U(len_w))
							+  + sizeof(int) * U(len_jw));
		if (!work) return NLOPT_OUT_OF_MEMORY;
		cgrad = work;
		c = cgrad + U(mpi1) * (n + 1);
		grad = c + mpi;
		xcur = grad + n+1;
		xprev = xcur + n;
		cgradtmp = xprev + n;
		w = cgradtmp + max_cdim*n;
		jw = (int *) (w + len_w);
		//memcpy(xcur, x, sizeof(double) * n);
		//memcpy(xprev, x, sizeof(double) * n);


	while (itr < opt->maxItr && ch > 0.01){
		// Update iteration counter
		itr++;

		// start timer
		t1 = MPI_Wtime();

		// Compute (a) obj+const, (b) sens, (c) obj+const+sens 
		ierr = physics->ComputeObjectiveConstraintsSensitivities(opt); CHKERRQ(ierr);
		// Compute objective scale
		if (itr==1){ 
			opt->fscale = 10.0/opt->fx; 
		}
		// Scale objectie and sens
		opt->fx = opt->fx*opt->fscale;
		VecScale(opt->dfdx,opt->fscale);

		// Filter sensitivities (chainrule)
		ierr = filter->Gradients(opt); CHKERRQ(ierr);

		// Sets outer movelimits on design variables
		ierr = oc->SetOuterMovelimit(opt->Xmin,opt->Xmax,opt->movlim,opt->x,
										 opt->xmin,opt->xmax); CHKERRQ(ierr);
		// Update design
		ierr = oc->Update(opt->x,opt->fx,opt->dfdx,opt->gx,opt->dgdx,opt->xmin,opt->xmax,c,grad,cgrad,xcur,w,jw,
					mpi,pi,mpi1,ni,fcur,acc,iter,mode,len_w,len_jw, state); CHKERRQ(ierr);
		

		// Inf norm on the design change
		ch = oc->DesignChange(opt->x,opt->xold);
		// Filter design field
		ierr = filter->FilterProject(opt); CHKERRQ(ierr);

		// stop timer
		t2 = MPI_Wtime();

		// Print to screen
		PetscPrintf(PETSC_COMM_WORLD,"It.: %i, obj.: %f, g[0]: %f, ch.: %f, time: %f\n",
				itr,opt->fx,opt->gx[0], ch,t2-t1);

		// Write field data: first 10 iterations and then every 20th
		if (itr<11 || itr%20==0){
			output->WriteVTK(opt->da_nodes,physics->GetStateField(),opt, itr);
		}
	}
	
	// Dump final design
	output->WriteVTK(opt->da_nodes,physics->GetStateField(),opt, itr+1);

	// STEP 7: CLEAN UP AFTER YOURSELF
	if (oc!=NULL){ delete oc;}
	delete output;
	delete filter;
	delete opt;
	delete physics;


	// Finalize PETSc / MPI
	PetscFinalize();
	return 0;
}






