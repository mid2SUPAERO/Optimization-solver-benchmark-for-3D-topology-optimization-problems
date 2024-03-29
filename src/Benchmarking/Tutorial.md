# Tutorial on how to manage Topopt Petsc (Niels Aage, Erik Andreassen, Boyan Stefanov Lazarov) and 
# to benchmark 3D topology optimization problems.

First and foremost, Topopt petsc consist of five C++ classes and a short main program that i will detail below.
# Class structure
1. TopOpt class : 
Contains informationon the optimization problem, gridsize, parameters and general settings.
2. LinearElasticity class : 
The physics class which solves the linear elasticity problem on a structured 3D grid using 8-node linear brick elements. 
It also contains methods necessary for the minimum compliance problem, ie objective, constraint and sensitivity calculations.
3. Filter/PDEFilter class :
are filter classes which contains both sensitivity (Sigmund 1997), density (Bruns and Tortorelli 2001; Bourdin 2001),
and PDE (Lazarov and Sigmund 2011) filters through a common interface.
4. MMA class :
containing a fully parallelized implementation of the Method of Moving Asymptotes (MMA) (Svanberg 1987), 
following the description given in Aage and Lazarov (2013).
5. MPIIO class :
A versatile output class capable of dumping arbitrary field data into a single binary file. 
# And finally :
The distribution also includes Python scripts that conveniently converts the binary outputdata to the VTU format 
(Schroeder and Martin 2003) that can be visualized in e.g. ParaView version 4 or newer (Ahrens et al. 2005).
That means one can visualize the optimize design via Paraview.

# How to compile and run the code Topopt petsc on the terminal.
First and foremost, you will need a makefile script to run it on the terminal. Don't worry, we already have one. 
To get access to the makefile, one should go back to src/Makefile/ directory
# How to use makefile!
One should first type on terminal : cd topopt/ (to be place in the Topopt petsc files) and then type : make or make ./topopt 
(to compile the code). To run the code on two processors using the default settings, one  should type :  mpiexec -np 2 ./topopt 
or mpirun -np 2 ./topopt

# How to vizualize the optimal design.
To vizualize the optimize design, one should :
1. Type cd .. (to be place outside ./topopt)
2. Type cp Topopt/*.py /tmpdir/name/Topopt-637443/ 
(where Topopt-614252 is for example your outputdata, name is your login to connect on calmip).
3. Type cd /tmpdir/name/Topopt-637443/
4. Type python bin2vtu.py 15 
(where 15 is for example you only want 1 15 30 45 55 ... outputdata). 
5. you can now go to /tmpdir/name/Topopt-637443/ and extract the file .vtu 
6. open Paraview and open file .vtu to vizualize.
7. press the button apply then In the upper right corner and last row, press Threshold and press button apply
8. choose scalars = xPhys or x (depending on what you want) fix your minimum between 0.1 and 0.9 and press button apply
9. if you want color then apply coloring by choosing again xPhys.

# Default settings :
The default problem is the minimum compliance cantilever problem as described in Aage and Lazarov (2013) on a 2x1x1 domain,
using sensitivity filter with radius 0.08 and a volume fraction of 0.12 percent. 
The contrast between solid and void is set to 1e+9, the test convergence is set to ||x_k -x_{k+1}|| L.T 0.01 
or  a maximum of 400 design cycles.

# How do you change a problem, ie cantilever to wheel or wheel to michell?.
One should go the source file : LinearElasticity.cc (LinearElasticity class) and inside the function : SetUpLoadAndBC(*),
and to the meat code : // Set the values.
To change default problem for example to a 3D wheel problem, one shoud go to src/Examples_to_Change_probs directory.
We do have 3D wheel problem, 3D michell problem and 3D cantilever (by default). 
And of course, one can also build his own new problem by changing //Set the values accordingly to obtain a new problem,
ie 3D MBB.

# How do you change domain's size ?.
One should go the source file : TopOpt.cc (TopOpt class) and inside the function : SetUp(*) 
and to the meat code : // SET DEFAULTS for FE mesh and levels for MG solver.

For example : xc[0] = xmin, xc[1]=xmax, xc[2]=ymin, xc[3]=ymax, xc[4]=zmin, xc[5]=zmax, 
so if you want 4x1x1 domain, one should change xc[0] = 0, xc[1]=4, xc[2]=0, xc[3]=1, xc[4]=0, xc[5]=1

# How do you change filter's size and volume fraction?!
One could repeat the same action done to change domain's size but instead modify the meat code :// SET DEFAULTS for optimization problems. 

# How to benchmark 3D topopogy optimization problems on HPC.
To benchmark 3D topopogy optimization problems, one should know how to change problems settings in Topopt Petsc:
1. One should know how to change problem.
2. one should know how to change domain's size, filter's size and volume fraction.
3. One should know how to change a solver. 
4. And one should know how to build up post-treatment or processing on matlab using performances profiles and data profiles.
We detail below one by one, how to achieve this goal.
1. One should go to line above : # How do you change a problem
2. To change domain's size one should go to line above : # How do you change domain's size ?.
But to change filter's size and volume fraction, one should use the script : topopt1.slurm.
To use the script one should go src/Makefile and then one should modify the setting acordingly.
3. To change a solver, ie MMA (default) to GCMMA or  to OC, one should go to MMA class and replace the class with GCMMA or OC. 
To use the GCMMA or OC class, one should go to src/
4. To build up post-treatement or processing on matlab, one should go to src/Benchmarking and select benchmarking.m

# How to use the script.slurm to generate 3D topology optimization problems.
Inside the script topopt1.slurm :
For a big size problem, ie over 100 million degrees of freedom (ddl), one should ask calmip (HPC) for
1. #SBATCH --nodes=40
2. #SBATCH --ntasks=800
3. #SBATCH --ntasks-per-node=20
4. #SBATCH --threads-per-core=1
And for a simpler size, ie less than 5 million ddl, one should ask for :
1. #SBATCH --nodes=2
2. #SBATCH --ntasks=72
3. #SBATCH --ntasks-per-node=36
4. #SBATCH --threads-per-core=1
The outputdata files are written outside ./topopt, in the file : /tmpdir/name/ .
To get the file from ./topopt, one should type on terminal : cd .. then cd /tmpdir/name/ .
You can also use WINSCP (software already install on your ICA's computer).


# For futur work
1. One should consider using an algorithm to generate problems setting
when using topopt1.slurm.
2. One could find in src/Makefile/makefile_nlopt sqp algorithm in nlopt with a makefile and source files to work with in other 
to change solver in topopt Petsc. The code isn't fully working but it compiles. 
(problem comes from a function call  length_work(*)) 
3. one could find debug makefile in src/Makefile/makefile_nlopt
 







