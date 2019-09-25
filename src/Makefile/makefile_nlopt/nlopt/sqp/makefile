

CXX=mpiicpc
RM=rm -rf  
CFLAGS=
LDFLAGS=-L/usr/local/nlopt/2.6.1/lib64 -lnlopt
NLOPT_DIR=/usr/local/nlopt/2.6.1/include
PETSC_DIR=/usr/local/petsc/3.7.4/impi
PETSC_ARCH=linux-intel-opt32
LIBS=-L$(PETSC_DIR)/lib -lpetsc -L$(NLOPT_DIR) -mkl
INCLUDE = -I/users/p18010/agyekum/Topopt -I$(PETSC_DIR)/include -I$(NLOPT_DIR)
FFLAGS=
CPPFLAGS=-I.
FPPFLAGS=
LOCDIR=
EXAMPLESC=
EXAMPLESF=
MANSEC=
CLEANFILES=
NP=



EXEC=topopt



all: $(EXEC)

topopt: main.o TopOpt.o LinearElasticity.o MMA.o Filter.o PDEFilter.o MPIIO.o  
	$(CXX) -o topopt main.o TopOpt.o LinearElasticity.o MMA.o Filter.o PDEFilter.o MPIIO.o $(LDFLAGS) $(LIBS)

main.o: 
	$(CXX) -o main.o -c main.cc $(CFLAGS) $(INCLUDE)      

TopOpt.o: 
	$(CXX) -o TopOpt.o -c TopOpt.cc $(CFLAGS) $(INCLUDE)
 	 
LinearElasticity.o: 
	$(CXX) -o LinearElasticity.o -c LinearElasticity.cc $(CFLAGS) $(INCLUDE)

MMA.o: 
	$(CXX) -o MMA.o -c MMA.cc $(CFLAGS) $(INCLUDE)
 
Filter.o: 
	$(CXX) -o Filter.o -c Filter.cc $(CFLAGS) $(INCLUDE)

PDEFilter.o: 
	$(CXX) -o PDEFilter.o -c PDEFilter.cc $(CFLAGS) $(INCLUDE)

MPIIO.o: 
	$(CXX) -o MPIIO.o -c MPIIO.cc $(CFLAGS) $(INCLUDE)







myclean:
	$(RM) topopt *.o output* binary* log* makevtu.pyc Restart* 




















