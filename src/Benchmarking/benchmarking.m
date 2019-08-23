%% Example Benchmarking 
% Summary of example objective
% we present here un example of how to produce a benchmarking as described in:
% Benchmarking Derivative-Free Optimization Algorithms
% Jorge J. More' and Stefan M. Wild
% SIAM J. Optimization, Vol. 20 (1), pp.172-191, 2009.
%
% The latest version of this subroutine is always available at
% http://www.mcs.anl.gov/~more/dfo/
% The authors would appreciate feedback and experiences from numerical
% studies conducted using this subroutine.
%
% Performance profiles were originally introduced in
% Benchmarking optimization software with performance profiles,
% E.D. Dolan and J.J. More', 
% Mathematical Programming, 91 (2002), 201--213.
%% Section 1 Build up to read Data files from a solver plug in TopOpt PETSc
% Description of first code block
% read file
NOM_Txt = []; % should contained a list of files name ( e.g. ["out_1.txt";"out_2.txt";....])
% reading file.txt into Matlab framework
np = 120; % number of problems
neval = 200; % number maximum evaluations
ns = 3; % number of solvers 
%
MatH  = zeros(neval,np,ns); % build up of the matrice H (we will explain mat_H at next step)
Matg  = zeros(neval,np,ns); % build up of the constraint matrice g (we will explain mat_g at next step)
%
Tmp   = zeros(neval,1); % temporary matrice to be use 
Tmpg  = Tmp;
Tmpv  = 1.0*ones(1,1);
%volume fraction : volfrac
vol  = ones(1,np,ns);

for kk=1:np
    out = [];
    outg  = [];
    outv  = [];
    filename  = NOM_Txt(kk);% output.txt of problem solve by an optimizer
    [fid,msg] = fopen(filename,'rt'); % output.txt can be read and write eg. r.t is a permission
    assert(fid>=3,msg); % debugging error code
    while ~feof(fid) % read output.txt until you reach end of output.file
        str = fgetl(fid); % read line by line of fid
        %
        vec   =  sscanf(str,'It.: %*d, obj.: %f%*[^\n]',[2,Inf]); % what to read line by line
        vecg  = sscanf(str,'It.: %*d, obj.: %*f, g[0] : %f%*[^\n]',[2,Inf]);
        vecv  = sscanf(str,'# -volfrac: %f%*[^\n]',[2,Inf]);
        %
        num   = numel(vec); % size of vec : number of elements in vec
        numv  = numel(vecv);
        % to append vec into list out
        if num
            out(end+1,1:num) = vec;
        end
        if num
            outg(end+1,1:num)  = vecg;
        end
        if numv
            outv(end+1,1:numv) = vecv;
        end
    end
    fclose(fid);
    Tmp(1:size(out))   = out; % to store temporary out
    Tmpg(1:size(outg)) = outg;
    Tmpv(1:size(outv)) = outv;
    MatH(:,kk,1)=Tmp; %e.g. for ns=1
    Matg(:,kk,1) = Tmpg;
    Vol(:,kk,1)  = Tmpv;
end    
    

%% Section 2 Post treatement
% we check if there is violation or not of the contraints
% counters
cpt = 0; cpt1 = 0; cpt2 = 0;
% we store the indices of where there is no violation of the contraints
indp = []; indp1 = []; indp2 = [];
for ii=1:120 
    for i=1:200
        if (abs(Matg(i,ii,1)) <= Vol(1,ii,1))
            cpt = cpt+1;
            indp(cpt) = i;
        end
        if (abs(Matg(i,ii,2)) <= Vol(1,ii,2))
            cpt1 = cpt1+1;
            indp1(cpt1) = i;
        end
        if (abs(Matg(i,ii,3)) <= Vol(1,ii,3)) 
            cpt2 = cpt2+1;
            indp2(cpt2) = i;
        end
    end
   
end

% build up H : matrice we will need as parameter for Performance profiles 
% and Data profiles
% here, we extract the min of every problem for each solver
for ii=1:120 
    for i=1:200
        if (MatH(i,ii,1) == 0)
            MatH(i,ii,1) = 100000;
        end
        if (MatH(i,ii,2) == 0)
            MatH(i,ii,2) = 100000;
        end
        if (MatH(i,ii,3) == 0)
            MatH(i,ii,3) = 100000;
        end
    end    
end
%
[m,IND]   = min(MatH(:,:,1));
[m0,IND0] = min(MatH(:,:,2));
[m1,IND1] = min(MatH(:,:,3));
% we affect the value the rest of the evaluation of a problem 
% for solver we have succeed to solver the problem before maximum
% evaluation np
for ii=1:120
    MatH(IND(ii):end,ii,1)  = m(ii);
    MatH(IND0(ii):end,ii,2) = m0(ii);
    MatH(IND1(ii):end,ii,3) = m1(ii);
end   
% build N : complexity of each problem depending on 
% number of elements in the mesh for each problem
n1 = 88*88*176;
n2 = 128*64*64;
n3 = n1;
%
Nelxyz = [n1;n2;n3];
%
N = zeros(np,1);
%
% Cantilever
N(1:15)  = n1+1;
N(16:30) = n2+1;
N(31:45) = n3+1;
% Michell
N(46:55) = n1+1;
N(56:65) = n2+1;
N(66:75) = n3+1;
% Wheel
N(76:90)   = n1+1;
N(91:105)  = n2+1;
N(106:120) = n3+1;
%% Section 2 Build up of Performance profiles and Data profiles
% Performance profiles :
% Performance profiles were originally introduced in
% Benchmarking optimization software with performance profiles,
% E.D. Dolan and J.J. More', 
% Mathematical Programming, 91 (2002), 201--213.
%
% The subroutine returns a handle to lines in a performance profile.
%
% H contains a three dimensional array of function values.
% H(f,p,s) = function value # f for problem p and solver s.
% gate is a positive constant reflecting the convergence tolerance.
% logplot=1 is used to indicate that a log (base 2) plot is desired.
gate = 1e-2; % eg. 1e-1, 1e-2, 1e-3
logplot = 10;
h_p = perf_profile(MatH,gate,logplot);
% Data profiles :
% The subroutine returns a handle to lines in a data profile.
%
% H contains a three dimensional array of function values.
% H(f,p,s) = function value # f for problem p and solver s.
% N is an np-by-1 vector of (positive) budget units. If simplex
% gradients are desired, then N(p) would be n(p)+1, where n(p) is
% the number of variables for problem p.
gate = 1e-2;
h_d = data_profile(MatH,N,gate);




