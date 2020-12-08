% PART 2: YALE FACES
clear all;close all, clc

%Load cropped faces data
CX=[];
croppedfaces = dir(fullfile('c:\Users\diana\Documents\AMATH584- Autumn 2020\\Homework\HW2\CroppedYale\yaleB*\yaleB*'));
%create for loop to import data
    for jj = 1 : length(croppedfaces)
        cropped_data = importdata(fullfile(croppedfaces(jj).folder, croppedfaces(jj).name ) );
        data1=reshape(cropped_data,1,32256);
        CX=[CX,data1]; %create matrix with loaded data
    end

%Reshape so each column is a new image
%This gives 2432 total images;
    X1=reshape(CX,32256,2432);

%%
%X1 is our data matrix; will be Xcropped after converting to double prec
Xcropped=double(X1); %convert to double precision
Xcropped=Xcropped'*Xcropped; %use correlation matrix
[m,n]=size(Xcropped); %compute data size
[u,s,v]=svd(Xcropped,'econ');  %perform the econ svd; this is full matrix
largestsvdmode=max(diag(s))  %leading order svd mode
%% 2a: Power Iter to find dominant eigenvector and eigenvalue. 

vec=randn(m,1); %initial guess
vec=vec/norm(vec); %normalize
for k=1:50
    w=Xcropped*vec;
    vec=w/norm(w);
    lambda(k)=(vec')*Xcropped*vec; %to keep track of lambda values
    if lambda(k)==largestsvdmode break 
    end
end

yalepowerlargesteig=max(lambda)  %find dominant eigenvalue
%vec is our dominant eigenvector; size 2432x1

%Compare to leading SVD mode
%Note: they are the same
figure(10)
subplot(2,1,1), plot((v(:,1)))
title('Leading SVD Mode')
subplot(2,1,2), plot((vec))
title('Dominant Eigenvector')
%% 2b: Use randomized sampling to produce the SVD matrices: U, S, V

% STAGE A
%construct a sampling matrix omega
[M,N]=size(Xcropped);
K=30; %K is the number of random projections we pick; subsampling
Omega=randn(N,K); %random projections
Y=Xcropped*Omega; %data matrix times the random projections
size(Y) %should be M by K

%Now we do a QR decomposition
[Q,R]=qr(Y,0);
size(Q) %should be size M by K aka 2432 by 30
 
%we have an orthonormal set of 30 vectors that span the column set

% STAGE B
%take matrix and project it onto column space and do svd
B=(Q')*Xcropped;
[U,S,V]=svd(B,'econ');
uapprox=Q*U; %to get back to the approximation of 
%what U wouldve been for the full decomposition

figure(11)
x=1:1:2432;
%compare the first 3 columns of U; 
%compare results from svd of full matrix
%and results from randomized sampling
subplot(3,1,1), plot(x,u(:,1),'k',x,uapprox(:,1),'r:')
title('U approximations for the first 3 columns compared to SVD results of full matrix')
subplot(3,1,2), plot(x,u(:,2),'c',x,uapprox(:,2),'b:')
subplot(3,1,3), plot(x,u(:,3),'g',x,uapprox(:,3),'k:')
%dashed lines are for approximations from random sampling

%Compare sigma plots
figure(12)
subplot(1,2,1), plot(diag(s),'ko','Linewidth',[2]); 
title('Singular Value Spectrum: true modes'), xlabel('mode')
subplot(1,2,2), plot(diag(S),'ko','Linewidth',[2]); 
title('Singular Value Spectrum: randomized sampling modes'), xlabel('mode')

figure(13)
subplot(1,2,1), plot(diag(s)/sum(diag(s)),'ko','Linewidth',[2]); 
%gives % variance within svd modes
title('Variance within true modes'), xlabel('mode')
subplot(1,2,2), plot(diag(S)/sum(diag(S)),'ko','Linewidth',[2]); 
title('Variance within randomized sampling modes'), xlabel('mode')
