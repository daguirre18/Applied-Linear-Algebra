clear, close, clc
%% PART I 

% Compute matrices we will work with
matrix1=randn(60,40);
matrix1cond=cond(matrix1); %condition # of matrix
%matrix 1 is well-conditioned; cond # is low
matrix2=randn(32,30);
matrix2cond=cond(matrix2); %condition # of matrix
%matrix 2 is well-conditioned; cond # is low

matrix3=randn(55,10);
matrix3=[matrix3 matrix3(:,1:10)]; %our matrix becomes size 55x20
matrix3cond=cond(matrix3); %condition # of matrix
%matrix 3 is ill-conditioned; cond # is very high

matrix4=randn(55,20);
matrix4(:,15)=matrix4(:,10); %make the 15th column be the same as the 10th 
matrix4cond=cond(matrix4); %condition # of matrix
%matrix 4 is ill-conditioned; cond # is very high

%% PART I - Develop modified Gram-Schmidt orthogonalization procedure

[Q1modgs,R1modgs]=modifiedgs(matrix1);
[Q2modgs,R2modgs]=modifiedgs(matrix2);
[Q3modgs,R3modgs]=modifiedgs(matrix3);
[Q4modgs,R4modgs]=modifiedgs(matrix4);
%% PART I - Using qrfactor.m code built in class

[Q1qrfactor,R1qrfactor]=qrfactor(matrix1);  
[Q2qrfactor,R2qrfactor]=qrfactor(matrix2); 
[Q3qrfactor,R3qrfactor]=qrfactor(matrix3);
[Q4qrfactor,R4qrfactor]=qrfactor(matrix4);
%% PART I - MATLAB's qr

[Q1mat,R1mat]=qr(matrix1);
[Q2mat,R2mat]=qr(matrix2);
[Q3mat,R3mat]=qr(matrix3);
[Q4mat,R4mat]=qr(matrix4);
%% Compare methods
%Use norms to compare errors
norm1modifiedgs=norm(Q1modgs*R1modgs-matrix1)
norm1qrfactor=norm(Q1qrfactor*R1qrfactor-matrix1)
norm1matlab=norm(Q1mat*R1mat-matrix1)

norm2modifiedgs=norm(Q2modgs*R2modgs-matrix2)
norm2qrfactor=norm(Q2qrfactor*R2qrfactor-matrix2)
norm2matlab=norm(Q2mat*R2mat-matrix2)


norm3modifiedgs=norm(Q3modgs*R3modgs-matrix3)
norm3qrfactor=norm(Q3qrfactor*R3qrfactor-matrix3)
norm3matlab=norm(Q3mat*R3mat-matrix3)

norm4modifiedgs=norm(Q4modgs*R4modgs-matrix4)
norm4qrfactor=norm(Q4qrfactor*R4qrfactor-matrix4)
norm4matlab=norm(Q4mat*R4mat-matrix4)


%% PART II

%PART II
dt=0.001;
x=[1.920:dt:2.080];
pLHS=(x-2).^9;
pRHS=x.^9-18*x.^8+144*x.^7-672*x.^6+2016*x.^5-4032*x.^4+5376*x.^3-4608*x.^2+2304*x-512;
figure(1)
subplot(1,2,1), plot(pRHS)
xlabel('x'), ylabel('p(x)')
title('Using RHS')
subplot(1,2,2), plot(pLHS)
title('Using LHS')
xlabel('x'), ylabel('p(x)')


%Check the difference between the two answers
for k=1:161
    diff(k,:)=pLHS(k)-pRHS(k);
end
%differences are very small


%% PART III

%PART III
%Construct random matrix where m>n; use A=randn(m,n)
A1=randn(5,4);
cond1=cond(A1);

A2=randn(8,6);
cond2=cond(A2);

A3=randn(15,10);
cond3=cond(A3);

A4=randn(50,40);
cond4=cond(A4);

A5=randn(200,150);
cond5=cond(A5);

A6=randn(3500,3480);
cond6=cond(A6);


%%

%lets use Achosen as our matrix
Achosen=randn(50,49);
%copy the first column of A and append it as the (n+1)th column of A 
[m,n] = size(Achosen); %figure out size of A matrix
A=[Achosen Achosen(:,1)];
condition=cond(A); %condition number
determinant=det(A); %determinant of the matrix
%%
%take the appended column and add noise to it
eps=10^-2;
random=rand(m,1);
Anoise1 = [A(:,1:n) A(:,n+1) +  eps*random];
condAnoise1=cond(Anoise1);
%note: see what happens to the condition number as a function of epsilon
eps=2*10^-2;
Anoise2 = [A(:,1:n) A(:,n+1) +  eps*random];
condAnoise2=cond(Anoise2);
eps=3*10^-2;
Anoise3 = [A(:,1:n) A(:,n+1) +  eps*random];
condAnoise3=cond(Anoise3);
eps=4*10^-2;
Anoise4 = [A(:,1:n) A(:,n+1) +  eps*random];
condAnoise4=cond(Anoise4);
%NOTE: condition # goes DOWN as we add noise 
%%
condnoisevector=[condition condAnoise1 condAnoise2 condAnoise3 condAnoise4]
figure(3)
bar(log(condnoisevector))
title('Condition # When Using Various Epsilon')
xticklabels({'no noise','\epsilon=1*10^{-2}','\epsilon=2*10^{-2}','\epsilon=3*10^{-2}','\epsilon=4*10^{-2}'})
ylabel('Condition Number')
xlabel('Noise Value')



