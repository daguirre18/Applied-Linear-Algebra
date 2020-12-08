%% PART 1- SYMMETRIC
clear all; close all; clc

%1a: FIND GROUND TRUTH
m=10;
A=randn(m,m);
A=A*A'; %will end up as size m by m
diff=A-A'; %should be 0 if A is symmetric

[V,D]=eigs(A,m); %to find ground truth eigenvalues & eigenvectors
%D is the diagonal matrix of the eigenvalues; 
%V's columns are the corresponding eigenvectors

truelargesteig=max(diag(D)) %value of the largest eigenvalue


% 1b: FIND LARGEST EIGENVALUE USING POWER ITER
%make initial vector
vec=randn(m,1);
vec=vec/norm(vec); %to make a vector unit length we do vector/norm(vector)

for k=1:30
    w=A*vec;
    vec=w/norm(w); %normalize
    lambda(k)=(vec')*A*vec; %to keep track of lambda values
    difference(k)=abs(truelargesteig-lambda(k));  %see how close we are to
    %the true largest eigenvalue
end

powerlargesteig=max(lambda)

% Compare the accuracy of the method as a function of iterations
figure(1)
plot(difference)
xlabel('Iteration Number'), 
ylabel('Difference between power iter & eigs value')
title('Accuracy of Power Iter Method in Finding Largest Eigenvalue')

%%
%1c: Find all 10 eigenvalues by using Rayleigh Quotient iteration and
%guessing initial "eigenvectors"

%pick initial v and lambda 
%vec1=randn(m,1); when we tried a completely random initial eigenvector, 
%our algorithm did not find any of the corresponding eigenvalues

%we will try strategically choosing initial eigenvector
vec1=V(:,1)+0.2;
vec1=vec1/norm(vec1); %to make vector unit length we do vector/norm(vector)
lambda1=vec1'*A*vec1; %corresponding Rayleigh quotient
iter1=0;
for k=1:30
    iter1=iter1+1;
    w1=(A-lambda1*eye(m,m))\vec1; %from (A-lambda^(k-1)*I)w=v^(k-1)
    vec1=w1/norm(w1); 
    lambda1=(vec1')*A*vec1; %keep track of lambda values; Rayleigh quotient
    difference1(k)=abs(D(1,1)-lambda1);
    if abs(D(1,1)-lambda1)<10^(-12)
        break
    end
end

%pick initial v and lambda
vec2=V(:,2)*2+0.2;
vec2=vec2/norm(vec2); 
lambda2=vec2'*A*vec2; %corresponding Rayleigh quotient
iter2=0;
for k=1:30
    iter2=iter2+1;
    w2=(A-lambda2*eye(m,m))\vec2; %from (A-lambda^(k-1)*I)w=v^(k-1)
    vec2=w2/norm(w2); 
    lambda2=(vec2')*A*vec2; %keep track of lambda values; Rayleigh quotient
    difference2(k)=abs(D(2,2)-lambda2); %accuracy
    if abs(D(2,2)-lambda2)<10^(-12)
        break
    end
end

vec3=V(:,3)+0.2;
vec3=vec3/norm(vec3); 
lambda3=vec3'*A*vec3; %corresponding Rayleigh quotient
iter3=0;
for k=1:30
    iter3=iter3+1;
    w3=(A-lambda3*eye(m,m))\vec3; %from (A-lambda^(k-1)*I)w=v^(k-1)
    vec3=w3/norm(w3); 
    lambda3=(vec3')*A*vec3; %keep track of lambda values; Rayleigh quotient
    difference3(k)=abs(D(3,3)-lambda3);
    if abs(D(3,3)-lambda3)<10^(-12)
        break
    end
end

vec4=V(:,4)+0.2;
vec4=vec4/norm(vec4);
lambda4=vec4'*A*vec4; %corresponding Rayleigh quotient
iter4=0;
for k=1:30
    iter4=iter4+1;
    w4=(A-lambda4*eye(m,m))\vec4; %from (A-lambda^(k-1)*I)w=v^(k-1)
    vec4=w4/norm(w4); 
    lambda4=(vec4')*A*vec4; %keep track of lambda values; Rayleigh quotient
    difference4(k)=abs(D(4,4)-lambda4);
    if abs(D(4,4)-lambda4)<10^(-12)
        break
    end
end

vec5=V(:,5)+0.2;
vec5=vec5/norm(vec5); 
lambda5=vec5'*A*vec5; %corresponding Rayleigh quotient
iter5=0;
for k=1:30
    iter5=iter5+1;
    w5=(A-lambda5*eye(m,m))\vec5; %from (A-lambda^(k-1)*I)w=v^(k-1)
    vec5=w5/norm(w5); 
    lambda5=(vec5')*A*vec5; %keep track of lambda values; Rayleigh quotient
    difference5(k)=abs(D(5,5)-lambda5);
    if abs(D(5,5)-lambda5)<10^(-12)
        break
    end
end

vec6=V(:,6)*2+0.2;
vec6=vec6/norm(vec6); 
lambda6=vec6'*A*vec6; %corresponding Rayleigh quotient
iter6=0;
for k=1:30
    iter6=iter6+1;
    w6=(A-lambda6*eye(m,m))\vec6; %from (A-lambda^(k-1)*I)w=v^(k-1)
    vec6=w6/norm(w6); 
    lambda6=(vec6')*A*vec6; %keep track of lambda values; Rayleigh quotient
    difference6(k)=abs(D(6,6)-lambda6);
    if abs(D(6,6)-lambda6)<10^(-12)
        break
    end
end

vec7=V(:,7)*15+0.2;
vec7=vec7/norm(vec7); 
lambda7=vec7'*A*vec7; %corresponding Rayleigh quotient
iter7=0;
for k=1:30
    iter7=iter7+1;
    w7=(A-lambda7*eye(m,m))\vec7; %from (A-lambda^(k-1)*I)w=v^(k-1)
    vec7=w7/norm(w7); 
    lambda7=(vec7')*A*vec7; %keep track of lambda values; Rayleigh quotient
    difference7(k)=abs(D(7,7)-lambda7);
    if abs(D(7,7)-lambda7)<10^(-12)
        break
    end
end

vec8=V(:,8)+0.2;
vec8=vec8/norm(vec8); 
lambda8=vec8'*A*vec8; %corresponding Rayleigh quotient
iter8=0;
for k=1:30
    iter8=iter8+1;
    w8=(A-lambda8*eye(m,m))\vec8; %from (A-lambda^(k-1)*I)w=v^(k-1)
    vec8=w8/norm(w8); 
    lambda8=(vec8')*A*vec8; %keep track of lambda values; Rayleigh quotient
    difference8(k)=abs(D(8,8)-lambda8);
    if abs(D(8,8)-lambda8)<10^(-12)
        break
    end
end

vec9=V(:,9)*20+0.2;
vec9=vec9/norm(vec9); 
lambda9=vec9'*A*vec9; %corresponding Rayleigh quotient
iter9=0;
for k=1:30
    iter9=iter9+1;
    w9=(A-lambda9*eye(m,m))\vec9; %from (A-lambda^(k-1)*I)w=v^(k-1)
    vec9=w9/norm(w9); 
    lambda9=(vec9')*A*vec9; %keep track of lambda values; Rayleigh quotient
    difference9(k)=abs(D(9,9)-lambda9);
    if abs(D(9,9)-lambda9)<10^(-12)
        break
    end
end

vec10=V(:,10)*15+0.2;
vec10=vec10/norm(vec10); 
lambda10=vec10'*A*vec10; %corresponding Rayleigh quotient
iter10=0;
for k=1:30
    iter10=iter10+1;
    w10=(A-lambda10*eye(m,m))\vec10; %from (A-lambda^(k-1)*I)w=v^(k-1)
    vec10=w10/norm(w10); 
    lambda10=(vec10')*A*vec10; %keep track of lambda; Rayleigh quotient
    difference10(k)=abs(D(10,10)-lambda10);
    if abs(D(10,10)-lambda10)<10^(-12)
        break
    end
end



%%
%Compare your accuracy of the method as a function of iterations
figure(2)
plot(difference1,'Linewidth',[2]), hold on
plot(difference2,'Linewidth',[2]), hold on
plot(difference3,'Linewidth',[2]), hold on
plot(difference4,'Linewidth',[2]), hold on
plot(difference5,'Linewidth',[2]), hold on
plot(difference6,'Linewidth',[2]), hold on
plot(difference7,'Linewidth',[2]), hold on
plot(difference8,'Linewidth',[2]), hold on
plot(difference9(:,2),'Linewidth',[2]), hold on
plot(difference10(:,2),'Linewidth',[2]), hold on
xlabel('iterations')
ylabel('difference to true eigenvalue')
title('Accuracy as a function of iterations')

%Note: after inspecting the final eigenvectors with the true ones, we see 
%that they match.
%%
%plot eigenvalues
lambdassymm=[lambda1 lambda2 lambda3 lambda4 lambda5 lambda6 lambda7 lambda8 lambda9 lambda10];
figure(3)
plot(diag(D),'o','Linewidth',[2]),hold on
plot(lambdassymm,'.')
%dots are the eigenvalues using Rayleigh
%circles are the true eigenvalues from eigs command
title('Eigenvalues Using Eigs Command VS Using Rayleigh Quotient')
xlabel('eigenvalue #'), ylabel('value')

%% Showing that using a random vector didnt work
%vec1=randn(m,1); when we tried a completely random initial eigenvector, 
%our algorithm did not find any of the corresponding eigenvalues
vec1=randn(m,1);
vec1=vec1/norm(vec1); 
lambda1=vec1'*A*vec1; %corresponding Rayleigh quotient
iter1=0;
for k=1:30
    iter1=iter1+1;
    w1=(A-lambda1*eye(m,m))\vec1; %from (A-lambda^(k-1)*I)w=v^(k-1)
    vec1=w1/norm(w1); 
    lambda1=(vec1')*A*vec1; %keep track of lambda values; Rayleigh quotient
    difference1(k)=abs(D(1,1)-lambda1);
    if abs(D(1,1)-lambda1)<10^(-12)
        break
    end
end

figure(34)
plot(diag(D(1,1)),'o','Linewidth',[2]),hold on
plot(lambda1,'.')
title('First Eigenvalue Using Eigs Command VS Using Rayleigh Quotient')
xlabel('eigenvalue #'), ylabel('value')