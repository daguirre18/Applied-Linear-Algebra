clear all; close all; clc

%TRAIN DATA

datatrain=load ('mnist_train.csv');
%create labels by taking the first column and all rows since thats how the 
%csv file is set up
labelstrain=datatrain(:,1);  %contains all the labels for the 60,000 rows

imagestrain=datatrain(:,2:end); %all rows and columns 2 to the end, 
%since column 1 is just the label
%we end up with size 60000x784
%each row is an image
Atrain=imagestrain'; %transpose so each column is an image

%to create image
%reshape(imagestrain(1,:),28,28); 
%images(1,:) will give you a row vector of 784 pixel values but we want 
%it to be a 28*28 image so we reshape; this represents the image for our 
%first pixel
%so to see images we use imagesc(reshape(imagestrain(1,:),28,28)')
%we can check if image is correct by using labels(row number)

% TEST DATA
datatest=load ('mnist_test.csv');
labelstest=datatest(:,1);  %contains all the labels for the 10,000 rows

imagestest=datatest(:,2:end); 
%we end up with size 10000x784
Atest=imagestest'; %each image is a column
%%
%create B matrix(output)
one=[1 0 0 0 0 0 0 0 0 0]';
two=[0 1 0 0 0 0 0 0 0 0]';
three=[0 0 1 0 0 0 0 0 0 0]';
four=[0 0 0 1 0 0 0 0 0 0]';
five=[0 0 0 0 1 0 0 0 0 0]';
six=[0 0 0 0 0 1 0 0 0 0]';
seven=[0 0 0 0 0 0 1 0 0 0]';
eight=[0 0 0 0 0 0 0 1 0 0]';
nine=[0 0 0 0 0 0 0 0 1 0]';
zero=[0 0 0 0 0 0 0 0 0 1]';
none=[0 0 0 0 0 0 0 0 0 0]';
Bgeneral=[one two three four five six seven eight nine zero]';

%we are working w/ an overdetermined system 
%(more rows than columns; tall skinny matrix)

%create B(for train); arrange each image according to their label
 B=[];
for j=1:60000 %because we have 60000 images
    %ex:the first label is 5, so the first column of B should be [0000100000]
    if labelstrain(j,1)==1;
        B(:,j)=one;
        %then that column in the B matrix is [1 0000.. ]
    elseif labelstrain(j,1)==2;
         B(:,j)=two;
    elseif labelstrain(j,1)==3;
         B(:,j)=three;
    elseif labelstrain(j,1)==4;
         B(:,j)=four;
    elseif labelstrain(j,1)==5;
         B(:,j)=five;
    elseif labelstrain(j,1)==6;
         B(:,j)=six;
    elseif labelstrain(j,1)==7;
         B(:,j)=seven;
    elseif labelstrain(j,1)==8;
         B(:,j)=eight;      
    elseif labelstrain(j,1)==9;
         B(:,j)=nine;
    else labelstrain(j,1)==0;
         B(:,j)=zero;
    end
end


%create Btest; arrange each image according to their label
 Btest=[];
for j=1:10000 %because we have 10000 images
    
    if labelstest(j,1)==1;
        Btest(:,j)=one;
    elseif labelstest(j,1)==2;
         Btest(:,j)=two;
    elseif labelstest(j,1)==3;
         Btest(:,j)=three;
    elseif labelstest(j,1)==4;
         Btest(:,j)=four;
    elseif labelstest(j,1)==5;
         Btest(:,j)=five;
    elseif labelstest(j,1)==6;
         Btest(:,j)=six;
    elseif labelstest(j,1)==7;
         Btest(:,j)=seven;
    elseif labelstest(j,1)==8;
         Btest(:,j)=eight;      
    elseif labelstest(j,1)==9;
         Btest(:,j)=nine;
    else labelstest(j,1)==0;
         Btest(:,j)=zero;
    end
end


%% Regressions using AX=B solvers

Atraintranspose=Atrain'; Btranspose=B';

lambda=0.01;%lambda will be used in the sparsity promoting algorithms 
%lambda is the L1 penalty

q=randperm(60000); %random permutation variable
Atrainrandom=Atraintranspose(q(1:60000),:);
Btrainrandom=Btranspose(q(1:60000),:);
%select 6,000 for each kfold to do 10fold validation=60,000 total
A1st6k=Atrainrandom(q(1:6000),:);
A2nd6k=Atrainrandom(q(6001:12000),:);
A3rd6k=Atrainrandom(q(12001:18000),:);
A4th6k=Atrainrandom(q(18001:24000),:);
A5th6k=Atrainrandom(q(24001:30000),:);
A6th6k=Atrainrandom(q(30001:36000),:);
A7th6k=Atrainrandom(q(36001:42000),:);
A8th6k=Atrainrandom(q(42001:48000),:);
A9th6k=Atrainrandom(q(48001:54000),:);
A10th6k=Atrainrandom(q(54001:60000),:);
B1st6k=Btrainrandom(q(1:6000),:);
B2nd6k=Btrainrandom(q(6001:12000),:);
B3rd6k=Btrainrandom(q(12001:18000),:);
B4th6k=Btrainrandom(q(18001:24000),:);
B5th6k=Btrainrandom(q(24001:30000),:);
B6th6k=Btrainrandom(q(30001:36000),:);
B7th6k=Btrainrandom(q(36001:42000),:);
B8th6k=Btrainrandom(q(42001:48000),:);
B9th6k=Btrainrandom(q(48001:54000),:);
B10th6k=Btrainrandom(q(54001:60000),:);
   
%%
% pinv on train data using 10fold
    a1_1=pinv(A1st6k)*B1st6k; 
    a1_2=pinv(A2nd6k)*B2nd6k; 
    a1_3=pinv(A3rd6k)*B3rd6k; 
    a1_4=pinv(A4th6k)*B4th6k; 
    a1_5=pinv(A5th6k)*B5th6k; 
    a1_6=pinv(A6th6k)*B6th6k; 
    a1_7=pinv(A7th6k)*B7th6k; 
    a1_8=pinv(A8th6k)*B8th6k; 
    a1_9=pinv(A9th6k)*B9th6k; 
    a1_10=pinv(A10th6k)*B10th6k;
    
%backlash on train data using 10fold
a2_1=A1st6k\B1st6k; 
a2_2=A2nd6k\B2nd6k; 
a2_3=A3rd6k\B3rd6k; 
a2_4=A4th6k\B4th6k; 
a2_5=A5th6k\B5th6k; 
a2_6=A6th6k\B6th6k; 
a2_7=A7th6k\B7th6k; 
a2_8=A8th6k\B8th6k; 
a2_9=A9th6k\B9th6k; 
a2_10=A10th6k\B10th6k; 


%lasso on train data using 10fold and lambda 0.01
%loop over columns of B
tic
for jj=1:10
    [a3_1(:,jj),stats(:,jj)]=lasso(A1st6k,B1st6k(:,jj),'Lambda',lambda);  
    [a3_2(:,jj),stats(:,jj)]=lasso(A2nd6k,B2nd6k(:,jj),'Lambda',lambda); 
    [a3_3(:,jj),stats(:,jj)]=lasso(A3rd6k,B3rd6k(:,jj),'Lambda',lambda); 
    [a3_4(:,jj),stats(:,jj)]=lasso(A4th6k,B4th6k(:,jj),'Lambda',lambda); 
    [a3_5(:,jj),stats(:,jj)]=lasso(A5th6k,B5th6k(:,jj),'Lambda',lambda); 
    [a3_6(:,jj),stats(:,jj)]=lasso(A6th6k,B6th6k(:,jj),'Lambda',lambda); 
    [a3_7(:,jj),stats(:,jj)]=lasso(A7th6k,B7th6k(:,jj),'Lambda',lambda); 
    [a3_8(:,jj),stats(:,jj)]=lasso(A8th6k,B8th6k(:,jj),'Lambda',lambda); 
    [a3_9(:,jj),stats(:,jj)]=lasso(A9th6k,B9th6k(:,jj),'Lambda',lambda); 
    [a3_10(:,jj),stats(:,jj)]=lasso(A10th6k,B10th6k(:,jj),'Lambda',lambda); 
    
end
toc

%ridge on train data using 10fold
%looping over B columns
tic
%l1 penalty of 0.01
for jjjj=1:10;
    a6_1(:,jjjj)=ridge(B1st6k(:,jjjj),A1st6k(:,2:end),0.01,0);
    a6_2(:,jjjj)=ridge(B2nd6k(:,jjjj),A2nd6k(:,2:end),0.01,0); 
    a6_3(:,jjjj)=ridge(B3rd6k(:,jjjj),A3rd6k(:,2:end),0.01,0);
    a6_4(:,jjjj)=ridge(B4th6k(:,jjjj),A4th6k(:,2:end),0.01,0);
    a6_5(:,jjjj)=ridge(B5th6k(:,jjjj),A5th6k(:,2:end),0.01,0);
    a6_6(:,jjjj)=ridge(B6th6k(:,jjjj),A6th6k(:,2:end),0.01,0);
    a6_7(:,jjjj)=ridge(B7th6k(:,jjjj),A7th6k(:,2:end),0.01,0);
    a6_8(:,jjjj)=ridge(B8th6k(:,jjjj),A8th6k(:,2:end),0.01,0);
    a6_9(:,jjjj)=ridge(B9th6k(:,jjjj),A9th6k(:,2:end),0.01,0);
    a6_10(:,jjjj)=ridge(B10th6k(:,jjjj),A10th6k(:,2:end),0.01,0);
    
end
toc

%% APPLY TO TEST DATA
Atesttranspose=Atest'; %makes it 10,000x784

%before applying, average the little a's found from using the training set
xforpinv=(a1_1+a1_2+a1_3+a1_4+a1_5+a1_6+a1_7+a1_8+a1_9+a1_10)/10; 
%should be 784x10
xforbackslash=(a2_1+a2_2+a2_3+a2_4+a2_5+a2_6+a2_7+a2_8+a2_9+a2_10)/10;
xforlasso=(a3_1+a3_2+a3_3+a3_4+a3_5+a3_6+a3_7+a3_8+a3_9+a3_10)/10;
xforridge=(a6_1+a6_2+a6_3+a6_4+a6_5+a6_6+a6_7+a6_8+a6_9+a6_10)/10;

Busingpinv=Atesttranspose*xforpinv;
Busingbackslash=Atesttranspose*xforbackslash;
Businglasso=Atesttranspose*xforlasso;
Busingridge=Atesttranspose*xforridge;

Btesttranspose=Btest';
       

%% CHECKING ACCURACY
% AKA NUMBER OF CORRECT CLASSIFICATIONS FOR EACH METHOD
maxBpinv=zeros(10000,1);
pinvindicesB=zeros(10000,1);

for tb=1:10000
    [maxBpinv(tb),pinvindicesB(tb)]=maxk(Busingpinv(tb,:),1); 
    %^ max values of B and their index
end

Bpinvfake=zeros(10000,10);

for m=1:10000
    if pinvindicesB(m)==1
        Bpinvfake(m,:)=[1;0;0;0;0;0;0;0;0;0];
    elseif pinvindicesB(m)==2
        Bpinvfake(m,:)=[0;1;0;0;0;0;0;0;0;0];
    elseif pinvindicesB(m)==3
         Bpinvfake(m,:)=[0;0;1;0;0;0;0;0;0;0];   
    elseif pinvindicesB(m)==4
         Bpinvfake(m,:)=[0;0;0;1;0;0;0;0;0;0];       
    elseif pinvindicesB(m)==5
         Bpinvfake(m,:)=[0;0;0;0;1;0;0;0;0;0];           
    elseif pinvindicesB(m)==6
        Bpinvfake(m,:)=[0;0;0;0;0;1;0;0;0;0];
    elseif pinvindicesB(m)==7
        Bpinvfake(m,:)=[0;0;0;0;0;0;1;0;0;0];
    elseif pinvindicesB(m)==8
        Bpinvfake(m,:)=[0;0;0;0;0;0;0;1;0;0];
    elseif pinvindicesB(m)==9
        Bpinvfake(m,:)=[0;0;0;0;0;0;0;0;1;0];
    else 
        Bpinvfake(m,:)=[0;0;0;0;0;0;0;0;0;1];
    end 
end
   
        
Bpnvaccuracysum=zeros(10000,1);
for r=1:10000
    if Btesttranspose(r,:)==Bpinvfake(r,:)
        Bpnvaccuracysum(r)=1;
    else
        Bpnvaccuracysum(r)=0;
    end
end
numbercorrectusingpinv=sum(Bpnvaccuracysum); %number of images correct out of 10,000

%for backslash 
maxBbackslash=zeros(10000,1);
backslashindicesB=zeros(10000,1);

for tb=1:10000
    [maxBbackslash(tb),backslashindicesB(tb)]=maxk(Busingbackslash(tb,:),1);
end

Bbackslashfake=zeros(10000,10);

for m=1:10000
    if backslashindicesB(m)==1
        Bbackslashfake(m,:)=[1;0;0;0;0;0;0;0;0;0];
    elseif backslashindicesB(m)==2
        Bbackslashfake(m,:)=[0;1;0;0;0;0;0;0;0;0];
    elseif backslashindicesB(m)==3
         Bbackslashfake(m,:)=[0;0;1;0;0;0;0;0;0;0];   
    elseif backslashindicesB(m)==4
         Bbackslashfake(m,:)=[0;0;0;1;0;0;0;0;0;0];       
    elseif backslashindicesB(m)==5
         Bbackslashfake(m,:)=[0;0;0;0;1;0;0;0;0;0];           
    elseif backslashindicesB(m)==6
        Bbackslashfake(m,:)=[0;0;0;0;0;1;0;0;0;0];
    elseif backslashindicesB(m)==7
        Bbackslashfake(m,:)=[0;0;0;0;0;0;1;0;0;0];
    elseif backslashindicesB(m)==8
        Bbackslashfake(m,:)=[0;0;0;0;0;0;0;1;0;0];
    elseif backslashindicesB(m)==9
        Bbackslashfake(m,:)=[0;0;0;0;0;0;0;0;1;0];
    else 
        Bbackslashfake(m,:)=[0;0;0;0;0;0;0;0;0;1];
    end 
end
   
        
Bbackslashaccuracysum=zeros(10000,1);
for r=1:10000
    if Btesttranspose(r,:)==Bbackslashfake(r,:)
        Bbackslashaccuracysum(r)=1;
    else
        Bbackslashaccuracysum(r)=0;
    end
end
numbercorrectusingbackslash=sum(Bbackslashaccuracysum);



%for lasso using lambda=0.01
maxBlassopart1=zeros(10000,1);
lassopart1indicesB=zeros(10000,1);

for tb=1:10000
    [maxBlassopart1(tb),lassopart1indicesB(tb)]=maxk(Businglasso(tb,:),1);
end

Blassopart1fake=zeros(10000,10);

for m=1:10000
    if lassopart1indicesB(m)==1
        Blassopart1fake(m,:)=[1;0;0;0;0;0;0;0;0;0];
    elseif lassopart1indicesB(m)==2
        Blassopart1fake(m,:)=[0;1;0;0;0;0;0;0;0;0];
    elseif lassopart1indicesB(m)==3
         Blassopart1fake(m,:)=[0;0;1;0;0;0;0;0;0;0];   
    elseif lassopart1indicesB(m)==4
         Blassopart1fake(m,:)=[0;0;0;1;0;0;0;0;0;0];       
    elseif lassopart1indicesB(m)==5
         Blassopart1fake(m,:)=[0;0;0;0;1;0;0;0;0;0];           
    elseif lassopart1indicesB(m)==6
        Blassopart1fake(m,:)=[0;0;0;0;0;1;0;0;0;0];
    elseif lassopart1indicesB(m)==7
        Blassopart1fake(m,:)=[0;0;0;0;0;0;1;0;0;0];
    elseif lassopart1indicesB(m)==8
        Blassopart1fake(m,:)=[0;0;0;0;0;0;0;1;0;0];
    elseif lassopart1indicesB(m)==9
        Blassopart1fake(m,:)=[0;0;0;0;0;0;0;0;1;0];
    else 
        Blassopart1fake(m,:)=[0;0;0;0;0;0;0;0;0;1];
    end 
end
        
Blassopart1accuracysum=zeros(10000,1);
for r=1:10000
    if Btesttranspose(r,:)==Blassopart1fake(r,:)
        Blassopart1accuracysum(r)=1;
    else
        Blassopart1accuracysum(r)=0;
    end
end
numbercorrectusinglassopart1=sum(Blassopart1accuracysum);

% using ridge with lambda 0.01
maxBridge=zeros(10000,1);
ridgeindicesB=zeros(10000,1);

for tb=1:10000
    [maxBridge(tb),ridgeindicesB(tb)]=maxk(Busingridge(tb,:),1);
end

Bridgefake=zeros(10000,10);

for m=1:10000
    if ridgeindicesB(m)==1
        Bridgefake(m,:)=[1;0;0;0;0;0;0;0;0;0];
    elseif ridgeindicesB(m)==2
        Bridgefake(m,:)=[0;1;0;0;0;0;0;0;0;0];
    elseif ridgeindicesB(m)==3
         Bridgefake(m,:)=[0;0;1;0;0;0;0;0;0;0];   
    elseif ridgeindicesB(m)==4
         Bridgefake(m,:)=[0;0;0;1;0;0;0;0;0;0];       
    elseif ridgeindicesB(m)==5
         Bridgefake(m,:)=[0;0;0;0;1;0;0;0;0;0];           
    elseif ridgeindicesB(m)==6
        Bridgefake(m,:)=[0;0;0;0;0;1;0;0;0;0];
    elseif ridgeindicesB(m)==7
        Bridgefake(m,:)=[0;0;0;0;0;0;1;0;0;0];
    elseif ridgeindicesB(m)==8
        Bridgefake(m,:)=[0;0;0;0;0;0;0;1;0;0];
    elseif ridgeindicesB(m)==9
        Bridgefake(m,:)=[0;0;0;0;0;0;0;0;1;0];
    else 
        Bridgefake(m,:)=[0;0;0;0;0;0;0;0;0;1];
    end 
end
   
        
Bridgeaccuracysum=zeros(10000,1);
for r=1:10000
    if Btesttranspose(r,:)==Bridgefake(r,:)
        Bridgeaccuracysum(r)=1;
    else
        Bridgeaccuracysum(r)=0;
    end
end
numbercorrectusingridge=sum(Bridgeaccuracysum); 
%number of images correct out of 10,000


%% 
%"By promoting sparsity, determine and rank which pixels in the MNIST set 
%are most informative for correctly labeling the digits." 

%lasso naturally promotes sparsity so we use lasso
%lasso on train data using 10fold and using various lambdas

lambda1=0.01; %what we used in part 1
lambda2=0.03;
lambda3=0.05;
lambda4=0.07;
lambda5=0.09;
lambda6=0.1;

for k=1:10  %to loop over columns
    [a1st6klambda1(:,k),stats(:,k)]=lasso(A1st6k,B1st6k(:,k),'Lambda',lambda1); 
    [a2nd6klambda1(:,k),stats(:,k)]=lasso(A2nd6k,B2nd6k(:,k),'Lambda',lambda1); 
    [a3rd6klambda1(:,k),stats(:,k)]=lasso(A3rd6k,B3rd6k(:,k),'Lambda',lambda1); 
    [a4th6klambda1(:,k),stats(:,k)]=lasso(A4th6k,B4th6k(:,k),'Lambda',lambda1); 
    [a5th6klambda1(:,k),stats(:,k)]=lasso(A5th6k,B5th6k(:,k),'Lambda',lambda1); 
    [a6th6klambda1(:,k),stats(:,k)]=lasso(A6th6k,B6th6k(:,k),'Lambda',lambda1); 
    [a7th6klambda1(:,k),stats(:,k)]=lasso(A7th6k,B7th6k(:,k),'Lambda',lambda1); 
    [a8th6klambda1(:,k),stats(:,k)]=lasso(A8th6k,B8th6k(:,k),'Lambda',lambda1); 
    [a9th6klambda1(:,k),stats(:,k)]=lasso(A9th6k,B9th6k(:,k),'Lambda',lambda1); 
    [a10th6klambda1(:,k),stats(:,k)]=lasso(A10th6k,B10th6k(:,k),'Lambda',lambda1); 

    %next lambda
    [a1st6klambda2(:,k),stats(:,k)]=lasso(A1st6k,B1st6k(:,k),'Lambda',lambda2); 
    [a2nd6klambda2(:,k),stats(:,k)]=lasso(A2nd6k,B2nd6k(:,k),'Lambda',lambda2); 
    [a3rd6klambda2(:,k),stats(:,k)]=lasso(A3rd6k,B3rd6k(:,k),'Lambda',lambda2); 
    [a4th6klambda2(:,k),stats(:,k)]=lasso(A4th6k,B4th6k(:,k),'Lambda',lambda2); 
    [a5th6klambda2(:,k),stats(:,k)]=lasso(A5th6k,B5th6k(:,k),'Lambda',lambda2); 
    [a6th6klambda2(:,k),stats(:,k)]=lasso(A6th6k,B6th6k(:,k),'Lambda',lambda2); 
    [a7th6klambda2(:,k),stats(:,k)]=lasso(A7th6k,B7th6k(:,k),'Lambda',lambda2); 
    [a8th6klambda2(:,k),stats(:,k)]=lasso(A8th6k,B8th6k(:,k),'Lambda',lambda2); 
    [a9th6klambda2(:,k),stats(:,k)]=lasso(A9th6k,B9th6k(:,k),'Lambda',lambda2); 
    [a10th6klambda2(:,k),stats(:,k)]=lasso(A10th6k,B10th6k(:,k),'Lambda',lambda2); 

    
    [a1st6klambda3(:,k),stats(:,k)]=lasso(A1st6k,B1st6k(:,k),'Lambda',lambda3);  
    [a2nd6klambda3(:,k),stats(:,k)]=lasso(A2nd6k,B2nd6k(:,k),'Lambda',lambda3); 
    [a3rd6klambda3(:,k),stats(:,k)]=lasso(A3rd6k,B3rd6k(:,k),'Lambda',lambda3); 
    [a4th6klambda3(:,k),stats(:,k)]=lasso(A4th6k,B4th6k(:,k),'Lambda',lambda3); 
    [a5th6klambda3(:,k),stats(:,k)]=lasso(A5th6k,B5th6k(:,k),'Lambda',lambda3); 
    [a6th6klambda3(:,k),stats(:,k)]=lasso(A6th6k,B6th6k(:,k),'Lambda',lambda3); 
    [a7th6klambda3(:,k),stats(:,k)]=lasso(A7th6k,B7th6k(:,k),'Lambda',lambda3); 
    [a8th6klambda3(:,k),stats(:,k)]=lasso(A8th6k,B8th6k(:,k),'Lambda',lambda3); 
    [a9th6klambda3(:,k),stats(:,k)]=lasso(A9th6k,B9th6k(:,k),'Lambda',lambda3); 
    [a10th6klambda3(:,k),stats(:,k)]=lasso(A10th6k,B10th6k(:,k),'Lambda',lambda3); 

    
    [a1st6klambda4(:,k),stats(:,k)]=lasso(A1st6k,B1st6k(:,k),'Lambda',lambda4); 
    [a2nd6klambda4(:,k),stats(:,k)]=lasso(A2nd6k,B2nd6k(:,k),'Lambda',lambda4); 
    [a3rd6klambda4(:,k),stats(:,k)]=lasso(A3rd6k,B3rd6k(:,k),'Lambda',lambda4); 
    [a4th6klambda4(:,k),stats(:,k)]=lasso(A4th6k,B4th6k(:,k),'Lambda',lambda4); 
    [a5th6klambda4(:,k),stats(:,k)]=lasso(A5th6k,B5th6k(:,k),'Lambda',lambda4); 
    [a6th6klambda4(:,k),stats(:,k)]=lasso(A6th6k,B6th6k(:,k),'Lambda',lambda4); 
    [a7th6klambda4(:,k),stats(:,k)]=lasso(A7th6k,B7th6k(:,k),'Lambda',lambda4); 
    [a8th6klambda4(:,k),stats(:,k)]=lasso(A8th6k,B8th6k(:,k),'Lambda',lambda4); 
    [a9th6klambda4(:,k),stats(:,k)]=lasso(A9th6k,B9th6k(:,k),'Lambda',lambda4); 
    [a10th6klambda4(:,k),stats(:,k)]=lasso(A10th6k,B10th6k(:,k),'Lambda',lambda4); 

    
    [a1st6klambda5(:,k),stats(:,k)]=lasso(A1st6k,B1st6k(:,k),'Lambda',lambda5);  
    [a2nd6klambda5(:,k),stats(:,k)]=lasso(A2nd6k,B2nd6k(:,k),'Lambda',lambda5); 
    [a3rd6klambda5(:,k),stats(:,k)]=lasso(A3rd6k,B3rd6k(:,k),'Lambda',lambda5); 
    [a4th6klambda5(:,k),stats(:,k)]=lasso(A4th6k,B4th6k(:,k),'Lambda',lambda5); 
    [a5th6klambda5(:,k),stats(:,k)]=lasso(A5th6k,B5th6k(:,k),'Lambda',lambda5); 
    [a6th6klambda5(:,k),stats(:,k)]=lasso(A6th6k,B6th6k(:,k),'Lambda',lambda5); 
    [a7th6klambda5(:,k),stats(:,k)]=lasso(A7th6k,B7th6k(:,k),'Lambda',lambda5); 
    [a8th6klambda5(:,k),stats(:,k)]=lasso(A8th6k,B8th6k(:,k),'Lambda',lambda5); 
    [a9th6klambda5(:,k),stats(:,k)]=lasso(A9th6k,B9th6k(:,k),'Lambda',lambda5); 
    [a10th6klambda5(:,k),stats(:,k)]=lasso(A10th6k,B10th6k(:,k),'Lambda',lambda5); 

    
    [a1st6klambda6(:,k),stats(:,k)]=lasso(A1st6k,B1st6k(:,k),'Lambda',lambda6);  
    [a2nd6klambda6(:,k),stats(:,k)]=lasso(A2nd6k,B2nd6k(:,k),'Lambda',lambda6); 
    [a3rd6klambda6(:,k),stats(:,k)]=lasso(A3rd6k,B3rd6k(:,k),'Lambda',lambda6); 
    [a4th6klambda6(:,k),stats(:,k)]=lasso(A4th6k,B4th6k(:,k),'Lambda',lambda6); 
    [a5th6klambda6(:,k),stats(:,k)]=lasso(A5th6k,B5th6k(:,k),'Lambda',lambda6); 
    [a6th6klambda6(:,k),stats(:,k)]=lasso(A6th6k,B6th6k(:,k),'Lambda',lambda6); 
    [a7th6klambda6(:,k),stats(:,k)]=lasso(A7th6k,B7th6k(:,k),'Lambda',lambda6); 
    [a8th6klambda6(:,k),stats(:,k)]=lasso(A8th6k,B8th6k(:,k),'Lambda',lambda6); 
    [a9th6klambda6(:,k),stats(:,k)]=lasso(A9th6k,B9th6k(:,k),'Lambda',lambda6); 
    [a10th6klambda6(:,k),stats(:,k)]=lasso(A10th6k,B10th6k(:,k),'Lambda',lambda6); 

end 

%average the values to get one loading matrix
%Ax=b
xlassolambda1=(a1st6klambda1+a2nd6klambda1+a3rd6klambda1+a4th6klambda1+a5th6klambda1+a6th6klambda1+a7th6klambda1+a8th6klambda1+a9th6klambda1+a10th6klambda1)/10;
xlassolambda2=(a1st6klambda2+a2nd6klambda2+a3rd6klambda2+a4th6klambda2+a5th6klambda2+a6th6klambda2+a7th6klambda2+a8th6klambda2+a9th6klambda2+a10th6klambda2)/10;
xlassolambda3=(a1st6klambda3+a2nd6klambda3+a3rd6klambda3+a4th6klambda3+a5th6klambda3+a6th6klambda3+a7th6klambda3+a8th6klambda3+a9th6klambda3+a10th6klambda3)/10;
xlassolambda4=(a1st6klambda4+a2nd6klambda4+a3rd6klambda4+a4th6klambda4+a5th6klambda4+a6th6klambda4+a7th6klambda4+a8th6klambda4+a9th6klambda4+a10th6klambda4)/10;
xlassolambda5=(a1st6klambda5+a2nd6klambda5+a3rd6klambda5+a4th6klambda5+a5th6klambda5+a6th6klambda5+a7th6klambda5+a8th6klambda5+a9th6klambda5+a10th6klambda5)/10;
xlassolambda6=(a1st6klambda6+a2nd6klambda6+a3rd6klambda6+a4th6klambda6+a5th6klambda6+a6th6klambda6+a7th6klambda6+a8th6klambda6+a9th6klambda6+a10th6klambda6)/10;


%will need these to check each lambda's accuracy
Businglambda1=Atesttranspose*xlassolambda1;
Businglambda2=Atesttranspose*xlassolambda2;
Businglambda3=Atesttranspose*xlassolambda3;
Businglambda4=Atesttranspose*xlassolambda4;
Businglambda5=Atesttranspose*xlassolambda5;
Businglambda6=Atesttranspose*xlassolambda6;

figure(1)
%plot the x for each lambda to show sparsity as lambda increases
subplot(2,3,1); plot(xlassolambda1,'k')
title('Pixel Weights with \lambda=0.01'), axis([0 784 -0.0005 0.0007])
xlabel(''); ylabel('weight')
subplot(2,3,2); plot(xlassolambda2,'k')
title('Pixel Weights with \lambda=0.03'), axis([0 784 -0.0005 0.0007])
xlabel('');
subplot(2,3,3); plot(xlassolambda3,'k'), axis([0 784 -0.0005 0.0007])
title('Pixel Weights with \lambda=0.05')
xlabel('');
subplot(2,3,4); plot(xlassolambda4,'k'), axis([0 784 -0.0005 0.0007])
title('Pixel Weights with \lambda=0.07')
xlabel('Pixel Number'); ylabel('weight')
subplot(2,3,5); plot(xlassolambda5,'k'), axis([0 784 -0.0005 0.0007])
title('Pixel Weights with \lambda=0.09')
xlabel('Pixel Number');
subplot(2,3,6); plot(xlassolambda6,'k'), axis([0 784 -0.0005 0.0007])
title('Pixel Weights with \lambda=0.1')
xlabel('Pixel Number');

%% ACCURACIES USING LASSO WITH DIFFERENT LAMBDAS
% lasso accuracies using different lambdas
maxBlambda1=zeros(10000,1);
lambda1indicesB=zeros(10000,1);

for tb=1:10000
    [maxBlambda1(tb),lambda1indicesB(tb)]=maxk(Businglambda1(tb,:),1); 
end

Blambda1fake=zeros(10000,10);

for m=1:10000
    if lambda1indicesB(m)==1
        Blambda1fake(m,:)=[1;0;0;0;0;0;0;0;0;0];
    elseif lambda1indicesB(m)==2
        Blambda1fake(m,:)=[0;1;0;0;0;0;0;0;0;0];
    elseif lambda1indicesB(m)==3
         Blambda1fake(m,:)=[0;0;1;0;0;0;0;0;0;0];   
    elseif lambda1indicesB(m)==4
         Blambda1fake(m,:)=[0;0;0;1;0;0;0;0;0;0];       
    elseif lambda1indicesB(m)==5
         Blambda1fake(m,:)=[0;0;0;0;1;0;0;0;0;0];           
    elseif lambda1indicesB(m)==6
        Blambda1fake(m,:)=[0;0;0;0;0;1;0;0;0;0];
    elseif lambda1indicesB(m)==7
        Blambda1fake(m,:)=[0;0;0;0;0;0;1;0;0;0];
    elseif lambda1indicesB(m)==8
        Blambda1fake(m,:)=[0;0;0;0;0;0;0;1;0;0];
    elseif lambda1indicesB(m)==9
        Blambda1fake(m,:)=[0;0;0;0;0;0;0;0;1;0];
    else 
        Blambda1fake(m,:)=[0;0;0;0;0;0;0;0;0;1];
    end 
end
   
        
Blambda1accuracysum=zeros(10000,1);
for r=1:10000
    if Btesttranspose(r,:)==Blambda1fake(r,:)
        Blambda1accuracysum(r)=1;
    else
        Blambda1accuracysum(r)=0;
    end
end
numbercorrectusinglambda1=sum(Blambda1accuracysum);


maxBlambda2=zeros(10000,1);
lambda2indicesB=zeros(10000,1);

for tb=1:10000
    [maxBlambda2(tb),lambda2indicesB(tb)]=maxk(Businglambda2(tb,:),1); 
end

Blambda2fake=zeros(10000,10);

for m=1:10000
    if lambda2indicesB(m)==1
        Blambda2fake(m,:)=[1;0;0;0;0;0;0;0;0;0];
    elseif lambda2indicesB(m)==2
        Blambda2fake(m,:)=[0;1;0;0;0;0;0;0;0;0];
    elseif lambda2indicesB(m)==3
         Blambda2fake(m,:)=[0;0;1;0;0;0;0;0;0;0];   
    elseif lambda2indicesB(m)==4
         Blambda2fake(m,:)=[0;0;0;1;0;0;0;0;0;0];       
    elseif lambda2indicesB(m)==5
         Blambda2fake(m,:)=[0;0;0;0;1;0;0;0;0;0];           
    elseif lambda2indicesB(m)==6
        Blambda2fake(m,:)=[0;0;0;0;0;1;0;0;0;0];
    elseif lambda2indicesB(m)==7
        Blambda2fake(m,:)=[0;0;0;0;0;0;1;0;0;0];
    elseif lambda2indicesB(m)==8
        Blambda2fake(m,:)=[0;0;0;0;0;0;0;1;0;0];
    elseif lambda2indicesB(m)==9
        Blambda2fake(m,:)=[0;0;0;0;0;0;0;0;1;0];
    else 
        Blambda2fake(m,:)=[0;0;0;0;0;0;0;0;0;1];
    end 
end
   
        
Blambda2accuracysum=zeros(10000,1);
for r=1:10000
    if Btesttranspose(r,:)==Blambda2fake(r,:)
        Blambda2accuracysum(r)=1;
    else
        Blambda2accuracysum(r)=0;
    end
end
numbercorrectusinglambda2=sum(Blambda2accuracysum);


maxBlambda3=zeros(10000,1);
lambda3indicesB=zeros(10000,1);

for tb=1:10000
    [maxBlambda3(tb),lambda3indicesB(tb)]=maxk(Businglambda3(tb,:),1); 
end

Blambda3fake=zeros(10000,10);

for m=1:10000
    if lambda3indicesB(m)==1
        Blambda3fake(m,:)=[1;0;0;0;0;0;0;0;0;0];
    elseif lambda3indicesB(m)==2
        Blambda3fake(m,:)=[0;1;0;0;0;0;0;0;0;0];
    elseif lambda3indicesB(m)==3
         Blambda3fake(m,:)=[0;0;1;0;0;0;0;0;0;0];   
    elseif lambda3indicesB(m)==4
         Blambda3fake(m,:)=[0;0;0;1;0;0;0;0;0;0];       
    elseif lambda3indicesB(m)==5
         Blambda3fake(m,:)=[0;0;0;0;1;0;0;0;0;0];           
    elseif lambda3indicesB(m)==6
        Blambda3fake(m,:)=[0;0;0;0;0;1;0;0;0;0];
    elseif lambda3indicesB(m)==7
        Blambda3fake(m,:)=[0;0;0;0;0;0;1;0;0;0];
    elseif lambda3indicesB(m)==8
        Blambda3fake(m,:)=[0;0;0;0;0;0;0;1;0;0];
    elseif lambda3indicesB(m)==9
        Blambda3fake(m,:)=[0;0;0;0;0;0;0;0;1;0];
    else 
        Blambda3fake(m,:)=[0;0;0;0;0;0;0;0;0;1];
    end 
end
   
        
Blambda3accuracysum=zeros(10000,1);
for r=1:10000
    if Btesttranspose(r,:)==Blambda3fake(r,:)
        Blambda3accuracysum(r)=1;
    else
        Blambda3accuracysum(r)=0;
    end
end
numbercorrectusinglambda3=sum(Blambda3accuracysum);



maxBlambda4=zeros(10000,1);
lambda4indicesB=zeros(10000,1);

for tb=1:10000
    [maxBlambda4(tb),lambda4indicesB(tb)]=maxk(Businglambda4(tb,:),1); 
end

Blambda4fake=zeros(10000,10);

for m=1:10000
    if lambda4indicesB(m)==1
        Blambda4fake(m,:)=[1;0;0;0;0;0;0;0;0;0];
    elseif lambda4indicesB(m)==2
        Blambda4fake(m,:)=[0;1;0;0;0;0;0;0;0;0];
    elseif lambda4indicesB(m)==3
         Blambda4fake(m,:)=[0;0;1;0;0;0;0;0;0;0];   
    elseif lambda4indicesB(m)==4
         Blambda4fake(m,:)=[0;0;0;1;0;0;0;0;0;0];       
    elseif lambda4indicesB(m)==5
         Blambda4fake(m,:)=[0;0;0;0;1;0;0;0;0;0];           
    elseif lambda4indicesB(m)==6
        Blambda4fake(m,:)=[0;0;0;0;0;1;0;0;0;0];
    elseif lambda4indicesB(m)==7
        Blambda4fake(m,:)=[0;0;0;0;0;0;1;0;0;0];
    elseif lambda4indicesB(m)==8
        Blambda4fake(m,:)=[0;0;0;0;0;0;0;1;0;0];
    elseif lambda4indicesB(m)==9
        Blambda4fake(m,:)=[0;0;0;0;0;0;0;0;1;0];
    else 
        Blambda4fake(m,:)=[0;0;0;0;0;0;0;0;0;1];
    end 
end
   
        
Blambda4accuracysum=zeros(10000,1);
for r=1:10000
    if Btesttranspose(r,:)==Blambda4fake(r,:)
        Blambda4accuracysum(r)=1;
    else
        Blambda4accuracysum(r)=0;
    end
end
numbercorrectusinglambda4=sum(Blambda4accuracysum);


maxBlambda5=zeros(10000,1);
lambda5indicesB=zeros(10000,1);

for tb=1:10000
    [maxBlambda5(tb),lambda5indicesB(tb)]=maxk(Businglambda5(tb,:),1); 
end

Blambda5fake=zeros(10000,10);

for m=1:10000
    if lambda5indicesB(m)==1
        Blambda5fake(m,:)=[1;0;0;0;0;0;0;0;0;0];
    elseif lambda5indicesB(m)==2
        Blambda5fake(m,:)=[0;1;0;0;0;0;0;0;0;0];
    elseif lambda5indicesB(m)==3
         Blambda5fake(m,:)=[0;0;1;0;0;0;0;0;0;0];   
    elseif lambda5indicesB(m)==4
         Blambda5fake(m,:)=[0;0;0;1;0;0;0;0;0;0];       
    elseif lambda5indicesB(m)==5
         Blambda5fake(m,:)=[0;0;0;0;1;0;0;0;0;0];           
    elseif lambda5indicesB(m)==6
        Blambda5fake(m,:)=[0;0;0;0;0;1;0;0;0;0];
    elseif lambda5indicesB(m)==7
        Blambda5fake(m,:)=[0;0;0;0;0;0;1;0;0;0];
    elseif lambda5indicesB(m)==8
        Blambda5fake(m,:)=[0;0;0;0;0;0;0;1;0;0];
    elseif lambda5indicesB(m)==9
        Blambda5fake(m,:)=[0;0;0;0;0;0;0;0;1;0];
    else 
        Blambda5fake(m,:)=[0;0;0;0;0;0;0;0;0;1];
    end 
end
   
        
Blambda5accuracysum=zeros(10000,1);
for r=1:10000
    if Btesttranspose(r,:)==Blambda5fake(r,:)
        Blambda5accuracysum(r)=1;
    else
        Blambda5accuracysum(r)=0;
    end
end
numbercorrectusinglambda5=sum(Blambda5accuracysum);


maxBlambda6=zeros(10000,1);
lambda6indicesB=zeros(10000,1);

for tb=1:10000
    [maxBlambda6(tb),lambda6indicesB(tb)]=maxk(Businglambda6(tb,:),1); 
end

Blambda6fake=zeros(10000,10);

for m=1:10000
    if lambda6indicesB(m)==1
        Blambda6fake(m,:)=[1;0;0;0;0;0;0;0;0;0];
    elseif lambda6indicesB(m)==2
        Blambda6fake(m,:)=[0;1;0;0;0;0;0;0;0;0];
    elseif lambda6indicesB(m)==3
         Blambda6fake(m,:)=[0;0;1;0;0;0;0;0;0;0];   
    elseif lambda6indicesB(m)==4
         Blambda6fake(m,:)=[0;0;0;1;0;0;0;0;0;0];       
    elseif lambda6indicesB(m)==5
         Blambda6fake(m,:)=[0;0;0;0;1;0;0;0;0;0];           
    elseif lambda6indicesB(m)==6
        Blambda6fake(m,:)=[0;0;0;0;0;1;0;0;0;0];
    elseif lambda6indicesB(m)==7
        Blambda6fake(m,:)=[0;0;0;0;0;0;1;0;0;0];
    elseif lambda6indicesB(m)==8
        Blambda6fake(m,:)=[0;0;0;0;0;0;0;1;0;0];
    elseif lambda6indicesB(m)==9
        Blambda6fake(m,:)=[0;0;0;0;0;0;0;0;1;0];
    else 
        Blambda6fake(m,:)=[0;0;0;0;0;0;0;0;0;1];
    end 
end
   
        
Blambda6accuracysum=zeros(10000,1);
for r=1:10000
    if Btesttranspose(r,:)==Blambda6fake(r,:)
        Blambda6accuracysum(r)=1;
    else
        Blambda6accuracysum(r)=0;
    end
end
numbercorrectusinglambda6=sum(Blambda6accuracysum);

%% New approach: choose certain important pixels to promote sparsity


lambda1=0.01; %what we used in part 1

for k=1:10  %to loop over columns
    [a1st6klambda1(:,k),stats(:,k)]=lasso(A1st6k,B1st6k(:,k),'Lambda',lambda1);  
    [a2nd6klambda1(:,k),stats(:,k)]=lasso(A2nd6k,B2nd6k(:,k),'Lambda',lambda1); 
    [a3rd6klambda1(:,k),stats(:,k)]=lasso(A3rd6k,B3rd6k(:,k),'Lambda',lambda1); 
    [a4th6klambda1(:,k),stats(:,k)]=lasso(A4th6k,B4th6k(:,k),'Lambda',lambda1); 
    [a5th6klambda1(:,k),stats(:,k)]=lasso(A5th6k,B5th6k(:,k),'Lambda',lambda1); 
    [a6th6klambda1(:,k),stats(:,k)]=lasso(A6th6k,B6th6k(:,k),'Lambda',lambda1); 
    [a7th6klambda1(:,k),stats(:,k)]=lasso(A7th6k,B7th6k(:,k),'Lambda',lambda1); 
    [a8th6klambda1(:,k),stats(:,k)]=lasso(A8th6k,B8th6k(:,k),'Lambda',lambda1); 
    [a9th6klambda1(:,k),stats(:,k)]=lasso(A9th6k,B9th6k(:,k),'Lambda',lambda1); 
    [a10th6klambda1(:,k),stats(:,k)]=lasso(A10th6k,B10th6k(:,k),'Lambda',lambda1); 
end 


%average to get loadings matrix
xlassolambda1=(a1st6klambda1+a2nd6klambda1+a3rd6klambda1+a4th6klambda1+a5th6klambda1+a6th6klambda1+a7th6klambda1+a8th6klambda1+a9th6klambda1+a10th6klambda1)/10;


%% TO SELECT IMPORTANT PIXELS

for p=1:784
pixellasso(p,:)=sum(norm(xlassolambda1(p,:))); 
%^criteria for determining importance
end

%rank pixels by importance
using50pixels=zeros(784,10);
[Bs,I]=maxk(pixellasso,50); %using 50 important pixels
sortingI=sort(I);

t=zeros(784,1);
for s=1:50  %of important pixels we want to keep
    for ss=1:784  %number of total pixels
        if ss==sortingI(s)
            t(ss)=t(ss)+1; %make 1
        else 
            t(ss)=t(ss)+0;  %make 0
        end
    end
end
    %t is now 784x1
    using50pixels=t.*xlassolambda1; %784x10; 
% ^ only keeping the ones from our matrix that are important
    
    %use with test data to get output matrix
    Busing50pixels=Atesttranspose*using50pixels;
    

    
 using100pixels=zeros(784,10);
[Bs,I]=maxk(pixellasso,100); %using 100 important pixels
sortingI=sort(I);

t=zeros(784,1);
for s=1:100  %of important pixels we want to keep
    for ss=1:784  %number of total pixels
        if ss==sortingI(s)
            t(ss)=t(ss)+1; %make 1
        else 
            t(ss)=t(ss)+0;  %make 0
        end
    end
end
    %t is now 784x1
    using100pixels=t.*xlassolambda1; %784x10; 
    
    %use in test
    Busing100pixels=Atesttranspose*using100pixels;
    
    
    
using200pixels=zeros(784,10);
[Bs,I]=maxk(pixellasso,200); %using 200 important pixels
sortingI=sort(I);

t=zeros(784,1);
for s=1:200  %of important pixels we want to keep
    for ss=1:784  %number of total pixels
        if ss==sortingI(s)
            t(ss)=t(ss)+1; %make 1
        else 
            t(ss)=t(ss)+0;  %make 0
        end
    end
end
    %t is now 784x1
    using200pixels=t.*xlassolambda1; 
    
    %use in test
    Busing200pixels=Atesttranspose*using200pixels;
    
    
using300pixels=zeros(784,10);
[Bs,I]=maxk(pixellasso,300); %using 300 important pixels
sortingI=sort(I);

t=zeros(784,1);
for s=1:300  %of important pixels we want to keep
    for ss=1:784  %number of total pixels
        if ss==sortingI(s)
            t(ss)=t(ss)+1; %make 1
        else 
            t(ss)=t(ss)+0;  %make 0
        end
    end
end
    %t is now 784x1
    using300pixels=t.*xlassolambda1;
    
    %use in test
    Busing300pixels=Atesttranspose*using300pixels;
    
    
using400pixels=zeros(784,10);
[Bs,I]=maxk(pixellasso,400); 
sortingI=sort(I);

t=zeros(784,1);
for s=1:400  %of important pixels we want to keep
    for ss=1:784  %number of total pixels
        if ss==sortingI(s)
            t(ss)=t(ss)+1; %make 1
        else 
            t(ss)=t(ss)+0;  %make 0
        end
    end
end
    %t is now 784x1
    using400pixels=t.*xlassolambda1; 
    
    %use in test
    Busing400pixels=Atesttranspose*using400pixels;
    
    
using500pixels=zeros(784,10);
[Bs,I]=maxk(pixellasso,500); 
sortingI=sort(I);

t=zeros(784,1);
for s=1:500  %of important pixels we want to keep
    for ss=1:784  %number of total pixels
        if ss==sortingI(s)
            t(ss)=t(ss)+1; %make 1
        else 
            t(ss)=t(ss)+0;  %make 0
        end
    end
end
    %t is now 784x1
    using500pixels=t.*xlassolambda1; 
    
    %use in test
    Busing500pixels=Atesttranspose*using500pixels;
    
using600pixels=zeros(784,10);
[Bs,I]=maxk(pixellasso,600); 
sortingI=sort(I);

t=zeros(784,1);
for s=1:600  %of important pixels we want to keep
    for ss=1:784  %number of total pixels
        if ss==sortingI(s)
            t(ss)=t(ss)+1; %make 1
        else 
            t(ss)=t(ss)+0;  %make 0
        end
    end
end
    %t is now 784x1
    using600pixels=t.*xlassolambda1; 
    
    %use in test
    Busing600pixels=Atesttranspose*using600pixels;
    
    
using700pixels=zeros(784,10);
[Bs,I]=maxk(pixellasso,784); 
sortingI=sort(I);

t=zeros(784,1);
for s=1:783  %of important pixels we want to keep
    for ss=1:784  %number of total pixels
        if ss==sortingI(s)
            t(ss)=t(ss)+1; %make 1
        else 
            t(ss)=t(ss)+0;  %make 0
        end
    end
end
    %t is now 784x1
    using700pixels=t.*xlassolambda1; 
    
    %use in test
    Busing700pixels=Atesttranspose*using700pixels;
       
    
%% Lasso accuracies using different number of pixels
maxBlasso50pixels=zeros(10000,1);
lasso50pixelsindicesB=zeros(10000,1);

for tb=1:10000
    [maxBlasso50pixels(tb),lasso50pixelsindicesB(tb)]=maxk(Busing50pixels(tb,:),1); %find max value in each row and indice
end

Blasso50pixelsfake=zeros(10000,10);  
%will be our predicted classification matrix 

for m=1:10000
    if lasso50pixelsindicesB(m)==1 
        %if indice is 1, then it will be shown as what's specified below
        %etc
        Blasso50pixelsfake(m,:)=[1;0;0;0;0;0;0;0;0;0];
    elseif lasso50pixelsindicesB(m)==2
        Blasso50pixelsfake(m,:)=[0;1;0;0;0;0;0;0;0;0];
    elseif lasso50pixelsindicesB(m)==3
         Blasso50pixelsfake(m,:)=[0;0;1;0;0;0;0;0;0;0];   
    elseif lasso50pixelsindicesB(m)==4
         Blasso50pixelsfake(m,:)=[0;0;0;1;0;0;0;0;0;0];       
    elseif lasso50pixelsindicesB(m)==5
         Blasso50pixelsfake(m,:)=[0;0;0;0;1;0;0;0;0;0];           
    elseif lasso50pixelsindicesB(m)==6
        Blasso50pixelsfake(m,:)=[0;0;0;0;0;1;0;0;0;0];
    elseif lasso50pixelsindicesB(m)==7
        Blasso50pixelsfake(m,:)=[0;0;0;0;0;0;1;0;0;0];
    elseif lasso50pixelsindicesB(m)==8
        Blasso50pixelsfake(m,:)=[0;0;0;0;0;0;0;1;0;0];
    elseif lasso50pixelsindicesB(m)==9
        Blasso50pixelsfake(m,:)=[0;0;0;0;0;0;0;0;1;0];
    else 
        Blasso50pixelsfake(m,:)=[0;0;0;0;0;0;0;0;0;1];
    end 
end
   
        
Blasso50pixelsaccuracysum=zeros(10000,1);
for r=1:10000
    if Btesttranspose(r,:)==Blasso50pixelsfake(r,:)
        Blasso50pixelsaccuracysum(r)=1;
        %if actual label and predicted label match
    else
        Blasso50pixelsaccuracysum(r)=0;
        %if actual label and predicted label dont match
    end
end
numbercorrectusinglasso50pixels=sum(Blasso50pixelsaccuracysum);



maxBlasso100pixels=zeros(10000,1);
lasso100pixelsindicesB=zeros(10000,1);

for tb=1:10000
    [maxBlasso100pixels(tb),lasso100pixelsindicesB(tb)]=maxk(Busing100pixels(tb,:),1);
end

Blasso100pixelsfake=zeros(10000,10);

for m=1:10000
    if lasso100pixelsindicesB(m)==1
        Blasso100pixelsfake(m,:)=[1;0;0;0;0;0;0;0;0;0];
    elseif lasso100pixelsindicesB(m)==2
        Blasso100pixelsfake(m,:)=[0;1;0;0;0;0;0;0;0;0];
    elseif lasso100pixelsindicesB(m)==3
         Blasso100pixelsfake(m,:)=[0;0;1;0;0;0;0;0;0;0];   
    elseif lasso100pixelsindicesB(m)==4
         Blasso100pixelsfake(m,:)=[0;0;0;1;0;0;0;0;0;0];       
    elseif lasso100pixelsindicesB(m)==5
         Blasso100pixelsfake(m,:)=[0;0;0;0;1;0;0;0;0;0];           
    elseif lasso100pixelsindicesB(m)==6
        Blasso100pixelsfake(m,:)=[0;0;0;0;0;1;0;0;0;0];
    elseif lasso100pixelsindicesB(m)==7
        Blasso100pixelsfake(m,:)=[0;0;0;0;0;0;1;0;0;0];
    elseif lasso100pixelsindicesB(m)==8
        Blasso100pixelsfake(m,:)=[0;0;0;0;0;0;0;1;0;0];
    elseif lasso100pixelsindicesB(m)==9
        Blasso100pixelsfake(m,:)=[0;0;0;0;0;0;0;0;1;0];
    else 
        Blasso100pixelsfake(m,:)=[0;0;0;0;0;0;0;0;0;1];
    end 
end
   
        
Blasso100pixelsaccuracysum=zeros(10000,1);
for r=1:10000
    if Btesttranspose(r,:)==Blasso100pixelsfake(r,:)
        Blasso100pixelsaccuracysum(r)=1;
    else
        Blasso100pixelsaccuracysum(r)=0;
    end
end
numbercorrectusinglasso100pixels=sum(Blasso100pixelsaccuracysum);


maxBlasso200pixels=zeros(10000,1);
lasso200pixelsindicesB=zeros(10000,1);

for tb=1:10000
    [maxBlasso200pixels(tb),lasso200pixelsindicesB(tb)]=maxk(Busing200pixels(tb,:),1);
end

Blasso200pixelsfake=zeros(10000,10);

for m=1:10000
    if lasso200pixelsindicesB(m)==1
        Blasso200pixelsfake(m,:)=[1;0;0;0;0;0;0;0;0;0];
    elseif lasso200pixelsindicesB(m)==2
        Blasso200pixelsfake(m,:)=[0;1;0;0;0;0;0;0;0;0];
    elseif lasso200pixelsindicesB(m)==3
         Blasso200pixelsfake(m,:)=[0;0;1;0;0;0;0;0;0;0];   
    elseif lasso200pixelsindicesB(m)==4
         Blasso200pixelsfake(m,:)=[0;0;0;1;0;0;0;0;0;0];       
    elseif lasso200pixelsindicesB(m)==5
         Blasso200pixelsfake(m,:)=[0;0;0;0;1;0;0;0;0;0];           
    elseif lasso200pixelsindicesB(m)==6
        Blasso200pixelsfake(m,:)=[0;0;0;0;0;1;0;0;0;0];
    elseif lasso200pixelsindicesB(m)==7
        Blasso200pixelsfake(m,:)=[0;0;0;0;0;0;1;0;0;0];
    elseif lasso200pixelsindicesB(m)==8
        Blasso200pixelsfake(m,:)=[0;0;0;0;0;0;0;1;0;0];
    elseif lasso200pixelsindicesB(m)==9
        Blasso200pixelsfake(m,:)=[0;0;0;0;0;0;0;0;1;0];
    else 
        Blasso200pixelsfake(m,:)=[0;0;0;0;0;0;0;0;0;1];
    end 
end
   
        
Blasso200pixelsaccuracysum=zeros(10000,1);
for r=1:10000
    if Btesttranspose(r,:)==Blasso200pixelsfake(r,:)
        Blasso200pixelsaccuracysum(r)=1;
    else
        Blasso200pixelsaccuracysum(r)=0;
    end
end
numbercorrectusinglasso200pixels=sum(Blasso200pixelsaccuracysum);



maxBlasso300pixels=zeros(10000,1);
lasso300pixelsindicesB=zeros(10000,1);

for tb=1:10000
    [maxBlasso300pixels(tb),lasso300pixelsindicesB(tb)]=maxk(Busing300pixels(tb,:),1);
end

Blasso300pixelsfake=zeros(10000,10);

for m=1:10000
    if lasso300pixelsindicesB(m)==1
        Blasso300pixelsfake(m,:)=[1;0;0;0;0;0;0;0;0;0];
    elseif lasso300pixelsindicesB(m)==2
        Blasso300pixelsfake(m,:)=[0;1;0;0;0;0;0;0;0;0];
    elseif lasso300pixelsindicesB(m)==3
         Blasso300pixelsfake(m,:)=[0;0;1;0;0;0;0;0;0;0];   
    elseif lasso300pixelsindicesB(m)==4
         Blasso300pixelsfake(m,:)=[0;0;0;1;0;0;0;0;0;0];       
    elseif lasso300pixelsindicesB(m)==5
         Blasso300pixelsfake(m,:)=[0;0;0;0;1;0;0;0;0;0];           
    elseif lasso300pixelsindicesB(m)==6
        Blasso300pixelsfake(m,:)=[0;0;0;0;0;1;0;0;0;0];
    elseif lasso300pixelsindicesB(m)==7
        Blasso300pixelsfake(m,:)=[0;0;0;0;0;0;1;0;0;0];
    elseif lasso300pixelsindicesB(m)==8
        Blasso300pixelsfake(m,:)=[0;0;0;0;0;0;0;1;0;0];
    elseif lasso300pixelsindicesB(m)==9
        Blasso300pixelsfake(m,:)=[0;0;0;0;0;0;0;0;1;0];
    else 
        Blasso300pixelsfake(m,:)=[0;0;0;0;0;0;0;0;0;1];
    end 
end
   
        
Blasso300pixelsaccuracysum=zeros(10000,1);
for r=1:10000
    if Btesttranspose(r,:)==Blasso300pixelsfake(r,:)
        Blasso300pixelsaccuracysum(r)=1;
    else
        Blasso300pixelsaccuracysum(r)=0;
    end
end
numbercorrectusinglasso300pixels=sum(Blasso300pixelsaccuracysum);


    
maxBlasso400pixels=zeros(10000,1);
lasso400pixelsindicesB=zeros(10000,1);

for tb=1:10000
    [maxBlasso400pixels(tb),lasso400pixelsindicesB(tb)]=maxk(Busing400pixels(tb,:),1);
end

Blasso400pixelsfake=zeros(10000,10);

for m=1:10000
    if lasso400pixelsindicesB(m)==1
        Blasso400pixelsfake(m,:)=[1;0;0;0;0;0;0;0;0;0];
    elseif lasso400pixelsindicesB(m)==2
        Blasso400pixelsfake(m,:)=[0;1;0;0;0;0;0;0;0;0];
    elseif lasso400pixelsindicesB(m)==3
         Blasso400pixelsfake(m,:)=[0;0;1;0;0;0;0;0;0;0];   
    elseif lasso400pixelsindicesB(m)==4
         Blasso400pixelsfake(m,:)=[0;0;0;1;0;0;0;0;0;0];       
    elseif lasso400pixelsindicesB(m)==5
         Blasso400pixelsfake(m,:)=[0;0;0;0;1;0;0;0;0;0];           
    elseif lasso400pixelsindicesB(m)==6
        Blasso400pixelsfake(m,:)=[0;0;0;0;0;1;0;0;0;0];
    elseif lasso400pixelsindicesB(m)==7
        Blasso400pixelsfake(m,:)=[0;0;0;0;0;0;1;0;0;0];
    elseif lasso400pixelsindicesB(m)==8
        Blasso400pixelsfake(m,:)=[0;0;0;0;0;0;0;1;0;0];
    elseif lasso400pixelsindicesB(m)==9
        Blasso400pixelsfake(m,:)=[0;0;0;0;0;0;0;0;1;0];
    else 
        Blasso400pixelsfake(m,:)=[0;0;0;0;0;0;0;0;0;1];
    end 
end
   
        
Blasso400pixelsaccuracysum=zeros(10000,1);
for r=1:10000
    if Btesttranspose(r,:)==Blasso400pixelsfake(r,:)
        Blasso400pixelsaccuracysum(r)=1;
    else
        Blasso400pixelsaccuracysum(r)=0;
    end
end
numbercorrectusinglasso400pixels=sum(Blasso400pixelsaccuracysum);

   
maxBlasso500pixels=zeros(10000,1);
lasso500pixelsindicesB=zeros(10000,1);

for tb=1:10000
    [maxBlasso500pixels(tb),lasso500pixelsindicesB(tb)]=maxk(Busing500pixels(tb,:),1);
end

Blasso500pixelsfake=zeros(10000,10);

for m=1:10000
    if lasso500pixelsindicesB(m)==1
        Blasso500pixelsfake(m,:)=[1;0;0;0;0;0;0;0;0;0];
    elseif lasso500pixelsindicesB(m)==2
        Blasso500pixelsfake(m,:)=[0;1;0;0;0;0;0;0;0;0];
    elseif lasso500pixelsindicesB(m)==3
         Blasso500pixelsfake(m,:)=[0;0;1;0;0;0;0;0;0;0];   
    elseif lasso500pixelsindicesB(m)==4
         Blasso500pixelsfake(m,:)=[0;0;0;1;0;0;0;0;0;0];       
    elseif lasso500pixelsindicesB(m)==5
         Blasso500pixelsfake(m,:)=[0;0;0;0;1;0;0;0;0;0];           
    elseif lasso500pixelsindicesB(m)==6
        Blasso500pixelsfake(m,:)=[0;0;0;0;0;1;0;0;0;0];
    elseif lasso500pixelsindicesB(m)==7
        Blasso500pixelsfake(m,:)=[0;0;0;0;0;0;1;0;0;0];
    elseif lasso500pixelsindicesB(m)==8
        Blasso500pixelsfake(m,:)=[0;0;0;0;0;0;0;1;0;0];
    elseif lasso500pixelsindicesB(m)==9
        Blasso500pixelsfake(m,:)=[0;0;0;0;0;0;0;0;1;0];
    else 
        Blasso500pixelsfake(m,:)=[0;0;0;0;0;0;0;0;0;1];
    end 
end
   
        
Blasso500pixelsaccuracysum=zeros(10000,1);
for r=1:10000
    if Btesttranspose(r,:)==Blasso500pixelsfake(r,:)
        Blasso500pixelsaccuracysum(r)=1;
    else
        Blasso500pixelsaccuracysum(r)=0;
    end
end
numbercorrectusinglasso500pixels=sum(Blasso500pixelsaccuracysum);


maxBlasso600pixels=zeros(10000,1);
lasso600pixelsindicesB=zeros(10000,1);

for tb=1:10000
    [maxBlasso600pixels(tb),lasso600pixelsindicesB(tb)]=maxk(Busing600pixels(tb,:),1);
end

Blasso600pixelsfake=zeros(10000,10);

for m=1:10000
    if lasso600pixelsindicesB(m)==1
        Blasso600pixelsfake(m,:)=[1;0;0;0;0;0;0;0;0;0];
    elseif lasso600pixelsindicesB(m)==2
        Blasso600pixelsfake(m,:)=[0;1;0;0;0;0;0;0;0;0];
    elseif lasso600pixelsindicesB(m)==3
         Blasso600pixelsfake(m,:)=[0;0;1;0;0;0;0;0;0;0];   
    elseif lasso600pixelsindicesB(m)==4
         Blasso600pixelsfake(m,:)=[0;0;0;1;0;0;0;0;0;0];       
    elseif lasso600pixelsindicesB(m)==5
         Blasso600pixelsfake(m,:)=[0;0;0;0;1;0;0;0;0;0];           
    elseif lasso600pixelsindicesB(m)==6
        Blasso600pixelsfake(m,:)=[0;0;0;0;0;1;0;0;0;0];
    elseif lasso600pixelsindicesB(m)==7
        Blasso600pixelsfake(m,:)=[0;0;0;0;0;0;1;0;0;0];
    elseif lasso600pixelsindicesB(m)==8
        Blasso600pixelsfake(m,:)=[0;0;0;0;0;0;0;1;0;0];
    elseif lasso600pixelsindicesB(m)==9
        Blasso600pixelsfake(m,:)=[0;0;0;0;0;0;0;0;1;0];
    else 
        Blasso600pixelsfake(m,:)=[0;0;0;0;0;0;0;0;0;1];
    end 
end
   
        
Blasso600pixelsaccuracysum=zeros(10000,1);
for r=1:10000
    if Btesttranspose(r,:)==Blasso600pixelsfake(r,:)
        Blasso600pixelsaccuracysum(r)=1;
    else
        Blasso600pixelsaccuracysum(r)=0;
    end
end
numbercorrectusinglasso600pixels=sum(Blasso600pixelsaccuracysum);


maxBlasso700pixels=zeros(10000,1);
lasso700pixelsindicesB=zeros(10000,1);

for tb=1:10000
    [maxBlasso700pixels(tb),lasso700pixelsindicesB(tb)]=maxk(Busing700pixels(tb,:),1);
end

Blasso700pixelsfake=zeros(10000,10);

for m=1:10000
    if lasso700pixelsindicesB(m)==1
        Blasso700pixelsfake(m,:)=[1;0;0;0;0;0;0;0;0;0];
    elseif lasso700pixelsindicesB(m)==2
        Blasso700pixelsfake(m,:)=[0;1;0;0;0;0;0;0;0;0];
    elseif lasso700pixelsindicesB(m)==3
         Blasso700pixelsfake(m,:)=[0;0;1;0;0;0;0;0;0;0];   
    elseif lasso700pixelsindicesB(m)==4
         Blasso700pixelsfake(m,:)=[0;0;0;1;0;0;0;0;0;0];       
    elseif lasso700pixelsindicesB(m)==5
         Blasso700pixelsfake(m,:)=[0;0;0;0;1;0;0;0;0;0];           
    elseif lasso700pixelsindicesB(m)==6
        Blasso700pixelsfake(m,:)=[0;0;0;0;0;1;0;0;0;0];
    elseif lasso700pixelsindicesB(m)==7
        Blasso700pixelsfake(m,:)=[0;0;0;0;0;0;1;0;0;0];
    elseif lasso700pixelsindicesB(m)==8
        Blasso700pixelsfake(m,:)=[0;0;0;0;0;0;0;1;0;0];
    elseif lasso700pixelsindicesB(m)==9
        Blasso700pixelsfake(m,:)=[0;0;0;0;0;0;0;0;1;0];
    else 
        Blasso700pixelsfake(m,:)=[0;0;0;0;0;0;0;0;0;1];
    end 
end
   
        
Blasso700pixelsaccuracysum=zeros(10000,1);
for r=1:10000
    if Btesttranspose(r,:)==Blasso700pixelsfake(r,:)
        Blasso700pixelsaccuracysum(r)=1;
    else
        Blasso700pixelsaccuracysum(r)=0;
    end
end
numbercorrectusinglasso700pixels=sum(Blasso700pixelsaccuracysum);


%% Plots

%important pixels considered: 50, 400, 700
figure(2)
for p=1:784
pixellasso50(p,:)=sum(norm(using50pixels(p,:)));
end
subplot(1,3,1); pcolor(reshape(pixellasso50,28,28)'); set(gca,'YDir','normal'); colormap(gray) 
        colorbar; title('Important Pixels: Lasso Using 50 Pixels')
for p=1:784
pixellasso300(p,:)=sum(norm(using300pixels(p,:)));
end

subplot(1,3,2); pcolor(reshape(pixellasso300,28,28)'); set(gca,'YDir','normal'); colormap(gray) 
        colorbar; title('Important Pixels: Lasso Using 400 Pixels')
        
subplot(1,3,3); pcolor(reshape(pixellasso,28,28)'); set(gca,'YDir','normal'); colormap(gray)
        colorbar; title('Important Pixels: Lasso Using 784 Pixels')
  
  
%Sparsity graphs w/different number of important pixels
figure(3)
subplot(1,3,1); plot(using50pixels,'k')
title('Using 50 Pixels'), xlabel('Pixel Number'), ylabel('Loading(Weight)')
subplot(1,3,2); plot(using400pixels,'k')
title('Using 400 Pixels'), xlabel('Pixel Number'), %ylabel('Loading(Weight)')
subplot(1,3,3); plot(using700pixels,'k')
title('Using 700 Pixels'), xlabel('Pixel Number'), %ylabel('Loading(Weight)')


%Do accuracy graph
accuracies=[39.43 43.56 66.74 73.83 78.30 78.30 78.47 78.47];
xlabelacc=[50 100 200 300 400 500 600 700];
figure(4)
plot(xlabelacc,accuracies,'LineWidth',[2]), axis([40 710 35 85])
title('Lasso Accuracy Depending on Important Pixels for Full Set')
xlabel('Number of Pixels Used')
ylabel('Accuracy in Percent')


%to see how each digit's features look with 400 important pixels used in
%the full set
figure(5)

    for kk=1:5
        subplot(2,5,kk); imagesc(reshape(using400pixels(:,kk),28,28)'); set(gca,'YDir','normal') 
        colorbar
       
    end
    for kk=6:10
            subplot(2,5,kk); imagesc(reshape(using400pixels(:,kk),28,28)'); set(gca,'YDir','normal') 
            colorbar
    end
    
    
    
%% Analysis with individual digits
    
%change the B for train and B for test to a vector (for each digit)
%update matrix B so that it has a 1 in the digit we are analyzing 
%and a 0 everywhere else
    B1train=[];  %for digit 1
for bb1=1:60000
    if labelstrain(bb1,1)==1
        B1train(bb1,1)=1; %1 represents it is the digit we want
    else 
        B1train(bb1,1)=0; %0 represents it is not the digit we want
    end
end

    B1test=[]; %for digit 1
for bb1=1:10000
    if labelstest(bb1,1)==1
        B1test(bb1,1)=1;
    else 
        B1test(bb1,1)=0;
    end
end

B2train=[]; %for digit 2
for bb2=1:60000
    if labelstrain(bb2,1)==2
        B2train(bb2,1)=1;
    else 
        B2train(bb2,1)=0;
    end
end

    B2test=[]; %for digit 2
for bb2=1:10000
    if labelstest(bb2,1)==2
        B2test(bb2,1)=1;
    else 
        B2test(bb2,1)=0;
    end
end

B3train=[];
for bb3=1:60000
    if labelstrain(bb3,1)==3
        B3train(bb3,1)=1;
    else 
        B3train(bb3,1)=0;
    end
end

    B3test=[];
for bb3=1:10000
    if labelstest(bb3,1)==3
        B3test(bb3,1)=1;
    else 
        B3test(bb3,1)=0;
    end
end

  B4train=[];
for bb4=1:60000
    if labelstrain(bb4,1)==4
        B4train(bb4,1)=1;
    else 
        B4train(bb4,1)=0;
    end
end

    B4test=[];
for bb4=1:10000
    if labelstest(bb4,1)==4

        B4test(bb4,1)=1;
    else 
        B4test(bb4,1)=0;
    end
end

    B5train=[];
for bb5=1:60000
    if labelstrain(bb5,1)==5
        B5train(bb5,1)=1;
    else 
        B5train(bb5,1)=0;
    end
end

    B5test=[];
for bb5=1:10000
    if labelstest(bb5,1)==5

        B5test(bb5,1)=1;
    else 
        B5test(bb5,1)=0;
    end
end

  B6train=[];
for bb6=1:60000
    if labelstrain(bb6,1)==6
        B6train(bb6,1)=1;
    else 
        B6train(bb6,1)=0;
    end
end

    B6test=[];
for bb6=1:10000
    if labelstest(bb6,1)==6

        B6test(bb6,1)=1;
    else 
        B6test(bb6,1)=0;
    end
end

   B7train=[];
for bb7=1:60000
    if labelstrain(bb7,1)==7
        B7train(bb7,1)=1;
    else 
        B7train(bb7,1)=0;
    end
end

    B7test=[];
for bb7=1:10000
    if labelstest(bb7,1)==7

        B7test(bb7,1)=1;
    else 
        B7test(bb7,1)=0;
    end
end

 B8train=[];
for bb8=1:60000
    if labelstrain(bb8,1)==8
        B8train(bb8,1)=1;
    else 
        B8train(bb8,1)=0;
    end
end

    B8test=[];
for bb8=1:10000
    if labelstest(bb8,1)==8

        B8test(bb8,1)=1;
    else 
        B8test(bb8,1)=0;
    end
end

    B9train=[];
for bb9=1:60000
    if labelstrain(bb9,1)==9
        B9train(bb9,1)=1;
    else 
        B9train(bb9,1)=0;
    end
end

    B9test=[];
for bb9=1:10000
    if labelstest(bb9,1)==9

        B9test(bb9,1)=1;
    else 
        B9test(bb9,1)=0;
    end
end

   B0train=[];
for bb0=1:60000
    if labelstrain(bb0,1)==0
        B0train(bb0,1)=1;
    else 
        B0train(bb0,1)=0;
    end
end

    B0test=[];
for bb0=1:10000
    if labelstest(bb0,1)==0

        B0test(bb0,1)=1;
    else 
        B0test(bb0,1)=0;
    end
end
%% Use all of matrix A and vector B with lasso

%digit1
Atrainrandom=Atraintranspose(q(1:60000),:);
B1trainrandom=B1train(q(1:60000),:);
%select 6,000 for each kfold to do 10fold validation
A1st6kone=Atrainrandom(q(1:6000),:);
A2nd6kone=Atrainrandom(q(6001:12000),:);
A3rd6kone=Atrainrandom(q(12001:18000),:);
A4th6kone=Atrainrandom(q(18001:24000),:);
A5th6kone=Atrainrandom(q(24001:30000),:);
A6th6kone=Atrainrandom(q(30001:36000),:);
A7th6kone=Atrainrandom(q(36001:42000),:);
A8th6kone=Atrainrandom(q(42001:48000),:);
A9th6kone=Atrainrandom(q(48001:54000),:);
A10th6kone=Atrainrandom(q(54001:60000),:);
B1st6kone=B1trainrandom(q(1:6000),:);
B2nd6kone=B1trainrandom(q(6001:12000),:);
B3rd6kone=B1trainrandom(q(12001:18000),:);
B4th6kone=B1trainrandom(q(18001:24000),:);
B5th6kone=B1trainrandom(q(24001:30000),:);
B6th6kone=B1trainrandom(q(30001:36000),:);
B7th6kone=B1trainrandom(q(36001:42000),:);
B8th6kone=B1trainrandom(q(42001:48000),:);
B9th6kone=B1trainrandom(q(48001:54000),:);
B10th6kone=B1trainrandom(q(54001:60000),:);

for jj=1:1 %its a vector so only going through the loop once
    [a3_1one(:,jj),stats(:,jj)]=lasso(A1st6kone,B1st6kone(:,jj),'Lambda',lambda1);  
    [a3_2one(:,jj),stats(:,jj)]=lasso(A2nd6kone,B2nd6kone(:,jj),'Lambda',lambda1); 
    [a3_3one(:,jj),stats(:,jj)]=lasso(A3rd6kone,B3rd6kone(:,jj),'Lambda',lambda1); 
    [a3_4one(:,jj),stats(:,jj)]=lasso(A4th6kone,B4th6kone(:,jj),'Lambda',lambda1); 
    [a3_5one(:,jj),stats(:,jj)]=lasso(A5th6kone,B5th6kone(:,jj),'Lambda',lambda1); 
    [a3_6one(:,jj),stats(:,jj)]=lasso(A6th6kone,B6th6kone(:,jj),'Lambda',lambda1); 
    [a3_7one(:,jj),stats(:,jj)]=lasso(A7th6kone,B7th6kone(:,jj),'Lambda',lambda1); 
    [a3_8one(:,jj),stats(:,jj)]=lasso(A8th6kone,B8th6kone(:,jj),'Lambda',lambda1); 
    [a3_9one(:,jj),stats(:,jj)]=lasso(A9th6kone,B9th6kone(:,jj),'Lambda',lambda1); 
    [a3_10one(:,jj),stats(:,jj)]=lasso(A10th6kone,B10th6kone(:,jj),'Lambda',lambda1);
    
end
%average to get 1 loading matrix
x1forlasso=((a3_1one)+(a3_2one)+(a3_3one)+(a3_4one)+(a3_5one)+(a3_6one)+(a3_7one)+(a3_8one)+(a3_9one)+(a3_10one))/10;

for ee=1:784
    if abs(x1forlasso(ee,1))>0.00003; 
% ^threshold to decide what pixels to keep
        x1lassotruncated(ee,1)=x1forlasso(ee,1);
    else
        x1lassotruncated(ee,1)=0;
    end
end
%xlassotruncated contains important pixel values and the rest of the pixels
%are 0

%DIGIT2

Atrainrandom=Atraintranspose(q(1:60000),:);
B2trainrandom=B2train(q(1:60000),:);
%select 6,000 for each kfold to do 10fold validation
A1st6ktwo=Atrainrandom(q(1:6000),:);
A2nd6ktwo=Atrainrandom(q(6001:12000),:);
A3rd6ktwo=Atrainrandom(q(12001:18000),:);
A4th6ktwo=Atrainrandom(q(18001:24000),:);
A5th6ktwo=Atrainrandom(q(24001:30000),:);
A6th6ktwo=Atrainrandom(q(30001:36000),:);
A7th6ktwo=Atrainrandom(q(36001:42000),:);
A8th6ktwo=Atrainrandom(q(42001:48000),:);
A9th6ktwo=Atrainrandom(q(48001:54000),:);
A10th6ktwo=Atrainrandom(q(54001:60000),:);
B1st6ktwo=B2trainrandom(q(1:6000),:);
B2nd6ktwo=B2trainrandom(q(6001:12000),:);
B3rd6ktwo=B2trainrandom(q(12001:18000),:);
B4th6ktwo=B2trainrandom(q(18001:24000),:);
B5th6ktwo=B2trainrandom(q(24001:30000),:);
B6th6ktwo=B2trainrandom(q(30001:36000),:);
B7th6ktwo=B2trainrandom(q(36001:42000),:);
B8th6ktwo=B2trainrandom(q(42001:48000),:);
B9th6ktwo=B2trainrandom(q(48001:54000),:);
B10th6ktwo=B2trainrandom(q(54001:60000),:);

tic
for jj=1:1
    [a3_1two(:,jj),stats(:,jj)]=lasso(A1st6ktwo,B1st6ktwo(:,jj),'Lambda',lambda1); 
    [a3_2two(:,jj),stats(:,jj)]=lasso(A2nd6ktwo,B2nd6ktwo(:,jj),'Lambda',lambda1); 
    [a3_3two(:,jj),stats(:,jj)]=lasso(A3rd6ktwo,B3rd6ktwo(:,jj),'Lambda',lambda1); 
    [a3_4two(:,jj),stats(:,jj)]=lasso(A4th6ktwo,B4th6ktwo(:,jj),'Lambda',lambda1); 
    [a3_5two(:,jj),stats(:,jj)]=lasso(A5th6ktwo,B5th6ktwo(:,jj),'Lambda',lambda1); 
    [a3_6two(:,jj),stats(:,jj)]=lasso(A6th6ktwo,B6th6ktwo(:,jj),'Lambda',lambda1); 
    [a3_7two(:,jj),stats(:,jj)]=lasso(A7th6ktwo,B7th6ktwo(:,jj),'Lambda',lambda1); 
    [a3_8two(:,jj),stats(:,jj)]=lasso(A8th6ktwo,B8th6ktwo(:,jj),'Lambda',lambda1); 
    [a3_9two(:,jj),stats(:,jj)]=lasso(A9th6ktwo,B9th6ktwo(:,jj),'Lambda',lambda1); 
    [a3_10two(:,jj),stats(:,jj)]=lasso(A10th6ktwo,B10th6ktwo(:,jj),'Lambda',lambda1); 
    
end
toc

x2forlasso=((a3_1two)+(a3_2two)+(a3_3two)+(a3_4two)+(a3_5two)+(a3_6two)+(a3_7two)+(a3_8two)+(a3_9two)+(a3_10two))/10;

for ee=1:784
    if abs(x2forlasso(ee,1))>0.00003; %threshold
        x2lassotruncated(ee,1)=x2forlasso(ee,1);
    else
        x2lassotruncated(ee,1)=0;
    end
end

%digit3

Atrainrandom=Atraintranspose(q(1:60000),:);
B3trainrandom=B3train(q(1:60000),:);
%select 6,000 for each kfold to do 10fold validation
A1st6kthree=Atrainrandom(q(1:6000),:);
A2nd6kthree=Atrainrandom(q(6001:12000),:);
A3rd6kthree=Atrainrandom(q(12001:18000),:);
A4th6kthree=Atrainrandom(q(18001:24000),:);
A5th6kthree=Atrainrandom(q(24001:30000),:);
A6th6kthree=Atrainrandom(q(30001:36000),:);
A7th6kthree=Atrainrandom(q(36001:42000),:);
A8th6kthree=Atrainrandom(q(42001:48000),:);
A9th6kthree=Atrainrandom(q(48001:54000),:);
A10th6kthree=Atrainrandom(q(54001:60000),:);
B1st6kthree=B3trainrandom(q(1:6000),:);
B2nd6kthree=B3trainrandom(q(6001:12000),:);
B3rd6kthree=B3trainrandom(q(12001:18000),:);
B4th6kthree=B3trainrandom(q(18001:24000),:);
B5th6kthree=B3trainrandom(q(24001:30000),:);
B6th6kthree=B3trainrandom(q(30001:36000),:);
B7th6kthree=B3trainrandom(q(36001:42000),:);
B8th6kthree=B3trainrandom(q(42001:48000),:);
B9th6kthree=B3trainrandom(q(48001:54000),:);
B10th6kthree=B3trainrandom(q(54001:60000),:);

tic
for jj=1:1
    [a3_1three(:,jj),stats(:,jj)]=lasso(A1st6kthree,B1st6kthree(:,jj),'Lambda',lambda1);  
    [a3_2three(:,jj),stats(:,jj)]=lasso(A2nd6kthree,B2nd6kthree(:,jj),'Lambda',lambda1); 
    [a3_3three(:,jj),stats(:,jj)]=lasso(A3rd6kthree,B3rd6kthree(:,jj),'Lambda',lambda1); 
    [a3_4three(:,jj),stats(:,jj)]=lasso(A4th6kthree,B4th6kthree(:,jj),'Lambda',lambda1); 
    [a3_5three(:,jj),stats(:,jj)]=lasso(A5th6kthree,B5th6kthree(:,jj),'Lambda',lambda1); 
    [a3_6three(:,jj),stats(:,jj)]=lasso(A6th6kthree,B6th6kthree(:,jj),'Lambda',lambda1); 
    [a3_7three(:,jj),stats(:,jj)]=lasso(A7th6kthree,B7th6kthree(:,jj),'Lambda',lambda1); 
    [a3_8three(:,jj),stats(:,jj)]=lasso(A8th6kthree,B8th6kthree(:,jj),'Lambda',lambda1); 
    [a3_9three(:,jj),stats(:,jj)]=lasso(A9th6kthree,B9th6kthree(:,jj),'Lambda',lambda1); 
    [a3_10three(:,jj),stats(:,jj)]=lasso(A10th6kthree,B10th6kthree(:,jj),'Lambda',lambda1); 
    
end
toc

x3forlasso=((a3_1three)+(a3_2three)+(a3_3three)+(a3_4three)+(a3_5three)+(a3_6three)+(a3_7three)+(a3_8three)+(a3_9three)+(a3_10three))/10;

for ee=1:784
    if abs(x3forlasso(ee,1))>0.00004; %threshold
        x3lassotruncated(ee,1)=x3forlasso(ee,1);
    else
        x3lassotruncated(ee,1)=0;
    end
end


%digit4
Atrainrandom=Atraintranspose(q(1:60000),:);
B4trainrandom=B4train(q(1:60000),:);
%select 6,000 for each kfold to do 10fold validation
A1st6four=Atrainrandom(q(1:6000),:);
A2nd6kfour=Atrainrandom(q(6001:12000),:);
A3rd6kfour=Atrainrandom(q(12001:18000),:);
A4th6kfour=Atrainrandom(q(18001:24000),:);
A5th6kfour=Atrainrandom(q(24001:30000),:);
A6th6kfour=Atrainrandom(q(30001:36000),:);
A7th6kfour=Atrainrandom(q(36001:42000),:);
A8th6kfour=Atrainrandom(q(42001:48000),:);
A9th6kfour=Atrainrandom(q(48001:54000),:);
A10th6kfour=Atrainrandom(q(54001:60000),:);
B1st6ktfour=B4trainrandom(q(1:6000),:);
B2nd6kfour=B4trainrandom(q(6001:12000),:);
B3rd6kfour=B4trainrandom(q(12001:18000),:);
B4th6kfour=B4trainrandom(q(18001:24000),:);
B5th6kfour=B4trainrandom(q(24001:30000),:);
B6th6kfour=B4trainrandom(q(30001:36000),:);
B7th6kfour=B4trainrandom(q(36001:42000),:);
B8th6kfour=B4trainrandom(q(42001:48000),:);
B9th6kfour=B4trainrandom(q(48001:54000),:);
B10th6kfour=B4trainrandom(q(54001:60000),:);


for jj=1:1
    [a3_1four(:,jj),stats(:,jj)]=lasso(A1st6four,B1st6ktfour(:,jj),'Lambda',lambda1);  
    [a3_2four(:,jj),stats(:,jj)]=lasso(A2nd6kfour,B2nd6kfour(:,jj),'Lambda',lambda1); 
    [a3_3four(:,jj),stats(:,jj)]=lasso(A3rd6kfour,B3rd6kfour(:,jj),'Lambda',lambda1); 
    [a3_4four(:,jj),stats(:,jj)]=lasso(A4th6kfour,B4th6kfour(:,jj),'Lambda',lambda1); 
    [a3_5four(:,jj),stats(:,jj)]=lasso(A5th6kfour,B5th6kfour(:,jj),'Lambda',lambda1); 
    [a3_6four(:,jj),stats(:,jj)]=lasso(A6th6kfour,B6th6kfour(:,jj),'Lambda',lambda1); 
    [a3_7four(:,jj),stats(:,jj)]=lasso(A7th6kfour,B7th6kfour(:,jj),'Lambda',lambda1); 
    [a3_8four(:,jj),stats(:,jj)]=lasso(A8th6kfour,B8th6kfour(:,jj),'Lambda',lambda1); 
    [a3_9four(:,jj),stats(:,jj)]=lasso(A9th6kfour,B9th6kfour(:,jj),'Lambda',lambda1); 
    [a3_10four(:,jj),stats(:,jj)]=lasso(A10th6kfour,B10th6kfour(:,jj),'Lambda',lambda1); 
    
end


x4forlasso=((a3_1four)+(a3_2four)+(a3_3four)+(a3_4four)+(a3_5four)+(a3_6four)+(a3_7four)+(a3_8four)+(a3_9four)+(a3_10four))/10;


for ee=1:784
    if abs(x4forlasso(ee,1))>0.00003; %threshold
        x4lassotruncated(ee,1)=x4forlasso(ee,1);
    else
        x4lassotruncated(ee,1)=0;
    end
end


%digit5
Atrainrandom=Atraintranspose(q(1:60000),:);
B5trainrandom=B5train(q(1:60000),:);
%select 6,000 for each kfold to do 10fold validation
A1st6five=Atrainrandom(q(1:6000),:);
A2nd6kfive=Atrainrandom(q(6001:12000),:);
A3rd6kfive=Atrainrandom(q(12001:18000),:);
A4th6kfive=Atrainrandom(q(18001:24000),:);
A5th6kfive=Atrainrandom(q(24001:30000),:);
A6th6kfive=Atrainrandom(q(30001:36000),:);
A7th6kfive=Atrainrandom(q(36001:42000),:);
A8th6kfive=Atrainrandom(q(42001:48000),:);
A9th6kfive=Atrainrandom(q(48001:54000),:);
A10th6kfive=Atrainrandom(q(54001:60000),:);
B1st6ktfive=B5trainrandom(q(1:6000),:);
B2nd6kfive=B5trainrandom(q(6001:12000),:);
B3rd6kfive=B5trainrandom(q(12001:18000),:);
B4th6kfive=B5trainrandom(q(18001:24000),:);
B5th6kfive=B5trainrandom(q(24001:30000),:);
B6th6kfive=B5trainrandom(q(30001:36000),:);
B7th6kfive=B5trainrandom(q(36001:42000),:);
B8th6kfive=B5trainrandom(q(42001:48000),:);
B9th6kfive=B5trainrandom(q(48001:54000),:);
B10th6kfive=B5trainrandom(q(54001:60000),:);

tic
for jj=1:1
    [a3_1five(:,jj),stats(:,jj)]=lasso(A1st6five,B1st6ktfive(:,jj),'Lambda',lambda1); 
    [a3_2five(:,jj),stats(:,jj)]=lasso(A2nd6kfive,B2nd6kfive(:,jj),'Lambda',lambda1); 
    [a3_3five(:,jj),stats(:,jj)]=lasso(A3rd6kfive,B3rd6kfive(:,jj),'Lambda',lambda1); 
    [a3_4five(:,jj),stats(:,jj)]=lasso(A4th6kfive,B4th6kfive(:,jj),'Lambda',lambda1); 
    [a3_5five(:,jj),stats(:,jj)]=lasso(A5th6kfive,B5th6kfive(:,jj),'Lambda',lambda1); 
    [a3_6five(:,jj),stats(:,jj)]=lasso(A6th6kfive,B6th6kfive(:,jj),'Lambda',lambda1); 
    [a3_7five(:,jj),stats(:,jj)]=lasso(A7th6kfive,B7th6kfive(:,jj),'Lambda',lambda1); 
    [a3_8five(:,jj),stats(:,jj)]=lasso(A8th6kfive,B8th6kfive(:,jj),'Lambda',lambda1); 
    [a3_9five(:,jj),stats(:,jj)]=lasso(A9th6kfive,B9th6kfive(:,jj),'Lambda',lambda1); 
    [a3_10five(:,jj),stats(:,jj)]=lasso(A10th6kfive,B10th6kfive(:,jj),'Lambda',lambda1); 
    
end
toc

x5forlasso=((a3_1five)+(a3_2five)+(a3_3five)+(a3_4five)+(a3_5five)+(a3_6five)+(a3_7five)+(a3_8five)+(a3_9five)+(a3_10five))/10;

for ee=1:784
    if abs(x5forlasso(ee,1))>0.00003; %threshold
        x5lassotruncated(ee,1)=x5forlasso(ee,1);
    else
        x5lassotruncated(ee,1)=0;
    end
end


%digit6
Atrainrandom=Atraintranspose(q(1:60000),:);
B6trainrandom=B6train(q(1:60000),:);
%select 6,000 for each kfold to do 10fold validation
A1st6six=Atrainrandom(q(1:6000),:);
A2nd6ksix=Atrainrandom(q(6001:12000),:);
A3rd6ksix=Atrainrandom(q(12001:18000),:);
A4th6ksix=Atrainrandom(q(18001:24000),:);
A5th6ksix=Atrainrandom(q(24001:30000),:);
A6th6ksix=Atrainrandom(q(30001:36000),:);
A7th6ksix=Atrainrandom(q(36001:42000),:);
A8th6ksix=Atrainrandom(q(42001:48000),:);
A9th6ksix=Atrainrandom(q(48001:54000),:);
A10th6ksix=Atrainrandom(q(54001:60000),:);
B1st6ktsix=B6trainrandom(q(1:6000),:);
B2nd6ksix=B6trainrandom(q(6001:12000),:);
B3rd6ksix=B6trainrandom(q(12001:18000),:);
B4th6ksix=B6trainrandom(q(18001:24000),:);
B5th6ksix=B6trainrandom(q(24001:30000),:);
B6th6ksix=B6trainrandom(q(30001:36000),:);
B7th6ksix=B6trainrandom(q(36001:42000),:);
B8th6ksix=B6trainrandom(q(42001:48000),:);
B9th6ksix=B6trainrandom(q(48001:54000),:);
B10th6ksix=B6trainrandom(q(54001:60000),:);

tic
for jj=1:1
    [a3_1six(:,jj),stats(:,jj)]=lasso(A1st6six,B1st6ktsix(:,jj),'Lambda',lambda1);  
    [a3_2six(:,jj),stats(:,jj)]=lasso(A2nd6ksix,B2nd6ksix(:,jj),'Lambda',lambda1); 
    [a3_3six(:,jj),stats(:,jj)]=lasso(A3rd6ksix,B3rd6ksix(:,jj),'Lambda',lambda1); 
    [a3_4six(:,jj),stats(:,jj)]=lasso(A4th6ksix,B4th6ksix(:,jj),'Lambda',lambda1); 
    [a3_5six(:,jj),stats(:,jj)]=lasso(A5th6ksix,B5th6ksix(:,jj),'Lambda',lambda1); 
    [a3_6six(:,jj),stats(:,jj)]=lasso(A6th6ksix,B6th6ksix(:,jj),'Lambda',lambda1); 
    [a3_7six(:,jj),stats(:,jj)]=lasso(A7th6ksix,B7th6ksix(:,jj),'Lambda',lambda1); 
    [a3_8six(:,jj),stats(:,jj)]=lasso(A8th6ksix,B8th6ksix(:,jj),'Lambda',lambda1); 
    [a3_9six(:,jj),stats(:,jj)]=lasso(A9th6ksix,B9th6ksix(:,jj),'Lambda',lambda1); 
    [a3_10six(:,jj),stats(:,jj)]=lasso(A10th6ksix,B10th6ksix(:,jj),'Lambda',lambda1); 
   
end
toc

x6forlasso=((a3_1six)+(a3_2six)+(a3_3six)+(a3_4six)+(a3_5six)+(a3_6six)+(a3_7six)+(a3_8six)+(a3_9six)+(a3_10six))/10;

for ee=1:784
    if abs(x6forlasso(ee,1))>0.00003; %threshold
        x6lassotruncated(ee,1)=x6forlasso(ee,1);
    else
        x6lassotruncated(ee,1)=0;
    end
end


Atrainrandom=Atraintranspose(q(1:60000),:);
B7trainrandom=B7train(q(1:60000),:);
%select 6,000 for each kfold to do 10fold validation
A1st6ksev=Atrainrandom(q(1:6000),:);
A2nd6ksev=Atrainrandom(q(6001:12000),:);
A3rd6ksev=Atrainrandom(q(12001:18000),:);
A4th6ksev=Atrainrandom(q(18001:24000),:);
A5th6ksev=Atrainrandom(q(24001:30000),:);
A6th6ksev=Atrainrandom(q(30001:36000),:);
A7th6ksev=Atrainrandom(q(36001:42000),:);
A8th6ksev=Atrainrandom(q(42001:48000),:);
A9th6ksev=Atrainrandom(q(48001:54000),:);
A10th6ksev=Atrainrandom(q(54001:60000),:);
B1st6ktsev=B7trainrandom(q(1:6000),:);
B2nd6ksev=B7trainrandom(q(6001:12000),:);
B3rd6ksev=B7trainrandom(q(12001:18000),:);
B4th6ksev=B7trainrandom(q(18001:24000),:);
B5th6ksev=B7trainrandom(q(24001:30000),:);
B6th6ksev=B7trainrandom(q(30001:36000),:);
B7th6ksev=B7trainrandom(q(36001:42000),:);
B8th6ksev=B7trainrandom(q(42001:48000),:);
B9th6ksev=B7trainrandom(q(48001:54000),:);
B10th6ksev=B7trainrandom(q(54001:60000),:);

tic
for jj=1:1
    [a3_1sev(:,jj),stats(:,jj)]=lasso(A1st6ksev,B1st6ktsev(:,jj),'Lambda',lambda1);  
    [a3_2sev(:,jj),stats(:,jj)]=lasso(A2nd6ksev,B2nd6ksev(:,jj),'Lambda',lambda1); 
    [a3_3sev(:,jj),stats(:,jj)]=lasso(A3rd6ksev,B3rd6ksev(:,jj),'Lambda',lambda1); 
    [a3_4sev(:,jj),stats(:,jj)]=lasso(A4th6ksev,B4th6ksev(:,jj),'Lambda',lambda1); 
    [a3_5sev(:,jj),stats(:,jj)]=lasso(A5th6ksev,B5th6ksev(:,jj),'Lambda',lambda1); 
    [a3_6sev(:,jj),stats(:,jj)]=lasso(A6th6ksev,B6th6ksev(:,jj),'Lambda',lambda1); 
    [a3_7sev(:,jj),stats(:,jj)]=lasso(A7th6ksev,B7th6ksev(:,jj),'Lambda',lambda1); 
    [a3_8sev(:,jj),stats(:,jj)]=lasso(A8th6ksev,B8th6ksev(:,jj),'Lambda',lambda1); 
    [a3_9sev(:,jj),stats(:,jj)]=lasso(A9th6ksev,B9th6ksev(:,jj),'Lambda',lambda1); 
    [a3_10sev(:,jj),stats(:,jj)]=lasso(A10th6ksev,B10th6ksev(:,jj),'Lambda',lambda1); 
   
end
toc

x7forlasso=((a3_1sev)+(a3_2sev)+(a3_3sev)+(a3_4sev)+(a3_5sev)+(a3_6sev)+(a3_7sev)+(a3_8sev)+(a3_9sev)+(a3_10sev))/10;

for ee=1:784
    if abs(x7forlasso(ee,1))>0.00003; %threshold
        x7lassotruncated(ee,1)=x7forlasso(ee,1);
    else
        x7lassotruncated(ee,1)=0;
    end
end


%digit8
Atrainrandom=Atraintranspose(q(1:60000),:);
B8trainrandom=B8train(q(1:60000),:);
%select 6,000 for each kfold to do 10fold validation
A1st6keig=Atrainrandom(q(1:6000),:);
A2nd6keig=Atrainrandom(q(6001:12000),:);
A3rd6keig=Atrainrandom(q(12001:18000),:);
A4th6keig=Atrainrandom(q(18001:24000),:);
A5th6keig=Atrainrandom(q(24001:30000),:);
A6th6keig=Atrainrandom(q(30001:36000),:);
A7th6keig=Atrainrandom(q(36001:42000),:);
A8th6keig=Atrainrandom(q(42001:48000),:);
A9th6keig=Atrainrandom(q(48001:54000),:);
A10th6keig=Atrainrandom(q(54001:60000),:);
B1st6kteig=B8trainrandom(q(1:6000),:);
B2nd6keig=B8trainrandom(q(6001:12000),:);
B3rd6keig=B8trainrandom(q(12001:18000),:);
B4th6keig=B8trainrandom(q(18001:24000),:);
B5th6keig=B8trainrandom(q(24001:30000),:);
B6th6keig=B8trainrandom(q(30001:36000),:);
B7th6keig=B8trainrandom(q(36001:42000),:);
B8th6keig=B8trainrandom(q(42001:48000),:);
B9th6keig=B8trainrandom(q(48001:54000),:);
B10th6keig=B8trainrandom(q(54001:60000),:);

tic
for jj=1:1
    [a3_1eig(:,jj),stats(:,jj)]=lasso(A1st6keig,B1st6kteig(:,jj),'Lambda',lambda1);  
    [a3_2eig(:,jj),stats(:,jj)]=lasso(A2nd6keig,B2nd6keig(:,jj),'Lambda',lambda1); 
    [a3_3eig(:,jj),stats(:,jj)]=lasso(A3rd6keig,B3rd6keig(:,jj),'Lambda',lambda1); 
    [a3_4eig(:,jj),stats(:,jj)]=lasso(A4th6keig,B4th6keig(:,jj),'Lambda',lambda1); 
    [a3_5eig(:,jj),stats(:,jj)]=lasso(A5th6keig,B5th6keig(:,jj),'Lambda',lambda1); 
    [a3_6eig(:,jj),stats(:,jj)]=lasso(A6th6keig,B6th6keig(:,jj),'Lambda',lambda1); 
    [a3_7eig(:,jj),stats(:,jj)]=lasso(A7th6keig,B7th6keig(:,jj),'Lambda',lambda1); 
    [a3_8eig(:,jj),stats(:,jj)]=lasso(A8th6keig,B8th6keig(:,jj),'Lambda',lambda1); 
    [a3_9eig(:,jj),stats(:,jj)]=lasso(A9th6keig,B9th6keig(:,jj),'Lambda',lambda1); 
    [a3_10eig(:,jj),stats(:,jj)]=lasso(A10th6keig,B10th6keig(:,jj),'Lambda',lambda1); 
   
end
toc

x8forlasso=((a3_1eig)+(a3_2eig)+(a3_3eig)+(a3_4eig)+(a3_5eig)+(a3_6eig)+(a3_7eig)+(a3_8eig)+(a3_9eig)+(a3_10eig))/10;

for ee=1:784
    if abs(x8forlasso(ee,1))>0.00003; %threshold
        x8lassotruncated(ee,1)=x8forlasso(ee,1);
    else
        x8lassotruncated(ee,1)=0;
    end
end


%digit9
Atrainrandom=Atraintranspose(q(1:60000),:);
B9trainrandom=B9train(q(1:60000),:);
%select 6,000 for each kfold to do 10fold validation
A1st6k9=Atrainrandom(q(1:6000),:);
A2nd6k9=Atrainrandom(q(6001:12000),:);
A3rd6k9=Atrainrandom(q(12001:18000),:);
A4th6k9=Atrainrandom(q(18001:24000),:);
A5th6k9=Atrainrandom(q(24001:30000),:);
A6th6k9=Atrainrandom(q(30001:36000),:);
A7th6k9=Atrainrandom(q(36001:42000),:);
A8th6k9=Atrainrandom(q(42001:48000),:);
A9th6k9=Atrainrandom(q(48001:54000),:);
A10th6k9=Atrainrandom(q(54001:60000),:);
B1st6kt9=B9trainrandom(q(1:6000),:);
B2nd6k9=B9trainrandom(q(6001:12000),:);
B3rd6k9=B9trainrandom(q(12001:18000),:);
B4th6k9=B9trainrandom(q(18001:24000),:);
B5th6k9=B9trainrandom(q(24001:30000),:);
B6th6k9=B9trainrandom(q(30001:36000),:);
B7th6k9=B9trainrandom(q(36001:42000),:);
B8th6k9=B9trainrandom(q(42001:48000),:);
B9th6k9=B9trainrandom(q(48001:54000),:);
B10th6k9=B9trainrandom(q(54001:60000),:);

tic
for jj=1:1
    [a3_1nin(:,jj),stats(:,jj)]=lasso(A1st6k9,B1st6kt9(:,jj),'Lambda',lambda1);  
    [a3_2nin(:,jj),stats(:,jj)]=lasso(A2nd6k9,B2nd6k9(:,jj),'Lambda',lambda1); 
    [a3_3nin(:,jj),stats(:,jj)]=lasso(A3rd6k9,B3rd6k9(:,jj),'Lambda',lambda1); 
    [a3_4nin(:,jj),stats(:,jj)]=lasso(A4th6k9,B4th6k9(:,jj),'Lambda',lambda1); 
    [a3_5nin(:,jj),stats(:,jj)]=lasso(A5th6k9,B5th6k9(:,jj),'Lambda',lambda1); 
    [a3_6nin(:,jj),stats(:,jj)]=lasso(A6th6k9,B6th6k9(:,jj),'Lambda',lambda1); 
    [a3_7nin(:,jj),stats(:,jj)]=lasso(A7th6k9,B7th6k9(:,jj),'Lambda',lambda1); 
    [a3_8nin(:,jj),stats(:,jj)]=lasso(A8th6k9,B8th6k9(:,jj),'Lambda',lambda1); 
    [a3_9nin(:,jj),stats(:,jj)]=lasso(A9th6k9,B9th6k9(:,jj),'Lambda',lambda1); 
    [a3_10nin(:,jj),stats(:,jj)]=lasso(A10th6k9,B10th6k9(:,jj),'Lambda',lambda1); 
   
end
toc

x9forlasso=((a3_1nin)+(a3_2nin)+(a3_3nin)+(a3_4nin)+(a3_5nin)+(a3_6nin)+(a3_7nin)+(a3_8nin)+(a3_9nin)+(a3_10nin))/10;

for ee=1:784
    if abs(x9forlasso(ee,1))>0.00003; %threshold
        x9lassotruncated(ee,1)=x9forlasso(ee,1);
    else
        x9lassotruncated(ee,1)=0;
    end
end

%digit0
Atrainrandom=Atraintranspose(q(1:60000),:);
B0trainrandom=B0train(q(1:60000),:);
%select 6,000 for each kfold to do 10fold validation
A1st6k0=Atrainrandom(q(1:6000),:);
A2nd6k0=Atrainrandom(q(6001:12000),:);
A3rd6k0=Atrainrandom(q(12001:18000),:);
A4th6k0=Atrainrandom(q(18001:24000),:);
A5th6k0=Atrainrandom(q(24001:30000),:);
A6th6k0=Atrainrandom(q(30001:36000),:);
A7th6k0=Atrainrandom(q(36001:42000),:);
A8th6k0=Atrainrandom(q(42001:48000),:);
A9th6k0=Atrainrandom(q(48001:54000),:);
A10th6k0=Atrainrandom(q(54001:60000),:);
B1st6kt0=B0trainrandom(q(1:6000),:);
B2nd6k0=B0trainrandom(q(6001:12000),:);
B3rd6k0=B0trainrandom(q(12001:18000),:);
B4th6k0=B0trainrandom(q(18001:24000),:);
B5th6k0=B0trainrandom(q(24001:30000),:);
B6th6k0=B0trainrandom(q(30001:36000),:);
B7th6k0=B0trainrandom(q(36001:42000),:);
B8th6k0=B0trainrandom(q(42001:48000),:);
B9th6k0=B0trainrandom(q(48001:54000),:);
B10th6k0=B0trainrandom(q(54001:60000),:);

tic
for jj=1:1
    [a3_1ze(:,jj),stats(:,jj)]=lasso(A1st6k0,B1st6kt0(:,jj),'Lambda',lambda1);  
    [a3_2ze(:,jj),stats(:,jj)]=lasso(A2nd6k0,B2nd6k0(:,jj),'Lambda',lambda1); 
    [a3_3ze(:,jj),stats(:,jj)]=lasso(A3rd6k0,B3rd6k0(:,jj),'Lambda',lambda1); 
    [a3_4ze(:,jj),stats(:,jj)]=lasso(A4th6k0,B4th6k0(:,jj),'Lambda',lambda1); 
    [a3_5ze(:,jj),stats(:,jj)]=lasso(A5th6k0,B5th6k0(:,jj),'Lambda',lambda1); 
    [a3_6ze(:,jj),stats(:,jj)]=lasso(A6th6k0,B6th6k0(:,jj),'Lambda',lambda1); 
    [a3_7ze(:,jj),stats(:,jj)]=lasso(A7th6k0,B7th6k0(:,jj),'Lambda',lambda1); 
    [a3_8ze(:,jj),stats(:,jj)]=lasso(A8th6k0,B8th6k0(:,jj),'Lambda',lambda1); 
    [a3_9ze(:,jj),stats(:,jj)]=lasso(A9th6k0,B9th6k0(:,jj),'Lambda',lambda1); 
    [a3_10ze(:,jj),stats(:,jj)]=lasso(A10th6k0,B10th6k0(:,jj),'Lambda',lambda1); 
   
end
toc

x0forlasso=((a3_1ze)+(a3_2ze)+(a3_3ze)+(a3_4ze)+(a3_5ze)+(a3_6ze)+(a3_7ze)+(a3_8ze)+(a3_9ze)+(a3_10ze))/10;

for ee=1:784
    if abs(x0forlasso(ee,1))>0.00003; %threshold
        x0lassotruncated(ee,1)=x0forlasso(ee,1);
    else
        x0lassotruncated(ee,1)=0;
    end
end

%% 
%ACCURACIES
B1forlasso=Atesttranspose*x1lassotruncated;  %apply model to test data
for uu=1:10000
    if B1forlasso(uu,:)>0.45;  %classify any value above this as a 1
        B1fake(uu,:)=1;
    else
        B1fake(uu,:)=0; %any value below 0.45 is 0
    end
end
B1accuracy=zeros(10000,1);
for yy=1:10000
    if B1fake(yy,:)==B1test(yy,:);
        B1accuracy(yy,:)=1;
    else
        B1accuracy(yy,:)=0;
    end 
end
%Bfake contains our digit classification predictions
digit1accuracy=(sum(B1accuracy)/10000)*100 %percent accuracy of digit 1

B2forlasso=Atesttranspose*x2lassotruncated;
for uu=1:10000
    if B2forlasso(uu,:)>0.45;
        B2fake(uu,:)=1;
    else
        B2fake(uu,:)=0;
    end
end
B2accuracy=zeros(10000,1);
for yy=1:10000
    if B2fake(yy,:)==B2test(yy,:);
        B2accuracy(yy,:)=1;
    else
        B2accuracy(yy,:)=0;
    end 
end
digit2accuracy=(sum(B2accuracy)/10000)*100

B3forlasso=Atesttranspose*x3lassotruncated;
for uu=1:10000
    if B3forlasso(uu,:)>0.45;
        B3fake(uu,:)=1;
    else
        B3fake(uu,:)=0;
    end
end
B3accuracy=zeros(10000,1);
for yy=1:10000
    if B3fake(yy,:)==B3test(yy,:);
        B3accuracy(yy,:)=1;
    else
        B3accuracy(yy,:)=0;
    end 
end
digit3accuracy=(sum(B3accuracy)/10000)*100

B4forlasso=Atesttranspose*x4lassotruncated;
for uu=1:10000
    if B4forlasso(uu,:)>0.45;
        B4fake(uu,:)=1;
    else
        B4fake(uu,:)=0;
    end
end
B4accuracy=zeros(10000,1);
for yy=1:10000
    if B4fake(yy,:)==B4test(yy,:);
        B4accuracy(yy,:)=1;
    else
        B4accuracy(yy,:)=0;
    end 
end
digit4accuracy=(sum(B4accuracy)/10000)*100

B5forlasso=Atesttranspose*x5lassotruncated;
for uu=1:10000
    if B5forlasso(uu,:)>0.45;
        B5fake(uu,:)=1;
    else
        B5fake(uu,:)=0;
    end
end
B5accuracy=zeros(10000,1);
for yy=1:10000
    if B5fake(yy,:)==B5test(yy,:);
        B5accuracy(yy,:)=1;
    else
        B5accuracy(yy,:)=0;
    end 
end
digit5accuracy=(sum(B5accuracy)/10000)*100

B6forlasso=Atesttranspose*x6lassotruncated;
for uu=1:10000
    if B6forlasso(uu,:)>0.45;
        B6fake(uu,:)=1;
    else
        B6fake(uu,:)=0;
    end
end
B6accuracy=zeros(10000,1);
for yy=1:10000
    if B6fake(yy,:)==B6test(yy,:);
        B6accuracy(yy,:)=1;
    else
        B6accuracy(yy,:)=0;
    end 
end
digit6accuracy=(sum(B6accuracy)/10000)*100

B7forlasso=Atesttranspose*x7lassotruncated;
for uu=1:10000
    if B7forlasso(uu,:)>0.45;
        B7fake(uu,:)=1;
    else
        B7fake(uu,:)=0;
    end
end
B7accuracy=zeros(10000,1);
for yy=1:10000
    if B7fake(yy,:)==B7test(yy,:);
        B7accuracy(yy,:)=1;
    else
        B7accuracy(yy,:)=0;
    end 
end
digit7accuracy=(sum(B7accuracy)/10000)*100

B8forlasso=Atesttranspose*x8lassotruncated;
for uu=1:10000
    if B8forlasso(uu,:)>0.45;
        B8fake(uu,:)=1;
    else
        B8fake(uu,:)=0;
    end
end
B8accuracy=zeros(10000,1);
for yy=1:10000
    if B8fake(yy,:)==B8test(yy,:);
        B8accuracy(yy,:)=1;
    else
        B8accuracy(yy,:)=0;
    end 
end
digit8accuracy=(sum(B8accuracy)/10000)*100

B9forlasso=Atesttranspose*x9lassotruncated;
for uu=1:10000
    if B9forlasso(uu,:)>0.45;
        B9fake(uu,:)=1;
    else
        B9fake(uu,:)=0;
    end
end
B9accuracy=zeros(10000,1);
for yy=1:10000
    if B9fake(yy,:)==B9test(yy,:);
        B9accuracy(yy,:)=1;
    else
        B9accuracy(yy,:)=0;
    end 
end
digit9accuracy=(sum(B9accuracy)/10000)*100

B0forlasso=Atesttranspose*x0lassotruncated;
for uu=1:10000
    if B0forlasso(uu,:)>0.45;
        B0fake(uu,:)=1;
    else
        B0fake(uu,:)=0;
    end
end
B0accuracy=zeros(10000,1);
for yy=1:10000
    if B0fake(yy,:)==B0test(yy,:);
        B0accuracy(yy,:)=1;
    else
        B0accuracy(yy,:)=0;
    end 
end
digit0accuracy=(sum(B0accuracy)/10000)*100


%% PLOTS
figure(6)
subplot(2,5,1); pcolor(reshape(x1lassotruncated,28,28)'); colorbar; 
title('Digit 1')
subplot(2,5,2); pcolor(reshape(x2lassotruncated,28,28)'); colorbar; 
title('Digit 2')
subplot(2,5,3); pcolor(reshape(x3lassotruncated,28,28)'); colorbar; 
title('Digit 3')
subplot(2,5,4); pcolor(reshape(x4lassotruncated,28,28)'); colorbar; 
title('Digit 4')
subplot(2,5,5); pcolor(reshape(x5lassotruncated,28,28)'); colorbar; 
title('Digit 5')
subplot(2,5,6); pcolor(reshape(x6lassotruncated,28,28)'); colorbar; 
title('Digit 6')
subplot(2,5,7); pcolor(reshape(x7lassotruncated,28,28)'); colorbar; 
title('Digit 7')
subplot(2,5,8); pcolor(reshape(x8lassotruncated,28,28)'); colorbar; 
title('Digit 8')
subplot(2,5,9); pcolor(reshape(x9lassotruncated,28,28)'); colorbar; 
title('Digit 9')
subplot(2,5,10); pcolor(reshape(x0lassotruncated,28,28)'); colorbar; 
title('Digit 0')

figure(7)
subplot(2,5,1); plot(x1lassotruncated); xlabel('Pixel Number'); 
ylabel('Loading weight'); title('Digit 1')
subplot(2,5,2); plot(x2lassotruncated); title('Digit 2')
subplot(2,5,3); plot(x3lassotruncated); title('Digit 3')
subplot(2,5,4); plot(x4lassotruncated); title('Digit 4')
subplot(2,5,5); plot(x5lassotruncated); title('Digit 5')
subplot(2,5,6); plot(x6lassotruncated); xlabel('Pixel Number'); 
ylabel('Loading weight'); title('Digit 6')
subplot(2,5,7); plot(x7lassotruncated); xlabel('Pixel Number'); 
title('Digit 7')
subplot(2,5,8); plot(x8lassotruncated); xlabel('Pixel Number'); 
title('Digit 8')
subplot(2,5,9); plot(x9lassotruncated); xlabel('Pixel Number'); 
title('Digit 9')
subplot(2,5,10); plot(x0lassotruncated); xlabel('Pixel Number'); 
title('Digit 0')