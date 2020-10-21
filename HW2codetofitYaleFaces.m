clear all; close all; clc
%Yale Faces 
%load data of cropped faces
CX=[];
croppedfaces = dir(fullfile...
('c:\Users\diana\Documents\AMATH584- Autumn 2020\\Homework\HW2\CroppedYale\yaleB*\yaleB*'));
%create for loop to import data
    for jj = 1 : length(croppedfaces)
        cropped_data = importdata(fullfile(croppedfaces(jj).folder, ...
            croppedfaces(jj).name ) );
        data1=reshape(cropped_data,1,32256);
        CX=[CX,data1]; %create matrix with loaded data
    end
    
   
       
%Reshape so each column is a new image
%This gives 2432 total images;
    X1=reshape(CX,32256,2432);

%figure(number)
%pixel size is 192x168, so we reshape again to see individual images
Xcroppedeachimage=reshape(X1,192,168,2432);
%last number on line below is for what image we wish to see
%imagesc(Xcroppedeachimage(:,:,2368)), colormap gray  %use to see
%individual images

% COMPUTE SVD
Xcropped=double(X1); %convert to double precision 
Xcroppedtrain=Xcropped(:,1:2368); %all images of the first 37 subjects
[m,n]=size(Xcroppedtrain); %compute data size
avgface=mean(Xcroppedtrain,2); %compute mean
Xcroppedtrain=Xcroppedtrain-repmat(avgface,1,n); %subtract mean
[u,s,v]=svd(Xcroppedtrain,'econ');  %perform the econ svd on matrix with
%37 subjects each column a new image
lambda=diag(s).^2; %produce diagonal variances


% SIGMA PLOTS-CROPPED
sig=diag(s);

%singular values
figure(1)
subplot(1,2,1), plot(sig,'ko','Linewidth',[1.5])
ylabel('sigma')
xlabel('modes')
title('Cropped Faces')
subplot(1,2,2), semilogy(sig,'ko','Linewidth',[1.5])
ylabel('sigma')
xlabel('modes')
% ENERGIES PLOT-CROPPED
energycropped=[];
for k=1:2368  %bc we used 2368 images
    energycropped=[energycropped;sum(sig(1:k))/sum(sig)];
end

figure(2)
x1=1:2368; %mode number
plot(x1,energycropped,'Linewidth',[1]);hold on
hold off
title('Cropped: Energy Contained in Number of Modes Used')
xlabel('Modes')
ylabel('Energy Contained')

figure(3)
subplot(1,4,1)
imagesc(reshape(avgface,192,168)), colormap gray  %avg mean subtracted face
title('Average Face')
set(gca,'XTick',[], 'YTick', [])
subplot(1,4,2)
imagesc(reshape(u(:,1),192,168)), colormap gray  %plot eigenface 
%(column of u is 1 so its the first eigenface)
title('First Eigenface')
set(gca,'XTick',[], 'YTick', [])
subplot(1,4,3)
imagesc(reshape(u(:,2),192,168)), colormap gray  %plot eigenface 
title('Second Eigenface')
set(gca,'XTick',[], 'YTick', [])
subplot(1,4,4)
imagesc(reshape(u(:,3),192,168)), colormap gray  %plot eigenface 
title('Third Eigenface')
set(gca,'XTick',[], 'YTick', [])

% Reconstructions
figure(4)
subplot(2,4,1)
imagesc(Xcroppedeachimage(:,:,2369)), colormap gray %plot the face we try 
%to reconstruct; from subject 38
title('Original Image')
set(gca,'XTick',[], 'YTick', [])

croppedtest=Xcropped(:,2369);%subject #38, first image
testfacecropped=croppedtest-avgface; %subtract avg face
reconcropface=zeros(32256,1600);
for r=[25 50 100 200 400 800 1600] %ranks to consider
    reconcropface(:,r)=avgface+(u(:,1:r)*(u(:,1:r)'*testfacecropped));

%these are approximate representations of image using eigenfaces basis
%of various order r
%only the columns corresponding to the ranks will matter; the rest will be
%zero bc of who we set up the matrix before the loop
end
%
%reconstruction of face using diff ranks
subplot(subplot(2,4,2))
imagesc(reshape(reconcropface(:,25),192,168)), colormap gray %rank 25
title('Rank 25')
set(gca,'XTick',[], 'YTick', [])
subplot(subplot(2,4,3))
imagesc(reshape(reconcropface(:,50),192,168)), colormap gray %rank 50
title('Rank 50')
set(gca,'XTick',[], 'YTick', [])
subplot(subplot(2,4,4))
imagesc(reshape(reconcropface(:,100),192,168)), colormap gray
title('Rank 100')
set(gca,'XTick',[], 'YTick', [])
subplot(subplot(2,4,5))
imagesc(reshape(reconcropface(:,200),192,168)), colormap gray
title('Rank 200')
set(gca,'XTick',[], 'YTick', [])
subplot(subplot(2,4,6))
imagesc(reshape(reconcropface(:,400),192,168)), colormap gray
title('Rank 400')
set(gca,'XTick',[], 'YTick', [])
subplot(subplot(2,4,7))
imagesc(reshape(reconcropface(:,800),192,168)), colormap gray
title('Rank 800')
set(gca,'XTick',[], 'YTick', [])
subplot(subplot(2,4,8))
imagesc(reshape(reconcropface(:,1600),192,168)), colormap gray
title('Rank 1600')
set(gca,'XTick',[], 'YTick', [])




%% UNCROPPED FACES
clear all
%load uncropped faces data using for loop
UX=[];
    uncroppedfaces = dir(fullfile...
        ('c:\Users\diana\Documents\AMATH584- Autumn 2020\Homework\HW2\yalefaces\subject*'));
    for jj = 1 : length(uncroppedfaces)
        uncropped_data = importdata(fullfile(uncroppedfaces(jj).folder, ...
            uncroppedfaces(jj).name ) );
        %uncropped_data.cdata includes the picture info
        data2=reshape(uncropped_data.cdata,1,77760);
        UX=[UX,data2]; %1x12830400; reshape into 77760x165 bc its 165 imgs
    end
%our data has 15 subjects with 11 images each = 165 total images    

%Reshape each image to a column vector
%This gives 165 images; each column=one image
    X1U=reshape(UX,243*320,165);

%pixel size is 243x320,so we reshape again if we want tosee individual imgs
Xuncropeachimage=reshape(X1U,243,320,165);
%imagesc(Xuncropeachimage(:,:,14)), colormap gray %plot individual images

% COMPUTE SVD
Xuncrop=double(X1U); %convert to double
Xuncroptrain=Xuncrop(:,1:154); %first 14 subjects(11 images each); 
%thus, 154 images total
[m,n]=size(Xuncroptrain); %compute data size
avgfaceuncrop=mean(Xuncroptrain,2); %compute mean for each row
Xuncroptrain=Xuncroptrain-repmat(avgfaceuncrop,1,n); %subtract mean; 
[u,s,v]=svd(Xuncroptrain,'econ');  %perform the econ svd
lambda=diag(s).^2; %produce diagonal variances

% SIGMA PLOTS-UNCROPPED
sig=diag(s);

figure(5)
%singular values
subplot(1,2,1), plot(sig,'ko','Linewidth',[1.5])
ylabel('sigma')
xlabel('modes')
title('Uncropped Faces')
subplot(1,2,2), semilogy(sig,'ko','Linewidth',[1.5])
ylabel('sigma')
xlabel('modes')

% ENERGIES PLOT- UNCROPPED
energyuncropped=[];
for k=1:154
    energyuncropped=[energyuncropped;sum(sig(1:k))/sum(sig)];
end

figure(6)
x2=1:154; %mode number
plot(x2,energyuncropped,'Linewidth',[1]);hold on
hold off
title('Uncropped Images Data: Energy Contained in Number of Modes Used')
xlabel('Modes')
ylabel('Energy Contained')

figure(7)
subplot(1,4,1)
imagesc(reshape(avgfaceuncrop,243,320)),colormap gray%avgmeansubtractedface
title('Average Face')
set(gca,'XTick',[], 'YTick', [])
subplot(1,4,2)
imagesc(reshape(u(:,1),243,320)), colormap gray  %plot eigenface 
%(column of u is 1 so its the first eigenface)
set(gca,'XTick',[], 'YTick', [])
title('First Eigenface')
subplot(1,4,3)
imagesc(reshape(u(:,2),243,320)), colormap gray  %plot eigenface 
set(gca,'XTick',[], 'YTick', [])
title('Second Eigenface')
subplot(1,4,4)
imagesc(reshape(u(:,3),243,320)), colormap gray  %plot eigenface 
set(gca,'XTick',[], 'YTick', [])
title('Third Eigenface')

% APPROXIMATE TEST IMAGE- UNCROPPED
figure (8) %plot true image
subplot(1,5,1)
imagesc(Xuncropeachimage(:,:,155)), colormap gray %plot individual image
title('Original Image')
set(gca,'XTick',[], 'YTick', [])


uncroppedtest=Xuncrop(:,155);%subject #15, first image
testfaceuncropped=uncroppedtest-avgfaceuncrop;

reconuncropface=zeros(77760,100);
for r=[10 30 60 100] %ranks to consider
 reconuncropface(:,r)=avgfaceuncrop+(u(:,1:r)*(u(:,1:r)'*testfaceuncropped));
    
%these are approximate representations of the image using eigenfaces basis
%of various order r
end

%image reconstructions using different ranks
subplot(subplot(1,5,2))
imagesc(reshape(reconuncropface(:,10),243,320)), colormap gray %rank 10
title('Rank 10')
set(gca,'XTick',[], 'YTick', [])
subplot(subplot(1,5,3))
imagesc(reshape(reconuncropface(:,30),243,320)), colormap gray %rank 30
title('Rank 30')
set(gca,'XTick',[], 'YTick', [])
subplot(subplot(1,5,4))
imagesc(reshape(reconuncropface(:,60),243,320)), colormap gray
title('Rank 60')
set(gca,'XTick',[], 'YTick', [])
subplot(subplot(1,5,5))
imagesc(reshape(reconuncropface(:,100),243,320)), colormap gray
title('Rank 100')
set(gca,'XTick',[], 'YTick', [])


