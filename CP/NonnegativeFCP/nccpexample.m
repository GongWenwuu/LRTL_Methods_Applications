clc
clear all
close all

load T.mat
N1 = 21; N2 = 21; N3 = 24;R=10;
A1 = max(0,randn(N1,R));
A2 = max(0,randn(N2,R));
A3 = max(0,randn(N3,R));
% generate tensor M using the above factor matrices
M = ktensor({A1,A2,A3});
M = full((arrange(M)));

%%
sr = 1; % percentage of samples
% randomly choose samples
known = randsample(N1*N2*N3,round(sr*N1*N2*N3));
data = M.data(known);
%%
ndim=[N1,N2,N3];
opts.tol = 1e-7; 
opts.maxit = 1000;r=21;
t0 = tic;
[A,Out] = ncpc(T,known,ndim,r,opts);
time = toc(t0);
%% Reporting
figure 
subplot(3,1,1)
plot(A1(:,2))
subplot(3,1,2)
plot(A2(:,2))
subplot(3,1,3)
plot(A3(:,2))





