addpath(genpath('solvers'));
addpath(genpath('test_video'));


%% load data
p = 0.7;
load('hall_qcif');
V=V(:,:,1:30);
X = V/max(abs(V(:)));
[n1,n2,n3] = size(X);
maxP = max(X(:));
Omega = find(rand(n1*n2*n3,1)<p);

%% tensor completion
[ TC,psnr] = TCTF_video( X,Omega);

fprintf('video PSNR:%.2f\n',psnr)

