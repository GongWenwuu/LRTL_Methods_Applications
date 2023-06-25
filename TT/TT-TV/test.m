dense_tensor = double(imread('data/Lena.bmp'));
sample_num = round(0.2*numel(dense_tensor));
fprintf('Sampling OD tensor with %4.1f%% known elements ...... \n',100*sample_ratio);
% Filter missing positions
idx = 1:numel(dense_tensor);
idx = idx(dense_tensor(:)>0);
% Artificial missing position
mask = sort(randperm(length(idx),sample_num));
arti_miss_idx = idx;
arti_miss_idx(mask) = [];
arti_miss_mv = dense_tensor(arti_miss_idx);
Omega = zeros(size(dense_tensor)); Omega(mask) = 1; Omega = boolean(Omega);
sparse_tensor = Omega.*dense_tensor;
fprintf('Known elements / total elements: %6d/%6d.\n',sample_num,numel(dense_tensor));
clear idx

Nway = [4 4 4 4 4 4 4 4 3]; 
I1 = 2; J1 = 2;
sizeData = size(dense_tensor);

Otrue  = CastImageAsKet22(dense_tensor, Nway, I1 ,J1 );
Oknown = CastImageAsKet22(Omega, Nway, I1, J1 );
Oknown = find( Oknown==1 );
Okn    = Otrue( Oknown );

Omiss = zeros( Nway );
Omiss( Oknown ) = Otrue( Oknown );
Omiss = CastKet2Image22( Omiss, 256, 256, I1, J1 );

opts=[];
opts.alpha  = weightTC(Nway);
opts.X0 = dense_tensor; 
opts.tol    = 1e-5;
opts.maxit_out  = 100;
opts.maxit_in  = 15;
opts.max_sigma = 10;
opts.max_beta = 10;
opts.rho    = 10^(-3);
opts.th     = 0.01;
opts.lambda = 0.5;
opts.beta0  = 0.01;
opts.sigma = 0.01;
opts.frame = 3;
opts.Level = 1;
opts.wLevel= 0.5;

[X_TT_Framelet, Out_TT_Framelet] = TT_Framelet(Okn, Oknown, Nway, opts);
est_tensor = CastKet2Image22(X_TT_Framelet,256,256,I1,J1);
[psnr, ssim, fsim, ergas, msam] = MSIQA(dense_tensor, est_tensor);

[~, pos] = tight_subplot(1,3,[.01 .0005],[.05 .05],[.05 .05]);
close;
figure(1)
subplot(1,3,1); ax = gca; ax.Position = pos{1};
imshow(uint8(dense_tensor));title('Original Image');
subplot(1,3,2); ax = gca; ax.Position = pos{2};
imshow(uint8(sparse_tensor));title('PSNR = 10.37, SSIM = 0.04');
subplot(1,3,3); ax = gca; ax.Position = pos{3};
imshow(uint8(est_tensor));title('PSNR = 25.09, SSIM = 0.61');


Lena_Eval_TT = zeros(8,12);
Lena_CPUTime_TT = zeros(1,12);
DLP = [0.1:0.1:0.9,0.93,0.95,0.99];
for MR = 1:length(DLP)
    rng('default')
    filename=['Lena_' num2str(MR) '.mat'];
    sample_ratio = 1- DLP(MR);
    sample_num = round(sample_ratio*numel(dense_tensor));
    fprintf('Sampling OD tensor with %4.1f%% known elements ...... \n',100*sample_ratio);
    % Filter missing positions 
    idx = 1:numel(dense_tensor);
    idx = idx(dense_tensor(:)>0);
    % Artificial missing position
    mask = sort(randperm(length(idx),sample_num));
    arti_miss_idx = idx;  
    arti_miss_idx(mask) = [];  
    arti_miss_mv = dense_tensor(arti_miss_idx);
    Omega = zeros(size(dense_tensor)); Omega(mask) = 1; Omega = boolean(Omega);
    sparse_tensor = Omega.*dense_tensor;
    Otrue  = CastImageAsKet22(dense_tensor, Nway, I1 ,J1 );
    Oknown = CastImageAsKet22(Omega, Nway, I1, J1 );
    Oknown = find( Oknown==1 );
    Okn    = Otrue( Oknown );

    Omiss = zeros( Nway );
    Omiss( Oknown ) = Otrue( Oknown );
    Omiss = CastKet2Image22( Omiss, 256, 256, I1, J1 );
    fprintf('Known elements / total elements: %6d/%6d.\n',sample_num,numel(dense_tensor));
    clear idx 

    t0 = tic;
    [X_TT_Framelet, Out_TT_Framelet] = TT_Framelet(Okn, Oknown, Nway, opts);
    Lena_CPUTime_TT(1,MR) = toc(t0);
    est_tensor = CastKet2Image22(X_TT_Framelet,256,256,I1,J1);
    save(filename,"est_tensor")
    rse = TensorNorm(est_tensor - dense_tensor,'fro')/TensorNorm(dense_tensor,'fro');
    nmae = norm(arti_miss_mv-est_tensor(arti_miss_idx),1) / norm(arti_miss_mv,1);
    rmse = sqrt((1/length(arti_miss_mv))*norm(arti_miss_mv-est_tensor(arti_miss_idx),2)^2);  
    [psnr, ssim, fsim, ergas, msam] = MSIQA(dense_tensor, est_tensor);
    Lena_Eval_TT(1,MR) = psnr; Lena_Eval_TT(2,MR) = ssim; Lena_Eval_TT(3,MR) = fsim; Lena_Eval_TT(4,MR) = ergas; 
    Lena_Eval_TT(5,MR) = rmse; Lena_Eval_TT(6,MR) = nmae; Lena_Eval_TT(7,MR) = msam; Lena_Eval_TT(8,MR) = rse;

end