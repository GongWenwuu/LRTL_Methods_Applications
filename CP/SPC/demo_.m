dense_tensor = double(imread('Lena.bmp'));
Lena_Eval_SPC = zeros(8,12);
Lena_CPUTime_SPC = zeros(1,12);
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
    fprintf('Known elements / total elements: %6d/%6d.\n',sample_num,numel(dense_tensor));
    clear idx 

    t0 = tic;
    TVQV = 'tv'; rho = [0.05 0.05 0.05]; K = 10; SNR = 25; nu = 0.01; maxiter = 500; tol = 1e-5; out_im  = 1; 
    [est_tensor, Z, G, U, histo, histo_R] = SPC(sparse_tensor,Omega,TVQV,rho,K,SNR,nu,maxiter,tol,out_im);
    save(filename,"est_tensor")
    Lena_CPUTime_SPC(1,MR) = toc(t0);
    rse = TensorNorm(est_tensor - dense_tensor,'fro')/TensorNorm(dense_tensor,'fro');
    nmae = norm(arti_miss_mv-est_tensor(arti_miss_idx),1) / norm(arti_miss_mv,1);
    rmse = sqrt((1/length(arti_miss_mv))*norm(arti_miss_mv-est_tensor(arti_miss_idx),2)^2);  
    [psnr, ssim, fsim, ergas, msam] = MSIQA(dense_tensor, est_tensor);
    Lena_Eval_SPC(1,MR) = psnr; Lena_Eval_SPC(2,MR) = ssim; Lena_Eval_SPC(3,MR) = fsim; Lena_Eval_SPC(4,MR) = ergas; 
    Lena_Eval_SPC(5,MR) = rmse; Lena_Eval_SPC(6,MR) = nmae; Lena_Eval_SPC(7,MR) = msam; Lena_Eval_SPC(8,MR) = rse;

end 