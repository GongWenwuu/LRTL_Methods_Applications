dense_tensor = double(imread('lena.bmp'));
Lena_Eval_HaLRTC = zeros(8,12);
Lena_CPUTime_HaLRTC = zeros(1,12);
DLP = [0.1:0.1:0.9,0.93,0.95,0.99];
for MR = 1:length(DLP)
    rng('default')
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
    alpha = [1, 1, 1e-3]; alpha = alpha / sum(alpha); rho = 1e-6;
    [est_tensor, ~] = HaLRTC(sparse_tensor,Omega,alpha,rho,500,1e-5);
    Lena_CPUTime_HaLRTC(1,MR) = toc(t0);
    rse = TensorNorm(est_tensor - dense_tensor,'fro')/TensorNorm(dense_tensor,'fro');
    nmae = norm(arti_miss_mv-est_tensor(arti_miss_idx),1) / norm(arti_miss_mv,1);
    rmse = sqrt((1/length(arti_miss_mv))*norm(arti_miss_mv-est_tensor(arti_miss_idx),2)^2);  
    [psnr, ssim, fsim, ergas, msam] = MSIQA(dense_tensor, est_tensor);
    Lena_Eval_HaLRTC(1,MR) = psnr; Lena_Eval_HaLRTC(2,MR) = ssim; Lena_Eval_HaLRTC(3,MR) = fsim; Lena_Eval_HaLRTC(4,MR) = ergas; 
    Lena_Eval_HaLRTC(5,MR) = rmse; Lena_Eval_HaLRTC(6,MR) = nmae; Lena_Eval_HaLRTC(7,MR) = msam; Lena_Eval_HaLRTC(8,MR) = rse;

end 