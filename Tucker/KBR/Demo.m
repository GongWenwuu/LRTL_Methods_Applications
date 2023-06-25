dense_tensor = double(imread('peppers.png'));
peppers_Eval_KBR = zeros(8,12);
peppers_CPUTime_KBR = zeros(1,12);
DLP = [0.1:0.1:0.9,0.93,0.95,0.99];
for MR = 1:length(DLP)
    rng('default')
    filename=['peppers_' num2str(MR) '.mat'];
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
    maxIter = 500; epsilon = 1e-5;
    beta = 1e-2; Par2.tol = epsilon; Par2.maxIter = maxIter; Par2.maxSubiter = 1; Par2.rho = 1.05;
    Par2.mu = beta*1e-3; Par2.lambda = 0.01;
    est_tensor = KBR_TC(sparse_tensor/255, Omega, Par2);
    est_tensor = est_tensor*255;
    peppers_CPUTime_KBR(1,MR) = toc(t0);
    save(filename,"est_tensor")
    rse = TensorNorm(est_tensor - dense_tensor,'fro')/TensorNorm(dense_tensor,'fro');
    nmae = norm(arti_miss_mv-est_tensor(arti_miss_idx),1) / norm(arti_miss_mv,1);
    rmse = sqrt((1/length(arti_miss_mv))*norm(arti_miss_mv-est_tensor(arti_miss_idx),2)^2);  
    [psnr, ssim, fsim, ergas, msam] = MSIQA(dense_tensor, est_tensor);
    peppers_Eval_KBR(1,MR) = psnr; peppers_Eval_KBR(2,MR) = ssim; peppers_Eval_KBR(3,MR) = fsim; peppers_Eval_KBR(4,MR) = ergas; 
    peppers_Eval_KBR(5,MR) = rmse; peppers_Eval_KBR(6,MR) = nmae; peppers_Eval_KBR(7,MR) = msam; peppers_Eval_KBR(8,MR) = rse;

end 