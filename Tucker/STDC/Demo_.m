dense_tensor = double(imread('Lena.bmp'));
DLP = [0.1:0.1:0.9,0.93,0.95,0.99];
Lena_Eval_STDC = zeros(6,13);
Lena_CPUTime_STDC = zeros(1,13);

for MR = 1:length(DLP)
    rng('default')
    filename=['Lena_' num2str(MR) '.mat'];
    MR_ = DLP(MR);
    idx = randperm(numel(dense_tensor));
    mark = zeros(size(dense_tensor));
    mark(idx(1:floor(MR_*numel(dense_tensor)))) = 1;
    mark = boolean(mark);
    Xm = dense_tensor;
    Xm(mark) = 0;
    
    % sample_ratio = 1- DLP(MR);
    % sample_num = round(sample_ratio*numel(dense_tensor));
    % fprintf('Sampling tensor with %4.1f%% known elements ...... \n',100*sample_ratio);
    % % Filter missing positions 
    % idx = 1:numel(dense_tensor);
    % idx = idx(dense_tensor(:)>0);
    % % Artificial missing position
    % mask = sort(randperm(length(idx),sample_num));
    % arti_miss_idx = idx;  
    % arti_miss_idx(mask) = [];  
    % arti_miss_mv = dense_tensor(arti_miss_idx);
    % Omega = zeros(size(dense_tensor)); Omega(mask) = 1; Omega = boolean(Omega);
    % sparse_tensor = Omega.*dense_tensor;
    % fprintf('Known elements / total elements: %6d/%6d.\n',sample_num,numel(dense_tensor));
    % clear idx 

    t0 = tic;
    Opts = initial_para(10^0.2,10^-0.8,0.1,2,25,300,true,false,false,{1,1,1},[1,1,1],{[],[],[]},size(dense_tensor));
    [~,~,info_STDC,est_tensor] = STDC(Xm,mark,Opts,dense_tensor,'image',[]);
    Lena_CPUTime_STDC(1,MR) = toc(t0);
    save(filename,"est_tensor")
    rse = TensorNorm(est_tensor - dense_tensor,'fro')/TensorNorm(dense_tensor,'fro');
    [psnr, ssim, fsim, ergas, msam] = MSIQA(dense_tensor, est_tensor);
    Lena_Eval_STDC(1,MR) = psnr; Lena_Eval_STDC(2,MR) = ssim; Lena_Eval_STDC(3,MR) = fsim; Lena_Eval_STDC(4,MR) = ergas; 
    Lena_Eval_STDC(5,MR) = msam; Lena_Eval_STDC(6,MR) = rse;

end