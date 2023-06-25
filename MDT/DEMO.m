load('signal.mat'); % 导入的原数据，变量名自动为x

T = length(x);

idd = randperm(T); % 构造随机缺失指标

x_missing = x;
x_missing(idd(1:floor(T/2))) = NaN; % 随机缺失序列数据

q = ones(T,1); % 是否为张量？
q(idd(1:floor(T/2))) = 0;

[Xest, histo, histR] = MDT_Tucker_incR(x_missing,q,50); % 输入是序列值x_missing,矩阵q，返回序列值

figure(1)
subplot(3,1,1)
plot(x);title('original signal');
subplot(3,1,2)
plot(x_missing);title('missing signal');
subplot(3,1,3)
plot(Xest);title('recovered signal');


