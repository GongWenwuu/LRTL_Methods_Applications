load('signal.mat'); % �����ԭ���ݣ��������Զ�Ϊx

T = length(x);

idd = randperm(T); % �������ȱʧָ��

x_missing = x;
x_missing(idd(1:floor(T/2))) = NaN; % ���ȱʧ��������

q = ones(T,1); % �Ƿ�Ϊ������
q(idd(1:floor(T/2))) = 0;

[Xest, histo, histR] = MDT_Tucker_incR(x_missing,q,50); % ����������ֵx_missing,����q����������ֵ

figure(1)
subplot(3,1,1)
plot(x);title('original signal');
subplot(3,1,2)
plot(x_missing);title('missing signal');
subplot(3,1,3)
plot(Xest);title('recovered signal');


