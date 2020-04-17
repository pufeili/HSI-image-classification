clear;close all;clc

dataset = "paviau"; % "indian"  "paviau"
train_ratio = 0.1 ;
[Index_train,Index_test] = DataCreate(dataset,train_ratio);
% 划分训练集，测试集 ，如果datasets文件夹里有，就不必执行这条命令

switch lower(dataset)
    case "paviau"
        load(strcat('./datasets/',"pavia_train_data",num2str(train_ratio),'.mat'))
        load(strcat('./datasets/',"pavia_train_label",num2str(train_ratio),'.mat'))
        load(strcat('./datasets/',"pavia_test_data",num2str(train_ratio),'.mat'))
        load(strcat('./datasets/',"pavia_test_label",num2str(train_ratio),'.mat'))
        type_num = length(unique(train_label));
    case "indian"
        pass
end

[train_data, mu, sigma] = featureCentralize(train_data); %%将样本标准化（服从N(0,1)分布）
test_data = bsxfun(@minus, test_data, mu);
test_data = bsxfun(@rdivide, test_data, sigma);   %%将测试样本标准化

%========================
method = [];
method.mode = 'pca';
method.K = 50;        %%降维的维数
svm = [];
tic;
[pca_train_data,U] = featureExtract(train_data,train_label,method,type_num);
pca_test_data = projectData(test_data, U, method.K);    %%将测试集按照训练集的映射方式映射到空间中
model = libsvmtrain(train_label,pca_train_data,'-s 0 -c 10^5 -t 2 -q'); % svmtrain 参数
pca_pred = libsvmpredict(test_label,pca_test_data,model); %得到预测的标签
svm = [svm mean(double(pca_pred == test_label)) * 100]; %得到一个准确度
fprintf('pca+svm Accuracy: %f\n', mean(svm));
pca_time = toc;
%=========================
% method = [];
% method.mode = 'lpp';
% method.K = 50;
% method.weightmode = 'binary';
% method.knn_k = 5;
% svm = [];
% 
% [lppB_train_data,U] = featureExtract(train_data,train_label,method,type_num);
% lppB_test_data = projectData(test_data, U, method.K);    %%将测试集按照训练集的映射方式映射到空间中
% model = libsvmtrain(train_label,lppB_train_data,'-s 0 -c 10^5 -t 2 -q');
% lppB_pred = libsvmpredict(test_label,lppB_test_data,model);
% svm = [svm mean(double(lppB_pred == test_label)) * 100];
% fprintf('lpp+svm Accuracy: %f\n', mean(svm));

% %===================================
% method = [];
% method.mode = 'lpp';
% method.K = 50;
% method.weightmode = 'heatkernel';
% method.t = 10;
% method.knn_k = 7;
% svm = [];
% 
% [lppH_train_data,U] = featureExtract(train_data,train_label,method,type_num);
% lppH_test_data = projectData(test_data, U, method.K);    %%将测试集按照训练集的映射方式映射到空间中
% model = libsvmtrain(train_label,lppH_train_data,'-s 0 -c 10^5 -t 2 -q');
% lppH_pred = libsvmpredict(test_label,lppH_test_data,model);
% svm = [svm mean(double(lppH_pred == test_label)) * 100];
% fprintf('lpp with heatkernel+svm Accuracy: %f\n', mean(svm));
% 
% %=====================================
% method = [];
% method.mode = 'ldpp';
% method.K = 50;
% method.mu = 1;
% method.gamma = 0.01; %0.001 0.1 1 10
% method.ratio_b = 1;
% method.ratio_w = 1;
% method.weightmode = 'binary';
% method.knn_k = 5;
% svm = [];
% 
% [ldppB_train_data,U] = featureExtract(train_data,train_label,method,type_num);
% ldppB_test_data = projectData(test_data, U, method.K);    %%将测试集按照训练集的映射方式映射到空间中
% model = libsvmtrain(train_label,ldppB_train_data,'-s 0 -c 10^5 -t 2 -q');
% ldppB_pred = libsvmpredict(test_label,ldppB_test_data,model);
% svm = [svm mean(double(ldppB_pred == test_label)) * 100];
% fprintf('ldpp with binary+svm Accuracy: %f\n', mean(svm));
% 
% %=======================================
% method = [];
% method.mode = 'ldpp';
% method.K = 50;
% method.mu = 0.0001;
% method.gamma = 0.0001;
% method.ratio_b = 0.9;
% method.ratio_w = 0.9;
% method.weightmode = 'heatkernel';
% method.t = 0.1;
% method.knn_k = 5;
% svm = [];
% 
% [ldppH_train_data,U] = featureExtract(train_data,train_label,method,type_num);
% ldppH_test_data = projectData(test_data, U, method.K);    %%将测试集按照训练集的映射方式映射到空间中
% model = libsvmtrain(train_label,ldppH_train_data,'-s 0 -c 10^5 -t 2 -q');
% ldppH_pred = libsvmpredict(test_label,ldppH_test_data,model);
% svm = [svm mean(double(ldppH_pred == test_label)) * 100];
% ========================================
% [cls_acc,AA,OA,Kappa,CM] = EvalPara(ldppH_pred,test_label);
% =========================================
% fprintf('ldpp with heatkernel+svm Accuracy: %f\n', mean(svm));


%%画图
i_pre = 0;
ID = pca_pred;  %ID = ldppH_pred;

load('./datasets/PaviaU.mat');%载入数据集
im = paviaU;
[I_row,I_line,I_high] = size(im);%读取数据集3维大小

gt_new = zeros(1,I_row*I_line);
 for i=1:type_num
   for j_train = 1:1:length(Index_train{i}) 
      gt_new(Index_train{i}(j_train))=i;
   end
   for j_test = 1:1:length(Index_test{i}) 
      gt_new(Index_test{i}(j_test))=ID(i_pre+j_test);
   end
   i_pre = i_pre+length(Index_test{i});
 end
%按SVM分类的结果将数据重新组合为2维矩阵      
gt_new1 = reshape(gt_new,I_row,I_line);

label2color(gt_new1,'uni'); %intia
figure,imshow(label2color(gt_new1,'uni'),[]),colormap jet
%不同的类别标上不同的颜色，输出图像。

load('./datasets/PaviaU_gt.mat')%载入分类标签矩阵
label2color(paviaU_gt,'uni');
figure,imshow(label2color(paviaU_gt,'uni'),[]),colormap jet





