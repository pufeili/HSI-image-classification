clear;close all;clc

load('PaviaU.mat');%载入数据集
im = paviaU;       
[I_row,I_line,I_high] = size(im);%读取数据集3维大小
im_all = im;
[I_row_all,I_line_all,I_high_all] = size(im_all);
im1_all = reshape(im_all,[I_row_all*I_line_all,I_high_all]);%重组为2维矩阵 
im1_all = im1_all';

load('PaviaU_gt.mat')%载入分类标签矩阵
no_classes = length(unique(paviaU_gt))-1;

RandSampled{1}=[133 373 100 100 100 101 100 100 100];  %2
RandSampled{2}= RandSampled{1}*2;  %4
RandSampled{3}= RandSampled{1}*3; %6
RandSampled{4}= RandSampled{1}*4; %8
RandSampled{5}= RandSampled{1}*5; %10


K = no_classes;
Train_Label = [];
Train_index = [];
for ii = 1: no_classes
%  index_ii =  find(indian_pines_gt == ii);
   index_ii =  find(paviaU_gt == ii);
   class_ii = ones(length(index_ii),1)* ii;
   Train_Label = [Train_Label class_ii'];
   Train_index = [Train_index index_ii'];   
end 
%按类别重新将数据集和对应标签排列为1维矩阵

RandSampled_Num=RandSampled{3};
tr_lab = [];
tt_lab = [];
tr_dat = [];
tt_dat = [];

Index_train = {};
Index_test = {};
for i = 1: K
    W_Class_Index = find(Train_Label == i);%  每一类的标签的位置
    Random_num = randperm(length(W_Class_Index));
    Random_Index = W_Class_Index(Random_num);%把每一类的样本数据随机打乱
    
    Tr_Index = Random_Index(1:RandSampled_Num(i)); %每个类选一定的数量即比例，并记下位置 
    Index_train{i} = Train_index(Tr_Index);%把一定比例的样本划为训练集
    
    Tt_Index = Random_Index(RandSampled_Num(i)+1 :end);
    Index_test{i} = Train_index(Tt_Index);%剩下的作为测试集

    tr_ltemp = ones(RandSampled_Num(i),1)'* i;
    tr_lab = [tr_lab tr_ltemp];
    tr_Class_DAT = im1_all(:,Train_index(Tr_Index));
    tr_dat = cat(2,tr_dat,tr_Class_DAT);%按类将训练集标签和原始数据排列为一一对应。
    
    tt_ltemp = ones(length(Index_test{i}),1)'* i;
    tt_lab = [tt_lab tt_ltemp];
    tt_Class_DAT = im1_all(:,Train_index(Tt_Index));
    tt_dat = cat(2,tt_dat,tt_Class_DAT);%按类将测试集标签和原始数据排列为一一对应。
end
[train_data,PS] = mapminmax(tr_dat,0,1);%将原始数据归一化处理
train_data=  train_data';    
train_label=tr_lab';

test_data = mapminmax('apply',tt_dat,PS);%归一化
test_data =test_data';
test_label  =  tt_lab'; 

%%降维与分类
type_num = no_classes;


% method = [];
% method.mode = 'pca';
% method.K = 50;
% [train_data,U] = featureExtract(train_data,train_label,method,type_num);
% test_data = projectData(test_data, U, method.K);%将测试集按照训练集的映射方式映射到空间中
% model = svmtrain(train_label,train_data,'-s 0 -c 10^5 t = 2'); %使用rbf核函数
% [pca_pre,~, ~] = svmpredict(test_label,test_data, model);
% [Acc_pca,AA_pca,OA_pca,Kap_pca,CM_pca] = EvalPara(pca_pre,test_label);
% save('pca_pred.mat','pca_pre');
% save('pca_cm.mat','CM_pca');
% fprintf('\n pca+svm Accuracy: %f\n',OA_pca);


% method = [];
% method.mode = 'lda';
% method.K = 100;
% [train_data,U] = featureExtract(train_data,train_label,method,type_num);
% test_data = projectData(test_data, U, method.K);%将测试集按照训练集的映射方式映射到空间中
% model = svmtrain(train_label,train_data,'-s 0 -c 10^5 t = 2'); %使用rbf核函数
% [lda_pre,~, ~] = svmpredict(test_label,test_data, model);
% [Acc_lda,AA_lda,OA_lda,Kap_lda,CM_lda] = EvalPara(lda_pre,test_label);
% save('lda_pred.mat','lda_pre');
% save('lda_cm.mat','CM_lda');
% fprintf('\n lda+svm Accuracy: %f\n',OA_lda);


% method = [];
% method.mode = 'lpp';
% method.K = 100;
% method.weightmode = 'binary';
% method.knn_k = 5;

method = [];
method.mode = 'lpp';
method.K = 100;
method.weightmode = 'heatkernel';
method.t = 10;
method.knn_k = 7;
[train_data,U] = featureExtract(train_data,train_label,method,type_num);
test_data = projectData(test_data, U, method.K);%将测试集按照训练集的映射方式映射到空间中
model = svmtrain(train_label,train_data,'-s 0 -c 10^5 t = 2'); %使用rbf核函数
[lpp_pre,ac, ~] = svmpredict(test_label,test_data, model);
[Acc_lpp,AA_lpp,OA_lpp,Kap_lpp,CM_lpp] = EvalPara(lpp_pre,test_label );
save('lpp_pred.mat','lpp_pre');
save('lpp_cm.mat','CM_lpp');
fprintf('\n lpp+svm Accuracy: %f\n',OA_lpp);

i_pre = 0;
ID = pred;
gt_new = zeros(1,I_row*I_line);
 for i=1:no_classes
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

label2color(gt_new1,'uni');
figure,imshow(label2color(gt_new1,'uni'),[]),colormap jet
%不同的类别标上不同的颜色，输出图像。

label2color(paviaU_gt,'uni');
figure,imshow(label2color(paviaU_gt,'uni'),[]),colormap jet





