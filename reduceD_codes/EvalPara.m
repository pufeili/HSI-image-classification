function [Acc_Cls,AA,OA,Kap,CM] = EvalPara(predicted_label,test_label )

%EVALPARA   生成相关的评估参数
%    
%                a = accaracy_all;  各个类的分类精度
%                b = AA_accaracy; 平均分类精度
%                c = OA_ACC;       整体分类精度                   
%                d = Kappa;          kappa系数
%                e = ConfuMatrix;   混淆矩阵
%
%

clsNum =  length(unique(test_label));
ID = predicted_label;

for i = 1:clsNum
  in_i=find(test_label==i);
  bb_i=length(find(ID(in_i)==i));
  aa_i=length(find(test_label==i));
  accaracy_all(i)=bb_i/aa_i;%计算每一类分类精度
end
OA_ACC=sum(ID==test_label)/length(test_label);%计算OA
AA_accaracy = mean(accaracy_all );%计算AA
accaracy_all=accaracy_all';
MatrixClassTable = [test_label,ID];
[ConfuMatrix,Kappa]=ClassifiEvaluate(MatrixClassTable,clsNum);%计算Kappa

Acc_Cls = accaracy_all;
OA = OA_ACC;
AA = AA_accaracy;
Kap = Kappa;
CM = ConfuMatrix;
 
end

