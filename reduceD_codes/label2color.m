function classif=label2color(label,data_name)
%=================================================================================
%用来标色绘图的函数
%输入参数:   label        :  像素的训练和预测标签
%           data_name    :  高光谱数据集名称
%                  'uni'       --  Pavia University scene
%                  'intia'     --  Indian Pines scene
%输出参数: classif       : 结果图
%=================================================================================
[w h]=size(label);

im=zeros(w,h,3);

switch lower(data_name)
    
    case 'uni'
        map=[255 192 192;0 255 0;0 255 255;0 128 0; 255 0 255;165 82 41;128 0 128;255 0 0;255 255 0];
        for i=1:w
            for j=1:h
                switch(label(i,j))
                    case(1)
                        im(i,j,:)=uint8(map(1,:));
                    case(2)
                        im(i,j,:)=uint8(map(2,:));
                    case(3)
                        im(i,j,:)=uint8(map(3,:));
                    case(4)
                        im(i,j,:)=uint8(map(4,:));
                    case(5)
                        im(i,j,:)=uint8(map(5,:));
                    case(6)
                        im(i,j,:)=uint8(map(6,:));
                    case(7)
                        im(i,j,:)=uint8(map(7,:));
                    case(8)
                        im(i,j,:)=uint8(map(8,:));
                    case(9)
                        im(i,j,:)=uint8(map(9,:));
                end
            end
        end
    case 'india'
        map=[140 67 46;0 0 255;255 100 0;80 255 123;164 75 155;101 174 255;118 254 172; 255 91 112;255,255,0;255 48 125;255 0 255;100 0 255;0 172 254;0 255 0;171 175 80;101 193 60];
        for i=1:w
            for j=1:h
                switch(label(i,j))
                    case(1)
                        im(i,j,:)=uint8(map(1,:));
                    case(2)
                        im(i,j,:)=uint8(map(2,:));
                    case(3)
                        im(i,j,:)=uint8(map(3,:));
                    case(4)
                        im(i,j,:)=uint8(map(4,:));
                    case(5)
                        im(i,j,:)=uint8(map(5,:));
                    case(6)
                        im(i,j,:)=uint8(map(6,:));
                    case(7)
                        im(i,j,:)=uint8(map(7,:));
                    case(8)
                        im(i,j,:)=uint8(map(8,:));
                    case(9)
                        im(i,j,:)=uint8(map(9,:));
                    case(10)
                        im(i,j,:)=uint8(map(10,:));
                    case(11)
                        im(i,j,:)=uint8(map(11,:));
                    case(12)
                        im(i,j,:)=uint8(map(12,:));
                    case(13)
                        im(i,j,:)=uint8(map(13,:));
                    case(14)
                        im(i,j,:)=uint8(map(14,:));   
                    case(15)
                        im(i,j,:)=uint8(map(15,:));   
                    case(16)
                        im(i,j,:)=uint8(map(16,:));   
                end
            end
        end

end

name=sprintf('classif_%s.tif',data_name);
im=uint8(im);
classif=uint8(zeros(w,h,3));
classif(:,:,1)=im(:,:,1);
classif(:,:,2)=im(:,:,2);
classif(:,:,3)=im(:,:,3);