function [CMsetMS] =  fun_MyGenerateCMsetsMS_m(hsi_img, wind_sz_set, gt_map)
%% Using cov and multiscale for samples augmentation
fprintf('Generate MCMs samples....\n');
[m,n,d] = size(hsi_img);
scale_num = length(wind_sz_set);   
tol = 1e-3;
Multi_scale_nblk = [];
Multi_scale_PS_H = [];
for i = 1:scale_num
 Multi_scale_nblk(i) = wind_sz_set(i)*wind_sz_set(i);   
 Multi_scale_PS_H(i) = floor(wind_sz_set(i)/2);  
end
multiscalemap = 1: wind_sz_set(end)*wind_sz_set(end);
multiscalemap = reshape(multiscalemap,[wind_sz_set(end) wind_sz_set(end) ]);
xx = Multi_scale_PS_H(end)+1;
yy = Multi_scale_PS_H(end)+1;
Multi_scale_im_Ex = padarray(hsi_img,[Multi_scale_PS_H(end) Multi_scale_PS_H(end)],'symmetric'  );    

%Gerate the neighbors for each scale
scale_index = {};
for ss = 1:scale_num
  index_temp = multiscalemap((xx-Multi_scale_PS_H(ss)):(xx+Multi_scale_PS_H(ss)),(yy-Multi_scale_PS_H(ss)):(yy+Multi_scale_PS_H(ss)));
  index_vec = index_temp(:)'; 
  scale_index{ss} = index_vec;
end
flag = 1;
no_classes = length(unique(gt_map))-1;
for i  = 1:no_classes
    idx_tmp = find(gt_map==i);
     for j = 1:length(idx_tmp)
          [X,Y] = ind2sub([m,n],idx_tmp(j));
           X_new = X+Multi_scale_PS_H(end);
           Y_new = Y+Multi_scale_PS_H(end);         
           X_range = [X_new-Multi_scale_PS_H(end) : X_new+Multi_scale_PS_H(end)];
           Y_range = [Y_new-Multi_scale_PS_H(end) : Y_new+Multi_scale_PS_H(end)];
           tt_Class_DAT_temp = Multi_scale_im_Ex(X_range,Y_range,:);
           [r,l,h]=size(tt_Class_DAT_temp);
           tt_Class_DAT = reshape(tt_Class_DAT_temp,[r*l,h])';           
           for k = 1: scale_num
                tmp_mat = tt_Class_DAT(:,scale_index{k});  
                tmp_mat = scale_func(tmp_mat);
                mean_mat = mean(tmp_mat,2);
                centered_mat = tmp_mat-repmat(mean_mat,1,size(tmp_mat,2));
                tmp = centered_mat*centered_mat'/((size(tmp_mat,2))-1);
                CMsetMS(:,:,flag)= tmp;

                flag = flag+1;
           end
     end
end

% % CMsetMS = zeros(d,d,  scale_num*m*n);
% flag = 1;
% for j = 1:m*n
%           [X,Y] = ind2sub([m,n],j);
%              if gt_map(X,Y) ~=0
%                    X_new = X+Multi_scale_PS_H(end);
%                    Y_new = Y+Multi_scale_PS_H(end);         
%                    X_range = [X_new-Multi_scale_PS_H(end) : X_new+Multi_scale_PS_H(end)];
%                    Y_range = [Y_new-Multi_scale_PS_H(end) : Y_new+Multi_scale_PS_H(end)];
%                    tt_Class_DAT_temp = Multi_scale_im_Ex(X_range,Y_range,:);
%                    [r,l,h]=size(tt_Class_DAT_temp);
%                    tt_Class_DAT = reshape(tt_Class_DAT_temp,[r*l,h])';           
%                    for i = 1: scale_num
%                         tmp_mat = tt_Class_DAT(:,scale_index{i});  
%                         tmp_mat = scale_func(tmp_mat);
%                         mean_mat = mean(tmp_mat,2);
%                         centered_mat = tmp_mat-repmat(mean_mat,1,size(tmp_mat,2));
%                         tmp = centered_mat*centered_mat'/((size(tmp_mat,2))-1);
%                         CMsetMS(:,flag)= SPD2Euclidean(tmp);
%                         CMsetMS_logm(:,flag) = SPD2Euclidean(logm(tmp+tol*eye(size(tmp,1))*(trace(tmp))));
%                         flag = flag+1;
%                    end
%              else
%                  continue;
%              end
%  end
fprintf('Done!\n')
end
