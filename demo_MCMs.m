clc;clear;

load('ind_MNF_20');
img = ind_MNF_20;
load('indian_pines_gt.mat')
gt_map = indian_pines_gt;
% load('pavia_MNF_20')
% img = pavia_MNF_20;
% load('paviaU_gt.mat')
% gt_map = paviaU_gt;
% load('salinas_MNF_20');
% load('Salinas_gt.mat')
% img = salinas_MNF_20;
% gt_map = salinas_gt;
% d = 20;% Can be adjusted;
% img = PCA_img(img, d);  % 




all_samples = [46,1428,830,237,483,730,28,478,20,972,2455,593,205,1265,386,93]; % indian pines
%all_samples = [6631,18649,2099,3064,1345,5029,1330,3682,947]; % pavia
%all_samples = [2009,3726,1976,1394,2678,3959,3579,11271,6203,3278,1068,1927,916,1070,7268,1807]; % salinas

no_classes = length(all_samples);
max_windows = 31;
labels = [];
for i = 1:no_classes
    tmp = ones(all_samples(i),1)*i;
    labels = [labels;tmp];
end
for iter = 1:5
train_number = ceil(all_samples*0.1);

[train_SL,test_SL,test_number]= GenerateSample(labels,train_number,no_classes);
wind_sz_set = 3:2:max_windows;
[CMsetMS] =  fun_MyGenerateCMsetsMS_m(img, wind_sz_set, double(gt_map));
numscale = length(wind_sz_set);
train_id = train_SL(1,:);
train_label = train_SL(2,:);
test_id = test_SL(1,:);
test_label = test_SL(2,:);
[trainidx_extend] = Ind_extend(train_id, numscale);
[testidx_extend] = Ind_extend(test_id, numscale);
train_samples = CMsetMS(:,:,trainidx_extend);
[trainlabel_extend] = Label_extend(train_label, numscale);
[testlabel_extend] = Label_extend(test_label, numscale);
test_samples = CMsetMS(:,:,testidx_extend);

data_reorder = cat(3,train_samples,test_samples);
[m,n,d] = size(data_reorder);
data_reorder_new = zeros(m,n,1,d);
mean_val = mean(train_samples(:));
%mean_val= mean(data_reorder,3);
std_val = std(train_samples(:));
for i = 1:d
    tmp = data_reorder(:,:,i);
   tmp = (tmp-mean_val)/std_val;
    data_reorder_new(:,:,1,i) = tmp;
end
labels_reorder = [trainlabel_extend,testlabel_extend];
set = [ones(1,size(train_samples,3)),2*ones(1,size(test_samples,3))];
data_mean = mean(train_samples,3);

images.data = data_reorder_new;
range = size(train_samples,3)+1:size(data_reorder_new,4);
images.test_samples = data_reorder_new(:,:,:,range);
images.num_scales = length(wind_sz_set);
images.test_labels = test_label;
images.data_mean = data_mean;
images.labels = labels_reorder;
images.set = set;
imdb.images = images;
meta.sets = {'train','test'};
meta.classes = {'1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16'};
%meta.classes = {'1','2','3','4','5','6','7','8','9'};
opts.expDir = 'data';
opts.batchNormalization = false;
opts.continue = false;

fprintf('Train CNN model....\n');
[net_bn, info_bn, OA, AA, Kappa] = cnn_run_new(opts,imdb,no_classes);
fprintf('Done\n');

result_OA(iter) = OA(end);
result_AA(iter) = AA(end);
result_Kappa(iter) = Kappa(end);
% filename = ['acc',num2str(max_flag),'.mat'];
% save(filename,'acc');
end


%save imdb images meta