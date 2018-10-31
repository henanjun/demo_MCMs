function [OA,AA,Kappa] = process_test(net,test_samples,num_scales,test_labels)

net.move('gpu') ;
net.mode = 'test' ;
net.conserveMemory = false;
n = size(test_samples,4);
batch_size = 512;
prediction_ms = [];
no_classes = max(test_labels);
for t = 1:batch_size:n
    batchStart = t; 
    batchEnd = min(t+batch_size-1, n) ;
    images = test_samples(:,:,:,batchStart:batchEnd);
    images = single(images);
    images = gpuArray(images);
    inputs = {'input',images};   
    net.eval(inputs) ;
    scores = squeeze(gather(net.vars(length(net.vars)-3).value));
    
    [~,tmp] = max(scores,[],1);
    prediction_ms = [prediction_ms,tmp];
end
[prediction] = MultiscaleMV(prediction_ms,num_scales);
% OA = cal_acc(prediction,test_labels);
[OA,Kappa,AA,~, ~] = calcError(test_labels-1,prediction-1,[1:no_classes]);
fprintf('test accuracy: %4f',OA)
fprintf('\n')
  
