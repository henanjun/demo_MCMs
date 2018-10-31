function net = net_init(varargin)
% CNN_MNIST_LENET Initialize a CNN similar for MNIST
opts.batchNormalization = false ;
opts.networkType = 'dagnn' ;
opts.weightInitMethod = 'xavier'; %  gaussian; xavier; xavierimproved
opts.scale = 1;
opts = vl_argparse(opts, varargin(1:end-1)) ;
no_classes = varargin{end};

rng('default');
rng(0) ;

net.layers = {} ;
net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{init_weight_new(opts,3,3,1,128, 'single'), zeros(1, 128, 'single')}}, ...
                           'stride', 1, ...
                           'pad', 0,'inputs','xno1') ;
% net.layers{end+1} = struct('type', 'pool', ...
%                            'method', 'max', ...
%                            'pool', [2 2], ...
%                            'stride', 2, ...
%                            'pad', 0) ;
net.layers{end+1} = struct('type', 'relu') ;

net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{init_weight_new(opts,3,3,128,64, 'single'),zeros(1,64,'single')}}, ...
                           'stride', 1, ...
                           'pad', 0) ;
% net.layers{end+1} = struct('type', 'pool', ...
%                            'method', 'max', ...
%                            'pool', [2 2], ...
%                            'stride', 2, ...
%                            'pad', 0) ;
net.layers{end+1} = struct('type', 'relu') ;

net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{init_weight_new(opts,16,16,64,128, 'single'),  zeros(1,128,'single')}}, ...
                           'stride', 1, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'relu') ;
net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{init_weight_new(opts,1,1,128,128, 'single'), zeros(1,128,'single')}}, ...
                           'stride', 1, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'relu') ;
net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{init_weight_new(opts,1,1,128,no_classes, 'single'), zeros(1,no_classes,'single')}}, ...
                           'stride', 1, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'softmaxloss') ;

% optionally switch to batch normalization
if opts.batchNormalization
  count = 0;
  net = insertBnorm(net, 1+count) ;
  count = count+1;
  net = insertBnorm(net, 4+count) ;
  count = count+1;
  net = insertBnorm(net, 7+count) ;
  count = count+1;
  net = insertBnorm(net, 9+count) ;
end

% Meta parameters
net.meta.inputSize = [20 20 1] ;
net.meta.trainOpts.learningRate = 0.001 ;
net.meta.trainOpts.numEpochs = 30 ;
net.meta.trainOpts.batchSize = 100 ;

% Fill in defaul values
net = vl_simplenn_tidy(net) ;

% Switch to DagNN if requested
switch lower(opts.networkType)
  case 'simplenn'
    % done
  case 'dagnn'
    net = dagnn.DagNN.fromSimpleNN(net, 'canonicalNames', true) ;
    net.addLayer('error', dagnn.Loss('loss', 'classerror'), {'prediction','label'}, 'error') ;
  otherwise
    assert(false) ;
end

% --------------------------------------------------------------------
function net = insertBnorm(net, l)
% --------------------------------------------------------------------
assert(isfield(net.layers{l}, 'weights'));
ndim = size(net.layers{l}.weights{1}, 4);
layer = struct('type', 'bnorm', ...
               'weights', {{ones(ndim, 1, 'single'), zeros(ndim, 1, 'single')}}, ...
               'learningRate', [1 1 0.05], ...
               'weightDecay', [0 0]) ;
net.layers{l}.weights{2} = [] ;  % eliminate bias in previous conv layer
net.layers = horzcat(net.layers(1:l), layer, net.layers(l+1:end)) ;
