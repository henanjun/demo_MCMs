function [net, info] = lenet_train(imdb, expDir)
% CNN_MNIST  Demonstrated MatConNet on MNIST using DAG
	%run(fullfile(fileparts(mfilename('fullpath')), '../../', 'matlab', 'vl_setupnn.m')) ;

	% some common options
	opts.train.batchSize = 128 ;
	opts.train.numEpochs = 100 ;
	opts.train.continue = false ;
	opts.train.gpus = 1 ;
	opts.train.learningRate = 0.001 ;
	opts.train.expDir = expDir;
	opts.train.numSubBatches = 1 ;
    opts.weightInitMethod = 'xavier'; %  gaussian; xavier; xavierimproved
    opts.scale = 1;
	% getBatch options
	bopts.useGpu = numel(opts.train.gpus) >  0 ;


	% network definition!
	% MATLAB handle, passed by reference
	net = dagnn.DagNN() ;
	net.addLayer('conv1', dagnn.Conv('size', [3 3 1 128], 'hasBias', true, 'stride', [1, 1], 'pad', [0 0 0 0]), {'input'}, {'conv1'},  {'conv1f'  'conv1b'});
	net.addLayer('pool1', dagnn.Pooling('method', 'max', 'poolSize', [2, 2], 'stride', [2 2], 'pad', [0 0 0 0]), {'conv1'}, {'pool1'}, {});
    net.addLayer('bn1', dagnn.BatchNorm(), {'pool1'}, {'bn1'},  {'bn1f'  'bn1b'})
    net.addLayer('relu1', dagnn.ReLU(), {'bn1'}, {'relu1'}, {});
    
	net.addLayer('conv2', dagnn.Conv('size', [3 3 128 64], 'hasBias', true, 'stride', [1, 1], 'pad', [0 0 0 0]), {'relu1'}, {'conv2'},  {'conv2f'  'conv2b'});
	net.addLayer('pool2', dagnn.Pooling('method', 'max', 'poolSize', [2, 2], 'stride', [2 2], 'pad', [0 0 0 0]), {'conv2'}, {'pool2'}, {});
    net.addLayer('bn2', dagnn.BatchNorm(), {'pool2'}, {'bn2'},  {'bn2f'  'bn2b'})
    net.addLayer('relu2', dagnn.ReLU(), {'bn2'}, {'relu2'}, {});
    
    net.addLayer('conv3', dagnn.Conv('size', [3 3 64 128], 'hasBias', true, 'stride', [1, 1], 'pad', [0 0 0 0]), {'relu2'}, {'conv3'},  {'conv3f'  'conv3b'});
    net.addLayer('bn3', dagnn.BatchNorm(), {'conv3'}, {'bn3'},  {'bn3f'  'bn3b'})
	net.addLayer('f1', dagnn.Conv('size', [1 1 128 128], 'hasBias', true, 'stride', [1, 1], 'pad', [0 0 0 0]), {'conv3'}, {'f1'},  {'f1_f'  'f1_b'});
    net.addLayer('bn4', dagnn.BatchNorm(), {'f1'}, {'bn4'},  {'bn4f'  'bn4b'})
    net.addLayer('relu_f1', dagnn.ReLU(), {'f1'}, {'relu_f1'}, {});
    %net.addLayer('drop1', dagnn.DropOut('rate', 0.1), {'relu_f1'}, {'drop1'}, {});
    net.addLayer('f2', dagnn.Conv('size', [1 1 128 128], 'hasBias', true, 'stride', [1, 1], 'pad', [0 0 0 0]), {'relu_f1'}, {'f2'},  {'f2_f'  'f2_b'});
    net.addLayer('bn5', dagnn.BatchNorm(), {'f2'}, {'bn5'},  {'bn5f'  'bn5b'})
    net.addLayer('relu_f2', dagnn.ReLU(), {'bn5'}, {'relu_f2'}, {});
    net.addLayer('f3', dagnn.Conv('size', [1 1 128 16], 'hasBias', true, 'stride', [1, 1], 'pad', [0 0 0 0]), {'relu_f2'}, {'f3'},  {'f3_f'  'f3_b'});
    
	net.addLayer('prediction', dagnn.SoftMax(), {'f3'}, {'prediction'}, {});
	net.addLayer('objective', dagnn.Loss('loss', 'log'), {'prediction', 'label'}, {'objective'}, {});
	net.addLayer('error', dagnn.Loss('loss', 'classerror'), {'prediction','label'}, 'error') ;
	% -- end of the network
    %net.removeLayer('drop1');
	% initialization of the weights (CRITICAL!!!!)
	initNet(net, opts);

	% do the training!
	info = cnn_train_dag_MV(net, imdb, @(i,b) getBatch(bopts,i,b), opts.train, 'val', find(imdb.images.set == 2)) ;
end

function initNet(net, opts)
	net.initParams();
	%
	for l=1:length(net.layers)
		% is a convolution layer?
		if(strcmp(class(net.layers(l).block), 'dagnn.Conv'))
			f_ind = net.layers(l).paramIndexes(1);
			b_ind = net.layers(l).paramIndexes(2);
            
            net.params(f_ind).value = init_weight(opts, size(net.params(f_ind).value), 'single');
			net.params(f_ind).learningRate = 1;
			net.params(f_ind).weightDecay = 1;

			net.params(b_ind).value = zeros(size(net.params(b_ind).value), 'single');
			net.params(b_ind).learningRate = 1;
			net.params(b_ind).weightDecay = 1;
        end
	end
end

% function on charge of creating a batch of images + labels
% function inputs = getBatch(opts, imdb, batch)
% 	images = imdb.images.data(:,:,batch) ;
% 	labels = imdb.images.labels(1,batch) ;
% 	if opts.useGpu > 0
%   		images = gpuArray(images) ;
% 	end
% 
% 	inputs = {'input', images, 'label', labels} ;
% end

function weights = init_weight(opts,sz, type)
% -------------------------------------------------------------------------
% See K. He, X. Zhang, S. Ren, and J. Sun. Delving deep into
% rectifiers: Surpassing human-level performance on imagenet
% classification. CoRR, (arXiv:1502.01852v1), 2015.
h=sz(1);w=sz(2);in=sz(3);out=sz(4);
switch lower(opts.weightInitMethod)
  case 'gaussian'
    sc = 0.01/opts.scale ;
    weights = randn(h, w, in, out, type)*sc;
  case 'xavier'
    sc = sqrt(3/(h*w*in)) ;
    weights = (rand(h, w, in, out, type)*2 - 1)*sc ;
  case 'xavierimproved'
    sc = sqrt(2/(h*w*out)) ;
    weights = randn(h, w, in, out, type)*sc ;
  otherwise
    error('Unknown weight initialization method''%s''', opts.weightInitMethod) ;
end
end
