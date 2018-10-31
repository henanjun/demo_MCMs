function [net, info,OA,AA,Kappa]  = cnn_run_new(params,imdb,no_classes)
%CNN_MNIST  Demonstrates MatConvNet on MNIST

% run(fullfile(fileparts(mfilename('fullpath')),...
%   '..', '..', 'matlab', 'vl_setupnn.m')) ;

opts.batchNormalization = false ;
opts.continue = false;
opts.network = [] ;
opts.networkType = 'dagnn' ;
opts.expDir = fullfile('data') ;
opts.imdbPath = fullfile( 'imdb.mat');
[opts, params] = vl_argparse(opts, params) ;

opts.train = struct() ;
opts = vl_argparse(opts, params) ;
if ~isfield(opts.train, 'gpus'), opts.train.gpus = []; end;
 opts.train.gpus = 1;

% --------------------------------------------------------------------
%                                                         Prepare data
% --------------------------------------------------------------------


net = net_init('batchNormalization', opts.batchNormalization, ...
    'networkType', opts.networkType,no_classes) ;
%imdb = load(opts.imdbPath);
% net.meta.classes.name = arrayfun(@(x)sprintf('%d',x),1:9,'UniformOutput',false) ;

% --------------------------------------------------------------------
%                                                                Train
% --------------------------------------------------------------------

switch opts.networkType
  case 'simplenn', trainfn = @cnn_train ;
  case 'dagnn', trainfn = @cnn_train_dag_MV ;
end

[net, info,OA,AA,Kappa] = trainfn(net, imdb, getBatch(opts), ...
  'expDir', opts.expDir, ...
  net.meta.trainOpts, ...
  opts.train, ...
  'val', find(imdb.images.set == 2)) ;

% --------------------------------------------------------------------
function fn = getBatch(opts)
% --------------------------------------------------------------------
switch lower(opts.networkType)
  case 'simplenn'
    fn = @(x,y) getSimpleNNBatch(x,y) ;
  case 'dagnn'
    bopts = struct('numGpus', numel(opts.train.gpus)) ;
    fn = @(x,y) getDagNNBatch(bopts,x,y) ;
end

% --------------------------------------------------------------------
function [images, labels] = getSimpleNNBatch(imdb, batch)
% --------------------------------------------------------------------
images = single(imdb.images.data(:,:,:,batch)) ;
labels = imdb.images.labels(1,batch) ;

% --------------------------------------------------------------------
function inputs = getDagNNBatch(opts, imdb, batch)
% --------------------------------------------------------------------
images = single(imdb.images.data(:,:,:,batch)) ;
labels = imdb.images.labels(1,batch) ;
if opts.numGpus > 0
  images = gpuArray(images) ;
end
inputs = {'input', images, 'label', labels} ;
