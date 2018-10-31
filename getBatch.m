function inputs = getBatch(opts, imdb, batch)
    images  = imdb.images.data(:,:,:,batch);
%     images = images - repmat( imdb.images.data_mean , [1,1,1,size(images,3)]);
    images = single(images);
    labels = imdb.images.labels(1,batch);
    if  opts.useGpu > 0
                images = gpuArray(images);
    end
    inputs = {'input',images,'label',labels};
end