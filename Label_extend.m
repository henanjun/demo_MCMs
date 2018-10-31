function [label_extend] = Label_extend(label, numscale)
% Generate the new index in the sequence of new samples


label_extend = zeros(1, length(label)*numscale);
for i = 1:length(label)
    label_extend((i-1)*numscale+1:i*numscale) =label(i);        
end

end