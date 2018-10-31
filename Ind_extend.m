function [id_extend] = Ind_extend(id, numscale)
% Generate the new index in the sequence of new samples


id_extend = zeros(1, length(id)*numscale);
for i = 1:length(id)
    range = (id(i)-1)*numscale+1: id(i)*numscale;
    id_extend((i-1)*numscale+1:i*numscale) =range;    
end
end