function [predict_label] = MultiscaleMV(predict_label_ex, num_scale)

predict_label = zeros(1,length(predict_label_ex)/num_scale);

for i = 1:length(predict_label)
    range = (i-1)*num_scale+1:i*num_scale;
    predict_label(i) = majorityvote(predict_label_ex(range) );
end

end