function acc = cal_acc(a,b)
if size(a,1)>size(a,2)
    a = a';
end
if size(b,1)>size(b,2)
    b = b';
end
acc = length(find(a-b==0))/length(a);
end