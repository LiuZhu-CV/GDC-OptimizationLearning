for i=1:4
    for j=1:8
    load(['im0' num2str(i) '_ker0' num2str(j) '.mat']);
    Pic=cat(3,y,y,y);
    Pic=im2uint8(Pic);
    imwrite(Pic,['im0' num2str(i) '_k0' num2str(j) '.png'],'png');
    end
end