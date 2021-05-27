
path = "C:"+File.separator+"Users"+File.separator+"lmorales-curiel"+File.separator+"Desktop"+File.separator+"a.tif";
open(path);
title=getTitle();
selectWindow(title);
saveAs("AVI... ", path);  	