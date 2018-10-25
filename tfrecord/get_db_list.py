import os
import random

root_dir = "/root/DB/VOC/VOC2012/JPEGImages"

total_imgset = os.listdir(root_dir)
total_imgset = [i[:-4] for i in total_imgset]

random.shuffle(total_imgset)

train_set = sorted(total_imgset[:2000])
val_set = sorted(total_imgset[2000:2500])

save_dir = "/root/DB/VOC/VOC2012/ImageSets/Main/"

f_train = open(save_dir + "train_2000.txt", "w")
f_val = open(save_dir + "val_500.txt", "w")

for train in train_set:
    f_train.write(train + "\n")

for val in val_set:
    f_val.write(val + "\n")

