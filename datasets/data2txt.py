import os
import cv2


data_path=r'E:\DataSets\DukeMTMC-reID\DukeMTMC-reID\bounding_box_train'

image_paths=[os.path.join(data_path,p)for p in os.listdir(data_path)]
nameid=[]
for image_path in image_paths:
    name=image_path.split('\\')[-1].split('_')[0]
    if name+'\n' not in nameid:
        nameid.append(name+'\n')

with open('train.txt', 'w', encoding='utf8') as f:
    f.writelines(nameid)



# _0087=[i for i in os.listdir(data_path) if i.startswith(('0087'))]
# print(_0087)
