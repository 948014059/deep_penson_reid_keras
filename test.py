from models.create_model import Create_Model
from  load_data import Person_Dataset
from utils import detect_image
import os
import time
import cv2
import matplotlib.pyplot as plt
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

test_image_path =r'E:\DataSets\DukeMTMC-reID\DukeMTMC-reID\query'
test_dataset_2=Person_Dataset(test_image_path,batch_size=1,train=False)

input_size=(215,90,3)
model,pred_model = Create_Model(inpt=input_size,num_classes=1812)
model.load_weights('logs\ep039-loss0.066.h5')

true_=0
for i in range(test_dataset_2.__len__()):
    image, _ = test_dataset_2.__getitem__(i)
    t1=time.time()
    same_l1 = detect_image(image[0], image[1], model=pred_model)
    t2= time.time()
    diff_l2 = detect_image(image[0], image[2], model=pred_model)

    plt.subplot(1, 3, 1)
    plt.imshow(np.array(image[0]))

    plt.subplot(1, 3, 2)
    plt.imshow(np.array(image[1]))

    plt.text(-12, -12, 'same:%.3f' % same_l1, ha='center', va='bottom', fontsize=11)

    plt.subplot(1, 3, 3)
    plt.imshow(np.array(image[2]))
    plt.text(-24, -12, 'diff:%.3f' % diff_l2, ha='center', va='bottom', fontsize=11)
    plt.show()
    input_=input('')


    print('0-1:%s 0-2:%s time:%s'%(same_l1,diff_l2,t2-t1))
    if same_l1<1 and diff_l2>1:
        true_+=1

print('准确率：%.2f'%(true_/test_dataset_2.__len__()))





