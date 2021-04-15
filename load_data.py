import numpy as np
import tensorflow.keras as k
import glob
import os
from tqdm import tqdm
import json
import math
import random
from  PIL import  Image
import cv2
from keras.utils import np_utils

def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a

def read_lines(path=r'datasets\train.txt'):
    with open(path,'r',encoding='utf8') as f:
        data= f.readlines()
    return data

class Person_Dataset(k.utils.Sequence):
    def __init__(self,image_path,batch_size,train=True,input_size=(215,90,3)):
        self.image_path = image_path
        self.num_len = len(os.listdir(image_path))
        self.batch_size = batch_size
        self.train = train
        self.json_data = self.load_data_path()
        self.json_key = list(self.json_data.keys())
        self.image_height,self.image_width,self.channel = input_size


    def load_data_path(self):
        if self.train:
            if not os.path.isfile('datasets/train.json'):
                label_data_dist={}
                label_names = read_lines()
                for labelname in tqdm(label_names):
                    labelname = labelname.strip('\n')
                    label_to_imagepaths= glob.glob(self.image_path+'\%s*'%labelname)
                    label_data_dist[labelname]=label_to_imagepaths
                    # print(label_to_imagepaths)
                json_ = json.dumps(label_data_dist)
                with open('datasets/train.json', 'w', encoding='utf8') as f:
                    f.writelines(json_)
                return  json_
            else:
                with open('datasets/train.json', 'r', encoding='utf8') as f:
                    json_ = json.loads(f.read())
                return  json_
        else:
            if not os.path.isfile('datasets/test.json'):
                label_data_dist={}
                label_names = read_lines(path=r'datasets\test.txt')
                for labelname in tqdm(label_names):
                    labelname = labelname.strip('\n')
                    label_to_imagepaths= glob.glob(self.image_path+'\%s*'%labelname)
                    label_data_dist[labelname]=label_to_imagepaths
                    # print(label_to_imagepaths)
                json_ = json.dumps(label_data_dist)
                with open('datasets/test.json', 'w', encoding='utf8') as f:
                    f.writelines(json_)
                return  json_
            else:
                with open('datasets/test.json', 'r', encoding='utf8') as f:
                    json_ = json.loads(f.read())
                return  json_


    # 随机增强数据
    def get_random_data(self, image, input_shape, jitter=.1, hue=.1, sat=1.3, val=1.3, flip_signal=True):
        '''

        :param image: PIL Image
        :param input_shape:  输入尺寸
        :param jitter: 裁剪
        :param hue: h
        :param sat: s
        :param val: v
        :param flip_signal: 翻转
        :return:
        '''
        image = image.convert("RGB")

        h, w = input_shape
        rand_jit1 = rand(1 - jitter, 1 + jitter)
        rand_jit2 = rand(1 - jitter, 1 + jitter)
        new_ar = w / h * rand_jit1 / rand_jit2

        # 随机裁剪图片
        scale = rand(0.9, 1.1)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)

        # 随机翻转图片
        flip = rand() < .5
        if flip and flip_signal:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        dx = int(rand(0, w - nw))
        dy = int(rand(0, h - nh))
        new_image = Image.new('RGB', (w, h), (128, 128, 128))

        new_image.paste(image, (dx, dy))
        image = new_image

        rotate = rand() < .5
        if rotate:
            angle = np.random.randint(-10, 10)
            a, b = w / 2, h / 2
            M = cv2.getRotationMatrix2D((a, b), angle, 1)
            image = cv2.warpAffine(np.array(image), M, (w, h), borderValue=[128, 128, 128])

            # hsv 颜色增强
        # hue = rand(-hue, hue)
        # sat = rand(1, sat) if rand() < .5 else 1 / rand(1, sat)
        # val = rand(1, val) if rand() < .5 else 1 / rand(1, val)
        # x = cv2.cvtColor(np.array(image, np.float32) / 255, cv2.COLOR_RGB2HSV)
        # x[..., 0] += hue * 360
        # x[..., 0][x[..., 0] > 1] -= 1
        # x[..., 0][x[..., 0] < 0] += 1
        # x[..., 1] *= sat
        # x[..., 2] *= val
        # x[x[:, :, 0] > 360, 0] = 360
        # x[:, :, 1:][x[:, :, 1:] > 1] = 1
        # x[x < 0] = 0
        # image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB) * 255
        image_data = image

        # 如果是单通道图片
        if self.channel == 1:
            image_data = Image.fromarray(np.uint8(image)).convert("L")
        # cv2.imshow("TEST",np.uint8(cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR)))
        # cv2.waitKey(0)
        return image_data


    def __len__(self):
        return math.ceil(self.num_len / float(self.batch_size))

    def __getitem__(self, item):
        images = np.zeros((self.batch_size, 3, self.image_height, self.image_width, self.channel))
        labels = np.zeros((self.batch_size, 3))

        for i in range(self.batch_size):
            c =np.random.choice(self.json_key,1)
            select_path = self.json_data[c[0]]

            while len(select_path)<2:
                c = np.random.choice(self.json_key, 1)
                select_path = self.json_data[c[0]]

            image_index = np.random.choice(select_path,2)

            image1 = Image.open(image_index[0])
            image1 = self.get_random_data(image1,[self.image_height,self.image_width])
            image1 = np.asarray(image1).astype(np.float64)/255.
            label = self.json_key.index(c[0])
            # print(label)
            images[i,0,:,:,:]=image1
            labels[i,0] =label

            image2 = Image.open(image_index[1])
            image2 = self.get_random_data(image2,[self.image_height,self.image_width])
            image2 = np.asarray(image2).astype(np.float64)/255.
            images[i, 1, :, :, :] = image2
            labels[i, 1] = label


            diff_c = np.random.choice(self.json_key,1)
            # print(c,diff_c)
            while diff_c[0] == c[0]:
                diff_c = np.random.choice(self.json_key, 1)

            diff_select_path = self.json_data[diff_c[0]]
            diff_c_image_path = np.random.choice(diff_select_path,1)

            diff_image = Image.open(diff_c_image_path[0])
            diff_image = self.get_random_data(diff_image,[self.image_height,self.image_width])
            diff_image = np.asarray(diff_image).astype(np.float64) / 255.
            diff_label = self.json_key.index(diff_c[0])
            images[i, 2, :, :, :] = diff_image

            labels[i, 2] = diff_label


            # print(label,diff_label)
            # print(image_index,diff_c_image_path)

        images1 = np.array(images)[:, 0, :, :, :]
        images2 = np.array(images)[:, 1, :, :, :]
        images3 = np.array(images)[:, 2, :, :, :]
        images = np.concatenate([images1, images2, images3], 0)

        labels1 = np.array(labels)[:, 0]
        labels2 = np.array(labels)[:, 1]
        labels3 = np.array(labels)[:, 2]
        labels = np.concatenate([labels1, labels2, labels3], 0)

        labels = np_utils.to_categorical(np.array(labels), num_classes=len(self.json_key))

        return images, {'Embedding': np.zeros_like(labels), 'Softmax': labels}


            # print(image_index,label)


# image_path =r'E:\DataSets\DukeMTMC-reID\DukeMTMC-reID\bounding_box_train'
# batch_size=1
# dataset=Person_Dataset(image_path,batch_size)
# # dataset.load_data_path()
# image,dict = dataset.__getitem__(1)
# embb=dict['Embedding']
# label = dict['Softmax']
#
# for i in range(3):
#     image_ = np.array(image[i]*255.,dtype='uint8')
#     image_ = cv2.cvtColor(image_,cv2.COLOR_RGB2BGR)
#     print(embb[i],label[i].argmax())
#     cv2.imshow('s',image_)
#     cv2.waitKey(0)




# print(image.shape)