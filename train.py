from  load_data import Person_Dataset
from loss.Triplet_loss import  triplet_loss
from models.create_model import Create_Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import tensorflow as tf
import tensorflow.keras as k
import random
from utils import detect_image
import  matplotlib.pyplot   as plt
import numpy as np

batch_size=64
input_size=(215,90,3)
epoch =100
init_epoch =39
train_image_path =r'E:\DataSets\DukeMTMC-reID\DukeMTMC-reID\bounding_box_train'
test_image_path =r'E:\DataSets\DukeMTMC-reID\DukeMTMC-reID\query'

train_dataset=Person_Dataset(train_image_path,batch_size)
# test_dataset = Person_Dataset(test_image_path,batch_size,train=False)
test_dataset_2=Person_Dataset(test_image_path,batch_size=1,train=False)


model,pred_model = Create_Model(inpt=input_size,num_classes=1812)
# model.summary()
model.compile(loss={'Embedding':triplet_loss(batch_size=batch_size),
                    'Softmax':"categorical_crossentropy"},
              optimizer=tf.keras.optimizers.Adam(lr=1e-5),
              metrics={'Softmax':'acc'})
model.load_weights('logs\ep039-loss0.066.h5')
# pred_model = Create_Model(inpt=input_size,num_classes=1812,train=False)

# 自定义回调函数
class Evaluator(k.callbacks.Callback):
    def __init__(self):
        self.accs = []
    def on_epoch_begin(self, epoch, logs=None):
        for i in range(3):
            plt.clf()
            radmon_int = random.randint(0,test_dataset_2.__len__()-1)
            image, _ = test_dataset_2.__getitem__(radmon_int)
            # print(image[0].shape)
            same_l1 = detect_image(image[0],image[1],model=pred_model)
            diff_l2 = detect_image(image[0],image[2],model=pred_model)

            plt.subplot(1, 3, 1)
            plt.imshow(np.array(image[0]))

            plt.subplot(1, 3, 2)
            plt.imshow(np.array(image[1]))

            plt.text(-12, -12, 'same:%.3f' % same_l1, ha='center', va='bottom', fontsize=11)

            plt.subplot(1, 3, 3)
            plt.imshow(np.array(image[2]))
            plt.text(-24, -12, 'diff:%.3f' % diff_l2, ha='center', va='bottom', fontsize=11)

            plt.savefig(r'image\test_epoch_%s_%s.png'%(epoch,i))

        # cv2.imwrite('train_img\epoch_%s_train.jpg'%epoch,train_img)
evaluator = Evaluator()






checkpoint_period = ModelCheckpoint(r'logs/' + 'ep{epoch:03d}-loss{loss:.3f}.h5',
                            monitor='val_loss', save_weights_only=True, save_best_only=True, period=1)
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=5, verbose=1)
early_stopping = EarlyStopping(monitor='loss', min_delta=0, patience=10, verbose=1)

model.fit(train_dataset,steps_per_epoch=train_dataset.__len__(),
          # validation_data=test_dataset,validation_steps=test_dataset.__len__(),
          epochs=epoch,
          initial_epoch=init_epoch,
          callbacks=[checkpoint_period,reduce_lr,early_stopping,evaluator],workers=4)

