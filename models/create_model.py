from  tensorflow.keras.applications import MobileNetV2
import tensorflow as tf
from  tensorflow.keras.layers import *
import tensorflow.keras.backend as K
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "-1"


def Create_Model(inpt=(128,128,3),train=True,num_classes=10,embedding_size=128, dropout_keep_prob=0.4):
    inpt = tf.keras.Input(inpt)
    base_model = MobileNetV2(include_top=True, input_tensor=inpt)
    out = base_model.get_layer('global_average_pooling2d').output
    x = Dropout(1.0 - dropout_keep_prob, name='Dropout')(out)
    # 全连接层到128
    # 128
    x = Dense(embedding_size, use_bias=False, name='Bottleneck')(x)
    x = BatchNormalization(momentum=0.995, epsilon=0.001, scale=False,
                           name='BatchNorm_Bottleneck')(x)

    # 创建模型
    model = tf.keras.Model(inpt, x, name='mobilenet')


    logits = Dense(num_classes)(model.output)
    softmax = Activation("softmax", name="Softmax")(logits)

    normalize = Lambda(lambda x: K.l2_normalize(x, axis=1), name="Embedding")(model.output)
    combine_model = tf.keras.Model(inpt, [softmax, normalize])


    x = Lambda(lambda x: K.l2_normalize(x, axis=1), name="Embedding")(model.output)
    model = tf.keras.Model(inpt, x)

    if train:
        return combine_model,model
    else:
        return model

# model= Create_Model(train=False)
# model.summary()