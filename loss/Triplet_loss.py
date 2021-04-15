import tensorflow as tf
from keras import backend as K
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def triplet_loss(alpha = 0.2, batch_size = 32):
    def _triplet_loss(y_true, y_pred):
        anchor, positive, negative = y_pred[:batch_size], y_pred[batch_size:int(2*batch_size)], y_pred[-batch_size:]
        # 同一张人脸的 欧几里得距离
        pos_dist = K.sqrt(K.sum(K.square(anchor - positive), axis=-1))
        # 不同人脸的 欧几里得距离
        neg_dist = K.sqrt(K.sum(K.square(anchor - negative), axis=-1))

        # Triplet Loss
        basic_loss = pos_dist - neg_dist + alpha #小
        idxs = tf.where(basic_loss > 0)


        select_loss = tf.gather_nd(basic_loss,idxs) #大

        # 小   大
        loss = K.sum(K.maximum(basic_loss, 0)) / tf.cast(tf.maximum(1, tf.shape(select_loss)[0]), tf.float32)
        return loss
    return _triplet_loss