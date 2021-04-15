import numpy as np
# import matplotlib.pyplot as plt
from PIL import Image

def letterbox_image(self, image, size):
    if self.input_shape[-1] == 1:
        image = image.convert("RGB")
    iw, ih = image.size
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    if self.input_shape[-1] == 1:
        new_image = new_image.convert("L")
    return new_image


def detect_image(image_1, image_2,model,np_data= True,input_shape=[215,90,3]):
        # ---------------------------------------------------#
        #   对输入图片进行不失真的resize
        # ---------------------------------------------------#
    if not np_data:
        image_1 = letterbox_image(image_1 ,[input_shape[1] ,input_shape[0]])
        image_2 = letterbox_image(image_2 ,[input_shape[1] ,input_shape[0]])

        # ---------------------------------------------------#
        #   进行图片的归一化
        # ---------------------------------------------------#
        image_1 = np.asarray(image_1).astype(np.float64 ) /255
        image_2 = np.asarray(image_2).astype(np.float64 ) /255

    photo1 = np.expand_dims(image_1 ,0)
    photo2 = np.expand_dims(image_2 ,0)
    # ---------------------------------------------------#
    #   图片传入网络进行预测
    # ---------------------------------------------------#
    output1 = model.predict(photo1)
    output2 = model.predict(photo2)

    # ---------------------------------------------------#
    #   计算二者之间的距离
    # ---------------------------------------------------#
    l1 = np.sqrt(np.sum(np.square(output1 - output2), axis=-1))

    # plt.subplot(1, 2, 1)
    # plt.imshow(np.array(image_1))
    #
    # plt.subplot(1, 2, 2)
    # plt.imshow(np.array(image_2))
    # plt.text(-12, -12, 'Distance:%.3f' % l1, ha='center', va= 'bottom' ,fontsize=11)
    # plt.show()
    return l1