# this code is just for testing
# ignore me

import tensorflow_datasets as tfds
import os
from utils import *
import matplotlib.pyplot as plt
import numpy as np
from model import MyModel
from dataset import data
os.environ["CUDA_VISIBLE_DEVICES"]="3"

def write_img(tensor,dir):
    tensor = tf.squeeze(tensor)
    img_data = tf.image.convert_image_dtype(tensor, dtype=tf.uint8)
    encode_image = tf.image.encode_jpeg(img_data)
    with tf.io.gfile.GFile(dir, 'wb') as f:
        f.write(encode_image.numpy())

if __name__=='__main__':

    config = read_config()
    scale = int(config['dataset'][-1:])
    checkpoint_path = 'checkpoint/' + config['dataset'][-10:] + '/model'

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(config['learning_rate'],
                                                                 decay_steps=config['decay_steps'],
                                                                 decay_rate=config['decay_rate'])

    model = MyModel(blocks=config['blocks'], channel=config['channel'], scale=scale)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss=Loss(),
        metrics=[PSNR(), SSIM()],
    )
    model.build((None, None, None, 3))

    model.load_weights(checkpoint_path)

    print(model.summary())
    trainset, valset, testset = data(name=config['dataset'])
    from PIL import Image
    import numpy as np

    def read_img(dir):
        x = np.array(Image.open(dir))
        x = tf.convert_to_tensor(x)
        x = tf.expand_dims(x,0)
        x = tf.cast(x, tf.float32) / 255.
        return x

    x = read_img('img/x.jpeg')
    z = read_img('img/z.jpeg')

    y = model.predict(x)
    y = (y[:,:,:,:3] + y[:,:,:,3:])/2.
    # y = tf.compat.v1.image.resize_bicubic(x,[1356,2040])

    write_img(y,'img/y.jpeg')

    eye = tf.image.crop_to_bounding_box(z,1356//2+55,1110,70,90)
    write_img(eye,'img/eye.jpeg')

    psnr = tf.image.psnr(y,z,1.0)
    ssim = tf.image.ssim(y,z,1.0)

    print(psnr,ssim)