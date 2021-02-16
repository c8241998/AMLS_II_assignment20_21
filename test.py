import os
from dataset import data
from utils import *
from model import MyModel
from tensorflow.keras.layers import Layer
os.environ["CUDA_VISIBLE_DEVICES"]="7,6,5,4,3,2,1"

class sum(Layer):
    def __init__(self):
        super(sum, self).__init__()
    def call(self,x):  # Defines the computation from inputs to outputs
        return (x[:,:,:,:3]+x[:,:,:,3:])/2.

class cat(Layer):
    def __init__(self):
        super(cat, self).__init__()
    def call(self,x):  # Defines the computation from inputs to outputs
        return tf.concat([x,x],axis=-1)

if __name__=='__main__':

    config = read_config()
    scale = 2
    checkpoint_path = 'checkpoint/bicubic_x2/model'

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(config['learning_rate'], decay_steps=config['decay_steps'],
                                                                 decay_rate=config['decay_rate'])

    # model1 = MyModel(blocks=config['blocks'], channel=config['channel'], scale=scale)
    # model1.compile(
    #     optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    #     loss=Loss(),
    #     metrics=[PSNR(), SSIM()],
    # )
    # model1.build((None,None,None,3))
    #
    # model1.load_weights(checkpoint_path)
    #
    # model2 = MyModel(blocks=config['blocks'], channel=config['channel'], scale=scale)
    # model2.compile(
    #     optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    #     loss=Loss(),
    #     metrics=[PSNR(), SSIM()],
    # )
    # model2.build((None, None, None, 3))
    #
    # model2.load_weights(checkpoint_path)
    #
    # model = tf.keras.models.Sequential([model1,sum(),model2])
    import tensorflow_hub as hub
    model = tf.keras.models.Sequential([
        hub.KerasLayer("https://tfhub.dev/captain-pool/esrgan-tf2/1", trainable=True),
        tf.keras.layers.Conv2D(filters=3, kernel_size=[1, 1], strides=[1, 1]),
        cat()
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss=Loss(),
        metrics=[PSNR(), SSIM()],
    )
    # model.build((None,None,None,3))
    # print(model.summary())



    trainset,valset,testset = data(name=config['dataset'])

    model.fit(
        trainset,
        epochs=config['epochs'],
        validation_data=valset,
        validation_freq=1,
    )

    eval = model.evaluate(valset)
    print(eval)