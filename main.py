import tensorflow as tf
from dataset import data
import tensorflow_hub as hub
import os
from utils import *
from model import MyModel
import datetime
os.environ["CUDA_VISIBLE_DEVICES"]="7,6,5,4,3,2,1"

if __name__=='__main__':

    # parameters
    config = read_config()
    scale = int(config['dataset'][-1:])
    checkpoint_path = 'checkpoint/' + config['dataset'][-10:] + '/model'
    log_dir = "logs/fit/" + config['dataset'] + '-' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # dataset
    trainset,valset,testset = data(name=config['dataset'])

    # model
    model = MyModel(blocks=config['blocks'],channel=config['channel'],scale=scale)

    # lr ExponentialDecay
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(config['learning_rate'],
                                                                 decay_steps=config['decay_steps'],
                                                                 decay_rate=config['decay_rate'])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss= Loss(),
        # loss = tf.keras.losses.mean_absolute_error,
        metrics=[PSNR(),SSIM()],
    )

    # resume checkpoint
    if config['resume']:
        model.build((None,None,None,3))
        model.load_weights(checkpoint_path)

    # save the best model checkpoint
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True)

    # tensorboard visulization
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # model fit
    model.fit(
        trainset,
        epochs=config['epochs'],
        validation_data=valset,
        validation_freq=1,
        callbacks = [model_checkpoint_callback,tensorboard_callback]
    )

    # model summary
    print(model.summary())



    # model = tf.keras.models.Sequential([
    #     tf.keras.layers.Conv2DTranspose(filters=128,kernel_size=[scale,scale],strides=[scale,scale],padding="valid",activation="relu"),
    #     tf.keras.layers.Conv2D(filters=128, kernel_size=[3, 3], strides=[1, 1],padding="same",activation="relu"),
    #     tf.keras.layers.Conv2D(filters=128, kernel_size=[3, 3], strides=[1, 1],padding="same",activation="relu"),
    #     tf.keras.layers.Conv2D(filters=3, kernel_size=[3, 3], strides=[1, 1],padding="same",activation="relu"),
    # ])