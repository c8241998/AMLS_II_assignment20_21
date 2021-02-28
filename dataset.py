import tensorflow_datasets as tfds
import tensorflow as tf
from utils import read_config

def normalize(lr,hr):
    lr = tf.cast(lr, tf.float32)/255.
    hr = tf.cast(hr, tf.float32)/255.
    return  lr,hr

def cut0(lr,hr):
    config = read_config()
    scale = int(config['dataset'][-1:])
    crop = config['crop']
    lr,hr = tf.image.resize_with_crop_or_pad(lr,crop,crop),tf.image.resize_with_crop_or_pad(hr,crop*scale,crop*scale)
    return lr,hr

def cut1(lr,hr):
    lr,hr = cut0(lr,hr)
    lr,hr = tf.image.flip_up_down(lr),tf.image.flip_up_down(hr)
    return lr,hr

def cut2(lr,hr):
    lr,hr = cut0(lr,hr)
    lr,hr = tf.image.rot90(lr),tf.image.rot90(hr)
    return lr,hr

def data_augmentation(ds,cut_index=0):
    cuts = [cut0, cut1, cut2]
    ds = ds.map(cuts[cut_index], num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return ds

def preprocess_train(ds):

    ds = ds.map(normalize, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.cache()
    ds = ds.shuffle(ds.__len__())
    ds = ds.batch(1)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

    return ds

def preprocess_val(ds):
    ds = ds.map(normalize, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.batch(1)
    ds = ds.cache()
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    return ds

def data(name="div2k/bicubic_x2"):

    (trainset, valset, testset), info = tfds.load(name=name,
                                                  split=["train", "validation[:50%]", "validation[50%:]"],
                                                  with_info=True,
                                                  as_supervised=True, shuffle_files=True,
                                                  data_dir='/scratch/uceezc4/tensorflow_datasets')
    print(info)

    for i in range(3):
        temp = data_augmentation(trainset,cut_index=i)
        train = train.concatenate(temp) if i > 0 else temp

    train = preprocess_train(train)
    val = preprocess_val(valset)
    test = preprocess_val(testset)

    return train,val,test

if __name__=='__main__':
    # train,val = data()
    # import matplotlib.pyplot as plt
    # for i,sample in enumerate(train.take(1)):
    #     _ = plt.imshow(sample[0][0,:,:,:])
    #     plt.savefig('0.png')
    #     _ = plt.imshow(sample[1][0, :, :, :])
    #     plt.savefig('1.png')

    (trainset, valset, testset), info = tfds.load(name="div2k/bicubic_x2", split=["train", "validation[:50%]","validation[50%:]"], with_info=True,
                                         as_supervised=True, shuffle_files=True,
                                         data_dir='/scratch/uceezc4/tensorflow_datasets')

    print(trainset.__len__(),valset.__len__(),testset.__len__())