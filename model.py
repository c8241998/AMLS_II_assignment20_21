from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Layer, BatchNormalization, ReLU
from tensorflow.keras import Model
from cbam import CBAM
import tensorflow as tf

class ResBlock(Layer):
    def __init__(self,channel):
        super(ResBlock, self).__init__()
        self.res = tf.keras.models.Sequential([
            Conv2D(filters=channel, kernel_size=[3, 3], strides=[1, 1], padding="same",activation="relu"),
            Conv2D(filters=channel, kernel_size=[3, 3], strides=[1, 1], padding="same",activation="relu"),
        ])

    def call(self, x):  # Defines the computation from inputs to outputs
        shorcut = x
        x = self.res(x)
        return x+shorcut

class MyModel(Model):
  def __init__(self,blocks=4,channel=128,scale=2):
    super(MyModel, self).__init__()
    self.deconv = Conv2DTranspose(filters=channel, kernel_size=[scale, scale], strides=[scale, scale], padding="valid", activation="relu")
    self.conv1 = Conv2D(filters=channel, kernel_size=[3, 3], strides=[1, 1],padding="same",activation="relu")
    self.conv2 = Conv2D(filters=channel, kernel_size=[3, 3], strides=[1, 1], padding="same", activation="relu")
    self.conv3 = Conv2D(filters=channel, kernel_size=[3, 3], strides=[1, 1], padding="same", activation="relu")
    self.conv4 = Conv2D(filters=channel, kernel_size=[3, 3], strides=[1, 1], padding="same", activation="relu")
    self.conv5 = Conv2D(filters=3, kernel_size=[3, 3], strides=[1, 1], padding="same", activation="relu")
    self.conv6 = Conv2D(filters=3, kernel_size=[3, 3], strides=[1, 1], padding="same", activation="relu")
    self.resblocks1 = [ResBlock(channel) for i in range(blocks)]
    self.resblocks2 = [ResBlock(channel) for i in range(blocks)]
    self.cbam = CBAM()

  def call(self, x):

      x = self.conv1(x)
      shortcut = x
      for resblock in self.resblocks1:
          x = resblock(x)
      x = self.conv2(x)
      x = shortcut + x

      x = self.deconv(x)
      x = self.cbam(x)

      x=self.conv3(x)
      shortcut = x
      for resblock in self.resblocks2:
          x = resblock(x)
      x=self.conv4(x)
      x = shortcut + x


      x1 = self.conv5(x)
      x2 = self.conv6(x)
      x= tf.concat([x1,x2],axis=-1)

      return x