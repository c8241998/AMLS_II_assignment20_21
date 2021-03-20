# cbam model

import keras.backend as K
import keras.layers as KL
from tensorflow.keras.layers import Layer
from tensorflow.keras import Model

# channel attention block in CBAM
class Channel_attention(Layer):
    def __init__(self):
        super(Channel_attention, self).__init__()
        self.reduction_ratio=0.125
        self.channel_axis = 1 if K.image_data_format() == "channels_first" else 3
        self.maxpool = KL.GlobalMaxPooling2D()
        self.avgpool = KL.GlobalAvgPool2D()
        self.sigmoid = KL.Activation('sigmoid')
        self.channel = 128
        self.Dense_One = KL.Dense(units=int(self.channel * self.reduction_ratio), activation='relu',
                                  kernel_initializer='he_normal',
                                  use_bias=True, bias_initializer='zeros')
        self.Dense_Two = KL.Dense(units=int(self.channel), activation='relu', kernel_initializer='he_normal',
                                  use_bias=True,
                                  bias_initializer='zeros')

    def call(self,input_xs):  # Defines the computation from inputs to outputs


        maxpool_channel = self.maxpool(input_xs)
        maxpool_channel = KL.Reshape((1, 1, self.channel))(maxpool_channel)
        avgpool_channel = self.avgpool(input_xs)
        avgpool_channel = KL.Reshape((1, 1, self.channel))(avgpool_channel)

        # max path
        mlp_1_max = self.Dense_One(maxpool_channel)
        mlp_2_max = self.Dense_Two(mlp_1_max)
        mlp_2_max = KL.Reshape(target_shape=(1, 1, int(self.channel)))(mlp_2_max)
        # avg path
        mlp_1_avg = self.Dense_One(avgpool_channel)
        mlp_2_avg = self.Dense_Two(mlp_1_avg)
        mlp_2_avg = KL.Reshape(target_shape=(1, 1, int(self.channel)))(mlp_2_avg)

        channel_attention_feature = KL.Add()([mlp_2_max, mlp_2_avg])
        channel_attention_feature = self.sigmoid(channel_attention_feature)
        return KL.Multiply()([channel_attention_feature, input_xs])

# spatial attention block in CBAM
class Spatial_attention(Layer):
    def __init__(self):
        super(Spatial_attention, self).__init__()
        self.max = KL.Lambda(lambda x: K.max(x, axis=3, keepdims=True))
        self.avg = KL.Lambda(lambda x: K.mean(x, axis=3, keepdims=True))
        self.cat = KL.Concatenate(axis=3)
        self.conv = KL.Conv2D(filters=1, kernel_size=(3, 3), padding="same", activation='sigmoid', kernel_initializer='he_normal', use_bias=False)

    def call(self, x):  # Defines the computation from inputs to outputs
        maxpool_spatial = self.max(x)
        avgpool_spatial = self.avg(x)
        max_avg_pool_spatial = self.cat([maxpool_spatial, avgpool_spatial])
        return self.conv(max_avg_pool_spatial)

# CBAM model definition
class CBAM(Model):
    def __init__(self):
        super(CBAM, self).__init__()
        self.channel_attention = Channel_attention()
        self.spatial_attention = Spatial_attention()

    def call(self, input_xs):  # Defines the computation from inputs to outputs
        channel_refined_feature = self.channel_attention(input_xs)
        spatial_attention_feature = self.spatial_attention(channel_refined_feature)
        refined_feature = KL.Multiply()([channel_refined_feature, spatial_attention_feature])
        return KL.Add()([refined_feature, input_xs])
