import tensorflow as tf

# PSNR Metric
class PSNR(tf.keras.metrics.Metric):
    def __init__(self, name='psnr', **kwargs):
        super(PSNR, self).__init__(name=name, **kwargs)
        self.psnr = self.add_weight(name='psnr', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred,sample_weight=None):
        psnr_ = tf.reduce_mean(tf.image.psnr(y_true, (y_pred[:,:,:,:3]+y_pred[:,:,:,3:])/2., max_val=1.0))
        self.psnr.assign_add(psnr_)
        self.count.assign_add(1)

    def result(self):
        return self.psnr / self.count


# SSIMMetric
class SSIM(tf.keras.metrics.Metric):
    def __init__(self, name='ssim', **kwargs):
        super(SSIM, self).__init__(name=name, **kwargs)
        self.ssim = self.add_weight(name='ssim', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred,sample_weight=None):
        ssim_ = tf.reduce_mean(tf.image.ssim((y_pred[:,:,:,:3]+y_pred[:,:,:,3:])/2., y_true, 1.0))
        self.ssim.assign_add(ssim_)  # output is 4x1 array
        self.count.assign_add(1)

    def result(self):
        return self.ssim / self.count

class Loss(tf.keras.losses.Loss):

  def call(self, y_true, y_pred):
      ssim = tf.reduce_mean(tf.image.ssim(y_true, y_pred[:,:,:,:3], 1.0))
      psnr = tf.reduce_mean(tf.image.psnr(y_true, y_pred[:,:,:,3:6], max_val=1.0))
      return - ( ssim + psnr/30 )

def read_config():
    import json
    with open("config.json", 'r') as load_f:
        load_dict = json.load(load_f)
        return load_dict