from dataset import data
import os
from utils import *
from model import MyModel
os.environ["CUDA_VISIBLE_DEVICES"]="0"

if __name__=='__main__':

    config = read_config() 
    scale = int(config['dataset'][-1:])
    checkpoint_path = 'checkpoint/' + config['dataset'][-10:] + '/model'

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(config['learning_rate'], decay_steps=config['decay_steps'],
                                                                 decay_rate=config['decay_rate'])

    model = MyModel(blocks=config['blocks'], channel=config['channel'], scale=scale)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss=Loss(),
        metrics=[PSNR(), SSIM()],
    )
    model.build((None,None,None,3))

    model.load_weights(checkpoint_path)

    print(model.summary())

    trainset,valset,testset = data(name=config['dataset'])
    eval = model.evaluate(testset)
    print(eval)