
import tensorflow as tf
from tensorflow.keras import layers,optimizers,datasets,Sequential,callbacks,losses
import os
import numpy as np


# os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class FCNloca(object):

    def __init__(self, img_rows=1000, img_cols=758):
    
        
        self.img_rows = img_rows
        self.img_cols = img_cols

    def datain(self):
        wave_train = np.load('./data.npy')
        loca_train = np.load('./label.npy')
        return wave_train, loca_train
    
    def get_network(self):
        model = Sequential([
            layers.Conv2D(64 , kernel_size=[3,3] , padding='same', activation=tf.nn.relu, kernel_initializer='he_uniform'),
            layers.MaxPool2D(pool_size=[2,2], padding='same'),
            layers.Conv2D(64 , kernel_size=[3,3] , padding='same', activation=tf.nn.relu, kernel_initializer='he_uniform'),
            layers.MaxPool2D(pool_size=[4,2], padding='same'),
            
            layers.Conv2D(128 , kernel_size=[3,3] , padding='same', activation=tf.nn.relu, kernel_initializer='he_uniform'),
            layers.MaxPool2D(pool_size=[2,3], padding='same'),
            layers.Conv2D(128 , kernel_size=[3,3] , padding='same', activation=tf.nn.relu, kernel_initializer='he_uniform'),
            layers.MaxPool2D(pool_size=[2,2], padding='same'),
            
            layers.Conv2D(256 , kernel_size=[3,3] , padding='same', activation=tf.nn.relu, kernel_initializer='he_uniform'),
            layers.MaxPool2D(pool_size=[2,2], padding='same'),
            layers.Conv2D(256 , kernel_size=[3,3] , padding='same', activation=tf.nn.relu, kernel_initializer='he_uniform'),
            layers.MaxPool2D(pool_size=[2,2], padding='same'),
            
            layers.Conv2D(512 , kernel_size=[3,3] , padding='same', activation=tf.nn.relu, kernel_initializer='he_uniform'),
            layers.MaxPool2D(pool_size=[2,2], padding='same'),
            layers.Conv2D(512 , kernel_size=[3,3] , padding='same', activation=tf.nn.relu, kernel_initializer='he_uniform'),
            layers.MaxPool2D(pool_size=[2,1], padding='same'),
            
            layers.Conv2D(1024 , kernel_size=[3,3] , padding='same', activation=tf.nn.relu, kernel_initializer='he_uniform'),
            layers.MaxPool2D(pool_size=[2,1], padding='same'),
            layers.Conv2D(1024 , kernel_size=[3,3] , padding='same', activation=tf.nn.relu, kernel_initializer='he_uniform'),
            layers.Dropout(0.3),
            layers.Conv2D(1024 , kernel_size=[3,3] , padding='same', activation=tf.nn.relu, kernel_initializer='he_uniform'),
            layers.Conv2D(512 , kernel_size=[3,3] , padding='same', activation=tf.nn.relu, kernel_initializer='he_uniform'),
            layers.UpSampling2D(size=(1, 2)),
            
            layers.Conv2D(256 , kernel_size=[3,3] , padding='same', activation=tf.nn.relu, kernel_initializer='he_uniform'),
            layers.UpSampling2D(size=(1, 2)),
            layers.Conv2D(128 , kernel_size=[3,3] , padding='same', activation=tf.nn.relu, kernel_initializer='he_uniform'),
            
            layers.UpSampling2D(size=(1, 2)),
            
            layers.Conv2D(32 , kernel_size=[3,3] , padding='same', activation=tf.nn.relu, kernel_initializer='he_uniform'),
            layers.UpSampling2D(size=(1, 2)),
            layers.Conv2D(32 , kernel_size=[3,3] , padding='same', activation=tf.nn.relu, kernel_initializer='he_uniform'),
            
            layers.UpSampling2D(size=(1, 2)),
            
            layers.Conv2D(8 , kernel_size=[3,3] , padding='same', activation=tf.nn.relu, kernel_initializer='he_uniform'),
            layers.Conv2D(8 , kernel_size=[3,3] , padding='same', activation=tf.nn.relu, kernel_initializer='he_uniform'),
            layers.UpSampling2D(size=(1, 2)),
            layers.Conv2D(3 , kernel_size=[3,3] , padding='same', activation=tf.nn.relu, kernel_initializer='he_uniform'),
            layers.Conv2D(3 , kernel_size=[3,3] , padding='same', activation=tf.nn.relu, kernel_initializer='he_uniform'),
            
            
            layers.Conv2D(3 , kernel_size=[3,3] , padding='same', activation='sigmoid')
            
            ])
        return model
            

    def train(self):
        print("loading data")
        wave_train, loca_train = self.datain()
        print("loading data done")
        model = self.get_network()
        print("got network")
        model.compile(optimizers.Adam(lr=0.0002), loss='binary_crossentropy', metrics=['accuracy'])
        model_checkpoint = callbacks.ModelCheckpoint('FCNloca_1D.hdf5', monitor='val_loss', verbose=1, save_best_only=True)

        print('Fitting model...')
        hist = model.fit(wave_train,loca_train,batch_size=8, epochs=100, verbose=1 , validation_split=0.15,
                          shuffle=True, callbacks=[model_checkpoint])
        
        with open('FCNloca.log', 'w') as f:
            f.write(str(hist.history))



if __name__ == '__main__':
    fcnloca = FCNloca()
    fcnloca.train()