#%%
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow.python.keras.engine import input_layer
from tensorflow.python.ops.gen_array_ops import one_hot, size
import tfutils
from tensorflow.keras.layers import Conv2D,Dense,Flatten,BatchNormalization
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.layers import Conv2DTranspose,Reshape,LeakyReLU
print(tf.__version__)

(x_train,y_train),(x_test,y_test) = tfutils.datasets.mnist.load_data(one_hot=False)
x_train = tfutils.datasets.mnist.load_subset([0],x_train,y_train)
x_test = tfutils.datasets.mnist.load_subset([0],x_test,y_test)
x = np.concatenate([x_train,x_test],axis=0)
# %%
tfutils.datasets.mnist.plot_ten_random_examples(plt,x,np.zeros((x.shape[0],1))).show()
# %%
size=28
noise_dim = 1
discriminator = Sequential([
    Conv2D(64,3,strides=2,input_shape=(28,28,1)),
    LeakyReLU(),BatchNormalization(),

    Conv2D(128,5,strides=2),LeakyReLU(),BatchNormalization(),

    Conv2D(256,5,strides=2),LeakyReLU(),BatchNormalization(),

    Flatten(),
    Dense(1,activation='sigmoid')
])
opt = tf.keras.optimizers.Adam(lr=2e-4,beta_1=0.5)
discriminator.compile(loss='binary_crossentropy',optimizer=opt,metrics=['accuracy'])
discriminator.summary()
# %%
generator = Sequential([
    Dense(256,activation='relu',input_shape=(noise_dim,)),
    Reshape((1,1,256)),
    Conv2DTranspose(256,5,activation='relu'),
    BatchNormalization(),
    Conv2DTranspose(128,5,activation='relu'),
    BatchNormalization(),

    Conv2DTranspose(64,5,activation='relu',strides=2),
    BatchNormalization(),
    Conv2DTranspose(32,5,activation='relu'),
    BatchNormalization(),
    Conv2DTranspose(1,4,activation='sigmoid')
])
generator.summary()
# %%
noise = np.random.randn(1,noise_dim)
gen_image = generator.predict(noise)[0]
plt.figure()
plt.imshow(np.reshape(gen_image,(28,28)),cmap='binary')
# %%
input_layer = tf.keras.layers.Input(shape=(noise_dim,))
gen_out = generator(input_layer)
disc_out = discriminator(gen_out)

gan = Model(input_layer,disc_out)
discriminator.trainable = False
gan.compile(loss='binary_crossentropy',optimizer=opt,metrics=['accuracy'])
gan.summary()
# %%
epochs = 25
batch_size = 128
steps_per_epoch = int(2*x.shape[0]/batch_size)
print('Steps per Epoch= ',steps_per_epoch)
# %%
dp = tfutils.plotting.DynamicPlot(plt,5,5,(8,8))
for e in range(0,epochs):
    dp.start_of_epoch(e)
    for step in range(0,steps_per_epoch):
        true_examples = x[int(batch_size/2)*step:int(batch_size/2)*(step+1)]
        true_examples = np.reshape(true_examples,(true_examples.shape[0],28,28,1))

        noise = np.random.randn(int(batch_size/2),noise_dim)
        generated_examples = generator.predict(noise)

        x_batch = np.concatenate([generated_examples,true_examples],axis=0)
        y_batch = np.array([0]*int(batch_size/2)+[1]*int(batch_size/2))

        indices = np.random.choice(range(batch_size),batch_size,replace=False)
        x_batch = x_batch[indices]
        y_batch = y_batch[indices]

        discriminator.trainable = True
        discriminator.train_on_batch(x_batch,y_batch)
        discriminator.trainable = False

        loss = gan.train_on_batch(noise,np.ones((int(batch_size/2),1)))
        _,acc = discriminator.evaluate(x_batch,y_batch,verbose=False)
        
    noise = np.random.randn(1,noise_dim)
    generated_example = generator.predict(noise)[0]
    
    dp.end_of_epoch(np.reshape(generated_example, (28, 28)), 'binary',
                   'DiscAcc:{:.2f}'.format(acc), 'GANLoss:{:.2f}'.format(loss))