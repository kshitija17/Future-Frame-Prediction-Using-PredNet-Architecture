import tensorflow as tf
import subprocess, os
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation

# import from data

from data import DataLoad
from model import prednet


# import from var
from var import  img_shape, num_pred_frames, time_steps, batch_size, lr
from var import train_dataset_url, test_dataset_url, train_source_url, test_source_url
from visualize import VisualizeData


# check for gpu
is_cuda_gpu_available = tf.config.list_physical_devices('GPU')


# load data
dl_obj = DataLoad()


train_data = dl_obj(train_dataset_url,train_source_url,time_steps)
test_data = dl_obj(test_dataset_url,test_source_url,time_steps)


train_data = np.array(train_data)
test_data = np.array(test_data)

# print("train data",np.array(train_data).shape)
# print("test data",np.array(test_data).shape)


# visualize video snippet sample data
vi_obj = VisualizeData()
sample_id = np.random.choice(range(len(train_data)),size=1)[0]
vi_obj(train_data[sample_id],10)


train_batch = dl_obj._create_batch(train_data,batch_size)
test_batch = dl_obj._create_batch(test_data,batch_size)



# prednet model

model = prednet()
optimizer = tf.keras.optimizers.Adam(learning_rate = lr)


###### Training ######

epochs = 25

train_loss = np.zeros(shape=(epochs))

for e in range(epochs):
    for batch in train_batch:
        with tf.GradientTape() as tape:
          loss, pred_frame_list = model(batch,num_pred_frames)

        gradients = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))
        train_loss[e] = loss

    print('Epoch:{}/{} ----loss : {}'.format(e,epochs,loss))



plt.plot(np.arange(0,epochs), train_loss, 'g', label='Training loss')
plt.title('Training loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()