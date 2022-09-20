# import model
from train import model


# import from data
from data import DataLoad
from model import prednet


# import from var
from var import  img_shape, num_pred_frames, time_steps, batch_size, lr
from var import  test_dataset_url,  test_source_url
from visualize import VisualizeData


import tensorflow as tf
import numpy as np

# from data import DataLoad

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation



# check for gpu
is_cuda_gpu_available = tf.config.list_physical_devices('GPU')
# print(is_cuda_gpu_available)

# load test data
dl_obj = DataLoad()
test_data = dl_obj(test_dataset_url, test_source_url, time_steps)
test_data = np.array(test_data)
test_batch = dl_obj._create_batch(test_data, batch_size)


# visualize video snippet
vi_obj = VisualizeData()


test_batch_sample_id = np.random.choice(range(np.array(test_batch).shape[0]), size=1)[0]
loss, pred_frame_list = model(test_batch[test_batch_sample_id], 10)

pred_frame_list = np.array(np.swapaxes(pred_frame_list, 0, 1))

display_sample_id = np.random.choice(range(len(pred_frame_list)), size=1)[0]

vi_obj(pred_frame_list[display_sample_id], 10 + num_pred_frames)

vi_obj(test_batch[test_batch_sample_id][display_sample_id], 10)

# generate video

frames = []  # for storing the generated images
sample_test = pred_frame_list[display_sample_id]

fig = plt.figure()
for i in range(len(sample_test)):
    frames.append([plt.imshow(sample_test[i], animated=True)])

ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True,
                                repeat_delay=1000)
ani.save('video.mp4')
plt.show()

# Calculate ssim for first 10 frames

ssim_value = tf.image.ssim(test_batch[test_batch_sample_id][display_sample_id], sample_test[:10], max_val=1,
                           filter_size=11,
                           filter_sigma=1.5, k1=0.01, k2=0.03)
print("SSIM values for first 10 frames")
tf.print(ssim_value, summarize=10)


plt.plot(np.arange(0,10), ssim_value, label='SSIM values')
plt.title(' structural similarity index measure (SSIM) for L0 loss')
plt.xlabel('Predicted Frames')
plt.ylabel('SSIM values')
plt.legend()
plt.show()
