import os




#dataset paths

dir = "kitti_hkl"
# dir = '/content/drive/MyDrive/kitti_hkl'

train_dataset_url = os.path.join(dir, "X_train.hkl")
train_source_url = os.path.join(dir, "sources_train.hkl")
test_dataset_url = os.path.join(dir,"X_test.hkl")
test_source_url = os.path.join(dir, "sources_test.hkl")


# image shape
img_shape = (128,160,3)


# hyperparameters

time_steps = 10
num_sequences = None
batch_size = 4

# training settings


#learning rate

lr = 0.001

# test setting

num_pred_frames = 10