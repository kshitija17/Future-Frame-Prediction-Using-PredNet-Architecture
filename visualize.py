
# import numpy as np

import matplotlib.pyplot as plt




class VisualizeData(object):
    def __init__(self):
        pass

    def __call__(self, sample, num_frames, groundtruth=False):
        # Construct a figure on which we will visualize the images.

        assert num_frames % 5 == 0, "The number of frames must be a multiple of 5 but{} is given.".format(num_frames)
        rows = num_frames / 5
        fig, axes = plt.subplots(int(rows), 5, figsize=(8, 6))

        # set the spacing between subplots
        plt.subplots_adjust(left=0.1,
                            bottom=0.1,
                            right=0.9,
                            top=0.9,
                            wspace=0.4,
                            hspace=0.4)

        for idx, ax in enumerate(axes.flat):
            ax.imshow(np.squeeze(sample[idx]))
            ax.set_title(f"Frame {idx + 1}")
            ax.axis("off")

        plt.show()