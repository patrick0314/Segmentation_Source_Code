import os
import cv2
import numpy as np
from skimage import data, segmentation, color
from skimage.future import graph
from matplotlib import pyplot as plt

def NCut(in_path, out_path):
    img = cv2.imread(in_path)

    labels1 = segmentation.slic(img, compactness=30, n_segments=400, start_label=1)
    out1 = color.label2rgb(labels1, img, kind='avg', bg_label=0)

    g = graph.rag_mean_color(img, labels1, mode='similarity')
    labels2 = graph.cut_normalized(labels1, g)
    out2 = color.label2rgb(labels2, img, kind='avg', bg_label=0)

    fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True, figsize=(6, 8))

    ax[0].imshow(out1.astype(np.uint8))
    ax[1].imshow(out2.astype(np.uint8))

    for a in ax:
        a.axis('off')

    fig.tight_layout()
    fig.savefig(out_path[:-4]+'.jpg')

if __name__ == '__main__':
    input_folder = 'pics'
    output_folder = 'results'
    filenames = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]
    for filename in filenames:
        NCut(os.path.join(input_folder, filename), os.path.join(output_folder, filename))