import cv2
import os
from sklearn.cluster import MeanShift, estimate_bandwidth
import numpy as np
from skimage.segmentation import mark_boundaries
from PIL import Image
from numpy import genfromtxt


def mean_shift(folder, index, alpha=1):
    im_list = ["BGRD.tif",
               "HSVD.tif",
               "LABD.tif",
               "YCrCbD.tif"]

    # without position information
    img = cv2.imread(os.path.join(folder, im_list[index]))
    h, w, _ = img.shape
    # normalization
    img_data = np.float64(img / 255).reshape(-1, 3)
    # mean shift
    # bandwidth = estimate_bandwidth(img_data, quantile=0.3, n_samples=500)
    # clustering = MeanShift(bandwidth=bandwidth, bin_seeding=True, max_iter=1).fit(img_data)
    # segments = clustering.cluster_centers_[clustering.labels_].reshape(h, w, -1)[:, :, :3]
    # label = clustering.labels_
    # print('cluster number =', len(np.unique(label)))
    # img_name = 'm_' + str(i) + '.png'
    # cv2.imwrite(os.path.join(folder, img_name), (segments*255).astype(np.uint8))
    # cv2.imshow("seg", segments)
    # cv2.waitKey()

    # with position information (contribution from my teammate)
    loc = np.float64(np.mgrid[0:h, 0:w])
    loc = loc.transpose(1, 2, 0)
    loc = np.reshape(loc, (-1, 2))
    # normalization
    loc[:, 0] /= h - 1
    loc[:, 1] /= w - 1
    loc *= alpha
    img_loc = np.concatenate((img_data, loc), axis=1)
    # mean shift
    bandwidth_loc = estimate_bandwidth(img_loc, quantile=0.3, n_samples=500)
    clustering_loc = MeanShift(bandwidth=bandwidth_loc, bin_seeding=True, max_iter=1).fit(img_loc)
    label_loc = clustering_loc.labels_
    label_loc = label_loc.reshape(h, w)
    mask = np.zeros((h, w))
    segments_loc = clustering_loc.cluster_centers_[clustering_loc.labels_].reshape(h, w, -1)[:, :, :3]
    print('cluster number =', len(np.unique(label_loc)))

    # show mask
    cluster_num = np.unique(label_loc)
    np.savetxt('label.csv', label_loc, fmt="%d", delimiter=',')
    for cluster in [0]:
        mask[np.where(label_loc == cluster)] = 1
    mask = np.uint8(mask)
    ori = cv2.imread('original_image/4.bmp')
    ori = np.uint8(ori)
    np.savetxt('mask.csv', mask, fmt="%d", delimiter=',')
    masked_img = mark_boundaries(ori, mask, mode='thick')
    cv2.imshow("mask", masked_img)
    cv2.waitKey()
    return clustering_loc, h, w


def finetune_weight(index):
    folder = 'tif/image' + str(index)
    img_path = 'tif/weight' + str(index)
    # Find the appropriate weight
    for alpha in [0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.3]:
        print(alpha)
        for i in range(4):
            clustering_loc, h, w = mean_shift(folder, i, alpha)
            segments_loc = clustering_loc.cluster_centers_[clustering_loc.labels_].reshape(h, w, -1)[:, :, :3]
            img_loc_name = str(alpha) + '_' + str(i) + '.png'
            cv2.imwrite(os.path.join(img_path, img_loc_name), (segments_loc*255).astype(np.uint8))


def show_mask():

    return


if __name__ == '__main__':
    # Find the appropriate weight
    # for folder_index in range(4):
    #     finetune_weight(folder_index)

    # segment image 1
    # folder = 'tif/image0'
    # segment rover and shadow
    # mean_shift(folder, 1, 0.6)
    # # segment rover
    # mean_shift(folder, 1, 1.25)
    # # segment shadow
    # mean_shift(folder, 2, 1.1)
    #
    # with open('./seg_res/mask_rs.csv') as f:
    #     mask_rs = np.loadtxt(f, dtype=int, delimiter=',')
    # with open('./seg_res/mask_r.csv') as f:
    #     mask_r = np.loadtxt(f, dtype=int, delimiter=',')
    # mask_s = mask_rs - mask_r
    # mask_s[0:920, :] = 0
    # ori = cv2.imread('BGR.jpg')
    # masked_img = mark_boundaries(ori, mask_s, mode='thick')
    # cv2.imshow("mask", masked_img)
    # cv2.waitKey()

    # segment image 2
    # folder = 'tif/image1'
    # mean_shift(folder, 0, 0.85)

    # segment image 3
    # folder = 'tif/image2'
    # mean_shift(folder, 1, 1.2)

    # segment image 4
    folder = 'tif/image3'
    mean_shift(folder, 0, 0.7)