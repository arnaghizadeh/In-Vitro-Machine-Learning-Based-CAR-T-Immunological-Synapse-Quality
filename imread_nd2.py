from nd2reader import ND2Reader
import cv2
import numpy as np
import os
import glob
import P


# Blue is F-actin fluorescence image (Channel 0)
# Green is Perforin fluorescence image (Channel 1)
# Red is tumor antigen fluorescence image (Channel 2)
# Purple is pZeta fluorescence image (Channel 3)
# DIC is the shape of cells (Channel 4)


def read_nd2_imgs(file):
    with ND2Reader(file) as images:
        imgs = [[] for c in range(images.sizes['c'])]
        for c in range(images.sizes['c']):
            for z in range(images.sizes['z']):
                image = images.get_frame_2D(c=c, t=0, z=z)
                imgs[c].append(image)
    imgs = np.asarray(imgs)
    return imgs


def map_mask_to_image(mask, img, color):
    mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
    mskd = img * mask
    clmsk = np.ones(mask.shape) * mask
    clmsk[:, :, 0] = clmsk[:, :, 0] * color[0] * 256
    clmsk[:, :, 1] = clmsk[:, :, 1] * color[1] * 256
    clmsk[:, :, 2] = clmsk[:, :, 2] * color[2] * 256
    img = img + 0.8 * clmsk - 0.8 * mskd
    return np.uint8(img)


def find_max_index(imgs, channel=2):
    max_value = -1000
    max_index = -1
    for i in range(imgs.shape[1]):
        if imgs[channel, i, :, :].max() > max_value:
            max_value = imgs[channel, i, :, :].max()
            max_index = i
    return max_index


def find_mask_overall(imgs, channels, index):
    c, z, h, w = imgs.shape
    mask_overall = np.zeros(shape=(h, w), dtype=np.float32)

    for i in channels:
        mask = imgs[i, index, :, :]
        mask = np.float32(mask)
        mask = mask - mask.min()
        mask = mask / mask.max()
        mask = cv2.normalize(np.uint8(mask * 255), None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        # mask = np.float32(mask)
        mask = mask - mask.min()
        mask = mask / mask.max()
        mask_overall += mask

    mask_overall = mask_overall - mask_overall.min()
    mask_overall = mask_overall / mask_overall.max()
    return mask_overall


def imshow_nd2(filename):
    if not os.path.exists('nd2_show'):
        os.mkdir('nd2_show')
    path = "../Datasets/protein/nd2"
    dataname = os.path.join(path, filename + '.nd2')
    imgs = read_nd2_imgs(dataname)
    index = find_max_index(imgs, channel=2)  # find the most antigen
    for i in range(4):
        mask = find_mask_overall(imgs, channels=[i], index=index)
        mask = np.uint8(mask * 255)
        mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
        cv2.imwrite(os.path.join('nd2_show', 'channel-{}.png'.format(i)), mask)
    cv2.imwrite(os.path.join('nd2_show', 'channel-4.png'),
                np.uint8(find_mask_overall(imgs, channels=[4], index=index) * 255))


def imshow_nd2_old(path, name):
    if not os.path.exists('nd2_show'):
        os.mkdir('nd2_show')
    # dataname = filenameos.path.join(path, filename+'.nd2')
    imgs = read_nd2_imgs(path)

    index = find_max_index(imgs, channel=2)  # find the most antigen
    """for i in range(4):
        mask = find_mask_overall(imgs, channels=[i], index=index)
        mask = np.uint8(mask*255)
        mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
        cv2.imwrite(os.path.join(P.nd2_png, name+'channel-{}.png'.format(i)), mask)"""

    for i in range(4):
        cv2.imwrite(os.path.join(P.nd2_png, name + 'channel-4.png'),
                    np.uint8(find_mask_overall(imgs, channels=[i], index=index) * 255))


def main():
    if not os.path.exists("images"):
        os.mkdir("images")
    path = P.source_dir

    names = os.listdir(path)
    print(names)

    print(path)
    for n in names:
        dataname = os.path.join(path, n)
        # print(p, "--", n)
        # imshow_nd2(p,n)
        imgs = read_nd2_imgs(dataname)
        index = find_max_index(imgs, channel=2)  # find the most antigen
        mask_overall = find_mask_overall(imgs, channels=[0], index=index)
        mask_overall = np.repeat(mask_overall[:, :, np.newaxis], repeats=3, axis=2)
        cv2.imwrite(os.path.join(P.nd2_png, '{}.png'.format(n)), np.uint8(mask_overall * 255))

    """
    for dataname in sorted(glob.glob(os.path.join(path,"*.nd2"))):
        print(dataname)

        imgs = read_nd2_imgs(dataname)
        index = find_max_index(imgs, channel=2) # find the most antigen
        # print("index:",index)
        mask_overall = find_mask_overall(imgs, channels=[0], index=index)
        mask_overall = np.repeat(mask_overall[:, :, np.newaxis], repeats=3, axis=2)
        # masked_img =  map_mask_to_image(mask_overall,img,(0.,0.,1.))
        # cv2.imshow('mask', np.uint8(mask_overall*255))
        # # cv2.imshow('mask', masked_img)
        # k = cv2.waitKey(0) & 0xFF
        # if k == ord('q'):
        #     cv2.destroyAllWindows()
        #     exit()
        savename = dataname.split('/')[-1].split('.')[0]
        cv2.imwrite(os.path.join(P.nd2_png,'{}.png'.format(savename)), np.uint8(mask_overall*255))"""


if __name__ == '__main__':
    main()
    # imshow_nd2('AF647-Streptavidin-Protein A AF488-Perforin AF568-P-Zeta AF405-F-actin-3')