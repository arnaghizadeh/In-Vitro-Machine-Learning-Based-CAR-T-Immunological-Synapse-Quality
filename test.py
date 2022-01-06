import torch
import os,sys
import numpy as np
import argparse
import cv2
import time
import colorsys
import random
from PIL import Image
import copy

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import P
import KGnet
from dataset_antigen import Protein as Antigen
import postprocessing
import nms


def parse_args():
    parser = argparse.ArgumentParser(description="InstanceHeat")
    parser.add_argument("--data_dir", help="data directory", default=P.data_dir, type=str)
    parser.add_argument("--png_path", help="data directory", default=P.save_path, type=str)
    parser.add_argument("--pretrained_weights", help="data directory", default=P.pretrained_weights, type=str)
    parser.add_argument("--postrained_weights", help="data directory", default=P.postrained_weights, type=str)
    parser.add_argument("--resume", help="resume file", default="end_model.pth", type=str)
    parser.add_argument('--input_h', type=int, default=512, help='input height')
    parser.add_argument('--input_w', type=int, default=512, help='input width')
    parser.add_argument('--save_img', type=bool, default=True, help='save img or not')
    parser.add_argument('--nms_thresh', type=float, default=0.3, help='nms_thresh')
    parser.add_argument('--seg_thresh', type=float, default=0.5, help='seg_thresh')
    parser.add_argument("--dataset", help="training dataset", default='antigen', type=str)
    args = parser.parse_args()
    return args

def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image
"""
def apply_rectangle(image):
    color = np.random.randint(0,255,3)
    cv2.rectangle(out_img, (int(x1), int(y1)), (int(x2), int(y2)), [color[0],color[1],color[2]], 2, 1)

    return image"""

def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors

class InstanceHeat(object):
    def __init__(self):
        self.model = KGnet.resnet50(pretrained=True)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def data_parallel(self):
        self.model = torch.nn.DataParallel(self.model)
    def load_weights(self, resume, args):
        self.model.load_state_dict(torch.load(os.path.join(args.postrained_weights, resume)))

    def map_mask_to_image(self, mask, img, color):
        # color = np.random.rand(3)
        mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        mskd = img * mask
        clmsk = np.ones(mask.shape) * mask
        clmsk[:, :, 0] = clmsk[:, :, 0] * color[0] * 256
        clmsk[:, :, 1] = clmsk[:, :, 1] * color[1] * 256
        clmsk[:, :, 2] = clmsk[:, :, 2] * color[2] * 256
        img = img + 1. * clmsk - 1. * mskd
        return np.uint8(img)

    def show_heat_mask(self, mask):
        mask = mask - np.min(mask)
        mask = mask / np.max(mask)
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        return heatmap

    def imshow_kp(self, kp, img_in):
        h,w = kp.shape[2:]
        img = cv2.resize(img_in, (w, h))
        colors = [(0,0,0.9),(0.9,0,0),(0.9,0,0.9),(0.9,0.9,0), (0.2,0.9,0.9)]
        for i in range(kp.shape[1]):
            img = self.map_mask_to_image(kp[0,i,:,:], img, color=colors[i])
        return img

    def imshow_kp_save(self, kp, img_in):
        h,w = kp.shape[2:]
        img = cv2.resize(img_in, (w, h))
        colors = [(0,0,0.9),(0.9,0,0),(0.9,0,0.9),(0.9,0.9,0), (0.2,0.9,0.9)]
        for i in range(kp.shape[1]):
            img = self.map_mask_to_image(kp[0,i,:,:], img, color=colors[i])
        return img

    def test_inference(self, args, image, bbox_flag=False):
        height, width, c = image.shape

        img_input = cv2.resize(image, (args.input_w, args.input_h))
        img_input = torch.FloatTensor(np.transpose(img_input.copy(), (2, 0, 1))).unsqueeze(0) / 255 - 0.5
        img_input = img_input.to(self.device)

        with torch.no_grad():
            begin = time.time()
            pr_c0, pr_c1, pr_c2, pr_c3, feat_seg = self.model.forward_dec(img_input)
            print("forward time is {:.4f}".format(time.time() - begin))
            pr_kp0, pr_short0, pr_mid0 = pr_c0
            pr_kp1, pr_short1, pr_mid1 = pr_c1
            pr_kp2, pr_short2, pr_mid2 = pr_c2
            pr_kp3, pr_short3, pr_mid3 = pr_c3

        torch.cuda.synchronize()
        skeletons0 = postprocessing.get_skeletons_and_masks(pr_kp0, pr_short0, pr_mid0)
        skeletons1 = postprocessing.get_skeletons_and_masks(pr_kp1, pr_short1, pr_mid1)
        skeletons2 = postprocessing.get_skeletons_and_masks(pr_kp2, pr_short2, pr_mid2)
        skeletons3 = postprocessing.get_skeletons_and_masks(pr_kp3, pr_short3, pr_mid3)

        skeletons0 = postprocessing.refine_skeleton(skeletons0)
        skeletons1 = postprocessing.refine_skeleton(skeletons1)
        skeletons2 = postprocessing.refine_skeleton(skeletons2)
        skeletons3 = postprocessing.refine_skeleton(skeletons3)


        kps = [pr_kp0,pr_kp1,pr_kp2,pr_kp3]


        bboxes = postprocessing.gather_skeleton(skeletons0, skeletons1, skeletons2, skeletons3)
        bboxes = nms.non_maximum_suppression_numpy(bboxes, nms_thresh=args.nms_thresh)
        if bbox_flag:
            return bboxes
        if bboxes is None:
            return None

        with torch.no_grad():
            predictions = self.model.forward_seg(feat_seg, [bboxes])
        predictions = self.post_processing(args, predictions, width, height)
        return [predictions,kps]

    def post_processing(self, args, predictions, image_w, image_h):
        if predictions is None:
            return predictions
        out_masks = []
        out_dets = []
        mask_patches, mask_dets = predictions
        for mask_b_patches, mask_b_dets in zip(mask_patches, mask_dets):
            for mask_n_patch, mask_n_det in zip(mask_b_patches, mask_b_dets):
                mask_patch = mask_n_patch.data.cpu().numpy()
                mask_det = mask_n_det.data.cpu().numpy()
                y1, x1, y2, x2, conf = mask_det
                y1 = np.maximum(0, np.int32(np.round(y1)))
                x1 = np.maximum(0, np.int32(np.round(x1)))
                y2 = np.minimum(np.int32(np.round(y2)), args.input_h - 1)
                x2 = np.minimum(np.int32(np.round(x2)), args.input_w - 1)

                mask = np.zeros((args.input_h, args.input_w), dtype=np.float32)
                mask_patch = cv2.resize(mask_patch, (x2 - x1, y2 - y1))

                mask[y1:y2, x1:x2] = mask_patch
                mask = cv2.resize(mask, (image_w, image_h))
                mask = np.where(mask >= args.seg_thresh, 1, 0)

                y1 = float(y1) / args.input_h * image_h
                x1 = float(x1) / args.input_w * image_w
                y2 = float(y2) / args.input_h * image_h
                x2 = float(x2) / args.input_w * image_w

                out_masks.append(mask)
                out_dets.append([y1,x1,y2,x2, conf])
        return [np.asarray(out_masks, np.float32), np.asarray(out_dets, np.float32)]



    def imshow_instance_segmentation(self,
                                     masks,
                                     dets,
                                     out_img,
                                     img_id=None,
                                     save_flag=False,
                                     show_box=False,
                                     save_path=None,
                                     gt_masks=None,
                                     keypoints=None):
        if gt_masks is not None:
            for mask in gt_masks:
                # mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
                # mskd = out_img * mask
                # clmsk = np.ones(mask.shape) * mask
                # clmsk[:, :, 0] = clmsk[:, :, 0] * 0 * 256
                # clmsk[:, :, 1] = clmsk[:, :, 1] * 0 * 256
                # clmsk[:, :, 2] = clmsk[:, :, 2] * 255 * 256
                # out_img = out_img + 1 * clmsk - 1 * mskd
                #apply_mask(image=out_img, mask=mask, color=(0, 1, 0), alpha=0.5)
                break


        colors = random_colors(masks.shape[0])
        for mask, det, color in zip(masks, dets, colors):
            color = np.random.rand(3)
            if show_box:
                y1,x1,y2,x2,conf = det
                #import random
                #cv2.rectangle(out_img, (int(x1), int(y1)), (int(x2), int(y2)), [random.randint(0,255),random.randint(0,255),random.randint(0,255)], 2, 1)
                #cv2.rectangle(out_img, (int(x1), int(y1)), (int(x2), int(y2)), [126, 108, 81], 2, 1)
                #cv2.rectangle(out_img, (int(x1), int(y1)), (int(x2), int(y2)), [0, 255, 0], 2, 1)
                #cv2.putText(out_img, "{:.4f}".format(conf), (int(x1),int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,255), 1, 1)
            # mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
            # mskd = out_img * mask
            # clmsk = np.ones(mask.shape) * mask
            # clmsk[:, :, 0] = clmsk[:, :, 0] * color[0] * 256
            # clmsk[:, :, 1] = clmsk[:, :, 1] * color[1] * 256
            # clmsk[:, :, 2] = clmsk[:, :, 2] * color[2] * 256
            #out_img = out_img + 1 * clmsk - 1 * mskd
            # print("numpy:",np.shape(out_img),np.shape(mask))
            #print(color)
            #out_img2 = copy.deepcopy(out_img)
            #apply_mask(image=out_img2, mask=mask, color=color, alpha=0.8)
            apply_mask(image=out_img, mask=mask, color=(0.519666,   0.85493216, 0.77586209), alpha=0.5)#(0.78737539, 0.9602264,  0.81372668)

            #cv2.imwrite(os.path.join(save_path, img_id + "_"+str(color)+".png"), np.uint8(out_img2))


            """heatmap = self.show_heat_mask(masks)
            heatmap = Image.fromarray(heatmap)
            heatmap.show()
            time.sleep(300)"""
            #contours, hierarchy = cv2.findContours(np.uint8(mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            # y1, x1, y2, x2, conf = det
            #cv2.drawContours(out_img, contours, -1, (color[0] * 256, color[1] * 256, color[2] * 256), 3)


        """for idx,kp in enumerate(keypoints):
            kp = kp.detach().cpu().numpy()
            out_img = self.imshow_kp(kp,out_img)
            cv2.imwrite(os.path.join(save_path, img_id + str(idx) +".png"), np.uint8(out_img))"""
        """
            img = Image.fromarray(img)
            img.show()
            time.sleep(100)

                #print(np.shape(pr_kp0))"""

        if save_flag:
            cv2.imwrite(os.path.join(save_path, img_id+".png"), np.uint8(out_img))
        """
        #imshow goes into infinite loop for some reason
        cv2.imshow('out_img', np.uint8(out_img))        
        cv2.imwrite('with_gt.png', np.uint8(out_img))
        k = cv2.waitKey(0)
        if k & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            exit(1)"""

    def test(self, args):
        self.load_weights(resume=args.resume, args=args)
        self.model = self.model.to(self.device)
        self.model.eval()

        if args.save_img:
            save_path = args.png_path+'save_result'
            if not os.path.exists(save_path):
                os.mkdir(save_path)
        else:
            save_path = None
        dsets = Antigen(data_dir=args.data_dir, phase='test')
        all_time = []
        for index in range(len(dsets)):
            time_begin = time.time()
            img = dsets.load_image(index)
            print(dsets.img_ids[index])
            predictions, kps = self.test_inference(args, img)
            if predictions is None:
                continue
            mask_patches, mask_dets = predictions
            gt_masks = dsets.load_gt_masks(dsets.load_annoFolder(dsets.img_ids[index]))
            #gt_masks = None
            all_time.append(time.time()-time_begin)
            self.imshow_instance_segmentation(mask_patches, mask_dets,
                                              out_img=img.copy(),
                                              img_id=dsets.img_ids[index],
                                              save_flag= args.save_img,
                                              save_path=save_path,
                                              gt_masks=gt_masks,
                                              show_box=True, keypoints = kps)
        print("Program Finished!")
        """
        all_time = all_time[1:]
        print('avg time is {}'.format(np.mean(all_time)))
        print('FPS is {}'.format(1./np.mean(all_time)))"""


if __name__ == '__main__':
    args = parse_args()
    object_is = InstanceHeat()
    object_is.test(args)
