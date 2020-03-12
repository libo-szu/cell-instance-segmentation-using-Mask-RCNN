


from keras import backend as K

K.tensorflow_backend._get_available_gpus()
import matplotlib
# Agg backend runs without a display
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
import os
import sys
import json
import datetime
import numpy as np
import skimage.io
from imgaug import augmenters as iaa

import tensorflow as tf
from mrcnn.config import Config
from mrcnn import utils
from mrcnn import model as modellib
# Root directory of the project
ROOT_DIR = os.path.abspath("../../")
from mrcnn import model as modellib
os.environ["CUDA_VISIBLE_DEVICES"] = '2,3'
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
config.gpu_options.allow_growth = True
sess = tf.Session(config = config)

class NucleusConfig(Config):
    """Configuration for training on the nucleus segmentation dataset."""
    # Give the configuration a recognizable name
    NAME = "nucleus"

    # Adjust depending on your GPU memory
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 4  # Background + nucleus

    # Number of training and validation steps per epoch
    STEPS_PER_EPOCH = 100000
    VALIDATION_STEPS = max(1, 1000)

    # Don't exclude based on confidence. Since we have two classes
    # then 0.5 is the minimum anyway as it picks between nucleus and BG
    DETECTION_MIN_CONFIDENCE = 0

    # Backbone network architecture
    # Supported values are: resnet50, resnet101
    BACKBONE = "resnet50"

    # Input image resizing
    # Random crops of size 512x512
    IMAGE_RESIZE_MODE = "crop"
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256
    IMAGE_MIN_SCALE = 2.0

    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)

    # ROIs kept after non-maximum supression (training and inference)
    POST_NMS_ROIS_TRAINING = 1000
    POST_NMS_ROIS_INFERENCE = 2000

    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.9

    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 64

    # Image mean (RGB)
    MEAN_PIXEL = np.array([43.53, 39.56, 48.22])

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask

    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 128

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 200



class NucleusInferenceConfig(NucleusConfig):
    # Set batch size to 1 to run one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # Don't resize imager for inferencing
    IMAGE_RESIZE_MODE = "pad64"
    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.7
def compute_dice(y_a,y_b):
    y_true_f = y_a.flatten()
    y_pred_f = y_b.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + 0.0000000001) / (np.sum(y_true_f) + np.sum(y_pred_f) + 0.0000000001)



class NucleusDataset(utils.Dataset):

    def load_nucleus(self, test_images_path = "/home/zhangwf/LIBOLIU/Mask_RCNN-master/samples/nucleus/isbi_data/Testing images/"):
        self.add_class("nucleus", 1, "Epithelial")
        self.add_class("nucleus", 2, "Lymphocyte")
        self.add_class("nucleus", 3, "Macrophage")
        self.add_class("nucleus", 4, "Neutrophil")
        list_imgs_path = []
        for file in os.listdir(test_images_path):
            for img_file in os.listdir(test_images_path + file + "/"):
                if (".png" in img_file):
                    list_imgs_path.append(test_images_path + file + "/" + img_file)
        for image_id in list_imgs_path:
            self.add_image(
                "nucleus",
                image_id=image_id,
                path=image_id)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "nucleus":
            return info["id"]
        else:
            super(self.__class__, self).image_reference(image_id)

def detect(test_images_path,log_dir,weights_path,save_path):
    count=0
    list_imgs_path = []
    for file in os.listdir(test_images_path):
        for img_file in os.listdir(test_images_path + file + "/"):
            if (".tif" in img_file):
                list_imgs_path.append(test_images_path + file + "/" + img_file)
    for tif_img_path in list_imgs_path:
        tif_img=cv2.imread(tif_img_path)
        cv2.imwrite(tif_img_path.replace("tif","png"),tif_img)


    config = NucleusInferenceConfig()
    config.display()
    model = modellib.MaskRCNN(mode="inference", config=config,model_dir=log_dir)
    model.load_weights(weights_path, by_name=True)
    list_imgs_path=[]
    for file in os.listdir(test_images_path):
        for img_file in os.listdir(test_images_path+file+"/"):
            if(".tif" in img_file):
                list_imgs_path.append(test_images_path+file+"/"+img_file.replace("tif","png"))
    dataset = NucleusDataset()
    dataset.load_nucleus(test_images_path)
    dataset.prepare()
    for image_id in dataset.image_ids:
        image_path=dataset.image_info[image_id]['path']
        # print(image_path
        image = dataset.load_image(image_id)
        image_shape=image.shape
        count1 = 0
        count2 = 0
        count3 = 0
        count4 = 0
        class_1_mask = np.zeros((image_shape[0], image_shape[1]), dtype=np.uint8)
        class_2_mask = np.zeros((image_shape[0], image_shape[1]), dtype=np.uint8)
        class_3_mask = np.zeros((image_shape[0], image_shape[1]), dtype=np.uint8)
        class_4_mask = np.zeros((image_shape[0], image_shape[1]), dtype=np.uint8)
        size=256
        step=128
        w,h=image_shape[0],image_shape[1]
        flag_w=w%step
        if(flag_w!=0):
            w_range=w//step+1
        else:
            w_range = w // step

        flag_h=h%step
        if(flag_h!=0):
            h_range=h//step+1
        else:
            h_range = h// step
        for i in range(w_range):
            for j in range(h_range):
                w_i=i*step+size
                if(w_i>w):
                    w_i=w
                h_j=j*step+size
                if(h_j>h):
                    h_j=h
                img1=image[i*step:w_i,j*step:h_j,:]
                img2=np.zeros((256,256,3))
                img2[:img1.shape[0],:img1.shape[1],:]=img1
                # print(img2.shape)
                r = model.detect([img2], verbose=0)[0]
                masks = r['masks']
                masks[masks == True] = 1
                masks[masks == False] = 0
                class_1 = np.zeros((256, 256), dtype=np.uint8)
                class_2 = np.zeros((256, 256), dtype=np.uint8)
                class_3 = np.zeros((256, 256), dtype=np.uint8)
                class_4 = np.zeros((256, 256), dtype=np.uint8)
                # print(image_shape)

                for k, item in enumerate(r['rois']):
                    now_mask = r['masks'][:, :, k]
                    now_mask[now_mask == True] = 1
                    now_mask[now_mask == False] = 0
                    now_score = r['scores'][k]
                    now_id = r['class_ids'][k]
                    # print(now_mask.shape)
                    for l in range(k + 1, len(r['scores'])):
                        new_mask = r['masks'][:, :, l]
                        new_score = r['scores'][l]
                        dice = compute_dice(now_mask, new_mask)
                        if (dice > 0.00000000001):
                            # print(dice)
                            if (new_score > now_score):
                                now_mask[new_mask == now_mask] = 0
                            else:
                                new_mask[new_mask == now_mask] = 0

                    # if (image_shape[0] < 256 and image_shape[1] < 256):
                    #     now_mask = now_mask[:image_shape[0], :image_shape[1]]
                    #     # plt.imshow(now_mask)
                    #     # plt.savefig("resize.png")
                    #     now_mask = now_mask.astype(np.uint8)
                    # else:
                    now_mask = now_mask.astype(np.uint8)

                    # print(now_mask.shape)
                        # print(now_mask.shape)
                        # now_mask = cv2.resize(now_mask, (h_original_img, w_original_img))
                        # now_mask[now_mask >= 0.1] = 1
                        # now_mask[now_mask < 0.1] = 0
                    # print(image1.shape)
                    # print(now_mask.shape)
                    g = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                    now_mask = cv2.morphologyEx(now_mask, cv2.MORPH_OPEN, g)

                    if (now_id == 1):
                        count1 += 1
                        class_1 += now_mask * count1
                    if (now_id == 2):
                        count2 += 1
                        class_2 += now_mask * count2
                    if (now_id == 3):
                        count3 += 1
                        class_3 += now_mask * count3
                    if (now_id == 4):
                        count4 += 1
                        class_4 += now_mask * count4
                    # cv2.imwrite("class_1.png",class_1)
                    # cv2.imwrite("class_2.png",class_2)
                    # cv2.imwrite("class_3.png",class_3)
                    # cv2.imwrite("class_4.png",class_4)
                class_1_mask[i*step:w_i,j*step:h_j]=class_1[:w_i-i*step,:h_j-j*step]
                class_2_mask[i*step:w_i,j*step:h_j]=class_2[:w_i-i*step,:h_j-j*step]
                class_3_mask[i*step:w_i,j*step:h_j]=class_3[:w_i-i*step,:h_j-j*step]
                class_4_mask[i*step:w_i,j*step:h_j]=class_4[:w_i-i*step,:h_j-j*step]
            # image1=cv2.resize(image,(256,256))
        img_path_split=image_path.split("/")
        file_base1=save_path+img_path_split[-2]+"/"
        file_base2=save_path+img_path_split[-2]+"/"+img_path_split[-1].replace(".png","")+"/"

        if(not os.path.exists(file_base1)):
            os.mkdir(file_base1)
        if(not os.path.exists(file_base2)):
            os.mkdir(file_base2)
        if(not os.path.exists(file_base2+"Epithelial/")):
            os.mkdir(file_base2+"Epithelial/")
        if(not os.path.exists(file_base2+"Lymphocyte/")):
            os.mkdir(file_base2+"Lymphocyte/")
        if(not os.path.exists(file_base2+"Macrophage/")):
            os.mkdir(file_base2+"Macrophage/")
        if(not os.path.exists(file_base2+"Neutrophil/")):
            os.mkdir(file_base2+"Neutrophil/")
        class_1_mask[class_1_mask>0]=1
        class_1_mask[class_1_mask<=0]=0


        class_2_mask[class_2_mask>0]=1
        class_2_mask[class_2_mask<=0]=0


        class_3_mask[class_3_mask>0]=1
        class_3_mask[class_3_mask<=0]=0


        class_4_mask[class_4_mask>0]=1
        class_4_mask[class_4_mask<=0]=0
        ret, labels1, stats, centroids = cv2.connectedComponentsWithStats(class_1_mask, connectivity=8)
        ret, labels2, stats, centroids = cv2.connectedComponentsWithStats(class_2_mask, connectivity=8)
        ret, labels3, stats, centroids = cv2.connectedComponentsWithStats(class_3_mask, connectivity=8)
        ret, labels4, stats, centroids = cv2.connectedComponentsWithStats(class_4_mask, connectivity=8)

        import scipy.io as scio
        if(np.sum(class_1_mask)>0):
            print(set(labels1[labels1!=0]))
            count+=1
            # cv2.imwrite(file_base2 + "Epithelial/" + str(count) + "mask.png",labels1*5)
            scio.savemat(file_base2+"Epithelial/"+str(count)+"mask.mat",{'__header__':b'MATLAB 5.0 MAT-file Platform: nt, Created on: Sun Feb 16 17:54:54 2020',
                                                                          '__version__': '1.0', '__globals__': [], 'n_ary_mask':labels1})
        if(np.sum(class_2_mask)>0):
            print(set(labels2[labels2!=0]))

            count+=1
            # cv2.imwrite(file_base2 + "Lymphocyte/" + str(count) + "mask.png",labels2*5)
            scio.savemat(file_base2+"Lymphocyte/"+str(count)+"mask.mat",{'__header__': b'MATLAB 5.0 MAT-file Platform: nt, Created on: Sun Feb 16 17:54:54 2020',
                                                         '__version__': '1.0', '__globals__': [], 'n_ary_mask':labels2})
        if(np.sum(class_3_mask)>0):
            count+=1
            # cv2.imwrite(file_base2 + "Macrophage/" + str(count) + "mask.png",labels3*5)
            scio.savemat(file_base2+"Macrophage/"+str(count)+"mask.mat",{'__header__': b'MATLAB 5.0 MAT-file Platform: nt, Created on: Sun Feb 16 17:54:54 2020',
                                                         '__version__': '1.0', '__globals__': [], 'n_ary_mask':labels3})
        if(np.sum(class_4_mask)>0):
            count+=1
            # cv2.imwrite(file_base2 + "Neutrophil/" + str(count) + "mask.png",labels4*5)
            scio.savemat(file_base2+"Neutrophil/"+str(count)+"mask.mat",{'__header__': b'MATLAB 5.0 MAT-file Platform: nt, Created on: Sun Feb 16 17:54:54 2020',
                                                         '__version__': '1.0', '__globals__': [], 'n_ary_mask':labels4})

if __name__=="__main__":
    test_images_path="/home/zhangwf/LIBOLIU/samples/nucleus/isbi_data/Testing images/"   #the path of the test image set
    log_dir="./log/"
    weights_path="mask_rcnn_nucleus_0104.h5"
    save_path="./SZU-HISTOPATH_MoNuSAC_test_results/"
    detect(test_images_path, log_dir, weights_path, save_path)

    # img=cv2.imread('class_2.png',0)
    # g=cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    # img_open=cv2.morphologyEx(img,cv2.MORPH_OPEN,g)
    # cv2.imwrite("kk.png",img_open)
