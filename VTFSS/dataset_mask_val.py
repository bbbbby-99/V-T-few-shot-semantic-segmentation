import random
import os
import torchvision
import torch
import cv2
from PIL import Image
import torchvision.transforms.functional as F
import torch.nn.functional as F_tensor
import numpy as np
from torch.utils.data import DataLoader
import time
from tqdm import tqdm


class Dataset(object):


    def __init__(self, data_dir, fold, input_size=[200, 200], normalize_mean=[0, 0, 0],
                 normalize_std=[1, 1, 1]):

        self.data_dir = data_dir

        self.input_size = input_size

        #random sample 500 pairs
        self.chosen_data_list_1 = self.get_new_exist_class_dict(fold=fold)
        chosen_data_list_2 = self.chosen_data_list_1[:]
        chosen_data_list_3 = self.chosen_data_list_1[:]

        random.shuffle(chosen_data_list_2)
        random.shuffle(chosen_data_list_3)


        self.chosen_data_list=self.chosen_data_list_1+ chosen_data_list_2+ chosen_data_list_3
        self.chosen_data_list=self.chosen_data_list[:500]


        self.binary_pair_list = self.get_binary_pair_list()#a dict of each class, which contains all imgs that include this class
        self.history_mask_list = [None] * 500
        self.query_class_support_list=[None] * 500
        for index in range (500):
            query_name=self.chosen_data_list[index][0]
            sample_class=self.chosen_data_list[index][1]
            support_img_list = self.binary_pair_list[sample_class]  # all img that contain the sample_class
            while True:  # random sample a support data
                support_name = support_img_list[random.randint(0, len(support_img_list) - 1)]
                if support_name != query_name:
                    break
            self.query_class_support_list[index]=[query_name,sample_class,support_name]



        self.initiaize_transformation(normalize_mean, normalize_std, input_size)
        pass

    def rgbnor(self,img):

        means=[0,0,0]
        stdevs =[0,0,0]
        for i in range(3):
            means[i] +=img[:,:,i].mean()
            stdevs[i]+=img[:,:,i].std()
        F.normalize(img, means, stdevs, inplace=False)
        return img


    def get_new_exist_class_dict(self, fold):
        new_exist_class_list = []

        # f = open(os.path.join(self.data_dir, 'Binary_map', 'split%1d.txt' % (fold)))
        # while True:
        #     item = f.readline()
        #     if item == '':
        #         break
        #     img_name = item[:10]
        #     cat = int(item[12:14])
        f = open(os.path.join(self.data_dir, 'Binary_map', 'split%1d.txt' % fold),encoding='UTF-8-sig').readlines()
        for l_idx in tqdm(range(len(f))):
            line = f[l_idx]
            line = line.strip()
            line_split = line.split()
            img_name = line_split[0]
            cat = int(line_split[1])
            new_exist_class_list.append([img_name, cat])
        return new_exist_class_list


    def initiaize_transformation(self, normalize_mean, normalize_std, input_size):
        self.ToTensor = torchvision.transforms.ToTensor()
        self.normalize = torchvision.transforms.Normalize(normalize_mean, normalize_std)

    def get_binary_pair_list(self):  # a list store all img name that contain that class
        binary_pair_list = {}
        for Class in range(1, 17):
            binary_pair_list[Class] = self.read_txt(
                os.path.join(self.data_dir, 'Binary_map', '%d.txt' % Class))
        return binary_pair_list

    def read_txt(self, dir):
        f = open(dir)
        out_list = []
        line = f.readline()
        while line:
            out_list.append(line.split()[0])
            line = f.readline()
        return out_list

    def __getitem__(self, index):

        # give an query index,sample a target class first
        query_name = self.query_class_support_list[index][0]
        sample_class = self.query_class_support_list[index][1]  # random sample a class in this img
        support_name=self.query_class_support_list[index][2]



        input_size = self.input_size[0]
        # random scale and crop for support
        scaled_size = int(random.uniform(1,1.5)*input_size)
        scale_transform_mask = torchvision.transforms.Resize([scaled_size, scaled_size], interpolation=Image.NEAREST)
        scale_transform_rgb = torchvision.transforms.Resize([scaled_size, scaled_size], interpolation=Image.BILINEAR)
        scale_transform_th = torchvision.transforms.Resize([scaled_size, scaled_size], interpolation=Image.BILINEAR)
        flip_flag = random.random()
        support_name_rgb = support_name
        support_name_rgb = support_name_rgb.replace('.png', '_rgb.png')
        support_name_th = support_name.replace('.png', '_th.png')

        image_th = cv2.imread(os.path.join(self.data_dir, 'seperated_images', support_name_th))
        image_th = Image.fromarray(cv2.cvtColor(image_th,cv2.COLOR_BGR2RGB))

        support_th = self.ToTensor(
                scale_transform_th(
                    self.flip(flip_flag,image_th)))

        support_th =self.rgbnor(support_th)
        support_rgb = self.normalize(self.ToTensor(
                scale_transform_rgb(
                    self.flip(flip_flag,
                              Image.open(
                                  os.path.join(self.data_dir, 'seperated_images', support_name_rgb))))))

        support_mask = self.ToTensor(
            scale_transform_mask(
                self.flip(flip_flag,
                          Image.open(
                              os.path.join(self.data_dir, 'Binary_map', str(sample_class), support_name )))))

        margin_h = random.randint(0, scaled_size - input_size)
        margin_w = random.randint(0, scaled_size - input_size)

        support_rgb = support_rgb[:, margin_h:margin_h + input_size, margin_w:margin_w + input_size]
        support_mask = support_mask[:, margin_h:margin_h + input_size, margin_w:margin_w + input_size]
        support_th = support_th[:, margin_h:margin_h + input_size, margin_w:margin_w + input_size]

        # random scale and crop for query
        scaled_size = 200

        scale_transform_mask = torchvision.transforms.Resize([scaled_size, scaled_size], interpolation=Image.NEAREST)
        scale_transform_rgb = torchvision.transforms.Resize([scaled_size, scaled_size], interpolation=Image.BILINEAR)
        scale_transform_th = torchvision.transforms.Resize([scaled_size, scaled_size], interpolation=Image.BILINEAR)
        flip_flag = 0#random.random()
        query_name_rgb = query_name
        query_name_rgb = query_name_rgb.replace('.png','_rgb.png')######
        query_name_th = query_name.replace('.png', '_th.png')

        image_thq = cv2.imread(os.path.join(self.data_dir, 'seperated_images', query_name_th))
        image_thq = Image.fromarray(cv2.cvtColor(image_thq, cv2.COLOR_BGR2RGB))

        query_th = self.ToTensor(
                scale_transform_th(
                    self.flip(flip_flag, image_thq)))

        query_th = self.rgbnor(query_th)
        query_rgb = self.normalize(self.ToTensor(
                scale_transform_rgb(
                    self.flip(flip_flag,
                              Image.open(
                                  os.path.join(self.data_dir, 'seperated_images', query_name_rgb))))))

        query_mask = self.ToTensor(
            scale_transform_mask(
                self.flip(flip_flag,
                          Image.open(
                              os.path.join(self.data_dir, 'Binary_map', str(sample_class),
                                           query_name)))))

        margin_h = random.randint(0, scaled_size - input_size)
        margin_w = random.randint(0, scaled_size - input_size)

        query_rgb = query_rgb[:, margin_h:margin_h + input_size, margin_w:margin_w + input_size]
        query_mask = query_mask[:, margin_h:margin_h + input_size, margin_w:margin_w + input_size]
        query_th = query_th[:, margin_h:margin_h + input_size, margin_w:margin_w + input_size]

        if self.history_mask_list[index] is None:

            history_mask=torch.zeros(2,26,26).fill_(0.0)

        else:

            history_mask=self.history_mask_list[index]



        return query_rgb,query_th ,query_mask, support_rgb,support_th, support_mask,history_mask,sample_class,index

    def flip(self, flag, img):
        if flag > 0.5:
            return F.hflip(img)
        else:
            return img

    def __len__(self):
        return 500
