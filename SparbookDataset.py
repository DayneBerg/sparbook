import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class SparbookDataset(Dataset):
    """
    TODO: Option to load all data in memory rather than lookup
    TODO: Implement gamma preprocessing / data augmentation
    """
    def __init__(self):
        self.lineheight = 27
        # self.freq_table = [1] * 94
        # construct list of index -> (doc name, line index)
        img_names = os.listdir(f"{os.getcwd()}\samples")
        self.datalist = []
        for img_name in img_names:
            with Image.open(f"{os.getcwd()}\samples\{img_name}") as cur_img:
                assert (cur_img.height % self.lineheight == 0), "sample's height was invalid"
                num = cur_img.height // self.lineheight
                exif_data_text = cur_img.getexif().get(270)
                assert (num == exif_data_text.count(chr(31))), "sample's height and description do not match"
                '''for c in exif_data_text:
                    i = ord(c) - 31
                    if i > 0:  # exclude Unit Separator
                        if i >= 65:  # correct for grave accent
                            i -= 1
                        self.freq_table[i - 1] += 1'''
                for i in range(num):
                    self.datalist.append((img_name, i))

    def __getitem__(self, index):
        (name, num) = self.datalist[index]
        filename = f"{os.getcwd()}\samples\{name}"
        img = Image.open(filename)

        # get data
        assert (img.height % self.lineheight == 0), "sample's height was invalid"
        # pad edges with median color
        median_color = round(np.median(np.asarray(img)).item())
        img_data = Image.new(mode="L", size=(img.width, img.height + 4), color=median_color)
        img_data.paste(img, (0, 2))
        img_data = img_data.crop((0, num * self.lineheight, img.width, (num + 1) * self.lineheight + 4))
        # size = (W, 31)
        '''img_data = img_data.resize((img.width, 16))'''

        img_data = torch.from_numpy(np.array(img_data, dtype=float)).permute(1, 0)
        # preprocessing

        img_data = img_data - median_color
        # result from dataloader should be N,W,27+4
        # median is zero, but norms of positive and negative values need to be equalized, respectively
        pos = torch.gt(img_data, 0.0)
        pos_norm = torch.sqrt(torch.sum(torch.square(torch.maximum(img_data, torch.zeros_like(img_data)))) / torch.sum(pos))
        neg = torch.lt(img_data, 0.0)
        neg_norm = torch.sqrt(torch.sum(torch.square(torch.minimum(img_data, torch.zeros_like(img_data)))) / torch.sum(neg))
        img_data = torch.where(pos, img_data / pos_norm, img_data / neg_norm)

        # get label
        exifdata = img.getexif()
        image_description = exifdata.get(270).split(chr(31))
        assert (len(image_description) > 1 + num), "sample's description was invalid"
        text_label = image_description[num]
        # result from dataloader should be N,max_num_chars<=80
        target = np.zeros(len(text_label), dtype=int)
        for i in range(len(text_label)):
            j = ord(text_label[i]) - 31
            assert (0 < j < 96), 'Image description contains unsupported characters.'
            if j >= 65:  # grave accent (65) should have been replaced with acute accent (8)
                j += -1
            target[i] = j
        return img_data, torch.from_numpy(target)

    def __len__(self):
        return len(self.datalist)


"""# gamma calculation to support later data augmentation
        hist = img.histogram()
        for i in range(1, len(hist)):
            hist[i] += hist[i-1]
        cumsum = 0.0
        num = 0
        for i in range(len(hist)):
            try:
                cumsum += math.log(hist[i]/hist[-1], (2 * i - 1)/(2 * len(hist)))
                num += 1
            except ValueError:
                pass
        gamma = cumsum / num"""
