import json
import os
import random

from torch.utils.data import Dataset

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

from data.utils import pre_caption,clean_report_mimic_cxr
import os,glob
import jsonlines


class pretrain_dataset(Dataset):
    def __init__(self, ann_file, laion_path, transform):

        self.ann_pretrain = []
        for file in ann_file:
            if "jsonl" in file:
                print('loading ' + file)
                with jsonlines.open(file) as f:
                    for ann in f:
                        pair = {}
                        # ann = json.load(list)
                        if 'PMC_OA' in file:
                            pair['image'] = os.path.join(
                                './data/PMC/PMC_OA/caption_T060_filtered_top4_sep_v0_subfigures',
                                ann['image'])
                            pair['caption'] = ann['caption']
                            self.ann_pretrain.append(pair)
                        elif 'MedICaT' in file:
                            pair['image'] = os.path.join(
                                './data/MedICaT/release/figures',
                                f"{ann['pdf_hash']}_{ann['fig_uri']}")
                            pair['caption'] = ann['s2_caption']
                            if os.path.exists(pair['image']):
                                self.ann_pretrain.append(pair)
            else:
                    print('loading ' + file)
                    ann = json.load(open(file, 'r'))
                    pair = {}
                    if 'roco_train' in file:
                        for i in ann:
                            pair['image'] = os.path.join("./data/ROCO/train/radiology/images", i['PMC_ID'])
                            pair['caption'] = i['caption']
                            if os.path.exists(pair['image']) and pair['image'].lower().endswith(".jpg"):
                                self.ann_pretrain.append(pair)
                    if 'roco_val' in file:
                        for i in ann:
                            pair['image'] = os.path.join("./data/ROCO/validation/radiology/images", i['PMC_ID'])
                            pair['caption'] = i['caption']
                            if os.path.exists(pair['image']) and pair['image'].lower().endswith(".jpg"):
                                self.ann_pretrain.append(pair)
                    if 'roco_test' in file:
                        for i in ann:
                            pair['image'] = os.path.join("./data/ROCO/test/radiology/images", i['PMC_ID'])
                            pair['caption'] = i['caption']
                            if os.path.exists(pair['image']) and pair['image'].lower().endswith(".jpg"):
                                self.ann_pretrain.append(pair)


                    # self.ann_pretrain += ann["train"]
                    # self.ann_pretrain += ann["val"]
                    # self.ann_pretrain += ann["test"]

        self.laion_path = laion_path
        if self.laion_path:
            self.laion_files = glob.glob(os.path.join(laion_path, '*.json'))

            print('loading ' + self.laion_files[0])
            with open(self.laion_files[0], 'r') as f:
                self.ann_laion = json.load(f)

            self.annotation = self.ann_pretrain + self.ann_laion
        else:
            self.annotation = self.ann_pretrain

        self.transform = transform

    def reload_laion(self, epoch):
        n = epoch%len(self.laion_files)
        print('loading '+self.laion_files[n])
        with open(self.laion_files[n],'r') as f:
            self.ann_laion = json.load(f)      
        
        self.annotation = self.ann_pretrain + self.ann_laion    
        
    
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):

        ann = self.annotation[index]
        image_path = ''.join(ann['image'])
        # image_path = ''.join(ann['image_path'])

        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        # caption = pre_caption(ann['report'])
        caption = clean_report_mimic_cxr(ann['caption'], 60)


        return image, caption