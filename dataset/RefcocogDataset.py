import json
import pandas
from skimage import io
from torch.utils.data import random_split
from typing import Sequence, Union

from PIL import Image, ImageDraw
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torch
import numpy as np


## Official Class used to train and test the model
class RefcocogDataset(Dataset):
    def __init__(self, base_path, split=None, transform=None, tokenization=None):
        annotation_path = base_path + "/annotations/"

        self.IMAGES_PATH = base_path + "/images/"
        self.transform = transform
        self.tokenization = tokenization

        # Load annotations and instances
        tmp_annotations = pandas.read_pickle(annotation_path + "refs(umd).p")
        tmp_instances = json.load(open(annotation_path + "instances.json", "r"))

        # Create dataframes
        annotations_dt = pandas.DataFrame.from_records(tmp_annotations) \
            .filter(items=["image_id", "split", "sentences", "ann_id"])

        instances_dt = pandas.DataFrame.from_records(tmp_instances['annotations'])

        # Add Explode to separate list-like sentences column and use them as separate samples
        # Create a new datapoint with every different phrase for the same image
        self.annotations = annotations_dt \
            .merge(instances_dt[["id", "bbox", "area", "segmentation"]], left_on="ann_id", right_on="id") \
            .explode('sentences', ignore_index=True) \
            .drop(columns="id")

        if split is not None:
            self.annotations = self.__get_annotations_by_split(split.lower())

    def getImage(self, sample):
        # Utility function to get the image from the dataset
        id = sample['idx'][0].item()
        item = self.annotations.iloc[id]
        image = self.__getimage(item.image_id)

        return image

    def getSentences(self, sample):
        # Utility function to get the sentences from the dataset
        id = sample['idx'][0].item()
        item = self.annotations.iloc[id]

        return self.__extract_sentences(item.sentences)

    def __computeGroundTruth(self, item):
        # Utility function to create a mask from the ground truth segmentation        
        image = self.__getimage(item.image_id)
        mask = Image.new("L", image.size)
        draw = ImageDraw.Draw(mask)
        draw.polygon(item.segmentation[0], fill="white", width=0)

        # Resize the computed mask to 640x640 to match the model input
        mask = mask.resize((640, 640))

        return self.__img_preprocess(mask)
    
    def __bbox_image(self, item, n_px: int = 224):
        # Utility function to create a mask from the ground truth bboxes
        image = self.__getimage(item.image_id)
        mask = Image.new("L", image.size)
        draw = ImageDraw.Draw(mask)
        
        rect_coords = [item.bbox[0], item.bbox[1], item.bbox[0] + item.bbox[2], item.bbox[1] + item.bbox[3]]
        draw.rectangle(rect_coords, fill="white", width=0)

        mask = mask.resize((640, 640))

        resized = T.Resize(n_px, interpolation=Image.BICUBIC)(mask)
        
        crop = T.CenterCrop(n_px)(resized)

        arr = torch.tensor(np.array(crop))

        return self.extract_bbox(arr)


    def extract_bbox(self, out):
        # Extract bbox coordinates from the an mask
        map = out.squeeze(0).squeeze(0).detach().cpu().numpy()
        # normalize map to [0, 1]
        map = (map - map.min()) / (map.max() - map.min())
        # threshold map
        map = (map > 0.8)
        x_min = 225
        y_min = 225
        x_max = 0
        y_max = 0
        for i in range(224):
            for j in range(224):
                if map[i][j] == True:
                    if i < y_min: y_min = i
                    if i > y_max: y_max = i
                    if j < x_min: x_min = j
                    if j > x_max: x_max = j
        
        return x_min, y_min, x_max, y_max
        
    
    def __img_preprocess(self, image: Image, n_px: int = 224, grid_px: int = 14):
        # Utility function to preprocess the input image
        resized = T.Resize(n_px, interpolation=Image.BICUBIC)(image)
        crop = T.CenterCrop(n_px)(resized)

        grid = T.Resize(grid_px, interpolation=Image.BICUBIC)(crop)

        arr = torch.tensor(np.asarray(grid))
        arr[arr <= 0.0] = 0 # make the image binary
        arr[arr > 0.0] = 1 # make the image binary
        return arr
    
    def __computeGroundTruthRefiner(self, item):
        image = self.__getimage(item.image_id)
        mask = Image.new("L", image.size)
        draw = ImageDraw.Draw(mask)
        draw.polygon(item.segmentation[0], fill="white", width=0)

        mask = mask.resize((640, 640))

        return self.__img_preprocess_refiner(mask)

    def __img_preprocess_refiner(self, image: Image, n_px: int = 224):
        resized = T.Resize(n_px, interpolation=Image.BICUBIC)(image)
        crop = T.CenterCrop(n_px)(resized)

        arr = torch.tensor(np.asarray(crop))
        arr[arr <= 0.0] = 0 # make the image binary
        arr[arr > 0.0] = 1 # make the image binary
        return arr

    def __get_train_annotations(self):
        return self.annotations[self.annotations.split == "train"].reset_index()

    def __get_annotations_by_split(self, split):
        return self.annotations[self.annotations.split == split].reset_index()

    def __getimage(self, id):
        return Image.open(self.IMAGES_PATH + "COCO_train2014_" + str(id).zfill(12) + ".jpg")

    def __extract_sentences(self, sentence):
        return f"{sentence['sent']}"

    def __tokenize_sents(self, sentences):
        return [self.tokenization(s) for s in sentences]

    def __len__(self):
        return self.annotations.shape[0]

    def __getitem__(self, idx):
        # Return {sample, sentence, id}, bbox
        # Return single sentence, probably preprocess needed so we do not waste data
        
        item = self.annotations.iloc[idx]
        image = self.__getimage(item.image_id).resize((640, 640))
        sentences = self.__extract_sentences(item.sentences)

        if self.transform:
            image = self.transform(image)

        if self.tokenization:
            sentences = self.__tokenize_sents(sentences)

        sample = {'idx': idx, 'image': image, 'sentences': sentences}

        return sample, {'bbox': self.__bbox_image(item), 'gt': self.__computeGroundTruth(item), 'gt_refiner':self.__computeGroundTruthRefiner(item)}
    