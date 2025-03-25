import json
import pandas
from PIL import Image
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np
import cv2
import matplotlib.pyplot as plt
import numpy as np 
from pycocotools import mask
import torch
from torchvision import transforms as T

class RefcocogDataset(Dataset):
    """ Custom Dataset for Refcocog """

    def __init__(self, data_path, split=None, transform=None) -> None:
        self.data_path = data_path
        self.annotation_path = self.data_path + "/annotations/"
        self.image_path = self.data_path + "/images/"
        self.split = split
        self.transform = transform

        # Load the data
        references = pandas.read_pickle(self.annotation_path + "refs(umd).p")
        references_df = pandas.DataFrame(references)
        instances = json.load(open(self.annotation_path + "instances.json", "r"))
        instances_df = pandas.DataFrame(instances["annotations"])

        self.annotations_df = references_df.merge(instances_df, left_on="ann_id", right_on="id").explode("sentences", ignore_index=True)
        self.annotations_df = self.annotations_df.filter(items=["image_id_x", "split", "sentences", "ann_id", "bbox", "area", "segmentation"])

        if self.split is not None:
            self.annotations_df = self.annotations_df[self.annotations_df["split"] == self.split.lower()]

    def __get_data__(self):
        """ returns the split and the annotations dataframe """
        return self.split, self.annotations_df

    def __len__(self):
        return len(self.annotations_df)

    def __get_image__(self, sample):
        image_id = sample["image_id_x"]
        file_name = "/COCO_train2014_" + str(image_id).zfill(12) + ".jpg"
        return Image.open(self.image_path + file_name)

    def __get_sentence__(self, sample):
        return sample["sentences"]["sent"]

    def __get_bbox__(self, sample):
        return sample["bbox"]

    def __get_token_mask__(self, sample):
        segmentation = sample["segmentation"]
        image = self.__get_image__(sample)

        # Get the segmentation mask
        rles = mask.frPyObjects(segmentation, image.size[1], image.size[0])
        rle = mask.merge(rles)
        m = mask.decode(rle)

        # Create the token mask
        token_mask = T.Resize(size=(14, 14), interpolation=Image.BICUBIC)(Image.fromarray(m))
        token_mask = torch.tensor(np.asarray(token_mask))
        token_mask[token_mask <= 0.0] = 0 # make the image binary
        token_mask[token_mask > 0.0] = 1 # make the image binary
        token_mask = token_mask.reshape(1, 14, 14)
        return token_mask

    def __get_segmentation__(self, sample):
        segmentation = sample["segmentation"]
        image = self.__get_image__(sample)

        # Get the segmentation mask
        rles = mask.frPyObjects(segmentation, image.size[1], image.size[0])
        rle = mask.merge(rles)
        m = mask.decode(rle)

        return m

    def __getitem__(self, idx, resized=False):
        """ returns a sample: image, sentence, bbox, token_mask, segmentation """
        sample = self.annotations_df.iloc[idx]

        image = self.__get_image__(sample)
        sentence = self.__get_sentence__(sample)
        bbox = self.__get_bbox__(sample)
        token_mask = self.__get_token_mask__(sample)
        segmentation = self.__get_segmentation__(sample)

        # Apply the transform
        if self.transform:
            image = self.transform(image)
        
        return image, sentence, bbox, token_mask, segmentation

    def __show_sample__(self, idx): 
        """ show a sample from the dataset """
        sample = self.annotations_df.iloc[idx]
        image = self.__get_image__(sample)
        sentence = self.__get_sentence__(sample)
        bbox = self.__get_bbox__(sample)
        token_mask = self.__get_token_mask__(sample)
        segmentation = self.__get_segmentation__(sample)

        int_bbox = [int(i) for i in bbox]
        image_bbox = np.array(image.copy())
        image_bbox = cv2.rectangle(image_bbox, (int_bbox[0], int_bbox[1]), (int_bbox[0]+int_bbox[2], int_bbox[1]+int_bbox[3]), (0, 255, 0), 2)

        plt.figure(figsize=(5, 5))
        plt.title(sentence + '\n' + str(bbox))
        plt.imshow(image_bbox)
        plt.show()

    def __show_segmentation__(self, idx):
        """ show the segmentation of a sample from the dataset """
        sample = self.annotations_df.iloc[idx]
        m = self.__get_segmentation__(sample)

        plt.figure(figsize=(5, 5))
        ax = plt.gca()
        ax.imshow(m)
        plt.show()

    def __show_token_mask__(self, idx):
        """ show the token mask of a sample from the dataset """
        sample = self.annotations_df.iloc[idx]
        token_mask = self.__get_token_mask__(sample)

        plt.figure(figsize=(5, 5))
        plt.imshow(token_mask[0])
        plt.show()

    def __show_sample_with_segmentation__(self, idx):
        """ show a sample from the dataset with segmentation """
        image, sentence, bbox, token_mask, segmentation = self.__getitem__(idx)

        int_bbox = [int(i) for i in bbox]
        image_bbox = np.array(image.copy())
        image_bbox = cv2.rectangle(image_bbox, (int_bbox[0], int_bbox[1]), (int_bbox[0]+int_bbox[2], int_bbox[1]+int_bbox[3]), (0, 255, 0), 2)

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 3, 1)
        plt.title(sentence + '\n' + str(bbox))
        plt.imshow(image_bbox)
        plt.subplot(1, 3, 2)
        ax = plt.gca()
        ax.imshow(segmentation)
        plt.subplot(1, 3, 3)
        plt.imshow(token_mask[0])
        plt.show()


class RefcocogDataset_RISCLIP(Dataset):
    """ Custom Dataset for Refcocog adapted to RISCLIP  """

    def __init__(self, data_path, split=None, transform=None, tokenizer= None, patch_per_image = 14) -> None:
        self.data_path = data_path
        self.annotation_path = self.data_path + "/annotations/"
        self.image_path = self.data_path + "/images/"
        self.split = split
        self.transform = transform
        self.tokenizer = tokenizer
        self.patch_per_image = patch_per_image

        # Load the data
        references = pandas.read_pickle(self.annotation_path + "refs(umd).p")
        references_df = pandas.DataFrame(references)
        instances = json.load(open(self.annotation_path + "instances.json", "r"))
        instances_df = pandas.DataFrame(instances["annotations"])

        self.annotations_df = references_df.merge(instances_df, left_on="ann_id", right_on="id").explode("sentences", ignore_index=True)
        self.annotations_df = self.annotations_df.filter(items=["image_id_x", "split", "sentences", "ann_id", "bbox", "area", "segmentation"])

        if self.split != None:
            self.annotations_df = self.annotations_df[self.annotations_df["split"] == self.split.lower()]

    def __get_data__(self):
        """ returns the split and the annotations dataframe """
        return self.split, self.annotations_df

    def __len__(self):
        return len(self.annotations_df)
    
    def __get_image__(self, sample):
        image_id = sample["image_id_x"]
        file_name = "/COCO_train2014_" + str(image_id).zfill(12) + ".jpg"
        return Image.open(self.image_path + file_name)
    
    def __get_sentence__(self, sample):
        return sample["sentences"]["sent"]
    
    def __get_bbox__(self, sample):
        return sample["bbox"]
    
    def __get_token_mask__(self, sample):
        segmentation = sample["segmentation"]
        image = self.__get_image__(sample)

        # Get the segmentation mask
        rles = mask.frPyObjects(segmentation, image.size[1], image.size[0])
        rle = mask.merge(rles)
        m = mask.decode(rle)

        # Create the token mask
        size = (self.patch_per_image, self.patch_per_image)
        token_mask = T.Resize(size, interpolation=Image.BICUBIC)(Image.fromarray(m))
        token_mask = torch.tensor(np.asarray(token_mask))
        token_mask[token_mask <= 0.0] = 0 # make the image binary
        token_mask[token_mask > 0.0] = 1 # make the image binary
        token_mask = token_mask.reshape(1, size[0], size[1])
        return token_mask
    
    def __get_segmentation__(self, sample):
        segmentation = sample["segmentation"]
        image = self.__get_image__(sample)

        # Get the segmentation mask
        rles = mask.frPyObjects(segmentation, image.size[1], image.size[0])
        rle = mask.merge(rles)
        m = mask.decode(rle)

        return m

    def __getitem__(self, idx):
        """ returns a sample: image, sentence, bbox, token_mask, segmentation """
        sample = self.annotations_df.iloc[idx]

        image = self.__get_image__(sample)
        sentence = self.__get_sentence__(sample)
        bbox = self.__get_bbox__(sample)
        token_mask = self.__get_token_mask__(sample)
        segmentation = self.__get_segmentation__(sample)

        image_width, image_height = image.size

        # Apply the transform to the image
        resized_image = cv2.resize(np.array(image), (224, 224))
        resized_image = Image.fromarray(resized_image)
        if self.transform != None:
            resized_image = self.transform(resized_image)
        
        # Apply the tokenizer to the sentence
        if self.tokenizer != None:
            sentence = self.tokenizer(sentence)

        # Resize the bbox
        resized_bbox = [bbox[0] / image_width * 224, bbox[1] / image_height * 224, bbox[2] / image_width * 224, bbox[3] / image_height * 224]
        resized_bbox = [int(i) for i in resized_bbox]
        resized_bbox = torch.tensor(resized_bbox).unsqueeze(0)

        # Apply the transform to the segmentation
        resized_segmentation = cv2.resize(segmentation, (224, 224))
        resized_segmentation = torch.tensor(resized_segmentation).unsqueeze(0)

        return {'image': resized_image,                             # tensor of shape (3, 224, 224)
                'sentence': sentence,                               # tensor
                'bbox': resized_bbox,                               # tensor of shape (1, 4)
                'token_mask': token_mask,                           # tensor of shape (1, 14, 14)
                'segmentation': resized_segmentation}               # tensor of shape (1, 224, 224)

    def __show_sample__(self, idx): 
        """ show a sample from the dataset """
        sample = self.annotations_df.iloc[idx]
        image = self.__get_image__(sample)
        sentence = self.__get_sentence__(sample)
        bbox = self.__get_bbox__(sample)
        token_mask = self.__get_token_mask__(sample)

        sample = self.__getitem__(idx)
        resized_image = sample['image']
        resized_bbox = sample['bbox'][0].numpy()
        resized_segmentation = sample['segmentation']
        token_mask = sample['token_mask']

        print('-------------------------------------------------')
        print("Target Object: ", sentence)
        print('-------------------------------------------------')

        plt.figure(figsize=(6, 6))
        # Plot the original image
        plt.subplot(2, 2, 1)
        plt.title("Original Image")
        int_bbox = [int(i) for i in bbox]
        image_bbox = np.array(image.copy())
        image_bbox = cv2.rectangle(image_bbox, (int_bbox[0], int_bbox[1]), (int_bbox[0]+int_bbox[2], int_bbox[1]+int_bbox[3]), (0, 255, 0), 2)
        plt.imshow(image_bbox)

        # Plot preprocessed image
        plt.subplot(2, 2, 2)
        plt.title("Preprocessed Image")
        plt.imshow(resized_image.numpy().transpose(1, 2, 0))
        plt.axis('off')

        # Plot the resized image
        plt.subplot(2, 2, 3)
        plt.title("Resized Image")
        resized_image_bbox = np.array(image.resize((224, 224)).copy())
        resized_image_bbox = cv2.rectangle(resized_image_bbox, (resized_bbox[0], resized_bbox[1]), (resized_bbox[0]+resized_bbox[2], resized_bbox[1]+resized_bbox[3]), (0, 255, 0), 2)
        plt.imshow(resized_image_bbox)
        plt.imshow(resized_segmentation[0], alpha=0.5)

        # Plot the token mask
        plt.subplot(2, 2, 4)
        plt.title("Token Mask")
        plt.imshow(token_mask[0])
        plt.show()
