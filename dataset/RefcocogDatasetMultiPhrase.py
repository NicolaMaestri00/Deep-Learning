import json
import clip
import pandas
import torch
from skimage import io
from torch.utils.data import random_split
from typing import Sequence, Union

from PIL import Image
from torch.utils.data import Dataset, DataLoader

MAX_NUM = 3


class RefcocogDataset(Dataset):
    def __init__(self, base_path, split=None, transform=None, tokenization=None):
        annotation_path = base_path + "/annotations/"

        self.IMAGES_PATH = base_path + "/images/"
        self.transform = transform
        self.tokenization = tokenization

        tmp_annotations = pandas.read_pickle(annotation_path + "refs(umd).p")
        tmp_instances = json.load(open(annotation_path + "instances.json", "r"))

        annotations_dt = pandas.DataFrame.from_records(tmp_annotations) \
            .filter(items=["image_id", "split", "sentences", "ann_id"])

        instances_dt = pandas.DataFrame.from_records(tmp_instances['annotations'])

        # Add Explode to separate list-like sentences column and use them as separate samples
        self.annotations = annotations_dt \
            .merge(instances_dt[["id", "bbox", "area"]], left_on="ann_id", right_on="id") \
            .drop(columns="id")


        if split is not None:
            if split.lower() == 'train':
                self.annotations = self.__get_train_annotations()

            if split.lower() == 'test':
                self.annotations = self.__get_test_annotations()

    def splitTrainVal(self, lengths: Sequence[Union[int, float]]):
        return random_split(self, lengths)

    def getImage(self, sample):
        id = sample['idx'][0].item()
        item = self.annotations.iloc[id]
        image = self.__getimage(item.image_id)

        return image

    def getSentences(self, sample):
        id = sample['idx'][0].item()
        item = self.annotations.iloc[id]

        return self.__extract_sentences(item.sentences)

    def __get_train_annotations(self):
        return self.annotations[self.annotations.split == "train"].reset_index()

    def __get_test_annotations(self):
        return self.annotations[self.annotations.split == "test"].reset_index()

    def __getimage(self, id):
        return Image.open(self.IMAGES_PATH + "COCO_train2014_" + str(id).zfill(12) + ".jpg")

    def __extract_sentences(self, sentences):
        return [f"a photo of a {s['sent']}" for s in sentences]

    def __tokenize_sents(self, sentences):
        return [self.tokenization(s) for s in sentences]

    def __len__(self):
        return self.annotations.shape[0]

    def __getitem__(self, idx):
        # Return {sample, sentence, id}, bbox
        # Return single sentence, probably preprocess needed so we do not waste data
        
        item = self.annotations.iloc[idx]
        image = self.__getimage(item.image_id)
        sentences = self.__extract_sentences(item.sentences)
        
        padding = MAX_NUM - len(sentences)
        print(padding)

        if self.transform:
            image = self.transform(image)

        if self.tokenization:
            sentences = self.__tokenize_sents(sentences)

        # Add padding to array of sentences
        for i in range(padding):
            sentences.append(0)
            # sentences.append(torch.zeros(sentences[0].size()))

        sample = {'idx': idx, 'image': image, 'sentences': sentences}

        return sample, item.bbox

def remove_padding(sentences):
    return [s for s in sentences if torch.mean(s, dtype=float) != 0]


if __name__ == "__main__":
    _, preprocess = clip.load("ViT-B/32")

    dataset = RefcocogDataset("../Dataset/refcocog", split="train", tokenization=clip.tokenize, transform=preprocess)
    train, val = random_split(dataset, [0.8, 0.2])
    train_dataloader = DataLoader(train)

    train_features, train_bbox = next(iter(train_dataloader))
    dataset.getImage(train_features).show()
    print(train_features['sentences'])
    print(remove_padding(train_features['sentences']))
    print(dataset.getSentences(train_features))
    print(len(train))
    print(len(val))
