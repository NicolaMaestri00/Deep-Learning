import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import model.locator.clip as clip
from dataset.RefcocogDataset import RefcocogDataset
from model.refiner.refiner import Refiner
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.patches import Rectangle

import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##########################
###### LOADL MODELS ######
##########################

locator_path = "./runs/ScaledImages/latest.pth"
refiner_path = "./model/epoch6/refiner_epoch_1.pth"

locator, preprocess = clip.load("ViT-B/16")
locator.init_adapters()
locator.load_state_dict(torch.load(locator_path, map_location=device))
locator = locator.to(device)
locator.to(torch.float32)

refiner = Refiner()
refiner.load_state_dict(torch.load(refiner_path, map_location=device))
refiner = refiner.to(device)
refiner.to(torch.float32)


# extract the bounding box from the segmentation map from the refiner
def extract_bbox(out):
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


# used to compute accuracy given the ground truth box and the predicted box
def computeIntersection(fx1, fy1, fx2, fy2, sx1, sy1, sx2, sy2):
    dx = min(fx2, sx2) - max(fx1, sx1)
    dy = min(fy2, sy2) - max(fy1, sy1)
    if (dx>=0) and (dy>=0):
        area = dx*dy
    else:
        area = 0
    return area


# compute IoU accuracy metric
def compute_accuracy(out, bbox):
    x, y, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
    x_min, y_min, x_max, y_max = extract_bbox(out)

    intersection = computeIntersection(x_min, y_min, x_max, y_max, x, y, x2, y2)
    area1 = (x_max-x_min)*(y_max-y_min)
    area2 = (x2-x)*(y2-y)
    union = area1 + area2 - intersection
    return intersection / union


##########################
###### EVALUATION ########
##########################

# evaluate on the test split of dataset
batch_size = 2
test_dataset = RefcocogDataset("./refcocog", split="test", transform=preprocess)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

acc = []
for i, (sample, bbox) in enumerate(test_loader):
    image = sample['image'].to(device)
    sentences = clip.tokenize(sample['sentences']).to(device)
    
    # locator returns the patch-level 16x16 probability map and 
    # features from the first 4 layers of CLIP visual encoder
    maps, fv = locator.encode(image, sentences)
    
    # refiner returns the pixel-level segmentation map
    out = refiner(maps, fv)

    for idx in range(out.shape[0]): # for each image in the batch...
        # ground truth bounding box
        box = bbox['bbox'][0][idx].item(), bbox['bbox'][1][idx].item(), bbox['bbox'][2][idx].item(), bbox['bbox'][3][idx].item()
        
        print(f'\tSent: {sample["sentences"][idx]}')

        accuracy = compute_accuracy(out[idx], box)
        acc.append(accuracy)

        print(f'[{i+1:^4}/{len(test_loader)}]\t[{idx+1}/{batch_size}] : {accuracy}')
        
        # show the results with image, sentence and ground truths
        plt.figure()
        plt.subplot(2, 3, 1)
        plt.imshow(maps[idx].squeeze(0).squeeze(0).detach().cpu().numpy())
        plt.subplot(2, 3, 2)
        plt.imshow(out[idx].squeeze(0).squeeze(0).detach().cpu().numpy())
        x_min, y_min, x_max, y_max = extract_bbox(out[idx])
        rect = Rectangle((x_min, y_min), x_max-x_min, y_max-y_min, linewidth=1, edgecolor='r', facecolor='none')
        plt.gca().add_patch(rect)
        plt.subplot(2, 3, 3)
        plt.imshow(sample['image'][idx].permute(1, 2, 0).numpy())
        rect = Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], linewidth=1, edgecolor='b', facecolor='none')
        plt.gca().add_patch(rect)
        plt.subplot(2, 3, 4)
        plt.imshow(bbox['gt'][idx])
        plt.subplot(2, 3, 5)
        plt.imshow(bbox['gt_refiner'][idx])
        plt.show()
    
print(f'\nAccuracy : {sum(acc)/len(acc)}')