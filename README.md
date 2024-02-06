# Machine-Learning

<!--
## Classic Machine Learning Techniques
-->

## Deep Learning
* Supervised Learning
    -  Shallow Neural Networks
    -  Deep Neural Networks
    -  Loss Functions
    -  Fitting Models
    -  Gradients and Initialization
    -  Measuring Performance
    -  Regularization
    -  Convolutional Neural Networks (CNN)
    -  Residual Networks (Res-Net)
    -  Transformers
    -  Graph Neural Networks

* Unsupervised learning
    -  Generative Adversial Networks (GAN)
    -  Normalizing Flows
    -  Variational Autoencoders
    -  Diffusion Models

* Reinforcement Learning

## Assignment
### Abstract
Visual Grounding is a challenging task which aims to locate an object in an image based on a natural language query. Despite impressive advances in computer vision and natural language processing, establishing meaningful connections between distinct elements of images and text to get a good comprehension of context is still a big research area. In our work, we explored some new strategies to solve the problem by laveraging the image-text alignment of Clip as a foundation model for new frameworks specialized in Visual Grounding.

### Introduction
First, we propose a baseline that combines the object detector Yolo [1] and CLIP [2] model without any other component. For each candidate object proposed by Yolo is computed a cosine similarity with the text query and the most similar one is chosen as output prediction. We then tried to break free from the object detector developing a framework based on heatmaps [3] to retrieve the target object and on SAM [4] to draw a proper bounding box. We finally implemented a new framework obtained by customizing the CLIP model to compute a low-level probability map from which we find a bounding box through some up-sampling convolutions [5]. Fine-tuning allowed us to exploit Clip strengths without the need for heavy training, therefore beyond the results, it seems to be the most promising direction.

### Related Work
- Visual Grounding
- Referring Image Segmentation
- Adapters and Fine-tuning

### Baseline BLIP + SAM
This baseline presents an alternative pipeline that removes the necessity for an object detector.
We employ the Lavis library to extract from each image a set of heat maps based on the text instances. These are then used to identify some points with high confidence to belong to the target object. Finally, we utilize SAM, Segment Anything Model, to obtain a mask of the whole object and the corresponding bounding box to return as output.

**Example**

![1](https://github.com/NicolaMaestri00/Deep-Learning/assets/104208237/142634d3-4b99-4c1f-9f26-1ad7f78323a1)

**Step 1: heatmap extraction based on the text query**

![2](https://github.com/NicolaMaestri00/Deep-Learning/assets/104208237/68e7cd10-1532-4b37-b297-c348fcd776f5)

**Step 2: Bounding Box extraction based on SAM**

![3](https://github.com/NicolaMaestri00/Deep-Learning/assets/104208237/a22cd2a5-c511-4a6b-8350-5067d8c59f68)

### Our implementation of RisClip
In our project, my team and I focused on enhancing object localization within images by utilizing the image-text alignment capabilities of Clip.
The network is subdivided into a Locator, which computes a low‑level probability map, and a Refiner, which upsamples this map to return a bounding box. 
Clip Vit 16 is used as frozen backbone for the Locator, with adapters to enhance cross-attention, whereas the Refiner consists of a series of convolutional layers. This drastically reduced the amount of training needed while achieving a considerable accuracy on RefCocog.
![5](https://github.com/NicolaMaestri00/Deep-Learning/assets/104208237/e4e1e6c1-bdcc-418a-a2b3-f76c1d01d0a9)

