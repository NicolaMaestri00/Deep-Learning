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
