# Deep Learning

## Final Assignent: *Transfer Learning for Visual Grounding*

Visual Grounding aims to locate an object in an image based on a natural language query. Our work leverages CLIP's image-text alignment to build new frameworks for this task. We first introduce a baseline that combines YOLO with CLIP by selecting the candidate object with the highest cosine similarity to the query. Next, we propose a detector-free approach that uses heatmaps for target retrieval and SAM to generate accurate bounding boxes. Finally, we customize CLIP to produce a low-level probability map refined through upsampling convolutions. Fine-tuning CLIP enables strong performance with minimal training, offering a promising direction for Visual Grounding research.

**Authors**: Nicola Maestri, Francesco Vaccari, Gabriele Stulzer

### Baseline Approach: *YOLO + CLIP*

Our baseline approach combines the YOLO object detector with CLIP to identify the target object. For each candidate proposed by YOLO, we compute the cosine similarity with the text query and select the object with the highest similarity as the final prediction.

[*Missing picture*]

### Detector-Free Approach: *BLIP + SAM*

This alternative pipeline removes the need for an object detector by leveraging the Lavis library to extract heat maps based on text instances. These heat maps highlight regions with high confidence of containing the target object, and the Segment Anything Model (SAM) is then used to generate a precise mask and corresponding bounding box.

**Example**

![1](https://github.com/NicolaMaestri00/Deep-Learning/assets/104208237/142634d3-4b99-4c1f-9f26-1ad7f78323a1)

**Step 1: heatmap extraction based on the text query**

![2](https://github.com/NicolaMaestri00/Deep-Learning/assets/104208237/68e7cd10-1532-4b37-b297-c348fcd776f5)

**Step 2: Bounding Box extraction based on SAM**

![3](https://github.com/NicolaMaestri00/Deep-Learning/assets/104208237/a22cd2a5-c511-4a6b-8350-5067d8c59f68)

### Customized CLIP Framework: *RisClip*

Our implementation of RisClip is a two-stage neural network composed of a Locator and a Refiner. The Locator uses a frozen CLIP ViT16 backbone with cross-attention adapters to compute a low-level probability map that highlights target regions. The Refiner, consisting of convolutional layers, upsamples this map to produce an accurate bounding box. This design effectively exploits CLIP's image-text alignment with minimal training, achieving notable accuracy on the RefCocog dataset.

![5](https://github.com/NicolaMaestri00/Deep-Learning/assets/104208237/e4e1e6c1-bdcc-418a-a2b3-f76c1d01d0a9)

_References_

1. ["CLIP"](https://arxiv.org/abs/2103.00020), Radford et al., 2021
2. ["YOLO"](https://ieeexplore.ieee.org/document/7780460), Redmon et al., 2016
3. ["BLIP"](https://arxiv.org/abs/2201.12086), Li et al., 2022
4. ["SAM"](https://arxiv.org/abs/2304.02643), Kirillov et al., 2023
5. ["RISCLIP"](https://arxiv.org/pdf/2306.08498v2), Kim et al., 2023

## Course Syllabus
- Shallow/Deep Neural Networks
- Fitting Models (Loss functions, optimizers, initializations, regularization)
- Model Evaluation (Evaluation Metrics)
- Architecuteres (Convolutional Neural Networks, Residual Neural Networks, Transformers)
- Generative Adversial Networks (GAN)
- Variational Autoencoders
- Diffusion Models
- Normalizing Flows

_References_
1. [Understanding Deep Learning](https://mitpress.mit.edu/9780262048644/understanding-deep-learning/), Simon J. D. Prince
