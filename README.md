# Deep Learning

## Final Assignent: *Transfer Learning for Visual Grounding*

Visual Grounding aims to locate an object in an image based on a natural language query. Our work leverages CLIP's image-text alignment to build new frameworks for this task. We first introduce a baseline that combines YOLO with CLIP by selecting the candidate object with the highest cosine similarity to the query. Next, we propose a detector-free approach that uses heatmaps for target retrieval and SAM to generate accurate bounding boxes. Finally, we customize CLIP to produce a low-level probability map refined through upsampling convolutions. Fine-tuning CLIP enables strong performance with minimal training, offering a promising direction for Visual Grounding research.

**Further Details**: refer to this [link]
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


## Course Syllabus
- Shallow/Deep Neural Networks
- Fitting Models (Loss functions, optimizers, initializations, regularization)
- Model Evaluation (Evaluation Metrics)
- Architecuteres (Convolutional Neural Networks, Residual Neural Networks, Transformers)
- Generative Adversial Networks (GAN)
- Variational Autoencoders
- Diffusion Models
- Normalizing Flows


_Project References_
1. Redmon, Joseph, Santosh Divvala, Ross Girshick, and Ali Farhadi. ["You only look once: Unified, real-time object detection."](https://ieeexplore.ieee.org/document/7780460) In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 779-788. 2016.

2. Radford, Alec, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry et al. "Learning transferable visual models from natural language supervision." In International conference on machine learning, pp. 8748-8763. PMLR, 2021.

3. Li, Junnan, Dongxu Li, Caiming Xiong, and Steven Hoi. "Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation." In International Conference on Machine Learning, pp. 12888-12900. PMLR, 2022.

4. Kirillov, Alexander, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao et al. "Segment anything." arXiv preprint arXiv:2304.02643 (2023).

5. Kim, Seoyeon, Minguk Kang, and Jaesik Park. ["RISCLIP: Referring Image Segmentation Framework using CLIP."](https://arxiv.org/pdf/2306.08498v2) arXiv preprint arXiv:2306.08498 (2023).

6. Vaswani, Ashish, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin. "Attention is all you need." Advances in neural information processing systems 30 (2017).
