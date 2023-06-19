# TensorFlow Deep Learning Repository for Computer Vision and Natural Language Processing

Welcome to my TensorFlow-based deep learning repository on GitHub! This repository is dedicated to advancing computer vision and natural language processing projects. Currently, I am focusing on computer vision tasks, including image classification and image segmentation, while also planning to explore natural text processing projects in the near future. Additionally, I aim to incorporate PyTorch into my workflow for these tasks, broadening the range of deep learning frameworks utilized.

## Key Features

1. **Computer Vision Projects:**  
   Within this repository, you will find a collection of projects focused on computer vision. I will be developing models for image classification and image segmentation using TensorFlow. By leveraging various techniques, such as feature extraction and fine-tuning, I will progressively enhance the models' performance. Notably, I will explore state-of-the-art algorithms like VGG16 and Xception, as well as other cutting-edge approaches, to achieve outstanding results in computer vision tasks. Additionally, I will incorporate batch normalization layers and residual connections to improve accuracy and model robustness. Stay tuned for more exciting advancements in this repository!

2. **Natural Language Processing Projects (Upcoming):**  
   In the near future, I plan to expand this repository to encompass natural language processing projects. Through the power of TensorFlow, I will delve into text classification, sentiment analysis, language generation, and more. Additionally, I will explore PyTorch as an alternative framework for these projects, allowing for a comprehensive understanding of deep learning across different platforms.

3. **Progressive Model Development:**  
   A key aspect of this repository is the iterative model development process. Beginning with basic models, I will gradually introduce advanced techniques to continually improve their performance. Through the integration of feature extraction, fine-tuning, and other optimization strategies, I aim to create models that excel in accuracy, efficiency, and versatility.

4. **Best Practices in Architecture:**  
   Implementing the latest and most effective architectural practices is a primary goal of this repository. I will stay up-to-date with the latest research and industry trends, ensuring that the models developed here adhere to the state-of-the-art in deep learning architecture design. By following these best practices, I strive to create models that are not only highly accurate but also scalable and interpretable.

Join me on this exciting journey into the world of deep learning with TensorFlow. Together, we will explore cutting-edge computer vision projects, venture into natural language processing, and utilize the best practices in deep learning architecture to push the boundaries of AI research and applications.


## Image Classification

 **Cats vs Dogs** a classification problem.<br />
Trying to emulate a real world problem by selecting only a fraction of the images available in this dataset. Using different architectures to get a remarkable generalization. I will start from a basic model of just a stack of Conv2D layers, and gradually make it more complex and deep by adding residual connection and batch normalization layers and at the end try features extraction and  fine-tune a VGG16 model.

 - [Basic Model](https://nbviewer.jupyter.org/github/antirrabia/Deep-Learning/blob/main/notebooks/CatsVsDogs_Basic.ipynb) - This is a very basic model composed of some *Conv2D* and *MaxPooling2D* layers. To reduce overfitting we include a *Dropout(0.5)* layer. We can see that filters increase in every layer while the features decrease. We started with 180, 180, and ended up with a 7, 7. This is a very common pattern to follow.
 - [Basic Model using augmentation](https://nbviewer.jupyter.org/github/antirrabia/Deep-Learning/blob/main/notebooks/CatsVsDogs_UsingAugmentation.ipynb) - The same architecture as the previous, but we incorporate an example of data augmentation to decrease even more over-fitting. 
 - [Feature extraction using VGG16 (No Augmentation)](https://github.com/antirrabia/Deep-Learning/blob/main/notebooks/CatsVsDogs_PreTrainedModel%28fast%29.ipynb) - With architecture, we try to increase the model's ability to generalize, by using the VGG16 model to extract features. I will simple run the data over the VGG16 model to get and store new. features 
 - [Feature extraction using VGG16 (With Augmentation)](https://github.com/antirrabia/Deep-Learning/blob/main/notebooks/CatsVsDogs_PreTrainedModel(UsingAugmentation).ipynb) - This time I will chain the VGG16 model in my model, so I can uses data augmentation to acquire a better performance.
 - [Fine-Tuning VGG16](https://github.com/antirrabia/Deep-Learning/blob/main/notebooks/CatsVsDogs(Fine-tuning-VGG16).ipynb) - Here I will fine-tune VGG16. I will freeze all of its layers, except for the last 3. Then I will retrain the entire model, using a small learning rate to avoid making too many changes to the parameter weights of VGG16.
- [Residual conections, batch normalization and distributed training on multiple GPUs](https://github.com/antirrabia/Deep-Learning/blob/main/notebooks/CatsVsDogs(DistributedTraining_ResidualConnections_BatchNormalization).ipynb) - I implemented architectural best practices, such as residual connections and batch normalization. I also defined a mirror strategy to train the model using multiple GPUs.

## Image Segmentation

**Oxford-IIIT Pets dataset** The dataset consists of images of 37 pet breeds, with 200 images per breed (~100 each in the training and test splits). Each image include the corresponding labels, and pixel-wise masks. The masks are class-labels for each pixel. Each pixel is given one of three categories.

- class 1: Pixel beloging to the pet.
- class 2: Pixel bordering the pat.
- class 3: None of the above/a surrounding pixel.

This project aims to provide a range of segmentation models, starting from basic architectures using Conv2D layers and Conv2DTranspose, to more advanced models incorporating state-of-the-art architectures like MobileNetV2 and other cutting-edge approaches.

Segmentation is a critical task in computer vision that involves dividing an image into meaningful regions or segments, enabling precise object recognition, understanding, and analysis. The Imaging Segmentation Toolkit offers a collection of models to facilitate accurate and efficient segmentation across diverse imaging applications.

 - [Basic Model](https://github.com/antirrabia/Deep-Learning/blob/main/notebooks/Basic_Segmentation_Model.ipynb) This is a simple model using Conv2D and Conv2DTranspose layers, along with some data augmentation.