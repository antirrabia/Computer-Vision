# Deep-Learning
Computer Vision using **TensorFlow**

### Classification
---
> **Cats vs Dogs** a classification problem.
Trying to emulate a real world problem by selecting only a fraction of the images available in this dataset. Using different architectures to get a remarkable generalization. I will start from a basic model of just a stack of Conv2D layers, and gradually make it more complex and deep by adding residual connection and batch normalization layers and at the end try features extraction and  fine-tune a VGG16 model.

 - [Basic Model](https://nbviewer.jupyter.org/github/antirrabia/Deep-Learning/blob/main/notebooks/CatsVsDogs_Basic.ipynb) - This is a very basic model composed of some *Conv2D* and *MaxPooling2D* layers. To reduce overfitting we include a *Dropout(0.5)* layer. We can see that filters increase in every layer while the features decrease. We started with 180, 180, and ended up with a 7, 7. This is a very common pattern to follow.
 - [Basic Model using augmentation](https://nbviewer.jupyter.org/github/antirrabia/Deep-Learning/blob/main/notebooks/CatsVsDogs_UsingAugmentation.ipynb) - The same architecture as the previous, but we incorporate an example of data augmentation to decrease even more over-fitting. 
 - [Feature extraction using VGG16 (No Augmentation)](https://github.com/antirrabia/Deep-Learning/blob/main/notebooks/CatsVsDogs_PreTrainedModel%28fast%29.ipynb) - With architecture, we try to increase the model's ability to generalize, by using the VGG16 model to extract features. I will simple run the data over the VGG16 model to get and store new. features 
 - [Feature extraction using VGG16 (With Augmentation)](https://github.com/antirrabia/Deep-Learning/blob/main/notebooks/CatsVsDogs_PreTrainedModel(UsingAugmentation).ipynb) - This time I will chain the VGG16 model in my model, so I can uses data augmentation to acquire a better performance.
 - [Fine-Tuning VGG16](https://github.com/antirrabia/Deep-Learning/blob/main/notebooks/CatsVsDogs(Fine-tuning-VGG16).ipynb) - Here I will fine-tune VGG16. I will freeze all of its layers, except for the last 3. Then I will retrain the entire model, using a small learning rate to avoid making too many changes to the parameter weights of VGG16.
- [Working on, residual block, batch normalization and distributed training on multiple GPUs ...](...)