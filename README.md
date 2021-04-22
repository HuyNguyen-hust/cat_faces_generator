# cat_faces_generator
A project that learns how to generate cat face images

# Datasets
Data used was downloaded from [Animal Faces](https://www.kaggle.com/andrewmvd/animal-faces)  
I only used cat data for this project

# How to run
Store all the cat images data to the folder cat  
to train the model: python cat_gan.py  
After each epoch it generates an image of cats and stores them into folder generated

# Some generated images
20 epochs:
![alt text](https://github.com/HuyNguyen-hust/cat_faces_generator/blob/main/generated/generated-images-0020.png)

40 epochs:
![alt text](https://github.com/HuyNguyen-hust/cat_faces_generator/blob/main/generated/generated-images-0040.png)


60 epochs:
![alt text](https://github.com/HuyNguyen-hust/cat_faces_generator/blob/main/generated/generated-images-0060.png)


80 epochs:
![alt text](https://github.com/HuyNguyen-hust/cat_faces_generator/blob/main/generated/generated-images-0080.png)

# Result 
Generator Loss vs Discriminator Loss over 80 epochs  
![alt text](https://github.com/HuyNguyen-hust/cat_faces_generator/blob/main/loss.jpg)
