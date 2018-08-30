# age-gender-determination-of-a-person-with-image-data-Convloution-netural-network-
CNN (used VGG 16 model and trained the weights from scratch)

This repositry has 1000 images and python files & metadata in csv format

clean_data.py would refer filenames.csv and age.csv to get the dataset .It will also check for the file with anamolies in dataset. 
It will resize all the data to 224,224,3 which would be the input to the model

creat_dataset.py splits the dataset to train test and validation data . It can also create tfrecord files of the dataset 

train_vgg16_model.py uses the dataset created and uses VGG16 architecture to start training the model and test it.

Used Keras library for the implementation and i have tried out this code on matab wiht alexnet and , i also tried out transfer learning
in tensorlayer.

For further info please contact me on nitheesh@asu.edu or +1-480-498-1497

If you want to increase the dataset you can dig in wiki_crop which has about 65000+ images.
