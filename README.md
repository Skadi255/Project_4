# Project 4 – 12 Lead EKG Image 

### Group 3 (Jose Gonzalez, Shannon Williams, Nancy Ulloa, Arle Alcid) 

## Introduction
For this project we were interested in creating an image machine learning model that classifies 12 lead EKG png files as either normal or abnormal.

## Data Sources
For this project, we pulled our data from PTB-XL, a large publically available electrocardiography dataset that was published to Physio Net. The data was collected with devices from Schiller AG over the course of a seven-year period between October 1989 to June 1996. The dataset contained over 21,799 ekg readings complimented by extensive metadata on patient demographics, likelihoods for diagnostic ECG statements, and cardiac infarction characteristics. https://physionet.org/content/ptb-xl/1.0.3/#files-panel

## Data Cleaning
The 12 lead EKG data were held in individual .hea files for each clinic trial particiant.  The metadata set also included a csv file containing a summary of each ekg reading including patient demographics information, filenames corresponding to the individual .hea ekg file paths, and scp_codes referring to the classifications of cardiac activity from the ekg readings. In order to to reduce the amount of data for our model we removed all clinic participants that had incomplete data in the csv file, and we used an indexer to only pull the first 1000 data points. Then we created a for loop simplifed that scp_codes column to label the ekg readings as either normal or abnormal. To generate the charts we created a for loop that matched the filename column to the file path of the .hea files and used the plot_wfdb() to generate the 12 lead ekg charts. An if statement was created inside the for loop the save the 12 lead ekg png files to their corresponding cardiac activity classifications folder, ekg_norm_png or ekg_abnorm_png. 

## Image Classification Models 

### Scikit-Learn
I created our first model using the scikit-learn package in python. I prepared the data by creating a for loop that iterates through all the images in the ekg_normal_png and ekg_abnorm_png folders and flattens the images from a matrix to a unidimentional array. Then I created a training and testing set using the train_test_split() funtion. Due to the limited processing power of my personal computer, I had to limit the training and testing data to 10% each. I set shuffle = Ture to remove as much bias as possible when reading the testing and training arrays an used the attribute stratify = labels to ensure the same proportion of normal and abnormal images were used from the original data to the training data. To train my images, I created multiple classifers using 3 gamma parameters (0.01, 0.001, 0.0001) and 4 C parameters (1, 10, 100, 1000) which would like hidden layers. By using the various parameters, the data iterates through 12 different classifers (as many combinations as gamma and C) and then chooses the best model. After running through the 12 classifiers the best model had C=1 and gamma=0.01 parameters, 52.27% of samples were correctly classifed.

### Tensor Flow - Google Colab
To try to make up for my computer inability to process the model I moved a copy of the data to my drive to get assistance from google colab. From there I created my batch sizes and structures the resolution of the image. Next using the tensorflow keras model I got my data and split it into eighty percent to train on (434 files) and twenty percent (86 files) to validate as the keras documentation puts it. The nature of how we classified the data the classes were set already one being normal ekgs reading and abnormal ekgs readings. Next we resize the images and place the images into the module. For my module our main focus was getting something to work so I used 3 layers set to relu and setting intermediately to resize and flatten the images to be processed. Then ran the model on 10 epochs. This model as expected was not very accurate with an accuracy score of sixty percent and many false negatives, but it did work as the building grounds to get a better understanding of image processing learning models and it enabled us to further optimize our model.

### Tensor Flow - Teachable Machines
To create a model that tests the confidence of predicting the classifications of individal images, I used the help of the image project with teachablemachines.com/train. I used the standard image model and imported 200 images from the normal and abnormal image folders and trained a model using 25 epochs, 32 batch size, and 0.001 learning rate. I imported the code from teaching machines into jupyter notebook and randomly selected photos to test in the model.
#####
Normal EKG:
- ekg_norm_png/01107_lr.png had a 95.24% confidence that the image is normal.
- ekg_norm_png/01505_lr.png had a 76.04% confidence that the image is normal.
- ekg_norm_png/01525_lr.png had a 86.21% confidence that the image is normal.
- ekg_norm_png/01553_lr.png had a 85.25% confidence that the image is normal.
- ekg_norm_png/01578_lr.png had a 92.84% confidence that the image is normal.
- ekg_norm_png/01629_lr.png had a 97.95% confidence that the image is normal.
- ekg_norm_png/01639_lr.png had a 77.80% confidence that the image is normal.
- ekg_norm_png/01645_lr.png had a 70.39% confidence that the image is normal.
- ekg_norm_png/01660_lr.png had a 96.95% confidence that the image is normal.
- ekg_norm_png/01664_lr.png had a 94.13% confidence that the image is normal.

Abnormal EKG:
- ekg_abnorm_png/01109_lr.png had a 99.70% confidence that the image is abnormal.
- ekg_abnorm_png/01111_lr.png had a 95.48% confidence that the image is abnormal.
- ekg_abnorm_png/01136_lr.png had a 99.99% confidence that the image is abnormal.
- ekg_abnorm_png/01146_lr.png had a 98.42% confidence that the image is abnormal.
- ekg_abnorm_png/01150_lr.png had a 97.52% confidence that the image is abnormal.
- ekg_abnorm_png/01151_lr.png had a 77.27% confidence that the image is abnormal.
- ekg_abnorm_png/01154_lr.png had a 91.79% confidence that the image is abnormal.
- ekg_abnorm_png/01156_lr.png had a 99.99% confidence that the image is abnormal.
- ekg_abnorm_png/01160_lr.png had a 79.75% confidence that the image is abnormal.
- ekg_abnorm_png/01183_lr.png had a 99.74% confidence that the image is abnormal.

### Nueral Network Model
A Neural Network model using a perceptron algorithm was created in order to classify whether an EKG sould read as a regular, or arrhtyhmic heartbeat
* Preprocess
  * -	The libraries used to create this model were Pandas, StandardScaler and train_test_split from sklearn, matplotlib

### 


### Limitations
1. 

### Outside Help 



### Contributions
Jose Gonzalez - Tensor Flow Colab Model
Shannon Williams - Optimization Model
Nancy Ulloa - Pre-trained Model
Arle Alcid - Skikit-Learn Model

