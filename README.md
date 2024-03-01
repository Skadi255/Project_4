# Project 4 – 12 Lead EKG Image Classifier

### Group 3 (Jose Gonzalez, Shannon Williams, Nancy Ulloa, Arle Alcid) 

## Introduction
For this project we were interested in creating an image machine learning model that classifies 12 lead EKG png files as either normal or abnormal. A 12 lead EKG captures the electrical activity of the heart to visualize normal and abnormal heart beats.

## Data Sources
For this project, we pulled our data from PTB-XL, a large publically available electrocardiography dataset that was published to Physio Net. The data was collected with devices from Schiller AG over the course of a seven-year period between October 1989 to June 1996. The dataset contained over 21,799 ekg readings complimented by extensive metadata on patient demographics, likelihoods for diagnostic ECG statements, and cardiac infarction characteristics. https://physionet.org/content/ptb-xl/1.0.3/#files-panel

## Data Cleaning (DataRetrev.ipynb)
The 12 lead EKG data were held in individual .hea files for each clinic trial particiant.  The metadata set also included a csv file containing a summary of each ekg reading including patient demographics information, filenames corresponding to the individual .hea ekg file paths, and scp_codes referring to the classifications of cardiac activity from the ekg readings. The csv file was imported into MongoDB and exported into jupyter notebook. In order to to reduce the amount of data for our model we removed all clinic participants that had incomplete data in the csv file, and we used an indexer to only pull the first 1000 data points. Then we created a for loop simplifed that scp_codes column to label the ekg readings as either normal or abnormal. To generate the charts we created a for loop that matched the filename column to the file path of the .hea files and used the plot_wfdb() to generate the 12 lead ekg charts. An if statement was created inside the for loop the save the 12 lead ekg png files to their corresponding cardiac activity classifications folder, ekg_norm_png or ekg_abnorm_png. 


## Image Classification Models 

### Tensor Flow - Google Colab
To try to make up for the computers inability to process the model, we moved a copy of the data to the computers drive to get assistance from google colab. From there we created batch sizes and structured the resolution of the image. Next, using the tensorflow keras model, we got the data and split it into eighty percent to train on (434 files) and twenty percent (86 files) to validate as the keras documentation puts it. Due to the nature of how we classified the data, the classes were set to either normal ekgs reading or abnormal ekgs readings. Then we resized the images and placed the images into the module. For the module our main focus was getting something to work, so we used 3 layers set to relu and setting intermediately to resize and flatten the images to be processed. Then ran the model on 10 epochs. This model as expected was not very accurate with an accuracy score of sixty percent and many false negatives, but it did work as the building grounds to get a better understanding of image processing learning models and it enabled us to further optimize our model.

### Scikit-Learn (skikit image classifier.ipynb)
A second model was created using the scikit-learn package in python. We prepared the data by creating a for loop that iterates through all the images in the ekg_normal_png and ekg_abnorm_png folders and flattens the images from a matrix to a unidimentional array. Then we created a training and testing set using the train_test_split() funtion. Due to the limited processing power of my personal computer, we had to limit the training and testing data to 10% each. We set shuffle = Ture to remove as much bias as possible when reading the testing and training arrays an used the attribute stratify = labels to ensure the same proportion of normal and abnormal images were used from the original data to the training data. To train the images, we created multiple classifers using 3 gamma parameters (0.01, 0.001, 0.0001) and 4 C parameters (1, 10, 100, 1000) which would like hidden layers. By using the various parameters, the data iterates through 12 different classifers (as many combinations as gamma and C) and then chooses the best model. Gamma values are the hyperparameter that controls the kernal coefficent for the RBF (Radial Basis Function) kermal, these calues represent dfferent levels of influence of individual training samples on the decision boundary. A lower gamma value indicates a smoother decision boundary, while a higher ganna value leads to a mode complex discision boundary. The C hyperparameter represents the regulation parameter. It controls the trad off between maximizing the margin and minimizing the classification error. A smaller C value leads to a wider margin by may allow more misclassifications, whie a larger C value results in a narrower margine but fewer misclassifications. After running through the 12 classifiers the best model had C=1 and gamma=0.01 parameters, 52.27% of samples were correctly classifed. 

### Tensor Flow - Teachable Machines Individual Images (tensor flow.ipynb)
To create a model that tests the confidence of predicting the classifications of individal images, we used the help of the image project with teachablemachines.com/train. We used the standard image model and imported 200 images from the normal and abnormal image folders and trained a model using 25 epochs, 32 batch size, and 0.001 learning rate. We imported the code from teaching machines into jupyter notebook and randomly selected photos to test in the model.

#####
Normal EKG:
- ekg_norm_png/01505_lr.png had a 76.04% confidence that the image is normal.
- ekg_norm_png/01525_lr.png had a 86.21% confidence that the image is normal.
- ekg_norm_png/01553_lr.png had a 85.25% confidence that the image is normal.
- ekg_norm_png/01578_lr.png had a 92.84% confidence that the image is normal.
- ekg_norm_png/01639_lr.png had a 77.80% confidence that the image is normal.
- ekg_norm_png/01645_lr.png had a 70.39% confidence that the image is normal.

Abnormal EKG:
- ekg_abnorm_png/01111_lr.png had a 95.48% confidence that the image is abnormal.
- ekg_abnorm_png/01151_lr.png had a 77.27% confidence that the image is abnormal.
- ekg_abnorm_png/01154_lr.png had a 91.79% confidence that the image is abnormal.
- ekg_abnorm_png/01160_lr.png had a 79.75% confidence that the image is abnormal.
- ekg_abnorm_png/01877_lr.png had a 90.67% confidence that the image is abnormal.
- ekg_abnorm_png/01849_lr.png had a 84.85% confidence that the image is abnormal.

### Deep Learning Optimized Image Classifier (MLM_tensorflow.ipynb)
To start we loaded in the data using a for loop that iterates through all ekg images in the normal and abnormal data folders and reshaped each image to have the same dimensions.  Then we preprocessed the data into 12 batches and then scaled the images via the map() function. We then split the data into a training set (40% of the original data) and a testing set (10% of the original data) and then shuffled it by skipping batches. To build the deep learning model, we created three hidden layers using the Conv2D() function, 'relu' and 'sigmoid' activations to fit the model, and 20 epochs. Overall, we wanted to see the loss of our model decrease and the accuracy of our model increase. Based on our loss plot we see there is an overall decrease in the loss of the (training) and val_loss (testing), indicating that our model is not overfitting the data. Based on our accuracy plot, we see there is an overall increase to over 75, indicating that our model is classifies images with over 75% accuracy. In order to evaluate the model we used the keras package and found our model had 72,72% precision and 69.56% recall.  Finally, we put our model to the test by loading in test images against our model. We randomly chose 'test_images/test_abnorm/01895_lr.png' and the image was correctly classifed as a normal ekg. 


### Neural Network Model (NeuralNetwork_model.ipynb)
A Neural Network model using a perceptron algorithm was created in order to classify whether an EKG sould read as a regular, or arrhtyhmic heartbeat
* Preprocess
   * The libraries used to create this model were Pandas, StandardScaler and train_test_split from sklearn, matplotlib
   * The DataFrame was cleaned out, to hold the most relevant columns pertaining to factors in which have a greater effect on heart health. patients with NAN readings for those columns, were dropped afterwards
   * X variable was made to hold the “ecg_id”, “age”, “sex”, “height” and “weight” of the patients
   * Y variable consisted of the corresponding “scp_codes_NORM” of each patient (whether their EKG was read as normal or arrhythmic)
   * “get_dummies” method was used on the new dataframe to convert the values into dummy/indicator variables
* Train
   * Model consisted of 3 hidden layers, with 15 nodes in the first layer, 10 in the second, and 5 in the third
   * ReLU activation function was used for the first hidden layer, and the “Sigmoid” activation function was used for the second and third hidden layer, as well as the Output layer.
* Validation
   * In order to validate the model, the X_train_scaled, and y_train variables were placed in a function with 25 epochs
* Predict
   * The model has a 2% loss with a level of 99.7% accuracy


### Limitations
1. Ram of personal laptops/desktops
 - Our biggest limitation was that the data we were trying to run our image classifiers on were too large and out personal devices simply did not have enought RAM or precessing pwoer to iteratur through all the data.
2. Image Sizes
- When we initially saved out images using the waveform basebase package in python we had to choose dimentions to save the images. In order to iterate through all of our images we had to decrease the dimensions of the images which caused the resolution of the photos to decline. We think this may have caused our model to inaccurately read the fine lines of the ekg images.
3. Training Size
  - The goal of the SkLearn model was to create multiple classification models (in our case 12 based on the gamma and C parameters) since there was too much data to iterate through we had to decrease our training and testing sizes. This likely affected the accuracy of our model since it was training on less than 100 images.
4. Variations of abnormal heart EKG readings 
  - Recall when we were cleaning the data we wanted to simplify all the image categories so we lumped together all images that were not classified as normal. There are many 'abnormal' cardiac conditions that may look normal on a 12 lead ekg reading. This may be the reason some images in the tensor flow individual image model predicted form images correctly with high condifidence by other images incorrectly. 


### Outside Help 
####
In order to complete our project we used help from Physio Net, our TAs, ChatGPT, and various youtube videos. 
- Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation [Online]. 101 (23), pp. e215–e220.
- Image classification with Python, Computer vierson engineer. https://www.youtube.com/watch?v=UuNGmhLpbCI
- Build a Depp CNN IMage Classifier with any images, Nicholas Renotte, https://www.youtube.com/watch?v=jztwpsIzEGc


### Contributions
Jose Gonzalez - Tensor Flow Colab Model
Arle Alcid - Skikit-Learn Model/Tensor Flow Individual Model
Shannon Williams - Optimization Model
Nancy Ulloa - Neural Network Model

