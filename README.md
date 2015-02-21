# PMLProject

## Practical Machine Learning Coursera Project

###Weight Lifting Exercises Performance Prediction

In this report Weight Lifting Exercises dataset is used to train machine learning algorithms to predict how well activities were performed. The dataset contains data from sensors in the users' glove, armband, lumbar belt and dumbbell, participants were asked to perform barbell lifts correctly and incorrectly in 5 different ways, these sensor data are then used to predict their performance. 

First, the dataset is explored, processed to transform into a tidy dataset to minimize runtime, then machine learning algorithms are built for prediction. Since the process is very time consuming, extra efforts are put into removing unnecessary data to reduce runtime.

Generalized Boosted Regression Models (gbm) and Random Forest (rf) are trained and tested to select the best one. Random Forest, selected for offering  higher accuracy, is cross validated and then tested for out of sample errors.

Finally, the Random Forest model is applied to SubmitTest data stored in pml-testing.csv to predict the performance of those 20 observasions, the results are saved in 20 files to be submitted for this project.

