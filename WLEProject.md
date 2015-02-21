# Weight Lifting Exercises Performance Prediction
## Executive Summary

In this report Weight Lifting Exercises dataset is used to train machine learning algorithms to predict how well activities were performed. The dataset contains data from sensors in the users' glove, armband, lumbar belt and dumbbell, participants were asked to perform barbell lifts correctly and incorrectly in 5 different ways, these sensor data are then used to predict their performance. 

First, the dataset is explored, processed to transform into a tidy dataset to minimize runtime, then machine learning algorithms are built for prediction. Since the process is very time consuming, extra efforts are put into removing unnecessary data to reduce runtime.

Generalized Boosted Regression Models (gbm) and Random Forest (rf) are trained and tested to select the best one. Random Forest, selected for offering  higher accuracy, is cross validated and then tested for out of sample errors.

Finally, the Random Forest model is applied to SubmitTest data stored in pml-testing.csv to predict the performance of those 20 observasions, the results are saved in 20 files to be submitted for this project.

## Data Processing



### Load data



```r
# Get data
training <- read.csv("pml-training.csv", header = TRUE)
SubmitTest  <- read.csv('pml-testing.csv')
dim(training)
```

```
## [1] 19622   160
```
The dataset contains a lot of unnecessary data which should be removed to reduce processing time. 

### Clean data 

The above data from pml-training.csv are preprocessed to minimize the number of variables but still capture a large amount of variability in the following steps:


```r
# Remove columns with over a 90% of NAs
NAPerColumn <- apply(training,2,function(x) {sum(is.na(x))});
training <- training[,which(NAPerColumn <  nrow(training)*0.9)];  
# Remove near zero variance predictors
NearZeroColumns <- nearZeroVar(training, saveMetrics = TRUE)
training <- training[, NearZeroColumns$nzv==FALSE]
# Remove irrelevant columns
training<-training[,7:ncol(training)]
# Convert classe into factor
training$classe <- factor(training$classe)
```

### Partition Data

The processed data are split into three partitions:

- TrainData 60% for training, creating the model 

- TestData 20% for testing and cross validation

- OOSData 20% for testing out of sample errors


```r
TrainIndex <- createDataPartition(y = training$classe, p=0.6,list=FALSE);
TrainData <- training[TrainIndex, ];
TestOOSData <- training[-TrainIndex, ];

TestOOSIndex <- createDataPartition(y = TestOOSData$classe, p=0.5,list=FALSE);
TestData <- TestOOSData[TestOOSIndex, ];
OOSData <- TestOOSData[-TestOOSIndex, ];
dim(TestData)
```

```
## [1] 3923   53
```

```r
dim(OOSData)
```

```
## [1] 3923   53
```

## Create machine learning models

Generalized Boosted Regression Models (gbm) and Random Forest are trained and tested on variable classe to select the one with highest accuracy. 

### Generalized Boosted Regression Model (gbm)


```r
set.seed(2)
ModelGBM <-train(classe ~ ., method = 'gbm', data = TrainData)
GBMAccuracy <- predict(ModelGBM , TestData)
print(confusionMatrix(GBMAccuracy, TestData$classe))
```
Generalized Boosted Regression Model (gbm) Overall Statistics
                                          
                Accuracy : 0.9587         
                  95% CI : (0.952, 0.9647)
     No Information Rate : 0.2845         
     P-Value [Acc > NIR] : < 2.2e-16      
                                          
                   Kappa : 0.9477         


### Random Forest Model (rf)


```r
set.seed(7)
ModelRF <- randomForest(classe ~ ., data = TrainData, important=TRUE, proximity=TRUE)    
RFAccuracy <- predict(ModelRF, TestData)
print(confusionMatrix(RFAccuracy, TestData$classe))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1115    7    0    0    0
##          B    1  751    3    0    0
##          C    0    1  681    2    0
##          D    0    0    0  641    1
##          E    0    0    0    0  720
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9962          
##                  95% CI : (0.9937, 0.9979)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9952          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9991   0.9895   0.9956   0.9969   0.9986
## Specificity            0.9975   0.9987   0.9991   0.9997   1.0000
## Pos Pred Value         0.9938   0.9947   0.9956   0.9984   1.0000
## Neg Pred Value         0.9996   0.9975   0.9991   0.9994   0.9997
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2842   0.1914   0.1736   0.1634   0.1835
## Detection Prevalence   0.2860   0.1925   0.1744   0.1637   0.1835
## Balanced Accuracy      0.9983   0.9941   0.9973   0.9983   0.9993
```

Random Forest, selected for having higher accuracy, is cross validated and then tested for out of sample errors.

### Random Forest Cross Validation

Random Forest Model cross validation is performed 10 times and 10 folds.


```r
set.seed(9)
ControlF <- trainControl(method = "repeatedcv", number = 10, repeats = 10)
ModelRFCV <- randomForest(classe ~ ., data = TrainData, important=TRUE, proximity=TRUE, trControl = ControlF)    
RFCVaccuracy <- predict(ModelRFCV, TestData)
print(confusionMatrix(RFCVaccuracy, TestData$classe))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1114    6    0    0    0
##          B    2  753    5    0    0
##          C    0    0  679    3    0
##          D    0    0    0  640    1
##          E    0    0    0    0  720
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9957          
##                  95% CI : (0.9931, 0.9975)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9945          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9982   0.9921   0.9927   0.9953   0.9986
## Specificity            0.9979   0.9978   0.9991   0.9997   1.0000
## Pos Pred Value         0.9946   0.9908   0.9956   0.9984   1.0000
## Neg Pred Value         0.9993   0.9981   0.9985   0.9991   0.9997
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2840   0.1919   0.1731   0.1631   0.1835
## Detection Prevalence   0.2855   0.1937   0.1738   0.1634   0.1835
## Balanced Accuracy      0.9980   0.9949   0.9959   0.9975   0.9993
```

### Random Forest Out of Sample Test

Random Forest Model Out of Sample Test is performed on the reserved OOSData which contains 20% of the original data set not used in the model training nor previous test.


```r
set.seed(22)
RFOOSAccuracy <- predict(ModelRF, OOSData)
print(confusionMatrix(RFOOSAccuracy, OOSData$classe))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1112    3    0    0    0
##          B    3  752    3    0    0
##          C    1    4  681    7    0
##          D    0    0    0  635    2
##          E    0    0    0    1  719
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9939          
##                  95% CI : (0.9909, 0.9961)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9923          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9964   0.9908   0.9956   0.9876   0.9972
## Specificity            0.9989   0.9981   0.9963   0.9994   0.9997
## Pos Pred Value         0.9973   0.9921   0.9827   0.9969   0.9986
## Neg Pred Value         0.9986   0.9978   0.9991   0.9976   0.9994
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2835   0.1917   0.1736   0.1619   0.1833
## Detection Prevalence   0.2842   0.1932   0.1767   0.1624   0.1835
## Balanced Accuracy      0.9977   0.9944   0.9960   0.9935   0.9985
```

## Conclusion

As expected, Random Forest out of sample test shows accuracy slightly lower than the in sample result but still good for prediction.

## Prediction Assignment Submission

The Random Forest model trained and tested above is now applied to SubmitTest data stored in pml-testing.csv to predict the performance of those 20 observasions, the results are saved to 20 files to be submitted for this project.


```r
SubmitAnswer <- predict(ModelRF, SubmitTest)

pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

pml_write_files(SubmitAnswer)
```






