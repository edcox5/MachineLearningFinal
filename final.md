Machine Learning Final Project
================
Edward Cox
June 9, 2017

Using Machine Learning to Predict the Manner of Exercise
--------------------------------------------------------

6 volunteers were outfitted with accelerometers on the belt, arm, forearm, and dumbbell and asked to perform lifts correctly and incorrectly in 5 different ways. The goal of this project is to identify the manner (A-E) in which the exercise was being performed.

We will build a model using random forests to predict outcomes.

``` r
set.seed(10101)
d <- read.csv("pml-training.csv")

# Cross validation method - random partitions
inTrain  <- createDataPartition(d$classe, p = 0.7, list = FALSE)
training <- d[inTrain,]
testing  <- d[-inTrain,]

# First 7 cols do not apply to model so we can remove them
training <- training[,-c(1:7)]
```

Remove covariates with near-zero variation
------------------------------------------

Variables that exhibit little or no variation are useless for prediction and should be removed from the model.

``` r
my.nzv <- nearZeroVar(training, saveMetrics = TRUE)
training <- training[,rownames(my.nzv[my.nzv$nzv == FALSE,])]
```

Handle missing values
---------------------

The train function fails if there are missing values (NAs). We must decide how to handle them.

``` r
# Handle missing values
sum(colSums(is.na(training))/nrow(training) > 0) # num cols that contain missing values
```

    ## [1] 44

``` r
sum(colSums(is.na(training))/nrow(training) == 0) # num cols that do not contain missing values
```

    ## [1] 53

``` r
sum(colSums(is.na(training))/nrow(training) > .9) # num cols that contain more than 90% missing values
```

    ## [1] 44

``` r
## Since every column that contains some missing values contains almost no data, remove these columns
training <- training[,colSums(is.na(training))/nrow(training) == 0]
```

Apply changes to testing set
----------------------------

If we remove a column from the training set, we must also remove it from the testing set.

``` r
# Remove same columns from testing set that we removed from training set
testing <- testing[,names(training)]
```

Train model
-----------

Our random forest model uses 52 variables to predict the "classe" of exercise.

``` r
modFit <- train(classe ~ ., data = training, method = "rf", ntree = 4)
```

    ## Loading required package: randomForest

    ## randomForest 4.6-12

    ## Type rfNews() to see new features/changes/bug fixes.

    ## 
    ## Attaching package: 'randomForest'

    ## The following object is masked from 'package:ggplot2':
    ## 
    ##     margin

Expected out-of-sample error
----------------------------

Let's estimate the accuracy of the model by looking at how well it predicts the classe variable in the testing partition.

``` r
confusionMatrix(testing$classe, data = predict(modFit, newdata = testing))
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1652   26    5    6    3
    ##          B   15 1085   23    3    5
    ##          C    3   18  976   16    2
    ##          D    1    5   18  934    9
    ##          E    3    5    4    5 1063
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9703          
    ##                  95% CI : (0.9656, 0.9745)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : <2e-16          
    ##                                           
    ##                   Kappa : 0.9624          
    ##  Mcnemar's Test P-Value : 0.4353          
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9869   0.9526   0.9513   0.9689   0.9824
    ## Specificity            0.9905   0.9903   0.9920   0.9933   0.9965
    ## Pos Pred Value         0.9764   0.9593   0.9616   0.9659   0.9843
    ## Neg Pred Value         0.9948   0.9886   0.9897   0.9939   0.9960
    ## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
    ## Detection Rate         0.2807   0.1844   0.1658   0.1587   0.1806
    ## Detection Prevalence   0.2875   0.1922   0.1725   0.1643   0.1835
    ## Balanced Accuracy      0.9887   0.9714   0.9716   0.9811   0.9895

Conclusions
-----------

The final model predicts the correct classe about 97% of the time, so the out-of-sample error rate is about 3%.
