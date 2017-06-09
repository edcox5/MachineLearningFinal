Using Machine Learning to Predict the Manner of Exercise
--------------------------------------------------------

6 volunteers were outfitted with accelerometers on the belt, arm, forearm, and dumbbell and asked to perform lifts correctly and incorrectly in 5 different ways. The goal of this project is to identify the manner (A-E) in which the exercise was being performed.

We will use a random partition to cross validate a model which uses regression and classification trees to predict outcomes.

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

Our classification tree model uses 52 variables to predict the "classe" of exercise.

    ## Loading required package: rpart

Plot tree
---------

Here's the plot of the dendrogram produced by the model.

``` r
fancyRpartPlot(modFit$finalModel)
```

![](final_files/figure-markdown_github/dendrogram-1.png)

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
    ##          A 1503  472  476  430  145
    ##          B   32  381   27  187  145
    ##          C  133  286  523  347  276
    ##          D    0    0    0    0    0
    ##          E    6    0    0    0  516
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.4967          
    ##                  95% CI : (0.4838, 0.5095)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.3427          
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.8978  0.33450  0.50975   0.0000  0.47689
    ## Specificity            0.6383  0.91761  0.78555   1.0000  0.99875
    ## Pos Pred Value         0.4967  0.49352  0.33419      NaN  0.98851
    ## Neg Pred Value         0.9402  0.85175  0.88356   0.8362  0.89446
    ## Prevalence             0.2845  0.19354  0.17434   0.1638  0.18386
    ## Detection Rate         0.2554  0.06474  0.08887   0.0000  0.08768
    ## Detection Prevalence   0.5142  0.13118  0.26593   0.0000  0.08870
    ## Balanced Accuracy      0.7681  0.62606  0.64765   0.5000  0.73782

Conclusions
-----------

The final model makes predictions using just 4 variables: roll\_belt, pitch\_forearm, magnet\_dumbbell\_y, and roll\_forearm. It predicts the correct classe about 50% of the time but cannot discern a classe of "D".
