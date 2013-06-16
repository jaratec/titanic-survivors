Titanic dataset - Predict survivors
========================================================

Data Set Information:
for each person in the training set, the following attributes are provided: survived,pclass,name,sex,age,sibsp,parch,ticket,fare,cabin,embarked.

load data:

```r
titanic.train.raw <- read.csv("train.csv")
```

save the original dataframe

```r
orig.df <- titanic.train.raw
```


first look at dataset:

```r
nrow(titanic.train.raw)  # records
```

```
## [1] 891
```

```r
ncol(titanic.train.raw)  # columns
```

```
## [1] 11
```

```r
ns <- names(titanic.train.raw)  # save names into a variable
ns
```

```
##  [1] "survived" "pclass"   "name"     "sex"      "age"      "sibsp"   
##  [7] "parch"    "ticket"   "fare"     "cabin"    "embarked"
```

```r
head(titanic.train.raw)
```

```
##   survived pclass                                                name
## 1        0      3                             Braund, Mr. Owen Harris
## 2        1      1 Cumings, Mrs. John Bradley (Florence Briggs Thayer)
## 3        1      3                              Heikkinen, Miss. Laina
## 4        1      1        Futrelle, Mrs. Jacques Heath (Lily May Peel)
## 5        0      3                            Allen, Mr. William Henry
## 6        0      3                                    Moran, Mr. James
##      sex age sibsp parch           ticket   fare cabin embarked
## 1   male  22     1     0        A/5 21171  7.250              S
## 2 female  38     1     0         PC 17599 71.283   C85        C
## 3 female  26     0     0 STON/O2. 3101282  7.925              S
## 4 female  35     1     0           113803 53.100  C123        S
## 5   male  35     0     0           373450  8.050              S
## 6   male  NA     0     0           330877  8.458              Q
```

```r
summary(titanic.train.raw)
```

```
##     survived         pclass    
##  Min.   :0.000   Min.   :1.00  
##  1st Qu.:0.000   1st Qu.:2.00  
##  Median :0.000   Median :3.00  
##  Mean   :0.384   Mean   :2.31  
##  3rd Qu.:1.000   3rd Qu.:3.00  
##  Max.   :1.000   Max.   :3.00  
##                                
##                                     name         sex           age       
##  Abbing, Mr. Anthony                  :  1   female:314   Min.   : 0.42  
##  Abbott, Mr. Rossmore Edward          :  1   male  :577   1st Qu.:20.12  
##  Abbott, Mrs. Stanton (Rosa Hunt)     :  1                Median :28.00  
##  Abelson, Mr. Samuel                  :  1                Mean   :29.70  
##  Abelson, Mrs. Samuel (Hannah Wizosky):  1                3rd Qu.:38.00  
##  Adahl, Mr. Mauritz Nils Martin       :  1                Max.   :80.00  
##  (Other)                              :885                NA's   :177    
##      sibsp           parch            ticket         fare      
##  Min.   :0.000   Min.   :0.000   1601    :  7   Min.   :  0.0  
##  1st Qu.:0.000   1st Qu.:0.000   347082  :  7   1st Qu.:  7.9  
##  Median :0.000   Median :0.000   CA. 2343:  7   Median : 14.5  
##  Mean   :0.523   Mean   :0.382   3101295 :  6   Mean   : 32.2  
##  3rd Qu.:1.000   3rd Qu.:0.000   347088  :  6   3rd Qu.: 31.0  
##  Max.   :8.000   Max.   :6.000   CA 2144 :  6   Max.   :512.3  
##                                  (Other) :852                  
##          cabin     embarked
##             :687    :  2   
##  B96 B98    :  4   C:168   
##  C23 C25 C27:  4   Q: 77   
##  G6         :  4   S:644   
##  C22 C26    :  3           
##  D          :  3           
##  (Other)    :186
```

types of columns?

```r
sapply(titanic.train.raw, class)
```

```
##  survived    pclass      name       sex       age     sibsp     parch 
## "integer" "integer"  "factor"  "factor" "numeric" "integer" "integer" 
##    ticket      fare     cabin  embarked 
##  "factor" "numeric"  "factor"  "factor"
```

how many survived? how many died?

```r
sum(subset(titanic.train.raw, select = c("survived")))
```

```
## [1] 342
```

```r
nrow(subset(titanic.train.raw, subset = (survived == 1)))
```

```
## [1] 342
```

```r
nrow(subset(titanic.train.raw, subset = (survived == 0)))
```

```
## [1] 549
```

attributes survived, pclass should be converted to factor
name should be converted to string/character
ticket does not seem to have much meaning (at least I don't see any); cabin could be useful in the prediction if we could identify location on ship (starboard, port, bow, back; deck), but this information is not readily available in the dataset; anyway most of cabin information is missing.

let's see correlations (numeric values)

```r
cor(titanic.train.raw[, c("survived", "pclass", "parch", "sibsp", "fare")])
```

```
##          survived   pclass   parch    sibsp    fare
## survived  1.00000 -0.33848 0.08163 -0.03532  0.2573
## pclass   -0.33848  1.00000 0.01844  0.08308 -0.5495
## parch     0.08163  0.01844 1.00000  0.41484  0.2162
## sibsp    -0.03532  0.08308 0.41484  1.00000  0.1597
## fare      0.25731 -0.54950 0.21622  0.15965  1.0000
```


let's make some groupings

```r
table(titanic.train.raw$survived, titanic.train.raw$sex)
```

```
##    
##     female male
##   0     81  468
##   1    233  109
```

```r
table(titanic.train.raw$survived, titanic.train.raw$pclass)
```

```
##    
##       1   2   3
##   0  80  97 372
##   1 136  87 119
```

```r
table(titanic.train.raw$survived, titanic.train.raw$embarked)
```

```
##    
##           C   Q   S
##   0   0  75  47 427
##   1   2  93  30 217
```

```r
table(titanic.train.raw$pclass, titanic.train.raw$embarked)
```

```
##    
##           C   Q   S
##   1   2  85   2 127
##   2   0  17   3 164
##   3   0  66  72 353
```

```r
table(titanic.train.raw$pclass, titanic.train.raw$sex)
```

```
##    
##     female male
##   1     94  122
##   2     76  108
##   3    144  347
```

```r
table(titanic.train.raw$sex, titanic.train.raw$embarked)
```

```
##         
##                C   Q   S
##   female   2  73  36 203
##   male     0  95  41 441
```


let's make some plots, colour by survived column

```r
library(ggplot2)
qplot(jitter(pclass), sex, data = titanic.train.raw, col = as.factor(survived), 
    xlab = "class", ylab = "gender")
```

![plot of chunk unnamed-chunk-8](figure/unnamed-chunk-81.png) 

```r
qplot(age, sex, data = titanic.train.raw, col = as.factor(survived), xlab = "age", 
    ylab = "gender")
```

```
## Warning: Removed 177 rows containing missing values (geom_point).
```

![plot of chunk unnamed-chunk-8](figure/unnamed-chunk-82.png) 

```r
qplot(age, jitter(pclass), data = titanic.train.raw, col = as.factor(survived), 
    xlab = "age", ylab = "class")
```

```
## Warning: Removed 177 rows containing missing values (geom_point).
```

![plot of chunk unnamed-chunk-8](figure/unnamed-chunk-83.png) 

```r
qplot(jitter(pclass), embarked, data = titanic.train.raw, col = as.factor(survived), 
    xlab = "class", ylab = "embarked")
```

![plot of chunk unnamed-chunk-8](figure/unnamed-chunk-84.png) 

```r
qplot(fare, embarked, data = titanic.train.raw, col = as.factor(survived), xlab = "fare", 
    ylab = "embarked")
```

![plot of chunk unnamed-chunk-8](figure/unnamed-chunk-85.png) 

```r
qplot(age, jitter(sibsp), data = titanic.train.raw, col = as.factor(survived), 
    xlab = "age", ylab = "siblings")
```

```
## Warning: Removed 177 rows containing missing values (geom_point).
```

![plot of chunk unnamed-chunk-8](figure/unnamed-chunk-86.png) 

```r
qplot(jitter(sibsp), sex, data = titanic.train.raw, col = as.factor(survived), 
    ylab = "gender", xlab = "siblings")
```

![plot of chunk unnamed-chunk-8](figure/unnamed-chunk-87.png) 


clean up dataset, make transformations

```r
titanic.train.raw$survived <- as.factor(titanic.train.raw$survived)
titanic.train.raw$pclass <- as.factor(titanic.train.raw$pclass)
titanic.train.raw$name <- as.character(titanic.train.raw$name)
titanic.train.raw[titanic.train.raw$embarked == "", c("embarked")] <- NA
```


missing data

```r
sum(!complete.cases(titanic.train.raw))
```

```
## [1] 179
```


a lot of missing data: age

split the dataset into a training set and a validation/test set,
validation set will be approx. 10% of the data available, the rest (90%) will constitute the training set

```r
set.seed(101)
indices <- seq_len(nrow(titanic.train.raw))
ind1 <- sample(indices, 89)
ind2 <- indices[!indices %in% ind1]
testData <- titanic.train.raw[ind1, ]
trainData <- titanic.train.raw[ind2, ]
```


load prediction dataset; clean it up in the same way

```r
predictionData <- read.csv("test.csv")
predictionData$pclass <- as.factor(predictionData$pclass)
predictionData$name <- as.character(predictionData$name)
predictionData$embarked <- factor(predictionData$embarked, levels = levels(trainData$embarked))
sapply(predictionData, class)
```

```
##      pclass        name         sex         age       sibsp       parch 
##    "factor" "character"    "factor"   "numeric"   "integer"   "integer" 
##      ticket        fare       cabin    embarked 
##    "factor"   "numeric"    "factor"    "factor"
```

```r
summary(predictionData)
```

```
##  pclass      name               sex           age            sibsp      
##  1:107   Length:418         female:152   Min.   : 0.17   Min.   :0.000  
##  2: 93   Class :character   male  :266   1st Qu.:21.00   1st Qu.:0.000  
##  3:218   Mode  :character                Median :27.00   Median :0.000  
##                                          Mean   :30.27   Mean   :0.447  
##                                          3rd Qu.:39.00   3rd Qu.:1.000  
##                                          Max.   :76.00   Max.   :8.000  
##                                          NA's   :86                     
##      parch            ticket         fare                   cabin    
##  Min.   :0.000   PC 17608:  5   Min.   :  0.0                  :327  
##  1st Qu.:0.000   113503  :  4   1st Qu.:  7.9   B57 B59 B63 B66:  3  
##  Median :0.000   CA. 2343:  4   Median : 14.5   A34            :  2  
##  Mean   :0.392   16966   :  3   Mean   : 35.6   B45            :  2  
##  3rd Qu.:0.000   220845  :  3   3rd Qu.: 31.5   C101           :  2  
##  Max.   :9.000   347077  :  3   Max.   :512.3   C116           :  2  
##                  (Other) :396   NA's   :1       (Other)        : 80  
##  embarked
##   :  0   
##  C:102   
##  Q: 46   
##  S:270   
##          
##          
## 
```


building models
logistic regression

```r
glm1 <- glm(survived ~ pclass + sex + age + embarked + sibsp + parch + fare, 
    family = binomial, data = trainData)
summary(glm1)
```

```
## 
## Call:
## glm(formula = survived ~ pclass + sex + age + embarked + sibsp + 
##     parch + fare, family = binomial, data = trainData)
## 
## Deviance Residuals: 
##    Min      1Q  Median      3Q     Max  
## -2.791  -0.620  -0.376   0.629   2.441  
## 
## Coefficients:
##             Estimate Std. Error z value Pr(>|z|)    
## (Intercept)  4.56602    0.56717    8.05  8.2e-16 ***
## pclass2     -1.40708    0.35241   -3.99  6.5e-05 ***
## pclass3     -2.47459    0.36099   -6.85  7.1e-12 ***
## sexmale     -2.59232    0.23355  -11.10  < 2e-16 ***
## age         -0.04742    0.00893   -5.31  1.1e-07 ***
## embarkedQ   -0.60819    0.62502   -0.97   0.3305    
## embarkedS   -0.30486    0.29073   -1.05   0.2944    
## sibsp       -0.43750    0.13821   -3.17   0.0015 ** 
## parch       -0.03820    0.13384   -0.29   0.7753    
## fare         0.00146    0.00265    0.55   0.5815    
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## (Dispersion parameter for binomial family taken to be 1)
## 
##     Null deviance: 871.33  on 644  degrees of freedom
## Residual deviance: 571.15  on 635  degrees of freedom
##   (157 observations deleted due to missingness)
## AIC: 591.2
## 
## Number of Fisher Scoring iterations: 5
```


drop from the model the variables that do not contribute to the prediction; drop age also (to many NA) 

```r
glm2 <- glm(survived ~ pclass + sex + sibsp, family = binomial, data = trainData)
summary(glm2)
```

```
## 
## Call:
## glm(formula = survived ~ pclass + sex + sibsp, family = binomial, 
##     data = trainData)
## 
## Deviance Residuals: 
##    Min      1Q  Median      3Q     Max  
## -2.265  -0.635  -0.482   0.633   2.555  
## 
## Coefficients:
##             Estimate Std. Error z value Pr(>|z|)    
## (Intercept)    2.486      0.243   10.25   <2e-16 ***
## pclass2       -0.979      0.263   -3.72    2e-04 ***
## pclass3       -1.856      0.224   -8.28   <2e-16 ***
## sexmale       -2.722      0.200  -13.59   <2e-16 ***
## sibsp         -0.283      0.101   -2.81    5e-03 ** 
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## (Dispersion parameter for binomial family taken to be 1)
## 
##     Null deviance: 1067.33  on 801  degrees of freedom
## Residual deviance:  739.58  on 797  degrees of freedom
## AIC: 749.6
## 
## Number of Fisher Scoring iterations: 5
```

```r
# prediction on test set
plr <- round(predict(glm2, type = "response", newdata = testData))
err_plr <- sum(testData$survived != plr)/nrow(testData)
sum(testData$survived != plr)  # number of missclassified records
```

```
## [1] 17
```

```r
err_plr  # error rate
```

```
## [1] 0.191
```


decision tree

```r
library(tree)
set.seed(101)
tree2 <- tree(survived ~ pclass + sex + age + sibsp + fare, data = trainData)
summary(tree2)
```

```
## 
## Classification tree:
## tree(formula = survived ~ pclass + sex + age + sibsp + fare, 
##     data = trainData)
## Number of terminal nodes:  8 
## Residual mean deviance:  0.807 = 515 / 638 
## Misclassification error rate: 0.181 = 117 / 646
```

```r
# visualize tree (in text mode)
tree2
```

```
## node), split, n, deviance, yval, (yprob)
##       * denotes terminal node
## 
##  1) root 646 900 0 ( 0.59 0.41 )  
##    2) sex: female 236 300 1 ( 0.25 0.75 )  
##      4) pclass: 3 92 100 0 ( 0.53 0.47 )  
##        8) fare < 23.0875 74 100 1 ( 0.45 0.55 ) *
##        9) fare > 23.0875 18  10 0 ( 0.89 0.11 ) *
##      5) pclass: 1,2 144  70 1 ( 0.06 0.94 ) *
##    3) sex: male 410 400 0 ( 0.79 0.21 )  
##      6) pclass: 2,3 319 300 0 ( 0.85 0.15 )  
##       12) age < 6.5 21  30 1 ( 0.38 0.62 )  
##         24) sibsp < 2 12   0 1 ( 0.00 1.00 ) *
##         25) sibsp > 2 9   6 0 ( 0.89 0.11 ) *
##       13) age > 6.5 298 200 0 ( 0.88 0.12 ) *
##      7) pclass: 1 91 100 0 ( 0.59 0.41 )  
##       14) age < 53 71 100 0 ( 0.51 0.49 ) *
##       15) age > 53 20  10 0 ( 0.90 0.10 ) *
```

```r
# plot tree
plot(tree2)
text(tree2)
```

![plot of chunk unnamed-chunk-15](figure/unnamed-chunk-15.png) 

```r
# prediction on test set
pt2 <- predict(tree2, newdata = testData, type = "class")
err_pt2 <- sum(testData$survived != pt2)/nrow(testData)
sum(testData$survived != pt2)  # number of missclassified records
```

```
## [1] 16
```

```r
err_pt2  # error rate
```

```
## [1] 0.1798
```


random forest

```r
library(randomForest)
```

```
## randomForest 4.6-7
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```r
set.seed(101)
# ommited fare and age, because of NA in the prediction/test data set
forest2 <- randomForest(survived ~ pclass + sex + sibsp, data = trainData, ntree = 501)
forest2
```

```
## 
## Call:
##  randomForest(formula = survived ~ pclass + sex + sibsp, data = trainData,      ntree = 501) 
##                Type of random forest: classification
##                      Number of trees: 501
## No. of variables tried at each split: 1
## 
##         OOB estimate of  error rate: 21.7%
## Confusion matrix:
##     0   1 class.error
## 0 435  60      0.1212
## 1 114 193      0.3713
```

```r
# prediction on test set
pf2 <- predict(forest2, newdata = testData)
err_pf2 <- sum(testData$survived != pf2)/nrow(testData)
sum(testData$survived != pf2)  # number of missclassified records
```

```
## [1] 17
```

```r
err_pf2  # error rate
```

```
## [1] 0.191
```


boosting

```r
library(ada)
```

```
## Loading required package: rpart
```

```r
set.seed(101)
ada3 <- ada(survived ~ pclass + sex + age + sibsp + embarked, data = trainData)
ada3
```

```
## Call:
## ada(survived ~ pclass + sex + age + sibsp + embarked, data = trainData)
## 
## Loss: exponential Method: discrete   Iteration: 50 
## 
## Final Confusion Matrix for Data:
##           Final Prediction
## True value   0   1
##          0 456  39
##          1  82 225
## 
## Train Error: 0.151 
## 
## Out-Of-Bag Error:  0.153  iteration= 35 
## 
## Additional Estimates of number of iterations:
## 
## train.err1 train.kap1 
##         38         38
```

```r
# prediction on test set
pa3 <- predict(ada3, newdata = testData)
err_pa3 <- sum(testData$survived != pa3)/nrow(testData)
sum(testData$survived != pa3)  # number of missclassified records
```

```
## [1] 14
```

```r
err_pa3  # error rate
```

```
## [1] 0.1573
```


make the predictions (exemple only for random forest, the others are similar)

```r
pred_pf2 <- as.numeric(predict(forest2, newdata = predictionData)) - 1
result_df <- data.frame(pred_pf2)
```

ensemble (combined model: decision tree, random forest, ada boosting)

```r
# decision tree
pred_pt2 <- as.numeric(predict(tree2, newdata = predictionData, type = "class")) - 
    1
# ada boosting
pred_pa3 <- as.numeric(predict(ada3, newdata = predictionData)) - 1
comb1 <- rep(0, nrow(predictionData))
for (i in 1:nrow(predictionData)) {
    # we assume random forest is better than the other algorithms
    comb1[i] <- if (pred_pa3[i] == pred_pt2[i]) 
        pred_pa3[i] else pred_pf2[i]
    # assume boosting is better than the other algorithms, then: comb1[i] <-
    # if (pred_pf2[i] == pred_pt2[i]) pred_pf2[i] else pred_pa3[i]
}
result_df <- data.frame(comb1)
```


save to predictions to csv file

```r
# comment out the line write.csv(result_df, file='predictionxx.csv',
# row.names=FALSE)
```

