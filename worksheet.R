
# data obtained from kaggle - titanic dataset

# Data Set Information:
# for each person in the training set, the followinfg attributes are provided: survived,pclass,name,sex,age,sibsp,parch,ticket,fare,cabin,embarked

# Goal:
# Build a model to classify/predict titanic records in order to predict who survived.

titanic.train.raw <- read.csv("train.csv")
# save the original dataframe
orig.df <- titanic.train.raw

# first look at dataset:
nrow(titanic.train.raw)
ncol(titanic.train.raw)
names(titanic.train.raw)
# 891 records/persons
# 11 columns
head(titanic.train.raw)
summary(titanic.train.raw)
# 177 NA on age column, 2 missing values on embarked column
# 314 females, 577 males
# types of columns?
sapply(titanic.train.raw, class)
# attributes survived, pclass should be converted to factor
# attribute name should be converted to string/character
# attributes ticket, cabin should be droped. ticket does not seem to have much meaning (at least I don't see any). cabin could be useful in the prediction if we could identify location on ship (starboard, port, bow, back; deck), but this information is not readily available in the dataset; anyway most of cabin information is missing.
# I'm tempted to drop the columns sibsp, parch; we'll see if their presence has any influence on the predictive power of the model(s).
# look at people whose fare is 0 (who are these people?):
subset(titanic.train.raw, subset=(fare<1))
# how many survived?
sum(subset(titanic.train.raw, select=c("survived")))
nrow(subset(titanic.train.raw, subset=(survived==1)))
# 342 survivors
nrow(subset(titanic.train.raw, subset=(survived==0)))
# 549 died
# let's see correlations (numeric values)
cor(titanic.train.raw[,c("survived", "pclass", "parch", "sibsp", "fare")])
#            survived      pclass      parch       sibsp       fare
#survived  1.00000000 -0.33848104 0.08162941 -0.03532250  0.2573065
#pclass   -0.33848104  1.00000000 0.01844267  0.08308136 -0.5494996
#parch     0.08162941  0.01844267 1.00000000  0.41483770  0.2162249
#sibsp    -0.03532250  0.08308136 0.41483770  1.00000000  0.1596510
#fare      0.25730652 -0.54949962 0.21622494  0.15965104  1.0000000
# let's make some groupings
table(titanic.train.raw$survived, titanic.train.raw$sex)
#    female male
#  0     81  468
#  1    233  109
# ratio of survival is way better for women than for men
table(titanic.train.raw$survived, titanic.train.raw$pclass)
#      1   2   3
#  0  80  97 372
#  1 136  87 119
# ratio of survival is better for higher class
table(titanic.train.raw$survived, titanic.train.raw$embarked)
#          C   Q   S
#  0   0  75  47 427
#  1   2  93  30 217
# C: 1.24 Q: 0.68 S: 0.5
# looks like if you embarked at C, you had better chances to survive;
# on the other hand this data (embarked) could be a confound for pclass
table(titanic.train.raw$pclass, titanic.train.raw$embarked)
table(titanic.train.raw$pclass, titanic.train.raw$sex)
table(titanic.train.raw$sex, titanic.train.raw$embarked)

# let's make some plots, colour by survived column
library(ggplot2)
qplot(jitter(pclass),sex,data=titanic.train.raw, col=as.factor(survived), xlab="class", ylab="gender")
qplot(age,sex,data=titanic.train.raw, col=as.factor(survived), xlab="age", ylab="gender")
qplot(age,jitter(pclass),data=titanic.train.raw, col=as.factor(survived), xlab="age", ylab="class")
qplot(jitter(pclass),embarked,data=titanic.train.raw, col=as.factor(survived), xlab="class", ylab="embarked")
qplot(fare,embarked,data=titanic.train.raw, col=as.factor(survived), xlab="fare", ylab="embarked")
qplot(age,jitter(sibsp),data=titanic.train.raw, col=as.factor(survived), xlab="age", ylab="siblings")
qplot(jitter(sibsp),sex,data=titanic.train.raw, col=as.factor(survived), ylab="gender", xlab="siblings")



# clean up dataset (don't forget to do the same to the test dataset)
# transform survived, pclass columns to factors
titanic.train.raw$survived <- as.factor(titanic.train.raw$survived)
titanic.train.raw$pclass <- as.factor(titanic.train.raw$pclass)
# transform name column to character
titanic.train.raw$name <- as.character(titanic.train.raw$name)
# is there some missing data? yes (177 on age).
sum(!complete.cases(titanic.train.raw)) # 177 (age NA)
# and don't forget the missing data (2) on embarked
titanic.train.raw[titanic.train.raw$embarked == "",c("embarked")] <- NA

# set a random seed
set.seed(101)
# randomly split dataset into training set and test (validation) set
# validation set will be approx. 10% of the data available, the rest (90%) will constitute the training set.
indices <- seq_len(nrow(titanic.train.raw))
ind1 <- sample(indices, 89)
ind2 <- indices[!indices %in% ind1]
testData <- titanic.train.raw[ind1,]
trainData <- titanic.train.raw[ind2,]

# names of columns
ns <- names(trainData)
ns

# summary train data
summary(trainData)
# ratio (subtract because conversion)
sum(as.numeric(trainData$survived) - 1) / nrow(trainData)
# 0.38
# summary test/validation data
summary(testData)
# ratio (subtract because conversion)
sum(as.numeric(testData$survived) - 1) / nrow(testData)
# 0.39


# load test/prediction dataset; clean it up in the same way
predictionData <- read.csv("test.csv")
predictionData$pclass <- as.factor(predictionData$pclass)
predictionData$name <- as.character(predictionData$name)
# defining the factor levels as in the training set (otherwise svm will not work)
predictionData$embarked <- factor(predictionData$embarked, levels = levels(trainData$embarked))
sapply(predictionData, class)
summary(predictionData)


# let's try logistic regression (with cross-validation ?)
glm2 <- glm(survived ~ ., family=binomial, data=reduced.train.data)
#Warning message:
#glm.fit: algorithm did not converge
# let's reduce the number of attributes
glm2 <- glm(survived ~ pclass + sex + fare + age, family=binomial, data=trainData)
glm2 <- glm(survived ~ pclass + sex + embarked + age, family=binomial, data=trainData)
glm2 <- glm(survived ~ pclass + sex + sibsp + age, family=binomial, data=trainData)
glm2 <- glm(survived ~ pclass + sex + parch + age, family=binomial, data=trainData)
glm2 <- glm(survived ~ pclass + sex + sibsp + fare, family=binomial, data=trainData)
summary(glm2)
# fare does not seem to be significant
# embarked isn't significant either
# parch the same, but apparently sibsp could be significant
glm2 <- glm(survived ~ pclass + sex + sibsp, family=binomial, data=trainData)
summary(glm2)
#cv2 <- cv.glm(trainData,glm2,K=9)
#cv2 <- cv.glm(reduced.train.data,glm2,K=9)
#summary(cv2)
plr <- round(predict(glm2, type="response", newdata=testData))
# NAs (in the test data) are a problem
# since the columns posing problems are embraked and age, drop them (get around them)
err_plr <- sum(testData$survived != plr) / nrow(testData)
sum(testData$survived != plr) # 17 missclassified out of 89
err_plr # 0.1910112 (or 0.19% error rate)


# let's try a (decision) tree prediction
library(tree)
tree1 <- tree(survived ~ pclass + sex + age + sibsp, data=trainData)
summary(tree1)
# result
#Classification tree:
#tree(formula = survived ~ pclass + sex + age + sibsp, data = trainData)
#Number of terminal nodes:  7 
#Residual mean deviance:  0.826 = 527.8 / 639
#Misclassification error rate: 0.1935 = 125 / 646 
# Plot tree
plot(tree1)
text(tree1)
# prediction on test set
pt1 <- predict(tree1,newdata=testData,type="class")
err_pt1 <- sum(testData$survived != pt1) / nrow(testData)
sum(testData$survived != pt1) # 18 missclassified out of 89
err_pt1 # 0.2022472 (or 0.2% error rate)

tree2 <- tree(survived ~ pclass + sex + age + sibsp + fare, data=trainData)
summary(tree2)
#Classification tree:
#tree(formula = survived ~ pclass + sex + age + sibsp + fare, 
#    data = trainData)
#Number of terminal nodes:  8 
#Residual mean deviance:  0.8071 = 514.9 / 638 
#Misclassification error rate: 0.1811 = 117 / 646 
# visualize tree (in text mode)
tree2
# Plot tree
plot(tree2)
text(tree2)
# prediction on test set
pt2 <- predict(tree2,newdata=testData,type="class")
err_pt2 <- sum(testData$survived != pt2) / nrow(testData)
sum(testData$survived != pt2) # 16 missclassified out of 89
err_pt2 # 0.1797753 (or 0.18% error rate)


# let's try a forest (default number of decision trees is 500)
library(randomForest)
forest1 <- randomForest(survived ~ pclass + sex + sibsp + fare, data=trainData, ntree=501)
forest1
# results:
#Call:
# randomForest(formula = survived ~ pclass + sex + sibsp + fare,      data = trainData, ntree = 501) 
#               Type of random forest: classification
#                     Number of trees: 501
#No. of variables tried at each split: 2
#
#        OOB estimate of  error rate: 19.45%
#Confusion matrix:
#    0   1 class.error
#0 449  46  0.09292929
#1 110 197  0.35830619
# show importance of variables:
round(importance(forest1, 2))
#       MeanDecreaseGini
#pclass               32
#sex                 101
#sibsp                15
#fare                 66
# prediction on test set
pf1 <- predict(forest1,newdata=testData)
# ommited age and embarked from the model, beacause of NAs
err_pf1 <- sum(testData$survived != pf1) / nrow(testData)
sum(testData$survived != pf1) # 17 missclassified out of 89
err_pf1 # 0.1910112 (or 0.19% error rate)

forest2 <- randomForest(survived ~ pclass + sex + sibsp, data=trainData, ntree=501)
forest2
# results:
#Call:
# randomForest(formula = survived ~ pclass + sex + sibsp, data = trainData,      ntree = 501) 
#               Type of random forest: classification
#                     Number of trees: 501
#No. of variables tried at each split: 1
#
#        OOB estimate of  error rate: 21.2%
#Confusion matrix:
#    0   1 class.error
#0 439  56   0.1131313
#1 114 193   0.3713355
pf2 <- predict(forest2,newdata=testData)
# ommited fare too, because of NA in the prediction/test data set
err_pf2 <- sum(testData$survived != pf2) / nrow(testData)
sum(testData$survived != pf2) # 17 missclassified out of 89
err_pf2 # 0.1910112 (or 0.19% error rate)



# let's try svm
library(e1071)
vec1 <- svm(survived ~ pclass + sex + age + sibsp + parch + fare + embarked, data=trainData)
vec1
# result:
#Call:
#svm(formula = survived ~ pclass + sex + age + sibsp + parch + fare + 
#    embarked, data = trainData, scale = F)
#
#Parameters:
#   SVM-Type:  C-classification 
# SVM-Kernel:  radial 
#       cost:  1 
#      gamma:  0.09090909 
#
#Number of Support Vectors:  331
pv1 <- predict(vec1,newdata=testData)
err_pv1 <- sum(testData$survived != pv1) / nrow(testData) # length of prediction set (67) does not match length of test set (89)
sum(testData$survived != pv1) # 18 missclassified out of 89
err_pv1 # 0.2022472 (or 0.2% error rate)

vec2 <- svm(survived ~ pclass + sex + age + sibsp, data=trainData)
vec2
# result:
#Call:
#svm(formula = survived ~ pclass + sex + age + sibsp, data = trainData)
#
#Parameters:
#   SVM-Type:  C-classification 
# SVM-Kernel:  radial 
#       cost:  1 
#      gamma:  0.1666667 
#
#Number of Support Vectors:  309
pv2 <- predict(vec2,newdata=testData)
# length of prediction set (67) does not match length of test set (89)

vec3 <- svm(survived ~ pclass + sex + sibsp + fare, data=trainData)
vec3
#Call:
#svm(formula = survived ~ pclass + sex + sibsp + fare, data = trainData
#
#Parameters:
#   SVM-Type:  C-classification 
# SVM-Kernel:  radial 
#       cost:  1 
#      gamma:  0.1666667 
#
#Number of Support Vectors:  375
pv3 <- predict(vec3,newdata=testData)
# length of prediction set (67) does not match length of test set (89)




# let's try boosting
library(ada)
ada1 <- ada(survived ~ pclass + sex + age + sibsp + fare + embarked, data=trainData)
ada1
# results:
#Call:
#ada(survived ~ pclass + sex + age + sibsp + fare + embarked, 
#    data = trainData)
#
#Loss: exponential Method: discrete   Iteration: 50 
#
#Final Confusion Matrix for Data:
#          Final Prediction
#True value   0   1
#         0 465  30
#         1  66 241
#
#Train Error: 0.12 
#
#Out-Of-Bag Error:  0.136  iteration= 45 
#
#Additional Estimates of number of iterations:
#
#train.err1 train.kap1 
#        43         45 
pa1 <- predict(ada1,newdata=testData)
err_pa1 <- sum(testData$survived != pa1) / nrow(testData)
sum(testData$survived != pa1) # 13 missclassified out of 89
err_pa1 # 0.1460674 (or 0.15% error rate)

ada2 <- ada(survived ~ pclass + sex + age + sibsp, data=trainData)
ada2
# results:
#Call:
#ada(survived ~ pclass + sex + age + sibsp, data = trainData)
#
#Loss: exponential Method: discrete   Iteration: 50 
#
#Final Confusion Matrix for Data:
#          Final Prediction
#True value   0   1
#         0 447  48
#         1  76 231
#
#Train Error: 0.155 
#
#Out-Of-Bag Error:  0.157  iteration= 50 
#
#Additional Estimates of number of iterations:
#
#train.err1 train.kap1 
#        24         48 
pa2 <- predict(ada2,newdata=testData)
err_pa2 <- sum(testData$survived != pa2) / nrow(testData)
sum(testData$survived != pa2) # 15 missclassified out of 89
err_pa2 # 0.1685393 (or 0.17% error rate)

ada3 <- ada(survived ~ pclass + sex + age + sibsp + embarked, data=trainData)
ada3
# results:
#Call:
#ada(survived ~ pclass + sex + age + sibsp + embarked, data = trainData)
#
#Loss: exponential Method: discrete   Iteration: 50 
#
#Final Confusion Matrix for Data:
#          Final Prediction
#True value   0   1
#         0 452  43
#         1  82 225
#
#Train Error: 0.156 
#
#Out-Of-Bag Error:  0.163  iteration= 37 
#
#Additional Estimates of number of iterations:
#
#train.err1 train.kap1 
#        36         50 
pa3 <- predict(ada3,newdata=testData)
err_pa3 <- sum(testData$survived != pa3) / nrow(testData)
sum(testData$survived != pa3) # 14 missclassified out of 89
err_pa3 # 0.1573034 (or 0.15% error rate)


# let's try apriori (association rules)
library(arules)
asr1 <- apriori(data=trainData)
asr1
# only works on factor data


# let's combine models (decision tree, random forest, ada-boosting)
# unfortunately svm does not build a result vector of compatible length (NAs ???)
# implementing voting
comb1 <- rep(NA,nrow(testData))
for (i in 1:nrow(testData)) {
  # we assume random forest is better than the other algorithms
  vote <- if (pa3[i] == pt2[i]) pa3[i] else pf2[i]
  comb1[i] <- if (vote == testData[i,]$survived) 1 else 0
}
err_cmb1 <- (nrow(testData) - sum(comb1)) / nrow(testData)
nrow(testData) - sum(comb1) # 16 missclassified out of 89
err_cmb1 # 0.1797753 (or 0.18% error rate)

comb2 <- rep(NA,nrow(testData))
for (i in 1:nrow(testData)) {
  # we assume boosting is better than the other algorithms
  vote <- if (pf2[i] == pt2[i]) pa3[i] else pa3[i]
  comb2[i] <- if (vote == testData[i,]$survived) 1 else 0
}
err_cmb2 <- (nrow(testData) - sum(comb2)) / nrow(testData)
nrow(testData) - sum(comb2) # 14 missclassified out of 89
err_cmb2 # 0.1573034 (or 0.16% error rate)


# write prediction files on test data
# convert to dataframe
# ada boosting
pred_pa2 <- as.numeric(predict(ada2, newdata=predictionData)) - 1
result_df <- data.frame(pred_pa2)
# decision tree
pred_pt1 <- as.numeric(predict(tree1,newdata=predictionData,type="class")) - 1
result_df <- data.frame(pred_pt1)
# random forest
pred_pf2 <- as.numeric(predict(forest2,newdata=predictionData)) - 1
result_df <- data.frame(pred_pf2)
# logistic regression
pred_plr <- round(predict(glm2,newdata=predictionData,type="response"))
result_df <- data.frame(pred_plr)
# ada boosting
pred_pa1 <- as.numeric(predict(ada1, newdata=predictionData)) - 1
result_df <- data.frame(pred_pa1)
# ensemble (combined model)
pred_pt2 <- as.numeric(predict(tree2,newdata=predictionData,type="class")) - 1
pred_pa3 <- as.numeric(predict(ada3, newdata=predictionData)) - 1
comb1 <- rep(0,nrow(predictionData))
for (i in 1:nrow(predictionData)) {
  # we assume random forest is better than the other algorithms
  #comb1[i] <- if (pred_pa3[i] == pred_pt2[i]) pred_pa3[i] else pred_pf2[i]
  # we assume boosting is better than the other algorithms
  comb1[i] <- if (pred_pf2[i] == pred_pt2[i]) pred_pf2[i] else pred_pa3[i]
}
result_df <- data.frame(comb1)


write.csv(result_df, file="prediction07.csv", row.names=FALSE)

