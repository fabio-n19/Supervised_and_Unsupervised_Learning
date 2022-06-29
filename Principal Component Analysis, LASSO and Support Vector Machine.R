update.packages(c("tidyverse", "tidyr", "readr"))
library(dplyr)

########################## Part 1 ###################################

### Variable Construction and Pretreatment of Data
#Find the columns of NAs and delete them from the data.
no_na <- m_d_806[ , colSums(is.na(m_d_806)) == 0]

#Removing "state", "asex", "triplet", "faminc"
drops <- c("state","asex", "triplet", "faminc", "...1")
no_var <- no_na[ , !(names(no_na) %in% drops)]
no_var$poverty

##Percentage of families with poverty level 7
pvrty_7 <- dplyr::filter(no_var, poverty %in% "7")
lvl_7 <- nrow(pvrty_7) 

allrows <- nrow(no_var)

pvrty_percentage <- (lvl_7/allrows)*100
print(pvrty_percentage)

##Redefining poverty as binary variable
pvrty_1 <- as.logical(no_var$poverty==7)
print(pvrty_1)
print(no_var$poverty)

pvrty <- as.integer(pvrty_1)

library(data.table)

pvrty <- data.frame(pvrty)
pvrty <- as.numeric(unlist(pvrty))
no_var$poverty=pvrty

###SPlitting the data into K-Folds
install.packages("cvTools")
install.packages("glmnet")
library(gamlr)
library(cvTools)
library(glmnet)
library(caret)
i <- 1
k <- 100
set.seed(40108865)
folds <- cut(seq(1,nrow(no_var)),breaks=100,labels=FALSE)
testIndexes <- which(folds==i,arr.ind=TRUE)
trainData <- no_var[testIndexes, ]
testData <- no_var[-testIndexes, ]

############################## Part 2 ###############################

### Supervised Learning Models
## Estimating poverty~. ^2

set.seed(40108865) 
x <- model.matrix(poverty~.^2,data=trainData)
model <- gamlr(x,trainData$poverty,family="binomial",lambda.min.ratio = 1e-4)
plot(model)

## K-Fold cross validation algorithm
set.seed(40108865)
cvmodel <- cv.gamlr(x,trainData$poverty,family="binomial",lambda.min.ratio = 1e-4)
plot(cvmodel)
best_lambda <- cvmodel$lambda.min
print(best_lambda)

Coeff <- coef(cvmodel, select ="min")
sum(Coeff!=0)

##Report the variables that give the greatest positive and negative impact
Coeff <- Coeff[-1, ]
Coeff[which.min(Coeff)]
Coeff[which.max(Coeff)]


#Plotting cross-validation selection process and model estimation without CV
par(mfrow=c(1,2))
plot(cvmodel)
plot(cvmodel$gamlr)

# Predict using model with the best lambda
x_valid <- model.matrix(poverty~.^2,data=testData)
predict_1=c(predict(model,newdata = x_valid,select=c("1se"),
                    type = "response"), lambda=best_lambda)


predict_1_d=ifelse(predict_1>0.9,1,0)
print(predict_1)
print(predict_1_d)

#Plotting ROC Curve

install.packages(c("ROCR", "cutoff"))
library(ROCR)
library(cutoff)
pred_1 <- predict(cvmodel,newdata=x_valid,select=c("1se"),type = "response")
lasso <- prediction(pred_1,testData$poverty)
perf_1 <- performance(lasso,'tpr','fpr')


#The optimal cutoff point
cut <- roc(score=c(pred1), class=testData$poverty)$cutoff
print(cut)

#Applying cutoff to LASSO and the false positive and
#false negative rate

pred_d1 <- ifelse(predict_1>cut,1,0)

#False Negative
sum(pred_d1[testData$poverty==0]==1)/sum(testData$poverty==0)*100
#Fast Positive
sum(pred_d1[testData$poverty==1]==0)/sum(testData$poverty==1)*100

#Using Support Vector Machine algorithm 
##Splitting training data into 5 folds
install.packages("e1071")
library(e1071)
svm_folds <- cut(seq(1,nrow(trainData)),breaks=5,labels=FALSE)
svmtestIndexes <- which(svm_folds==i,arr.ind=TRUE)
svmtrainData <- trainData[svmtestIndexes, ]
svmtestData <- trainData[-svmtestIndexes, ]

svm_model_j_1 <- svm(poverty~.^2, data=svmtrainData, kernel = "linear", cost = 3^(1:10/2))

svm_pred <- predict(svm_model_j_1, newdata = svmtestData)

opt_cost <- sum(diag(table(svm_pred,
                           svmtestData$poverty)))/length(svmtestData$poverty)

# Apply optimal cost to predict the outcome in the testing sample

svm_model_opt <- svm(poverty~.^2, data=svmtrainData, kernel = "linear", cost = opt_cost)


################## Part 3 --Unsupervised Learning ###################

install.packages(c("factoextra","naniar", "VIM", "missMDA"))

library(cluster)
library(factoextra)
library(missMDA)
library(naniar)
library(VIM)

##Part 1 -- Remove the outcome variable from testing sample
drop_pca <- "poverty"
testData <- testData[ , !(names(testData) %in% drop_pca)]
pca <- prcomp(testData, center=TRUE, scale=TRUE)
summary(pca)
fviz_contrib(pca, choice = "var", top = 20)

#Part 2 -- Composition of first two principal components
fviz_contrib(pca, choice = "var", axes = 1, top = 10)
fviz_contrib(pca, choice = "var", axes = 1, top = 10)
fviz_pca_var(pca,
             col.var = "contrib", # Color by contributions to the PC
             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
             repel = TRUE )

#Part 3 -- Randomly selecting 1000 families from training sample
j <- 1000
trained1000 <- testData[sample(1:nrow(testData), size = j,
                               replace = FALSE),]
trained1000pca <- prcomp(trained1000, center=TRUE, scale=TRUE)

fviz_pca_var(trained1000pca,
             col.var = "contrib", # Color by contributions to the PC
             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
             repel = TRUE )

#Part 4 -- Computing the euclidean matrix
trained1000pca <- as.matrix(unlist(trained1000pca))
trained1000pca <- as.matrix.data.frame(unlist(trained1000pca))
dist(trained1000pca, method = "euclidean", p=2)