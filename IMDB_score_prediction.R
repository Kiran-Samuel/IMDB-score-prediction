######################################################################################################
##                        Movie Rating Range Prediction Using Machine Learning Algorithms
######################################################################################################

##################################################################################
##          Install packages necessary for the analysis
##################################################################################

# Clearing environment and suppressing warnings
rm(list=ls())
options(warn=-1)


if (!require('dplyr')) install.packages('dplyr')
if (!require('tidyverse')) install.packages('tidyverse')
if (!require('pryr')) install.packages('pryr')
if (!require('data.table')) install.packages('data.table')
if (!require('ggplot2')) install.packages('ggplot2')
if (!require('Metrics')) install.packages('Metrics')
if (!require('randomForest')) install.packages('randomForest')
if (!require('rfUtilities')) install.packages('rfUtilities')
if (!require('rpart')) install.packages('rpart')
if (!require('rpart.plot')) install.packages('rpart.plot')
if (!require('caret')) install.packages('caret')
if (!require('GGally')) install.packages('GGally')
if (!require('Boruta')) install.packages('Boruta')
if (!require('e1071')) install.packages('e1071')
if (!require('faraway')) install.packages('faraway')


##################################################################################
##                        Load libraries
##################################################################################

library(dplyr)                          # For dataset manipulations
library(tidyverse)
library(pryr)                           # memory handling
library(data.table)
library(ggplot2)                        # for plots
library(Metrics)
library(randomForest)                   # for classification using random forest  
library(rfUtilities)
library(rpart)                          # recursive partitioning
library(rpart.plot)                     # plotting decision trees
library(caret)
library(GGally)                         # for plots
library(Boruta)                         # feature selection
library(e1071)
library(faraway)                        # for calculating VIF
library(rstudioapi)                     # to extract path for reading the file

options(scipen = 999)                   # to prevent scientific notations on any numerical variable 

mem_used()

########################################################################################################################
##                      Loading Source file
########################################################################################################################

########################################################################################################################
##                        Cleaned DATASET
########################################################################################################################

# The dataset used here is cleaned version of the original dataset


# setting working directory
setwd(dirname(getActiveDocumentContext()$path))

getwd()
movie_collection <- tryCatch(read.csv('./movie_collection.csv'),   # movie_collection contains a collection of roughly 4000 movies after cleaning
                             warning = function(c) {
                               warning('File not found')})


str(movie_collection)                        # the structure of the dataset
summary(movie_collection)                    # summary statistics

object_size(movie_collection)                # shows how much memory movie_collection occupies

############################################################################################
##                      Feature Selection
############################################################################################


# Removing the default indexes created when saved as a csv
movie_collection <- subset(movie_collection, select = -c(1))

# The profit variable is removed from as it was introduced only for data exploratory purposes
movie_collection <- subset(movie_collection, select = -c(18))

# checking the unique entries for the factor variables
sum(uniqueN(movie_collection$director))
sum(uniqueN(movie_collection$actor_1))
sum(uniqueN(movie_collection$actor_2))
sum(uniqueN(movie_collection$actor_3))
sum(uniqueN(movie_collection$title))


selected_features <- subset(movie_collection, select = -c(1:2, 6:12))  # removing categorical variables as they are very diverse to be 
                                                                       # used for prediction
str(selected_features)
# Correlation for the variables in the dataset
round(cor(selected_features),2)
# Looking at some higher values for correlation, it can be seen that there is a high positive correlation between voted_users
# and user_reviews, and also a very high correlation can be seen between cast_FB_likes and actor_1_FB_likes 
# VIF is used to check these if correlations are statistically significant 

mymodel <- lm(imdb_score ~., selected_features)

# Calculating Variance Inflation Factor to check for multicollinearity
vif(mymodel)
# It can be seen that actor_1_FB_likes, actor_2_FB_likes, cast_FB_likes  have VIF values greater than 10
# For severe multicollinearity, the VIF value will be greater than 10
# So they are removed
selected_features <- subset(selected_features, select = -c(10:11, 13))

# Checking for the lowest and highest IMDb score
range(selected_features$imdb_score)
mean(selected_features$imdb_score)


# Splitting the imdb scores into different bins for easy classification of movies
# The movies are categorized into 4 bins:
# 1) (0,4)  - Bad movies
# 2) (4,6)  - Below average movies
# 3) (6,8)  - Good movies
# 4) (8,10) - Excellent movies
selected_features$score_range <- cut(selected_features$imdb_score, breaks = c(0,4,6,8,10))

# Removing the imdb_score column as we do not need it anymore
selected_features <- subset(selected_features, select = -c(2))

#Feature selection using Boruta

set.seed(111)
boruta <- Boruta(score_range ~., data = selected_features, doTrace = 2, maxRuns = 500)

plot(boruta, las = 2, cex.axis = 0.6)
# Ideally importance of shadow attribute should be close to 0, or it might almost have non zero values
# Green are the confirmed attributes
# blue are the shadow attributes
# If an attribute was considered unimportant, it would be coloured red

# In the plot there are no unimportant variables and all the variables are deemed important
# by Boruta

plotImpHistory(boruta)
# the green areas are confirmed or important attributes
# These have much higher importance than the shadow attributes which are 
# the blue colored lines


attStats(boruta)
# All the attributes were found to be important than the shadow attribute
# as seen in the normHits column

# Displaying the final confirmed attributes
finalvars = getSelectedAttributes(boruta, withTentative = F)
getConfirmedFormula(boruta)

# Pairplot to see distribution of the different classes
pair <- ggpairs(
  data = selected_features,
  columns = 1:10,
  mapping = ggplot2::aes(color = score_range),
  upper = list(continuous = 'cor'),
  lower = list(continuous = 'points'),
  diag = list(continuous = 'densityDiag'),)
show(pair)
# The classes are very closely overlapped and there is no clear demarcation
# showing a clear case of non linear classification. So Random forest, decision tree and KNN
# were used for classification of the different classes

####################################################################################################
##                          Modeling
####################################################################################################

# Data Partitioning

# Partitioning data with 80:20 ratio, 80% for train and 20% for test
set.seed(222)
sample_size <- floor(0.80*nrow(selected_features))
train_idx <- sample(seq_len(nrow(selected_features)), size = sample_size)

# split the dataset in training and test data
movie_train <- selected_features[train_idx,]
movie_test <- selected_features[-train_idx,]

##########################################################################################################
##                                  [1]  Decision Tree
##########################################################################################################

# build the model

model.rpart <- rpart(score_range ~ .,method = "class", data = movie_train,)
printcp(model.rpart)

# Predict the data
pred.rpart <- predict(model.rpart,
                      newdata = movie_test, 
                      type = "class")

#Confusion matrix
confusionMatrix(pred.rpart, movie_test$score_range)

# Confusion matrix
table.rpart <- table('Actual' = movie_test$score_range, 'Predicted' = pred.rpart)
table.rpart

# from the confusion matrix, the model does not classify the (0,4] score at all, but does well in the other ranges

# error rate and accuracy
error_rate.rpart <- sum(movie_test$score_range != pred.rpart) / nrow(movie_test)   

# Accuracy of model before pruning
dec_tree_acc <- round((1 - error_rate.rpart)*100,2)                           
sum((diag(table.rpart))/nrow(movie_test))*100                           # to confirm the accuracy

# more info
printcp(model.rpart)
summary(model.rpart)

# print model result
prp(model.rpart, extra = 100, main = "Classification Tree")

####################################################################################################  
##                                          Tree Pruning
####################################################################################################

# Plotting the Cost parameter (CP) table
plotcp(model.rpart)

# Printing the Cost parameter (CP) table
print(model.rpart$cptable)

# Retrieving the optimal and lowest cp value based on cross-validated error
cp_least_xerror.rpart <- model.rpart$cptable[which.min(model.rpart$cptable[,"xerror"]),"CP"]
cp_least_xerror.rpart                                      

#Pruning model based on optimal cp value
model.rpart_pruned <- prune(tree = model.rpart, cp = cp_least_xerror.rpart)


pred.rpart_pruned <- predict(model.rpart_pruned,
                             newdata = movie_test, 
                             type = "class")


table.rpart_pruned <- table('Actual' = movie_test$score_range, 'Predicted' = pred.rpart_pruned)
table.rpart_pruned

# Accuracy of the model
pruned_tree_acc <- round(sum(diag(table.rpart_pruned))/nrow(movie_test)*100, 2)   


#########################################################################################################
##                                     [2]  Random Forest
#########################################################################################################

# Build random forest model
set.seed(789)
rf <- randomForest(score_range ~., data = movie_train)
print(rf)
# random forest method is chosen as classification because score_range is a factor variable
# ntree = 500 (default)
# mtry =  (For classification, the default is around square root of number of features or variables)
# OOB is the data which the tree has not seen.
# From the confusion matrix, predictions are good for (6,8] and (8,10]
# but errors are highest for (0,4]) and (4,6])
# Therefore,the model predicts score ranges (6,8] and (8,10] better than others

# prediction and confusion matrix on train data
pred_train <- predict(rf, movie_train)
head(pred_train)                          # prediction by model

head(movie_train$score_range)        # actual value
# It can be seen that the first 6 predictions are 100% accurate, which is a coincidence

confusionMatrix(pred_train, movie_train$score_range)
# no mis-classification in train data at 95% confidence interval
# This accuracy is very high because all 2984 data points are already seen by the model

# prediction and confusion matrix on test data
pred_test <- predict(rf, movie_test)                                             
confusionMatrix(pred_test, movie_test$score_range)

# From the sensitivity values it can be seen that it is best for (6,8] and (4,6] 
# but for class (0,4], it is the least

# Confusion Matrix
rf_pre_tune_xtab <- table(Actual = movie_test$score_range, Predicted = pred_test)
rf_pre_tune_xtab

# Accuracy of model
rf_pre_tune_acc <- sum((diag(rf_pre_tune_xtab))/nrow(movie_test))*100


#Error rate of random forest
plot(rf)
legend('topright', colnames(rf$err.rate), col=1:5, fill=1:5)

# The black line depicts the Out Of Bag(OOB) error rate which is about 30%
# The error rate of the classes (6,8] and (8,10] are the least while the model is not able 
# to predict class (0,4]  which has the highest error rate

# As the number of trees increases, it is noted that the OOB error 
#initially drops down and becomes constant. And we are cannot improve the error after about 200 trees 

########################################################################################
##                                      Tuning
########################################################################################

# Tune mtry

set.seed(1234)
t <- tuneRF(movie_train[,-10], movie_train[,10],
            stepFactor = 0.5,
            plot = TRUE,
            ntreeTry = 200,
            trace = TRUE,
            improve = 0.05)

best_mtry <- t[t[, 2] == min(t[, 2]), 1]
print(t)
print(best_mtry)

rf_tuned <- randomForest(score_range ~., data = movie_train,
                         ntree = 200,
                         mtry = best_mtry,
                         importance = TRUE,
                         proximity = TRUE)            
rf_tuned

# train data
pred_train1 <- predict(rf_tuned, movie_train)
head(pred_train1)                          # prediction

head(movie_train$score_range)      # actual

confusionMatrix(pred_train1, movie_train$score_range)

#test data
pred_test1 <- predict(rf_tuned, movie_test)
confusionMatrix(pred_test1, movie_test$score_range)


# Confusion Matrix
rf_tune_xtab <- table(Actual = movie_test$score_range, Predicted = pred_test1)
rf_tune_xtab

# Accuracy of tuned model
rf_tune_acc <- sum((diag(rf_tune_xtab))/nrow(movie_test))*100
rf_acc <- rf_tune_acc                                          # accuracy of random forest model

# no of nodes for trees
hist(treesize(rf_tuned),
     main = "No of nodes for trees",
     col = "green") 
# There are more than 40 trees about 460-500 nodes in them

# variable importance
varImpPlot(rf_tuned,
           pch = 18,
           col = "red",
           sort =T,
           cex = 1,
           main = "Variable Importance")

#Mean decrease accuracy
# If we remove voted_users while making the tree, the accuracy will be drastically affected
# Voted_users and gross has maximum contribution to accuracy
# while critic_reviews is not very important for the prediction

#Gini captures how pure the nodes are at the end of the tree without each variable
#Gini decreases by a large amount if we remove voted_users
# voted_users and gross are very important for this parameter

# Visualization of importance of each variable using Mean decrease accuracy
importance <- importance(rf_tuned)
var_importance <- data.frame(Variables = row.names(importance), 
                             Importance = round(importance[ ,'MeanDecreaseAccuracy'],2))

# Assigning rank to each variable based on their importance
importance_rank <- var_importance %>%
  mutate(Rank = paste0('#',dense_rank(desc(Importance))))

p3 <- ggplot(importance_rank, aes(x = reorder(Variables, Importance), 
                                  y = Importance, fill = Importance)) +
  geom_bar(stat='identity') + 
  geom_text(aes(x = Variables, y = 0.5, label = Rank),
            hjust=0, vjust=0.55, size = 4, colour = 'red') +
  geom_text(aes(label = signif(Importance, digits = 6)),
            hjust=1.5, vjust=0.55, size = 4, colour = 'white') +
  labs(x = 'Variables') + 
  labs(title = "Importance of each variable") +
  coord_flip()
show(p3)


# Partial dependency plots based on the most important variable: voted_users

rf.partial.prob(x = rf_tuned, pred.data = movie_test, xname = "voted_users",
                which.class = "(0,4]")

rf.partial.prob(x = rf_tuned, pred.data = movie_test, xname = "voted_users",
                which.class = "(8,10]")

# From the graphs it is clear that more the voted_users, the greater the probability 
# that the movie can be classified in the (8,10] class , while lesser the voted_users, the more
# probability that the movie can be classified in the (0,4] class. 

#########################################################################################################
##                    [3]    K- Nearest Neighbors (KNN)
#########################################################################################################

str(selected_features)

# KNN model
trControl <- trainControl(method = "repeatedcv",
                          number = 10,
                          repeats = 3)
# reapeated cross validation
# number of resampling iterations = 10
# repeats is the complete set of folds to repeat the cross validation
set.seed(456)
fit <- train(score_range ~.,
             data = movie_train,
             method ="knn",
             tuneLength = 20,
             trControl = trControl,
             preProcess = c("center","scale"))  # standardizing the values

# The data is normalized by subtracting mean from each value and then divide it by standard deviation
# center is mean and scale is standard deviation


# model performance
fit
# a 10 fold cross validation is repeated 3 times, in each cross validation
# training data is split into 10 folds and 9 of them are used for creating the model
# and the remaining are used for accessing the model.


plot(fit, main ="Optimal k value")       # plotting optimal k value

knn_varimp <- varImp(fit)
plot(knn_varimp)                     # shows importance of each variable on each different classes

#prediction for test data
test_pred <- predict(fit, newdata = movie_test)

# confusion matrix ans statistics of test data
confusionMatrix(test_pred, movie_test$score_range)

# Confusion Matrix
knn_xtab <- table(Actual = movie_test$score_range, Predicted = test_pred)
knn_xtab

# Accuracy of model
knn_acc <- sum((diag(knn_xtab))/nrow(movie_test))*100

###################################################################################################
##                      Visualization of accuracy of classifiers
###################################################################################################

Models <- c("Decision Tree", "Random Forest", "KNN")
accuracy <- c(round(dec_tree_acc,4), round(rf_acc,4), round(knn_acc,4))


mod.data <- data.frame(Models, accuracy)

ggplot(aes(x = reorder(Models, -accuracy), y = accuracy, fill = Models), data = mod.data) + 
  labs(y= "Accuracy%", x = "Classifier") + geom_bar(stat = "identity") + geom_col(colour = "black")  + 
  labs(title = "Accuracy of models") +
  geom_text(aes(label = signif(accuracy, digits = 4)), nudge_y = 1.5) + 
  theme(axis.text.x = element_text(angle = 0, hjust = 0.5)) 

##################################################################################################
