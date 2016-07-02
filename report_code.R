### report code:
library(caret)
library(doMC)
library(randomForest)
registerDoMC(cores = 4)
set.seed(1979)
options(warn = -1)

# setwd("~/Documents/courses/Practical Machine Learning")
options(stringsAsFactors = FALSE)
pml <- read.csv("../pml-training.csv", header = TRUE)


## the first variable is empty
pml <- pml[, 2:ncol(pml)]

inTrain <- createDataPartition(y = pml$classe, p = .8, list = FALSE)
training <- pml[inTrain,]
testing <- pml[-inTrain,]


natest <- function(c){
  return(as.character(summary(c)[7]))
}

createNAcolmap <- function(dat){
  ### set all columns as numerics
  for(i in 6:(ncol(dat)-1)){
    dat[, i] <- as.numeric(dat[, i])
  }
  ### get summary data of each column
  nacol <- list()
  for(i in 1:ncol(dat)){
    nacol[[i]] <- natest(dat[, i])
  }
  map <- unlist(nacol)
  names(map) <- colnames(dat)
  return(map[!is.na(map)])  
}


########################################### clean data ######################################
nacolmap <- createNAcolmap(training)
elim_name <- c("raw_timestamp_part_2", "cvtd_timestamp", "new_window", "raw_timestamp_part_1", "user_name") ### eliminate after series column is created 

clean <- function(dat){
  nonas <- dat[,!(colnames(dat) %in% names(nacolmap))]
  return(nonas[, !(colnames(nonas) %in% elim_name)])
}

cleantraining <- clean(training)
cleantraining$classe <- factor(cleantraining$classe)
cleantesting <- clean(testing) 


##################################### model training and tuning ####################
# pp <- preProcess(cleantraining, method = "pca")
# trainingpp <- predict(pp, newdata = cleantraining)
# rf2 <- randomForest(classe~., data = trainingpp, ntree = 100)
# testpp <- predict(pp, newdata = cleantesting[, 1:53])  ## run pca on the validation set not including the classe variable
# predb <- predict(rf2, newdata = testpp)
# confusionMatrix(predb, cleantesting$classe)  ## 98.02%


## model 1
rf1 <- randomForest(classe~ ., data = cleantraining, ntree = 100) ##1.5 minues
# rf1 <- train(classe~ ., method = "rf", data = cleantraining, ntree = 100)
confusionMatrix(predict(rf1, newdata = cleantesting[, 1:53]), cleantesting$classe)

## model 2
pp <- preProcess(cleantraining, method = "pca")
trainingpp <- predict(pp, newdata = cleantraining)
rf2 <- randomForest(classe~., data = trainingpp, ntree = 100)
# rf2 <- train(classe ~., method ="rf", data = trainingpp, ntree = 100)
testpp <- predict(pp, newdata = cleantesting[, 1:53])
confusionMatrix(predict(rf2, newdata = testpp), cleantesting$classe) 

## model 3
rf3 <- train(classe~., method = "rf", data = cleantraining, trControl = trainControl(method = "repeatedcv", number = 20), ntree=100)
confusionMatrix(predict(rf3, cleantesting[, 1:53]), cleantesting$classe)


