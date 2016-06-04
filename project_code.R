
library(caret)
library(doMC)
library(randomForest)

registerDoMC(cores = 4)

set.seed(1979)

setwd("~/Documents/courses/Practical Machine Learning")
options(stringsAsFactors = FALSE)



pml <- read.csv("pml-training.csv", header = TRUE)

summary(pml)
## shows that there are 3 types of timestamp and some variables are characters. They should probably be numeric. and the almost all of the numeric
## variables have 19216 NAs byros_forearm_z does not... Lets find out:
table(pml$classe, pml$cvtd_timestamp, pml$user_name)  ### shows that the users performed the exercise "correctly" first, this time stamp is not a good classifier

pml <- subset(pml, new_window == "no")

### get rid of the empty column: 
pml <- pml[, 2:ncol(pml)]

### split into training and test sets
inTrain <- createDataPartition(y = pml$classe, p = .8, list = FALSE)
training <- pml[inTrain,]
testing <- pml[-inTrain,]



####################################### cleaning ####################################
################################# functions for cleaning ############################
### create a vector of columns with na values:
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

### turn the timestamp into just series data:
# notime <- function(dat){
#   map <- as.numeric(1:length(unique(dat$raw_timestamp_part_1)))
#   names(map) <- as.character(unique(dat$raw_timestamp_part_1))
#   dat$series <- map[as.character(dat$raw_timestamp_part_1)]
#   return(dat)
# }

########################################### clean data ######################################
nacolmap <- createNAcolmap(training)
elim_name <- c("raw_timestamp_part_2", "cvtd_timestamp", "new_window", "raw_timestamp_part_1", "user_name") ### eliminate after series column is created 

clean <- function(dat){
  dat$classe <- factor(dat$classe)
  nonas <- dat[,!(colnames(dat) %in% names(nacolmap))]
  return(nonas[, !(colnames(nonas) %in% elim_name)])
}

cleantraining <- clean(training)
cleantesting_ref <- clean(testing)
cleantesting <- cleantesting_ref[, 1:53]



####################################### run models ###########################################
# motherfucker <- cleantraining[, !(colnames(cleantraining) %in% "classe")]
# for(i in 1:ncol(motherfucker)){
#   motherfucker[, i] <- as.numeric(motherfucker[, i])
# }
# classe <- cleantraining$classe
# mf <- cbind(motherfucker, classe)

### rf without any pre-processing
rf1 <- randomForest(classe~ ., data = cleantraining, ntree = 100) ##1.5 minues
# rf1.1 <- train(classe~ ., method = "rf", data = cleantraining, ntree = 100)
preda <- predict(rf1, newdata = cleantesting)
confusionMatrix(preda, cleantesting_ref$classe)  ### accuracy 99.77


### preprocess
pp <- preProcess(cleantraining, method = "pca")
trainingpp <- predict(pp, newdata = cleantraining)
rf2 <- randomForest(classe~., data = trainingpp, ntree = 100)
# rf2 <- train(classe ~., method ="rf", data = trainingpp, ntree = 100)
testpp <- predict(pp, newdata = cleantesting)
predb <- predict(rf2, newdata = testpp)
confusionMatrix(predb, cleantesting_ref$classe)  ## 98.02%

### cross validation
rf3 <- train(classe~., method = "rf", data = cleantraining, trControl = trainControl(method = "repeatedcv", number = 20), ntree=100)
confusionMatrix(predict(rf3, cleantesting), cleantesting_ref$classe)  ## 99.77%

############################ try all 3 without the series variable
# cleantraining2 <- cleantraining[, 1:(ncol(cleantraining)-1)]
# rf4 <- randomForest(classe~ ., data = cleantraining2, ntree = 100) ##1.5 minues
# rf4.1 <- train(classe~ ., method = "rf", data = cleantraining2, ntree = 100)
# confusionMatrix(predict(rf4, cleantesting[, colnames(cleantesting) != "series"]), cleantesting$classe)
# confusionMatrix(predict(rf4.1, cleantesting[, colnames(cleantesting) != "series"]), cleantesting$classe)

# ### preprocess
# pp2 <- preProcess(cleantraining2, method = "pca")
# trainingpp2 <- predict(pp2, newdata = cleantraining2)
# rf5 <- randomForest(classe~., data = trainingpp2, ntree = 100)
# rf5.1 <- train(classe ~., method ="rf", data = trainingpp2, ntree = 100)


# ### cross validation
# rf6 <- train(classe~., method = "rf", data = cleantraining2, trControl = trainControl(method = "repeatedcv", number = 20), ntree=100)
# confusionMatrix(predict(rf6, cleantesting[, colnames(cleantesting) != "series"]), cleantesting$classe)
# ###################### test with variable importance ###############################
# rf7varImport <- train(classe~., method = "rf", data = cleantraining2, ntree = 100, importance = TRUE)
# plot(rf7varImport)

# varimpobj <- varImp(rf7varImport)$importance
# varimpobj$ave <- apply(varimpobj, 1, mean)
# varimpobj <- varimpobj[order(varimpobj$ave, decreasing = TRUE), ]
# varnames <- rownames(varimpobj)[1:28]
# varimportsub <- cleantraining[, c("classe", varnames)]
# lastrf <- train(classe~., data = varimportsub, ntree = 100)

# testsub <- cleantesting[, c("classe", varnames)]
# pred1 <- predict(lastrf, newdata = testsub)
# confusionMatrix(pred1, testsub$classe)

# I eliminated the series variable

# original paper used 10 random forests calculated with 10 trees each with bagging
# classifer tested with 10 fold cross-validation with windows with 0.5s overlap


################## run testing data
realtest <- read.csv("pml-testing.csv")
realtest <- realtest[, 2:ncol(realtest)]

clean2 <- function(dat){
#   dat$classe <- factor(dat$classe)
  nonas <- dat[,!(colnames(dat) %in% names(nacolmap))]
  return(nonas[, !(colnames(nonas) %in% elim_name)])
}

testdata <- clean2(realtest)
realtestpp <- predict(pp, newdata = testdata[, 1:53])
predictions <- predict(rf2, newdata = realtestpp)

pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

pml_write_files(predictions)