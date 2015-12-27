### report code:
library(caret)
library(doMC)
registerDoMC(cores = 4)
set.seed(1979)

setwd("~/Documents/courses/Practical Machine Learning")
options(stringsAsFactors = FALSE)
pml <- read.csv("pml-training.csv", header = TRUE)


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
elim_name <- c("raw_timestamp_part_2", "cvtd_timestamp", "new_window", "num_window", "raw_timestamp_part_1", "user_name") ### eliminate after series column is created 

clean <- function(dat){
  dat$classe <- factor(dat$classe)
  nonas <- dat[,!(colnames(dat) %in% names(nacolmap))]
  return(nonas[, !(colnames(nonas) %in% elim_name)])
}

cleantraining2 <- clean(training)
cleantesting <- clean(testing)

##################################### model training and tuning ####################
rf7varImport <- train(classe~., method = "rf", data = cleantraining2, ntree = 100, importance = TRUE)

############ variable importance ################
varimpobj <- varImp(rf7varImport)$importance
varimpobj$ave <- apply(varimpobj, 1, mean)
varimpobj <- varimpobj[order(varimpobj$ave, decreasing = TRUE), ]
varnames <- rownames(varimpobj)[1:28]
varimportsub <- cleantraining2[, c("classe", varnames)]

############# new model ###################
lastrf <- train(classe~., data = varimportsub, ntree = 100)

testsub <- cleantesting[, c("classe", varnames)]
pred1 <- predict(lastrf, newdata = testsub)


