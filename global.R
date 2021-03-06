
##--------------------------------Data Processing Starts------------------------
#Step 1: Data Selector
selectData <- function (input){
    if(input$radio == 2){
        req(input$file1)
        df <- read.csv(input$file1$datapath, sep = ",", stringsAsFactors = FALSE)
        df
    }else{
        df <- PimaIndiansDiabetes
        levels(df[, "diabetes"])[levels(df[, "diabetes"])=="pos"] <- "1"
        levels(df[, "diabetes"])[levels(df[, "diabetes"])=="neg"] <- "0"
        df
    }
}

#Step 2: Data Pre-Processing [Convert into factors]
convertToFactors <- function(inputData, input){
  ### Convert outcome variable to a factor
  outcome <- input$in2
  inputData[outcome] <- lapply(inputData[outcome], function(x) as.factor(x))

  #Convert all data frame character columns to factors. [Is this right thing to do?]
  inputData[sapply(inputData, is.character)] <- lapply(inputData[sapply(inputData, is.character)], as.factor)  
  
  #Convert all data frame "integer" columns to factors. [Is this right thing to do?]
  inputData[sapply(inputData, is.integer)] <- lapply(inputData[sapply(inputData, is.integer)], as.factor)
  
  inputData
}

##--------------------------------------------------------------------
#Step 3: Data Pre-Processing [Impute missing data]
imputeMissingData <- function(inputData, input){
  dataImputationOption <- input$in4
  if(dataImputationOption == "Predictive Mean Matching"){
    imputeMissingDataMice(inputData, input)
  }
  else if(dataImputationOption == "Option2"){
    imputeMissingDataOption2(inputData, input)
  }
}

#Step 3.1: Data Pre-Processing [Impute missing data: Mice]
imputeMissingDataMice <- function(inputData, input){
  #print("Inside imputeMissingDataMice()")
  predictors <- unlist((input$mychooser)[2])
  my.data <- inputData[, predictors]
  nums <- sapply(my.data, is.numeric)
  #my.data stroes all numeric columns from predictors set
  my.data <- my.data[, nums]
  #Find missing data pattern
  missing.data.count <- sum(is.na(my.data))
  #Numeric Predictors
  numeric.predictors <- colnames(my.data)
  
  if(missing.data.count == 0){
    inputData
  }
  else{
    # Perform predictive imputation on all numeric columns i.e. on all data in my.data
    mice_mod <- mice(my.data, method='pmm', printFlag = FALSE)
    # Save the complete output 
    mice_output <- complete(mice_mod)
    
    #Copy over data from data frame "mice_output" to data frame "inputData"
    #print(paste0("Missing Count - Before", sum(is.na(inputData[, numeric.predictors]))))
    for (i in 1:length(numeric.predictors)){
      inputData[, numeric.predictors[i]] <- mice_output[, numeric.predictors[i]]
    }
    #print(paste0("Missing Count - After", sum(is.na(inputData[, numeric.predictors]))))
    
    inputData
  }
}

#Step 3.2: Data Pre-Processing [Impute missing data: Option 2]
imputeMissingDataOption2 <- function(inputData, input){
  #print("Inside imputeMissingDataOption2()")
  inputData
  ##To be done
}
##--------------------------------------------------------------------

#Step 4: Data Pre-Processing [Top Level Function]
preProcessData <- function(inputData, input){
  processedData.1 <- imputeMissingData(inputData, input)
  processedData.2 <- convertToFactors(processedData.1, input)
  processedData.2
}

#Step 5: Split data into training and test datasets
splitData <- function(inputData, input){
    sample <- sample.split(inputData[, input$in2], SplitRatio = 0.75)
    train = subset(inputData, sample == TRUE)
    test = subset(inputData, sample == FALSE)
    list(train, test)
}

#Step 6: Derive Training Data
getTrainingData <- function(input){
  preProcessedData <- preProcessData(selectData(input), input)
  splitData(preProcessedData, input)[[1]]
}

#Step 7: Derive Test Data
getTestData <- function(input){
  preProcessedData <- preProcessData(selectData(input), input)
  splitData(preProcessedData, input)[[2]]
}

##--------------------------------Data Processing Ends------------------------

##--------------------------------Model Building Starts-----------------------
#Build Formula
modelFormula <- function(input){
    y <- input$in2
    x <- paste0(unlist((input$mychooser)[2]), collapse = "+")
    f <- as.formula(paste(y, x, sep="~"))
}

#Logistic Regression Model
logitFunc <- function(input){
    glm(modelFormula(input), data = getTrainingData(input), family = "binomial")
}

#Naive Bayes Model
naiveBayesFunc <- function(input){
  #set.seed(100)
  naiveBayes(modelFormula(input), data = getTrainingData(input))
}

#Neural Networks Model
nnetFunc <- function(input){
    nnet(modelFormula(input), data = getTrainingData(input), size=10, decay = 0.025, maxit = 10000)
}

#Support Vector Machines Model
svmFunc <- function(input){
  svm(modelFormula(input), data = getTrainingData(input))
}

#Confusion Matrix for Naive Bayes
predict.naivebayes.fulldata <- function(input){
    test.data <- getTestData(input)
    outcome <- input$in2
    naivebayes.predicted.fulldata <- predict(naiveBayesFunc(input), test.data)
    table(naivebayes.predicted.fulldata, test.data[,outcome], dnn = c("Predicted", "Actual"))
}

#Confusion Matrix for Logistic
predict.logit.fulldata <- function(input){
  test.data <- getTestData(input)
  outcome <- input$in2
  logit.predicted.fulldata <- predict(logitFunc(input), test.data)
  logit.predicted.fulldata <- ifelse(logit.predicted.fulldata < 0.7, 0, 1)
  table(logit.predicted.fulldata, test.data[,outcome], dnn = c("Predicted", "Actual"))
}

#Confusion Matrix for Neural Networks
predict.nnet.fulldata <- function(input){
  test.data <- getTestData(input)
  outcome <- input$in2
  nnet.predicted.fulldata <- predict(nnetFunc(input), test.data)
  nnet.predicted.fulldata <- ifelse(nnet.predicted.fulldata < 0.7, 0, 1)
  table(nnet.predicted.fulldata, test.data[,outcome], dnn = c("Predicted", "Actual"))
}

#Confusion Matrix for SVM
predict.svm.fulldata <- function(input){
  test.data <- getTestData(input)
  outcome <- input$in2
  svm.predicted.fulldata <- predict(svmFunc(input), test.data)
  #svm.predicted.fulldata <- ifelse(svm.predicted.fulldata < 0.7, 0, 1)
  table(svm.predicted.fulldata, test.data[,outcome], dnn = c("Predicted", "Actual"))
}

#Predict Single Value
predict.single.value <- function(model, inputData) {
  predict(model, inputData)
}

##--------------------------------Model Building Ends-------------------------

##--------------------------------Model Prediction Starts---------------------
sliderControl <- function(controlName, minVal, maxVal, value){
    
    sliderInput(
      controlName,
      controlName,
      min = minVal,
      max = maxVal,
      value = value
    )
}

selectInputControl <- function(controlName, choices){
  selectInput(
    controlName,
    controlName,
    choices = choices
  )
}

constructInputDataFrame <- function(input){
  inputDataVars <- getTestData(input)[0, unlist((input$mychooser)[2])]
  predictorNames <- names(inputDataVars)
  
  for (i in 1:length(predictorNames)){
    inputDataVars[1, predictorNames[i]] <- input[[predictorNames[i]]]
  }
  print(inputDataVars)
  inputDataVars
}

#This list will hold input controls for "Model Prediction" tab
deriveInputControls <- function(input){
  inputControls <- list()
  inputDataVars <- getTestData(input)[, unlist((input$mychooser)[2])]
  predictorNames <- names(inputDataVars)
  
  for (i in 1:length(predictorNames)){
    #Return levels for factor
    if(is.factor(inputDataVars[, predictorNames[i]])){
      inputControls[[i]] <- selectInputControl(predictorNames[i], levels(inputDataVars[, predictorNames[i]]))
    }
    
    if(is.numeric(inputDataVars[, predictorNames[i]])){
      inputControls[[i]] <- sliderControl(predictorNames[i], 
                                          min(inputDataVars[, predictorNames[i]]), 
                                          max(inputDataVars[, predictorNames[i]]), 
                                          mean(inputDataVars[, predictorNames[i]])
                                          )
    }
  }
  inputControls
}

##--------------------------------Model Prediction Ends-----------------------

##--------------------------------Model Performance Starts--------------------
# Logistic Regression Performance
perf_logistic <- function(model, testData, dependentVar, measure, x.measure){
  predict_logistic <- predict(model, newdata = testData ,type = 'response')
  pred_logistic <- prediction(predict_logistic, testData[,dependentVar])
  
  #perf_auc <- performance(pred_logistic, measure="auc", x.measure = "none")
  #print(paste0("Logistic AUC=",perf_auc@y.values[[1]]))
  
  performance(pred_logistic, measure, x.measure)  
}

# Naive Bayes Performance
perf_nb <- function(model, testData, dependentVar, measure, x.measure){
  predict_nb <- predict(model, newdata = testData, type="raw")
  predict_nb <- predict_nb[,2]  ## c(p0 , p1)
  pred_nb <- prediction( predict_nb,  testData[,dependentVar])
  performance(pred_nb, measure, x.measure)
}

# NNet Performance
perf_nnet <- function(model, testData, dependentVar, measure, x.measure){
  predict_nnet <- predict(model, newdata = testData)
  predict_nnet <- predict_nnet[,1]
  pred_nnet <- prediction(predict_nnet, testData[,dependentVar] )
  performance(pred_nnet, measure, x.measure)
}

#SVM Performance
perf_svm <- function(model, testData, dependentVar, measure, x.measure){
  predict_svm <- predict(model, newdata = testData)
  pred_svm <- prediction(as.numeric(predict_svm), testData[,dependentVar] )
  performance(pred_svm, measure, x.measure)
}

performanceTable <- function(input){
  y <- data.frame()
  
  test_data <- getTestData(input)
  dependentVar <- input$in2
  
  logit.TP <- predict.logit.fulldata(input)["1","1"]
  logit.FP <- predict.logit.fulldata(input)["1","0"]
  logit.TN <- predict.logit.fulldata(input)["0","0"]
  logit.FN <- predict.logit.fulldata(input)["0","1"]  
  logit.perf.auc <- perf_logistic(logitFunc(input), test_data, dependentVar, "auc", "none")
  logit.auc <- logit.perf.auc@y.values[[1]]
  
  naivebayes.TP <- predict.naivebayes.fulldata(input)["1","1"]
  naivebayes.FP <- predict.naivebayes.fulldata(input)["1","0"]
  naivebayes.TN <- predict.naivebayes.fulldata(input)["0","0"]
  naivebayes.FN <- predict.naivebayes.fulldata(input)["0","1"] 
  naivebayes.perf.auc <- perf_nb(naiveBayesFunc(input), test_data, dependentVar, "auc", "none")
  naivebayes.auc <- naivebayes.perf.auc@y.values[[1]]
  
  nnet.TP <- predict.nnet.fulldata(input)["1","1"]
  nnet.FP <- predict.nnet.fulldata(input)["1","0"]
  nnet.TN <- predict.nnet.fulldata(input)["0","0"]
  nnet.FN <- predict.nnet.fulldata(input)["0","1"] 
  nnet.perf.auc <- perf_nnet(nnetFunc(input), test_data, dependentVar, "auc", "none")
  nnet.auc <- nnet.perf.auc@y.values[[1]]
  
  svm.TP <- predict.svm.fulldata(input)["1","1"]
  svm.FP <- predict.svm.fulldata(input)["1","0"]
  svm.TN <- predict.svm.fulldata(input)["0","0"]
  svm.FN <- predict.svm.fulldata(input)["0","1"]
  svm.perf.auc <- perf_svm(svmFunc(input), test_data, dependentVar, "auc", "none")
  svm.auc <- svm.perf.auc@y.values[[1]]
  
  y["AUC", "Logistic"] <- round(logit.auc, digits = 3)
  y["Accuracy", "Logistic"] <- round( ((logit.TP + logit.TN) / (logit.TP + logit.FP + logit.TN + logit.FN)), digits = 3) 
  y["True Positive Rate/Sensitivity", "Logistic"] <- round((logit.TP / (logit.TP+logit.FP)), digits = 3)
  y["True Negative Rate/Specificity", "Logistic"] <- round((logit.TN / (logit.TN + logit.FN)), digits = 3)
  y["False Positive Rate", "Logistic"] <- round(1- (logit.TN / (logit.TN + logit.FN)), digits = 3)
  y["False Negative Rate", "Logistic"] <- round(1-(logit.TP / (logit.TP+logit.FP)), digits = 3)
  
  y["AUC", "Naive Bayes"] <- round(naivebayes.auc, digits = 3)
  y["Accuracy", "Naive Bayes"] <- round( ((naivebayes.TP + naivebayes.TN) / (naivebayes.TP + naivebayes.FP + naivebayes.TN + naivebayes.FN)), digits = 3) 
  y["True Positive Rate/Sensitivity", "Naive Bayes"] <- round((naivebayes.TP / (naivebayes.TP + naivebayes.FP)), digits = 3)
  y["True Negative Rate/Specificity", "Naive Bayes"] <- round((naivebayes.TN / (naivebayes.TN + naivebayes.FN)), digits = 3)  
  y["False Positive Rate", "Naive Bayes"] <- round(1-(naivebayes.TN / (naivebayes.TN + naivebayes.FN)), digits = 3)
  y["False Negative Rate", "Naive Bayes"] <- round(1-(naivebayes.TP / (naivebayes.TP + naivebayes.FP)), digits = 3)
  
  y["AUC", "Neural Networks"] <- round(nnet.auc, digits = 3)
  y["Accuracy", "Neural Networks"] <- round( ((nnet.TP + nnet.TN) / (nnet.TP + nnet.FP + nnet.TN + nnet.FN)), digits = 3) 
  y["True Positive Rate/Sensitivity", "Neural Networks"] <- round((nnet.TP / (nnet.TP + nnet.FP)), digits = 3)
  y["True Negative Rate/Specificity", "Neural Networks"] <- round((nnet.TN / (nnet.TN + nnet.FN)), digits = 3)  
  y["False Positive Rate", "Neural Networks"] <- round(1-(nnet.TN / (nnet.TN + nnet.FN)), digits = 3)
  y["False Negative Rate", "Neural Networks"] <- round(1-(nnet.TP / (nnet.TP + nnet.FP)), digits = 3)
  
  y["AUC", "SVM"] <- round(svm.auc, digits = 3)
  y["Accuracy", "SVM"] <- round( ((svm.TP + svm.TN) / (svm.TP + svm.FP + svm.TN + svm.FN)), digits = 3) 
  y["True Positive Rate/Sensitivity", "SVM"] <- round((svm.TP / (svm.TP + svm.FP)), digits = 3)
  y["True Negative Rate/Specificity", "SVM"] <- round((svm.TN / (svm.TN + svm.FN)), digits = 3)    
  y["False Positive Rate", "SVM"] <- round(1-(svm.TN / (svm.TN + svm.FN)), digits = 3)
  y["False Negative Rate", "SVM"] <- round(1-(svm.TP / (svm.TP + svm.FP)), digits = 3)
  
  y
}
##--------------------------------Model Performance Ends--------------------