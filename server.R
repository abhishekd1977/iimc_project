
# This is the server logic for a Shiny web application.
# You can find out more about building applications with Shiny here:
#
# http://shiny.rstudio.com
#

library(shiny)
library(DT)
library(mlbench)
library(e1071)
library(caTools)
library(nnet)
library(mice)
library(ROCR)
library(ggplot2)
library(plotROC)
data(PimaIndiansDiabetes)
source("chooser.R")

##
function(input, output) {
  output$out2 <- renderPrint(input$in2)
  output$out3 <- renderPrint(input$in3)
  output$selection <- renderPrint(input$mychooser)
  
  #Logistic Regression Tab
  output$fields.lr <- renderUI({
      fluidPage(
          fluidRow(
              h4("Model Summary"),
              tableOutput("logitTable"),
              tags$br(),
              tableOutput("nTextLogistic")
          )
      )
  })
  
  #Naive Bayes Tab
  output$fields.nb <- renderUI({
      fluidPage(
          fluidRow(
              h4("Model Summary"),
              tableOutput("naiveBayesTable")
          )
      )
  })  
  
  #Neural Networks Tab
  output$fields.nnet <- renderUI({
      fluidPage(
          fluidRow(
              h4("Model Summary"),
              tableOutput("nnetTable")
          )
      )
  })
  
  #SVM Tab
  output$fields.svm <- renderUI({
    fluidPage(
      fluidRow(
        h4("Model Summary"),
        tableOutput("svmTable")
      )
    )
  })  
  
  #This shows all the contents(Train + Test) from the dataset
  output$contents <- DT::renderDataTable({
      DT::datatable(selectData(input))
  })
  
  #This shows summary of all the contents(Train + Test) from the dataset
  output$summary <- renderPrint({
      summary(selectData(input))
  })
  
  ##-------------------------------------------------
  logitModel <- eventReactive(input$actionTrain, {
      logitFunc(input)
  })
  
  logitModelCoeff <- eventReactive(input$actionTrain, {
    as.data.frame(summary(logitModel())$coeff)
  })
  
  logitModelTable <- eventReactive(input$actionTrain, {
    as.data.frame(predict.logit.fulldata(input, input$in2))
  })
  
  output$nPlotLogistic <- renderPlot({
      plot(logitModel())
  })
  
  output$nTextLogistic <- renderTable({
    logitModelCoeff()
  })

  #Confusion Matrix for Logistic
  output$logitTable <- renderTable({
    logitModelTable()
  })
  ##-------------------------------------------------  
  naiveBayesModel <- eventReactive(input$actionTrain, {
      naiveBayesFunc(input)
  })
 
  naiveBayesModelTable <- eventReactive(input$actionTrain, {
    as.data.frame(predict.naivebayes.fulldata(input, input$in2))
  })
  
  #Confusion Matrix for Naive Bayes
  output$naiveBayesTable <- renderTable({
    naiveBayesModelTable()
  })
  ##---------------------------------------------------
  nnetModel <- eventReactive(input$actionTrain, {
      nnetFunc(input)
  })
  
  nnetModelTable <- eventReactive(input$actionTrain, {
    as.data.frame(predict.nnet.fulldata(input, input$in2))
  })  
  
  #Confusion Matrix for Neural Networks
  output$nnetTable <- renderTable({
    nnetModelTable()
  })
  #-----------------------------------------------------------------------------
  svmModel <- eventReactive(input$actionTrain, {
    svmFunc(input)
  })
  
  svmModelTable <- eventReactive(input$actionTrain, {
    as.data.frame(predict.svm.fulldata(input, input$in2))
  })  
  
  #Confusion Matrix for SVM
  output$svmTable <- renderTable({
    svmModelTable()
  })
  #-----------------------------------------------------------------------------
  # ROC Curves
  roc <- eventReactive(input$actionTrain,{
    test_data <- getTestData(input)
    
    # Logistic Regression
    classifier_logistic <- logitModel()
    predict_logistic <- predict(classifier_logistic, newdata = test_data ,type = 'response')
    dependentVar <- all.vars(classifier_logistic$formula)[1]
    pred_logistic <- prediction(predict_logistic, test_data[,dependentVar])
    perf_logistic <- performance(pred_logistic, 'tpr', 'fpr')
    
    # Naive Bayes
    classifier_Nb <- naiveBayesModel()
    predict_nb <- predict(classifier_Nb, newdata = test_data, type="raw")
    predict_nb <- predict_nb[,2]  ## c(p0 , p1)
    pred_nb <- prediction( predict_nb,  test_data[,dependentVar])
    perf_nb <- performance(pred_nb,"tpr","fpr")
    
    #ANN
    classifier_nnet <- nnetModel()
    predict_nnet <- predict(classifier_nnet, newdata = test_data)
    predict_nnet <- predict_nnet[,1]
    pred_nnet <- prediction(predict_nnet, test_data[,dependentVar] )
    perf_nnet <- performance(pred_nnet,"tpr","fpr")
    
    #SVM
    classifier_svm <- svmModel()
    predict_svm <- predict(classifier_svm, newdata = test_data)
    pred_svm <- prediction(as.numeric(predict_svm), test_data[,dependentVar] )
    perf_svm <- performance(pred_svm,"tpr","fpr")
    
    # Plots
    plot(perf_logistic, main = "ROC Curve", col = 'red', text.adj = c(-0.2,1.7))
    plot(perf_nb , add = TRUE, col = 'green')
    plot(perf_nnet , add = TRUE, col = 'blue')
    plot(perf_svm , add = TRUE, col = 'black')
    legend("bottomright", 
           legend = c("Logistic", "Naive Bayes", "Neural Net", "SVM"),
           col = c("red", "green", "blue", "black"),
           pt.cex = 2, 
           cex = 1.2, 
           text.col = "black", 
           horiz = F
    )    
  })
  
  output$nPlotClassifierROC <- renderPlot({
    roc()
  })  
  #-----------------------------------------------------------------------------
  
  #Data Selection Tab
  output$dataselector <- renderUI({
  fluidPage(
      fluidRow(
          column(12,      
                    # Horizontal line ----
                    tags$br(),
                    
                    radioButtons("radio", label = h5("Use example data or upload your data:"),
                               choices = list("Load Example dataset" = 1, 
                                              "Upload your dataset" = 2), 
                               selected = 1),
                    
                    # Horizontal line ----
                    tags$hr(),
                    
                    # Input: Select a file ----
                    fileInput("file1", h5("Choose CSV File"),
                            multiple = TRUE,
                            accept = c("text/csv",
                                       "text/comma-separated-values,text/plain",
                                       ".csv")),
                    
                    tabsetPanel(
                      tabPanel("Data Snapshot", DT::dataTableOutput("contents")),
                      tabPanel("Data Summary", verbatimTextOutput("summary")),
                      tabPanel("Missing Data Pattern", "This panel is intentionally left blank")
                    ),
                 tags$br()
                )
            )
        )
  })
  
  #-----------------------------------------------------------------------------
  #Model Configuration Tab
  output$fields <- renderUI({
    fluidPage(
      fluidRow(
        column(4,
          # Horizontal line ----
          tags$br(),
          h4("Step 1"),
          h5("Choose Predictors:"),
          chooserInput("mychooser", "Available frobs", "Selected frobs",
                       names(selectData(input)), c(), size = 10, multiple = TRUE
          )#,
          #tags$br(),
          #verbatimTextOutput("selection")
        ),
        column(4,
               # Horizontal line ----
               tags$br(),
               h4("Step 2"),
               h5("Choose Outcome Variable:"),
               selectInput('in2', 'Options', names(selectData(input)), selectize=FALSE),
               verbatimTextOutput('out2')
        ),
        column(4,
               # Horizontal line ----
               tags$br(),
               h4("Step 3"),
               h5("Choose Prediction Models:"),
               selectInput('in3', 'Options', c("Logistic", "Naive Bayes", "Neural Networks", "SVM"), multiple=TRUE, selectize=TRUE)#,
               #verbatimTextOutput('out3')
        )
      ),
      fluidRow(
        tags$hr(),
        column(4,
               # Horizontal line ----
               tags$br(),
               h4("Step 4"),
               h5("Choose Data Imputation Method:"),
               selectInput('in4', 'Options', c("Predictive Mean Matching", "Option2"), selectize=FALSE)
        ),
        column(4,
               # Horizontal line ----
               tags$br(),
               h4("Step 5"),
               h5("Train the Model(s):"),
               tags$br(),
               actionButton("actionTrain", label = "Train Now !", class = "btn-primary")
        ),
        column(4,
               # Horizontal line ----
               tags$br(),
               h4("Step 6"),
               h5("Validate the Model(s):"),
               tags$br(),
               actionButton("actionValidate", label = "Validate Now !", class = "btn-primary")
        )
      ),
      fluidRow(
        tags$hr(),
        h4("Model Summary"),
        plotOutput("nPlotClassifierROC")
      )
    )
  })
  #-----------------------------------------------------------------------------
  #Model Prediction Tab
  output$modelprediction <- renderUI({
    fluidPage(
      # Application title
      h4("Model Predictors"),
      sidebarLayout(
        # Sidebar with a slider input
        sidebarPanel(
          deriveInputControls(input),
          actionButton("actionPredict", label = "Predict Now ! ", class = "btn-primary")
        ),
        # Show a table of the predicted values
        mainPanel(
          tableOutput("predictedValuesTable")
        )
      )
    )
  })
  
  #Create table of predicted values for 4 models
  predicted.values.table <- eventReactive(input$actionPredict,{
    inputDataFrame <- constructInputDataFrame(input)
    
    logit.predicted.value <- predict.single.value(logitFunc(input), inputDataFrame)
    naivebayes.predicted.value <- predict.single.value(naiveBayesFunc(input), inputDataFrame)
    nnet.predicted.value <- predict.single.value(nnetFunc(input), inputDataFrame)
    svm.predicted.value <- predict.single.value(svmFunc(input), inputDataFrame)
    
    Model <- c("Logistic", "NaiveBayes", "NeuralNet", "SVM")
    PredictedValue <- c(logit.predicted.value, naivebayes.predicted.value, nnet.predicted.value, svm.predicted.value)
    data.frame(Model, PredictedValue)
  })
  
  output$predictedValuesTable <- renderTable({
    predicted.values.table()
  })
  #-----------------------------------------------------------------------------
}