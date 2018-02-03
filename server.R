
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
                      tabPanel("Data Summary", verbatimTextOutput("summary"))
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
               selectInput('in4', 'Options', c("Mice"), selectize=FALSE)
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
        plotOutput("nPlotLogistic")
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
          sliderInput("var1",
                      "Glucose:",
                      min = min(PimaIndiansDiabetes$glucose),
                      max = max(PimaIndiansDiabetes$glucose),
                      value = median(PimaIndiansDiabetes$glucose)),
          sliderInput("var2",
                      "Insulin:",
                      min = min(PimaIndiansDiabetes$insulin),
                      max = max(PimaIndiansDiabetes$insulin),
                      value = median(PimaIndiansDiabetes$insulin))          
        ),
        # Show a plot of the generated distribution
        mainPanel(
          plotOutput("distPlot")
        )
      )
    )
  })
  
  #Sample Code, Take it out!
  output$distPlot <- renderPlot({
    hist(rnorm(input$var1))
  })
  #-----------------------------------------------------------------------------
}