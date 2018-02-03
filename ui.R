
# This is the user-interface definition of a Shiny web application.
# You can find out more about building applications with Shiny here:
#
# http://shiny.rstudio.com
#

library(shiny)
library(DT)
library(shinythemes)
source("chooser.R")

radioSelection <- textOutput("radioSelection")

shinyUI(
  tagList(
    shinythemes::themeSelector(),
    navbarPage(
      theme = "yeti",  # <--- To use a theme, uncomment this
      "Prediction Machine",
      tabPanel("Classification",
               mainPanel(
                 tabsetPanel(
                   tabPanel("Data Selector", uiOutput("dataselector")),
                   tabPanel("Model Configuration", uiOutput("fields")),
                   tabPanel("Model Prediction", uiOutput("modelprediction")),
                   tabPanel("Logistic Regression", uiOutput("fields.lr")),
                   tabPanel("Naive Bayes", uiOutput("fields.nb")),
                   tabPanel("Neural Networks", uiOutput("fields.nnet")),
                   tabPanel("SVM", uiOutput("fields.svm"))
                 )
               )
      ),
      tabPanel("Regression", "This panel is intentionally left blank"),
      tabPanel("Clustering", "This panel is intentionally left blank"),
      tabPanel("Principal Component Analysis", "This panel is intentionally left blank")
    )
  )
)
