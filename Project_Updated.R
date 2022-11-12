### -------------------- Credit Card fraud Detection ----------------------- ###

# ---------------------------- Importing Libraries --------------------------- #

library(class)      # for classification
library(dplyr)      # for data manipulation
library(ranger)     # for faster implementation of random forests
library(caret)      # for classification and regression training
library(caTools)    # for splitting data into training and test set
library(data.table) # for converting data frame to table for faster execution
library(ggplot2)    # for basic plot
library(corrplot)   # for plotting correlation plot between elements
library(rpart)      # for regression trees
library(rpart.plot) # for plotting decision tree
library(e1071)      # for functions for statistic and probabilistic 
library(thematic)
library(plotly)

# -------------------- Importing the credit card data set -------------------- #

getwd()                                            # check the working directory
setwd("D:/7th Semester/R Project")                 # set the working directory
credit_card <- read.csv("Credit_Card_Cleaned.csv") # reading the data set
credit_card = credit_card[1:18]                    # feature selection
View(credit_card)                                  # view the data set 

# ----------------------------- Data Exploration ----------------------------- #

str(credit_card)    # structure of data set
head(credit_card)   # print first 6 obs of the data set
tail(credit_card)   # print last 6 obs of the data set 
names(credit_card)  # display the columns names

# convert the Class to factor as it has 0 (non-frauds) and 1 (frauds)
credit_card$Class = as.factor(credit_card$Class)
# get the distribution of fraud and legit transactions in the data set
table(credit_card$Class)
# get the percentage of fraud and legit transactions in the data set
prop.table(table(credit_card$Class))
# labeling the classes
labels <- c("Legitimate:","Fraud:")
labels <- paste(labels, round(100*prop.table(table(credit_card$Class)),2))
labels
labels <- paste0(labels,"%")
labels

summary(credit_card)        # summarizing the count of the frauds and non-frauds 
sum(is.na(credit_card))     # check for any NA values
summary(credit_card$Amount) # view summary of amount
var(credit_card$Amount)     # view variance of amount column
sd(credit_card$Amount)      #  view the standard deviation of amount column
# renaming column names
colnames(credit_card) <- c("Time","V1","V2","V3","V4","V5","V6","V7","V8","V9",
                           "V10","V11","V12","V13","V14","V15","Amount","Class")

# ----------------------------- Data visualization --------------------------- #

# pie chart of credit card transactions
pie(table(credit_card$Class),labels, col = c("dark green","red"),
    main = "Pie chart of Credit Card Transactions")

# histogram on amounts
hist(credit_card$Amount, col = 'light green')

# amount less than 100
hist(credit_card$Amount[credit_card$Amount < 100], col = 'sky blue')

# distribution of class labels
common_theme <- theme(plot.title = element_text(hjust = 0.5, face = "bold"))
p <- ggplot(credit_card, aes(x = Class)) + geom_bar() 
       + ggtitle("Number of class labels") + common_theme
print(p)

# distribution of transactions across time
credit_card %>% ggplot(aes(x = Time, fill = factor(Class))) + 
  geom_histogram(bins = 100) + 
  labs(x = 'Time in seconds since first transaction', y = 'No. of transactions') + 
  ggtitle('Distribution of time of transaction by class') + 
  facet_grid(Class ~ ., scales = 'free_y') + common_theme

# distribution of transactions amount by class 
q <- ggplot(data, aes(x = Class, y = Amount)) 
            + geom_boxplot() 
            + ggtitle("Distribution of transaction amount by class") 
            + common_theme
print(q)

# correlation of anonymity variables and amount
credit_card$Class <- as.numeric(credit_card$Class)
correlations = cor(credit_card[,-1], method = "pearson")
corrplot(correlations, number.cex = .9, method = "circle", type = "full",
         tl.cex=0.8,tl.col = "black")

# ---------------------------- Data Pre-processing --------------------------- #

# scaling the data using standardization and remove the first column (time)
credit_card$Amount <- scale(credit_card$Amount)
new_credit_card <- credit_card[, -c(1)]
head(new_credit_card)
# change 'Class' variable to factor
new_credit_card$Class <- as.factor(new_credit_card$Class)
levels(new_credit_card$Class) <- c("Not Fraud", "Fraud")
new_credit_card

# ------------------------------ Data Modeling ------------------------------- #

set.seed(123)
# taking 80 of values as training data and rest 20 are test data
data_sample = sample.split(new_credit_card$Class,SplitRatio=0.70)
# training data
train_data = subset(new_credit_card,data_sample==TRUE)
dim(train_data)
# testing data
test_data = subset(new_credit_card,data_sample==FALSE)
dim(test_data)

# visualize the training data
train_data %>% ggplot(aes(x = factor(Class), y = prop.table(stat(count)), 
                      fill = factor(Class))) + geom_bar(position = "dodge") +
                      scale_y_continuous(labels = scales::percent) +
                      labs(x = 'Class', y = 'Percentage',
                          title = 'Training Class distributions') + theme_grey()

# ---------------------------- Fitting Algorithms ---------------------------- # 

         # --------------------- KNN Algorithm ------------------- #

credit_card =credit_card[1:18]    #feature scaling
str(credit_card)
# splitting data 
indexs = createDataPartition(credit_card$Class,p=.70,list = F)
indexs
View(indexs)
# training data
train_data = credit_card[indexs,]
train_data
# testing data
test_data = credit_card[-indexs,]
test_data
x_train = train_data[,-18]
y_train = train_data[,18]
x_test = test_data[,-18]
y_test = test_data[,18]
str(y_test)

# knn model
knn_algo = knn(x_train, x_test, y_train, k = 54) 
# k = 54 as the square root of 2999 is approx 54
knn_algo

# Confusion Matrix
cm = table(test_data$Class,knn_algo)
cm

# Model Evaluation
confusionMatrix(cm)

# Accuracy
acc_knn <- sum(diag(cm)) / sum(cm)
print(paste('Accuracy for test is found to be', round(acc_knn,3) ,"%"))



     # --------------------- Naive Bayer's Algorithm ------------------- #

# Splitting data into train and test data
split <- sample.split(credit_card, SplitRatio = 0.75)
train_data <- subset(credit_card, split == "TRUE")
test_data <- subset(credit_card, split == "FALSE")

# Feature Scaling
train_scale <- scale(train_data[, 2:18])
test_scale <- scale(test_data[, 2:18])

# Fitting Naive Bayes Model to training data set

set.seed(120)  # Setting Seed
classifier_data <- naiveBayes(Class~ ., data = train_data)
classifier_data

# Predicting on test data'
naive_pred <- predict(classifier_data, newdata = test_data)

# Confusion Matrix
cm <- table(test_data$Class, naive_pred)
cm

# Model Evaluation
confusionMatrix(cm)

# Accuracy
acc_naive <- sum(diag(cm)) / sum(cm)
print(paste('Accuracy for test is found to be', round(acc_naive,3) ,"%"))




    # --------------------- Decision Tree Algorithm ------------------- #

# Splitting data into train and test data
split <- sample.split(credit_card, SplitRatio = 0.75)
train_data <- subset(credit_card, split == "TRUE")
test_data <- subset(credit_card, split == "FALSE")

# Fitting Decision Tree Model to training data set
decisionTree_model <- rpart(Class ~ . , credit_card, method = 'class')
predicted_val <- predict(decisionTree_model, credit_card, type = 'class')
probability <- predict(decisionTree_model, credit_card, type = 'prob')
rpart.plot(decisionTree_model)

predict_model <- predict(decisionTree_model,test_data,type = 'class')
m_at <- table(test_data$Class,predict_model)
m_at

# Confusion Matrix
confusionMatrix(table(test_data$Class,predict_model))

# Accuracy
acc_dt <- sum(diag(m_at)) / sum(m_at)
print(paste('Accuracy for test is found to be',round(acc_dt,4),"%"))


      # --------------------- Neutral Network ------------------- #

credit_card$Class = as.numeric(credit_card$Class)
hist(credit_card$Class,col = 'light blue')

set.seed(123)
# taking 80 of values as training data and rest 20 are test data
data_sample = sample.split(credit_card$Class,SplitRatio=0.70)
# training data
train_data = subset(credit_card,data_sample==TRUE)
dim(train_data)
# testing data
test_data = subset(credit_card,data_sample==FALSE)
dim(test_data)

library(neuralnet)
#intsall the package neuralnet for neural network implementation and 
#load it to the program

credit_model <- neuralnet(Class ~ .,data = train_data)
#training the simplest multilayer feedforward network with 
#only a single hidden node

plot(credit_model)

model_results = neuralnet::compute(credit_model,test_data[2:18])
#It returns a list with two components: $neurons, which stores the
#neurons for each layer in the network, and 
#$net.result, which stores the predicted values.
predicted_class <- model_results$net.result

cor(predicted_class, test_data$Class)
#Correlations close to 1 indicate strong linear relationships between two variables.

credit_model2 <- neuralnet(Class ~ ., data = train_data, hidden = c(5,4))
plot(credit_model2)

model_results2 <- neuralnet::compute(credit_model2, test_data[2:18])
predicted_class2 <- model_results2$net.result
cor(predicted_class2, test_data$Class)


# --------------------- Analyzing the Accuracy of Model ---------------------- # 


library(ascii)     # used to format as table and outlines
data_framee <- data.frame(Algorithm = rep(c('KNN', 'Naive Bayers', 
                                            'Decision Tree', 
                                            'Single Nuteral Network',
                                            'Multiple Nuteral Network')),
                 Accuracy = c("98%","95%","99%","31%","45%"))
data_framee

# print the table of accuracy using ASCII library
print(ascii(df), type = "rest")


# -------------------------------- THE END ----------------------------------- # 
