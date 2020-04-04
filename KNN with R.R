# Project 1
# 1. Consider the training and test data in the files 1-tranining-data.csv
# and 1-test-data.csv, respectively, for a classification problem with two classes.
# 1.(a) Fit KNN with K = 1,6,...,496.

# Brief look at the data
train.1 <- read.csv('/1-training_data.csv')
test.1 <- read.csv('/1-test_data.csv')

str(train.1)
head(train.1)

str(test.1)
head(test.1)

# split data into X's and Y's 
# training data
train.X.1 <- train.1[,1:2]
train.Y.1 <- train.1[,3]
dim(train.X.1)
head(train.X.1)
head(train.Y.1)

# test data
test.X.1 <- test.1[,1:2]
test.Y.1 <- test.1[,3]
dim(test.X.1)
head(test.X.1)
head(test.Y.1)

plot(train.X.1, main = 'Plot of training data', xlab = "x.1", ylab = "x.2", 
     col = ifelse(train.Y.1 ==	"yes", "green", "red"))
legend("bottomleft", lty = 1, col = c("green", "red"), legend = c("yes", "no"), cex = 0.75)

# Apply KNN
library(class)

ks.1 <- seq(1, 496, by = 5)   # define values of k that we want to try
nks.1 <- length(ks.1)     # number of k's we want to try
# get a numeric vector of length same as number of k's we want to try to store error rates
err.rate.train.1 <- numeric(length = nks.1)        # store training error rate in this variable 
err.rate.test.1 <- numeric(length = nks.1)

# Assign ks to names of two error rate vectors to make them named numetic vectors
names(err.rate.train.1) <- names(err.rate.test.1) <- ks.1   

# Fit KNN with K = 1, 6, ..., 494
for (i in seq(along = ks.1)) {     # seq(along = ks): length of ks
  set.seed(1)
  mod.train.1 <- knn(train.X.1, train.X.1, train.Y.1, k = ks.1[i])
  set.seed(1)
  mod.test.1 <- knn(train.X.1, test.X.1, train.Y.1, k = ks.1[i])
  err.rate.train.1[i] <- mean(mod.train.1 != train.Y.1)
  err.rate.test.1[i] <- mean(mod.test.1 != test.Y.1)
}

###########################################################################################
# (b) Plot training and test error rates against K. Explain what you observe. 
# Is it consistent with what you expect from the class?

# Plot training and test error rates against K
plot(ks.1, err.rate.train.1, main = 'Training & Test Error Rates against K',
     xlab = "Number of nearest neighbors", ylab = "Training Error Rates", type = "b", 
     ylim = range(c(err.rate.train.1, err.rate.test.1)), col = "blue", pch = 20)
lines(ks.1, err.rate.test.1, type="b", col="purple", pch = 20)
legend("bottomright", lty = 1, col = c("blue", "purple"), legend = c("training", "test"))

###########################################################################################
# (c) What is the optimal value of K? 
# What are the training and test error rates associated with the optimal K?

result.1 <- data.frame(ks.1, err.rate.train.1, err.rate.test.1)
result.opt.1<-result.1[err.rate.test.1 == min(result.1$err.rate.test.1), ]
dim(result.opt.1)

###########################################################################################
# (d) Make a plot of the training data that also shows the decision boundary for the optimal K.
# Comment on what you observe. Does the decision boundary seem sensible?
  
# knn function could compute proportions of class 1 for each point
n.grid.1 <- 50
# compute the probability for each point in the grid 
# seq(from, to, length)
# set up 50 equal space points on x1
x1.grid.1 <- seq(f = min(train.X.1[, 1]), t = max(train.X.1[, 1]), l = n.grid.1) 
# set up 50 equal space points on x2
x2.grid.1 <- seq(f = min(train.X.1[, 2]), t = max(train.X.1[, 2]), l = n.grid.1)
# make a 50*50 grid (could make more points) 
# 100*100 grid would be more smooth but computation would spend more time.
grid.1 <- expand.grid(x1.grid.1, x2.grid.1)

# Fit knn with k = 256 (best k value) and get probabilities for class 1 for each point in the grid
k.opt.1 <- 256
set.seed(1)
mod.opt.1 <- knn(train.X.1, grid.1, train.Y.1, k = k.opt.1, prob = T)
prob.1 <- attr(mod.opt.1, "prob") # prob is voting fraction for winning class
prob.1 <- ifelse(mod.opt.1 == "yes", prob.1, 1 - prob.1) # now it is voting fraction for y == "yes"
prob.1 <- matrix(prob.1, n.grid.1, n.grid.1)

# Plot the contour
plot(train.X.1, main = "Decision Boundary for the Optimal K = 256", 
     col = ifelse(train.Y.1 == "yes", "green", "red"))
contour(x1.grid.1, x2.grid.1, prob.1, levels = 0.5, labels = "", 
        xlab = "", ylab = "",  add = T)

###########################################################################################
# Problem 2
# install the keras package with its TensorFlow backend
# devtools::install_github("rstudio/keras")

#library(keras)
#install_keras()

#library(keras)
#cifar <- dataset_cifar10()     # read the data
#str(cifar)

x.train <- read.csv('/cifar_x_train.csv')
y.train <- read.csv('/cifar_y_train.csv')

x.test <- read.csv('/cifar_x_test.csv')
y.test <- read.csv('/cifar_y_test.csv')

# Check the strtures of the data sets
str(x.train)
str(y.train)
str(x.test)
str(y.test)

# Change the datatype of y
y.train <- y.train[,1]
length(y.train)
y.train <- as.factor(y.train)
str(y.train)

class(y.test[,1])
y.test <- y.test[,1]
length(y.test)
y.test <- as.factor(y.test)
str(y.test)

# (a) Fit KNN with K = 50, 100, 200, 300, 400 and examine the test error rates. 
# (Feel free to explore additional values of K.)
# Apply KNN
library(class)
ks.2 <- c(1, 25, 50, 100, 150, 200, 300, 400)  # define values of k that we want to try
# length(ks.2)
# ks.2
nks.2 <- length(ks.2)     # number of k's we want to try
nks.2

# get a numeric vector of length same as number of k's we want to try to store error rates
# err.rate.train.2 <- numeric(length = nks.2)        # store training error rate in this variable 
err.rate.test.2 <- numeric(length = nks.2)

# Assign ks to names of two error rate vectors to make them named numetic vectors
names(err.rate.train.2) <- names(err.rate.test.2) <- ks.2  
# print(err.rate.train.2)
print(err.rate.test.2)

library(class)
# Fit KNN with 1st value of K
i <- 1
# set.seed(1)
# mod.train.2 <- knn(x.train, x.train, y.train, k = ks.2[i])
set.seed(1)
mod.test.2 <- knn(x.train, x.test, y.train, k = ks.2[i])
# err.rate.train.2[i] <- mean(mod.train.2 != y.train)
err.rate.test.2[i] <- mean(mod.test.2 != y.test)
cat("i = ", i, '\n')
cat("Error rates for k = ", ks.2[i], '\n')
#cat("Training error rate = ", err.rate.train.2[i], '\n')
cat("Test error rate = ", err.rate.test.2[i], '\n')

# Fit KNN with 2nd value of K
i <- 2
# set.seed(1)
# mod.train.2 <- knn(x.train, x.train, y.train, k = ks.2[i])
set.seed(1)
mod.test.2 <- knn(x.train, x.test, y.train, k = ks.2[i])
# err.rate.train.2[i] <- mean(mod.train.2 != y.train)
err.rate.test.2[i] <- mean(mod.test.2 != y.test)
cat("i = ", i, '\n')
cat("Error rates for k = ", ks.2[i], '\n')
# cat("Training error rate = ", err.rate.train.2[i], '\n')
cat("Test error rate = ", err.rate.test.2[i], '\n')

# Fit KNN with 3rd value of K
i <- 3
# set.seed(1)
# mod.train.2 <- knn(x.train, x.train, y.train, k = ks.2[i])
set.seed(1)
mod.test.2 <- knn(x.train, x.test, y.train, k = ks.2[i])
# err.rate.train.2[i] <- mean(mod.train.2 != y.train)
err.rate.test.2[i] <- mean(mod.test.2 != y.test)
cat("i = ", i, '\n')
cat("Error rates for k = ", ks.2[i], '\n')
#cat("Training error rate = ", err.rate.train.2[i], '\n')
cat("Test error rate = ", err.rate.test.2[i], '\n')

# Fit KNN with 4th value of K
i <- 4
# set.seed(1)
# mod.train.2 <- knn(x.train, x.train, y.train, k = ks.2[i])
set.seed(1)
mod.test.2 <- knn(x.train, x.test, y.train, k = ks.2[i])
# err.rate.train.2[i] <- mean(mod.train.2 != y.train)
err.rate.test.2[i] <- mean(mod.test.2 != y.test)
cat("i = ", i, '\n')
cat("Error rates for k = ", ks.2[i], '\n')
#cat("Training error rate = ", err.rate.train.2[i], '\n')
cat("Test error rate = ", err.rate.test.2[i], '\n')

# Fit KNN with 5th value of K
i <- 5
# set.seed(1)
# mod.train.2 <- knn(x.train, x.train, y.train, k = ks.2[i])
set.seed(1)
mod.test.2 <- knn(x.train, x.test, y.train, k = ks.2[i])
# err.rate.train.2[i] <- mean(mod.train.2 != y.train)
err.rate.test.2[i] <- mean(mod.test.2 != y.test)
cat("i = ", i, '\n')
cat("Error rates for k = ", ks.2[i], '\n')
#cat("Training error rate = ", err.rate.train.2[i], '\n')
cat("Test error rate = ", err.rate.test.2[i], '\n')

# Fit KNN with 6th value of K
i <- 6
# set.seed(1)
# mod.train.2 <- knn(x.train, x.train, y.train, k = ks.2[i])
set.seed(1)
mod.test.2 <- knn(x.train, x.test, y.train, k = ks.2[i])
# err.rate.train.2[i] <- mean(mod.train.2 != y.train)
err.rate.test.2[i] <- mean(mod.test.2 != y.test)
cat("i = ", i, '\n')
cat("Error rates for k = ", ks.2[i], '\n')
#cat("Training error rate = ", err.rate.train.2[i], '\n')
cat("Test error rate = ", err.rate.test.2[i], '\n')

# Fit KNN with 7th value of K
i <- 7
# set.seed(1)
# mod.train.2 <- knn(x.train, x.train, y.train, k = ks.2[i])
set.seed(1)
mod.test.2 <- knn(x.train, x.test, y.train, k = ks.2[i])
# err.rate.train.2[i] <- mean(mod.train.2 != y.train)
err.rate.test.2[i] <- mean(mod.test.2 != y.test)
cat("i = ", i, '\n')
cat("Error rates for k = ", ks.2[i], '\n')
#cat("Training error rate = ", err.rate.train.2[i], '\n')
cat("Test error rate = ", err.rate.test.2[i], '\n')

# Fit KNN with 8th value of K
i <- 8
set.seed(1)
mod.test.2 <- knn(x.train, x.test, y.train, k = ks.2[i])
err.rate.test.2[i] <- mean(mod.test.2 != y.test)
cat("i = ", i)
cat("Error rates for k = ", ks.2[i], '\n')
cat("Test error rate = ", err.rate.test.2[i], '\n')

# Print training and test error rates
print("Test error rate: ")
print(err.rate.test.2)

# Plot test error rates against k
plot(ks.2, err.rate.test.2, main = "Test Error Rates against K",
     xlab = "Number of nearest neighbors", ylab = "Test Error Rates", type = "b", 
     ylim = range(c(err.rate.test.2)), col = "blue", pch = 20)

# Get error rates for test data
result.2 <- data.frame(ks.2, err.rate.test.2)
result.2[err.rate.test.2 == min(result.2$err.rate.test.2), ]

############################################################################################
# (b) For the best value of K (among the ones you have explored), 
# examine the confusion matrix and comment on your findings.

# Assign best value of K
# cat("Best value of K: k.opt.2")
k.opt.2 <- 25
set.seed(1)

# Predict for test data
mod.test.opt.2 <- knn(x.train, x.test, y.train, k = k.opt.2)
table(mod.test.opt.2, y.test)
mean(mod.test.opt.2 != y.test)
