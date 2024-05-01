if (!require(dplyr)) install.packages("dplyr")
if (!require(caret)) install.packages("caret", dependencies = TRUE)

library(caret)
library(dplyr)


apple_quality <- read.csv("A:/learning/project/datamining/apple/apple_quality.csv")
print(apple_quality)

dim(apple_quality)

apple_quality <- apple_quality[,-which(names(apple_quality)=="A_id")]

dim(apple_quality)

apple_quality <- apple_quality[-nrow(apple_quality), ]

dim(apple_quality)

missing_count <- colSums(is.na(apple_quality))
print(missing_count)

str(apple_quality)

apple_quality$Acidity <- as.numeric(apple_quality$Acidity)
str(apple_quality)

summary(apple_quality)

numeric_data <- apple_quality[,sapply(apple_quality,is.numeric)]
corr_matrix <- cor(numeric_data)
print(corr_matrix)

pairs(numeric_data)

apple_quality$Quality <- ifelse(apple_quality$Quality == "good",1,
                                ifelse(apple_quality$Quality == "bad",0,apple_quality$Quality))
print(apple_quality)

apple_quality$Quality <- as.numeric(apple_quality$Quality)

min_max_scaling <- function(x){
  return((x - min(x)) / (max(x) - min(x)))
}
apple_quality_subset <- apple_quality[,-ncol(apple_quality)]
scaled_apple_quality_subset <- as.data.frame(lapply(apple_quality_subset,min_max_scaling))
scaled_apple_quality <- cbind(scaled_apple_quality_subset,apple_quality[,ncol(apple_quality)])
print(scaled_apple_quality)

scaled_apple_quality$Quality <- scaled_apple_quality$`apple_quality[, ncol(apple_quality)]`

summary(scaled_apple_quality)

scaled_apple_quality <- scaled_apple_quality[,-which(names(scaled_apple_quality)=="apple_quality[, ncol(apple_quality)]")]

summary(scaled_apple_quality)

print(scaled_apple_quality)

# Assuming scaled_apple_quality is your dataset and it includes a target variable named 'class'
set.seed(123)  # for reproducibility

# Create indices for the training set
trainIndex <- createDataPartition(scaled_apple_quality$Quality, p = 0.7, list = FALSE, times = 1)

# Create the training and testing datasets
trainData <- scaled_apple_quality[trainIndex, ]
testData <- scaled_apple_quality[-trainIndex, ]
# Check the number of rows in each set
nrow(trainData)
nrow(testData)

# Perform PCA on the training data, assuming it is already scaled
pca_train <- prcomp(trainData[, -which(names(trainData) == "Quality")], center = TRUE, scale. = FALSE)

# Examine the summary to decide how many components to retain
summary(pca_train)

# Using the first two principal components
pc_train <- pca_train$x[, 1:2]

# K-means clustering
set.seed(123)
kmeans_result <- kmeans(pc_train, centers = 2, nstart = 25)

# Project the test data onto the PCA model developed from the training data
pc_test <- predict(pca_train, newdata = testData[, -which(names(testData) == "Quality")])

# Retain the same principal components as for the training data
pc_test <- pc_test[, 1:2]

# Function to assign clusters based on the nearest centroid
assign_cluster <- function(point, centers) {
  apply(centers, 1, function(center) sum((point - center)^2)) %>%
    which.min()
}

# Apply this function to each row in the PCA-transformed test data
test_clusters <- apply(pc_test, 1, assign_cluster, centers = kmeans_result$centers)
# Optionally, compare against known labels if available
table(Predicted = test_clusters, Actual = testData$Quality)

# Visualizing the clustering
plot(pc_test, col = test_clusters, pch = 19, main = "Test Data Cluster Assignments")
points(kmeans_result$centers, col = 1:2, pch = 4, cex = 3, lwd = 2)

result_table <- table(Predicted = test_clusters, Actual = testData$Quality)
print(result_table)

# Assuming two classes and two clusters for simplicity
cluster_labels <- apply(result_table, 1, function(row) names(which.max(row)))
names(cluster_labels) <- levels(testData$Quality)[cluster_labels]  # Match to the actual class names
print(cluster_labels)

mapped_test_clusters <- factor(test_clusters, labels = names(cluster_labels))

if (length(cluster_labels) > 0) {
  mapped_test_clusters <- factor(test_clusters, labels = cluster_labels)
  print(table(mapped_test_clusters, testData$Quality))
} else {
  print("Cluster labels could not be determined correctly.")
}

if (exists("mapped_test_clusters")) {
  accuracy <- sum(mapped_test_clusters == testData$Quality) / length(testData$Quality)
  print(paste("Accuracy:", accuracy))
}


library(caret)
confusionMatrix(data = mapped_test_clusters, reference = testData$Quality)



