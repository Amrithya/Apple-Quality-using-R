library(caret)
library(dplyr)

apple_quality <- read.csv("A:/learning/sem 2/data mining/project/apple/apple_quality.csv")

# Finding the shape of the dataset
shape <- dim(apple_quality)
print(shape)

# head() gives us the first few rows of the dataset
head(apple_quality)

# For basic statistics, we use summary()
summary(apple_quality)

# Checking the total number of values in each column
col_values <- sapply(apple_quality, function(x) length(x))

# Checking if there are any missing values in each columns
missing_values <- colSums(is.na(apple_quality))

# Printing the tail of the data set
tail(apple_quality)

# na.omit() removes the rows with missing values
apple_quality <- na.omit(apple_quality)
missing_values <- colSums(is.na(apple_quality))

# Removing the column named "A_id" as it is duplicated
apple_quality <- apple_quality[, !(names(apple_quality) == "A_id")]
head(apple_quality)

# converting type of Acidity to double
apple_quality$Acidity <- as.numeric(apple_quality$Acidity)

# Converting Quality to binary (`GOOD = 1` and `BAD = 0`)
apple_quality$Quality <- ifelse(apple_quality$Quality == "good", 1, 0)

# Checking the distribution of Quality 
quality_counts <- table(apple_quality$Quality)
print(quality_counts)
quality_counts_df <- as.data.frame(quality_counts)
names(quality_counts_df) <- c("Quality", "Count")
ggplot(quality_counts_df, aes(x = factor(Quality), y = Count)) +
  geom_bar(stat = "identity", fill = "skyblue", color = "black") +
  labs(title = "Frequency of 1s and 0s in Quality",
       x = "Quality",
       y = "Count")

# plot_distribution is used to plot the distribution of values in each column
plot_distribution <- function(data, col_name) {
  if(is.numeric(data[[col_name]])) {
    ggplot(data, aes(x = !!sym(col_name))) +
      geom_histogram(binwidth = 0.5, fill = "skyblue", color = "black") +
      labs(title = paste("Distribution of", col_name),
           x = col_name,
           y = "Frequency")
  } else {
    ggplot(data, aes(x = factor(!!sym(col_name)))) +
      geom_bar(fill = "skyblue", color = "black") +
      labs(title = paste("Distribution of", col_name),
           x = col_name,
           y = "Frequency")
  }
}
for (col_name in names(apple_quality)) {
  plots_list[[col_name]] <- plot_distribution(apple_quality, col_name)
}
grid.arrange(grobs = plots_list, ncol = 3)

# Plotting correlation matrix for all numeric columns
correlation_matrix <- cor(apple_quality)
corrplot(correlation_matrix, method = 'square', order = 'AOE', addCoef.col = 'black', tl.pos = 'd',
         cl.pos = 'n', col = COL2('BrBG'))

# Pair wise plotting between all features.
pairs(apple_quality, col = "blue", pch = 16, labels = colnames(apple_quality))

# Plotting boxplot to find outliers
par(mfrow=c(1, ncol(apple_quality)))
for(i in 1:ncol(apple_quality)) {
  if(is.numeric(apple_quality[, i])) {
    boxplot(apple_quality[, i], main=names(apple_quality)[i])
  }
}

# Removing outliers using Z_Scores
apple_quality <- as.data.frame(apple_quality)
outlier_indices <- list()
for (col_name in names(apple_quality)) {
  z_scores <- scale(apple_quality[[col_name]])
  threshold <- 3
  outliers <- which(abs(z_scores) > threshold)
  outlier_indices[[col_name]] <- outliers
}
all_outliers <- unique(unlist(outlier_indices))
apple_quality <- apple_quality[-all_outliers, ]


# Separating target and feature variables
target_variable <- apple_quality[, ncol(apple_quality)]
feature_variables <- apple_quality[, -ncol(apple_quality)]
shape_feature_variables <- dim(feature_variables)

# Splitting the dataset into train-test
train_index <- createDataPartition(target_variable, p = 0.8, list = FALSE)
train_data <- apple_quality[train_index, ]
test_data <- apple_quality[-train_index, ]

#Kmeans

fviz_nbclust(apple_quality, kmeans, method = "silhouette")
set.seed(240)  # for reproducibility
apple_quality$Quality <- as.numeric(apple_quality$Quality)
train_data$Quality <-as.numeric(train_data$Quality)
test_data$Quality <-as.numeric(test_data$Quality)
k2 <- kmeans(apple_quality, centers = 2, nstart = 25)
str(k2)
fviz_cluster(k2, data = apple_quality)
head(apple_quality)
feature_train_data <- train_data[, !names(train_data) %in% "Quality"]
pca_result <- prcomp(feature_train_data, center = TRUE, scale. = TRUE)
summary(pca_result)
plot(pca_result, type = "lines")
biplot(pca_result)
fviz_pca_ind(pca_result)
pca_scores <- pca_result$x
explained_variance <- summary(pca_result)$importance[2,]
cum_explained_variance <- cumsum(explained_variance)
num_components <- which(cum_explained_variance >= 0.8)[1]
pca_scores <- pca_result$x[, 1:num_components]
feature_test_data <- test_data[, !names(test_data) %in% "target_variable"]

test_data_pca <- predict(pca_result, newdata = feature_test_data)

test_data_pca <- test_data_pca[, 1:num_components]

get_cluster_assignment <- function(point, centers) {
  distances <- apply(centers, 1, function(center) sum((point - center)^2))
  return(which.min(distances))
}

test_clusters <- apply(test_data_pca, 1, get_cluster_assignment, centers = kmeans_result$centers)

test_data$predicted_cluster <- test_clusters
test_data$Quality_predicted <- ifelse(test_data$predicted_cluster == 1, 1, 0)

true_clusters <- test_data$cluster # replace with your actual true cluster column name
accuracy <- mean(test_data$Quality == test_data$Quality_predicted)
print(paste("Accuracy of cluster prediction on test data:", accuracy))

#KNN

train_data <- apple_quality[train_index, ]
test_data <- apple_quality[-train_index, ]
train_data$Quality <- as.factor(train_data$Quality)
test_data$Quality <- as.factor(test_data$Quality)

knn_results <- train(Quality ~ ., data = train_data, method = "knn",
                     trControl = train_control,
                     preProcess = "scale",
                     tuneGrid = expand.grid(k = 1:10))

# Extract accuracies and corresponding k values
results <- knn_results$results
k_values <- results$k
accuracies <- results$Accuracy

print(accuracies)

# Create a data frame for plotting
plot_data <- data.frame(k = k_values, Accuracy = accuracies)

# Plot accuracy vs k
ggplot(plot_data, aes(x = k, y = Accuracy)) +
  geom_line() +
  geom_point(shape = 19) +
  labs(title = "KNN Accuracy vs. k",
       x = "Number of Neighbors (k)",
       y = "Accuracy") +
  theme_minimal()

train_control <- trainControl(method = "cv", number = 10)  # 10-fold cross-validation

# Train the model using k = 7
knn_model <- train(Quality ~ ., data = train_data, method = "knn",
                   trControl = train_control,
                   preProcess = "scale",  # Ensure features are scaled
                   tuneGrid = data.frame(k = 7))  # Using k = 7
predictions <- predict(knn_model, newdata = test_data)
accuracy <- sum(predictions == test_data$Quality) / nrow(test_data)
cat("Accuracy of the KNN model with k=7: ", accuracy * 100, "%\n")








