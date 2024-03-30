install.packages("VIM")
install.packages("e1071")
install.packages("caret")
install.packages("Metrics")
install.packages("ggplot2")
library(e1071)
library(caret)
library(VIM)
library(Metrics)
library(ggplot2)
df<-read.csv('C:\\Users\\Pritom Chandra Dey\\Desktop\\Data Science\\car_evaluation.csv')
summary(df) 
any_missing<-any(is.na(df))
missing_in_each_column<-sapply(df, function(x) sum(is.na(x)))
print(any_missing)
print(missing_in_each_column)
aggr_plot<-aggr(df, col = c('green', 'red'), numbers = TRUE, sortVars = TRUE, labels = names(df), cex.axis = 0.7, gap = 3)
barplot(table(df$class), col = "skyblue", main = "Class Distribution")

str(df)
df_1<-df
unique(df_1$doors)
unique(df_1$persons)
unique(df_1$lug_boot)
unique(df_1$safety)

df_1$doors<-gsub("2", "two", df_1$doors)
df_1$doors<-gsub("3", "three", df_1$doors)
df_1$doors<-gsub("4", "four", df_1$doors)
df_1$persons<-gsub(2, "two", df_1$persons)
df_1$persons<-gsub(4, "four", df_1$persons)

response_var <- df_1[["class"]]
features <- df_1[, !names(df_1) %in% "class"]
chi_square_results <- lapply(features, function(feature) {
  chisq.test(feature, response_var)
})

print(chi_square_results)

p_values <- sapply(chi_square_results, function(result) result$p.value)
significant_attributes <- names(p_values[p_values < 0.05])

naive_bayes_model <- naiveBayes(as.formula(paste("class", "~", 
                                                 paste(significant_attributes, 
                                                       collapse = "+"))), df_1)

set.seed(48)
index <- createDataPartition(df_1[["class"]], p = 0.8, list = FALSE)
train_data <- df_1[index, ]
test_data <- df_1[-index, ]
naive_bayes_model_train_test <- naiveBayes(as.formula(paste("class", "~", 
                                                            paste(significant_attributes, 
                                                                  collapse = "+"))), train_data)
predictions <- predict(naive_bayes_model_train_test, test_data)
test_data[["class"]] <- factor(test_data[["class"]], levels = levels(predictions))


conf_matrix_tts <- confusionMatrix(predictions, test_data[["class"]])
print(conf_matrix_tts)

accuracy_tts <- conf_matrix_tts$overall["Accuracy"]
cat("Accuracy of Naive Bayes using train-test approach:", round(accuracy_tts * 100, 2), "%\n")
print(conf_matrix_tts)

class_recall <- sapply(unique(test_data[["class"]]), function(class) {
  TP <- conf_matrix_tts$table[class, class]
  FN <- sum(conf_matrix_tts$table[class, ]) - TP
  recall <- TP / (TP + FN)
  return(round(recall, 3))
})
class_names <- unique(df_1[["class"]])
cat("Class\tRecall\n")
for (i in seq_along(class_names)) {
  cat(paste(class_names[i], "\t", class_recall[i], "\n"))
}
class_precision <- sapply(unique(test_data[["class"]]), function(class) {
  TP <- conf_matrix_tts$table[class, class]
  FP <- sum(conf_matrix_tts$table[, class]) - TP
  precision <- TP / (TP + FP)
  return(round(precision, 3))
})
cat("Class\tPrecision\n")
for (i in seq_along(class_names)) {
  cat(paste(class_names[i], "\t", class_precision[i], "\n"))
}
class_f_measure <- sapply(seq_along(class_names), function(i) {
  f_measure <- 2 * (class_precision[i] * class_recall[i]) / (class_precision[i] + class_recall[i])
  return(ifelse(is.nan(f_measure) | is.infinite(f_measure), 0, f_measure))  # Handle potential division by zero
})
cat("Class\tF-Measure\n")
for (i in seq_along(class_names)) {
  cat(paste(class_names[i], "\t", round(class_f_measure[i], 3), "\n"))
}
conf_matrix_df_tts <- as.data.frame(as.table(conf_matrix_tts))
ggplot(conf_matrix_df_tts, aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), vjust = 1) +
  labs(title = "Confusion Matrix of Train-Test Split", x = "Reference", y = "Prediction") +
  scale_fill_gradient(low = "lightgreen", high = "green") +
  theme_minimal()



set.seed(48)
filtered_data <- df_1[, c(significant_attributes, "class")]
formula <- as.formula(paste("class ~", paste(significant_attributes, collapse = " + ")))
num_folds <- 10
folds <- createFolds(filtered_data$class, k = num_folds)
all_predictions <- vector("list", length = num_folds)
all_actual <- vector("list", length = num_folds)

for (fold in 1:num_folds) {
  train_indices <- unlist(folds[-fold])
  test_indices <- unlist(folds[fold])
  train_data_cv <- filtered_data[train_indices, ]
  test_data_cv <- filtered_data[test_indices, ]
  nb_model_cv <- naiveBayes(formula, data = train_data_cv, laplace = 1)
  predictions_cv <- predict(nb_model_cv, newdata = test_data_cv)
  all_predictions[[fold]] <- as.vector(predictions_cv)
  all_actual[[fold]] <- as.vector(test_data_cv$class)
}

predictions_cv <- do.call(c, all_predictions)
actual_cv <- do.call(c, all_actual)

conf_matrix_cv <- table(predictions_cv, actual_cv)
print(conf_matrix_cv)
accuracy_cv <- sum(diag(conf_matrix_cv)) / sum(conf_matrix_cv)
cat("Accuracy of Naive Bayes using 10-fold CV approach:", round(accuracy_cv * 100, 2), "%\n")

TP_cv <- diag(conf_matrix_cv)
FP_cv <- rowSums(conf_matrix_cv) - TP_cv  
FN_cv <- colSums(conf_matrix_cv) - TP_cv  

precision_cv <- TP_cv / (TP_cv + FP_cv)
precision_cv <- round(precision_cv, 3)

recall_cv <- TP_cv / (TP_cv + FN_cv)
recall_cv <- round(recall_cv, 3)

f_measure_cv <- (2 * precision_cv * recall_cv) / (precision_cv + recall_cv)
f_measure_cv <- round(f_measure_cv, 3)

results_cv <- data.frame(Precision = precision_cv,
                         Recall = recall_cv,
                         F_measure = f_measure_cv)
print(results_cv)

conf_matrix_df_cv<-conf_matrix_cv

conf_matrix_df_cv <- as.data.frame(as.table(conf_matrix_df_cv))
ggplot(conf_matrix_df_cv, aes(x = actual_cv, y = predictions_cv, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), vjust = 1) +
  labs(title = "Confusion Matrix of 10-fold CV", x = "Reference", y = "Prediction") +
  scale_fill_gradient(low = "lightblue", high = "navy") +
  theme_minimal()

