#!/usr/bin/env Rscript
args = commandArgs(trailingOnly=TRUE)

library("caret")
library("party") 

setwd("/home/autoprep")
# Read the experiments results
# The columns are: DatasetID, meta-features_1, ..., meta-features_n, algorithm, transformation_1, ..., transformation_n
# - dataset ID: the Id of the dataset
# - meta-features_1, ..., meta-features_n: the value of the dataset meta-features
# - algorithm: the used mining algorithm (NB, KNN, RF)
# - transformation_1, ..., transformation_n: the operator (chosen by SMBO) for each of the transformations in the prototype
union <- read.csv(paste("./resources/raw_results/", args[1],"/exploratory_analysis/meta_learning_input.csv", sep = ""), header = TRUE)
# Drop the dataset IDs
union <- union[,-1]

# For each of the transformations we create an ad-hoc training-set, dropping all the columns that regard the other transformations
encode <- union[ , -which(names(union) %in% c("features", "impute", "normalize", "discretize", "rebalance"))]
features <- union[ , -which(names(union) %in% c("encode", "impute", "normalize", "discretize", "rebalance"))]
impute <- union[ , -which(names(union) %in% c("encode", "features", "normalize", "discretize", "rebalance"))]
normalize <- union[ , -which(names(union) %in% c("encode", "features", "impute", "discretize", "rebalance"))]
discretize <- union[ , -which(names(union) %in% c("encode", "features", "impute", "normalize", "rebalance"))]
rebalance <- union[ , -which(names(union) %in% c("encode", "features", "impute", "normalize", "discretize"))]
rebalance <- rebalance[ , -which(names(rebalance) %in% c("MinSkewnessOfNumericAtts", "NumberOfSymbolicFeatures", "NumberOfFeatures", "MinorityClassPercentage"))]

set.seed(30)
ctreeFitFeatures <- train(features ~ ., 
                          data = features, 
                          method = "ctree2", 
                          na.action = na.pass, 
                          trControl = trainControl(method = "cv"))
pdf(paste(c("./resources/artifacts/", args[1],"/Figure10.pdf"), collapse = ""), width = 15, height = 6)
plot(ctreeFitFeatures$finalModel)
dev.off()

ctreeFitRebalance <- train(rebalance ~ ., 
                           data = rebalance, 
                           method = "ctree2", 
                           na.action = na.pass, 
                           trControl = trainControl(method = "cv"))
pdf(paste(c("./resources/artifacts/", args[1],"/Figure11.pdf"), collapse = ""), width = 15, height = 6)
plot(ctreeFitRebalance$finalModel)
dev.off()
