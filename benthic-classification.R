# Load required libraries
library(mlr)
library(raster)
library(Boruta)
library(class)
library(e1071)
library(nnet)
library(randomForest)
library(rattle)

# Import bathymetry and backscatter rasters and assign to variables 'a' and 'b'.
a <- raster("../input/raster/1_bathy.tif")
b <- raster("../input/raster/2_back.tif")

# Create a 3x3 matrix window and assign to variable 'm'
m <- matrix(c(0, 1, 0, 1, 0, 1, 0, 1, 0), nrow = 3)

# Calculate Moran's I for bathymetry and backscatter and assign to variables 'mor1' and 'mor2'.
mor1 <- MoranLocal(a, m) 
mor2 <- MoranLocal(b, m) 

# Calculate roughness for bathymetry and backscatter, assign to variables 'rough1' and 'rough2'.
rough1 <- focal(a, w = m, fun = function(x, ...) max(x) - min(x), pad = TRUE, padValue = NA, na.rm = TRUE)
rough2 <- focal(b, w = m, fun = function(x, ...) max(x) - min(x), pad = TRUE, padValue = NA, na.rm = TRUE)

# Calculate Bathymetric Position Index for bathymetry, assign to variable 'bpi'.
bpi <- terrain(a, opt = 'TPI', neighbors = 8)

# Save resultant rasters to project folder. 
writeRaster(mor1, filename = "../input/raster/8_bathy_moran_i.tif", format = "GTiff", overwrite = TRUE)
writeRaster(mor2, filename = "../input/raster/9_back_moran_i.tif", format = "GTiff", overwrite = TRUE)
writeRaster(rough1, filename = "../input/raster/3_bathy_rough.tif", format = "GTiff", overwrite = TRUE)
writeRaster(rough2, filename = "../input/raster/4_back_rough.tif", format = "GTiff", overwrite = TRUE)
writeRaster(bpi, filename = "../input/raster/15_bpi.tif", format = "GTiff", overwrite = TRUE)

# Open training/testing dataset and assign to variable 'data_full'.
data_full <- read.csv("../input/csv/data_full.csv")

# Split the dataset into two separate matrices (training and testing) for later use.
# Assign to variables 'data_train_all' and 'data_test_all'. Column 17 is omitted, 
# as it refers to whether the data is part of training or testing set.
data_train_all <- data_full[1:656, -17]
data_test_all <- data_full[657:960, -17]

# Remove ‘row.names’ column from ‘data_test_all’.
row.names(data_test_all) <- seq(nrow(data_test_all))

# Inspect datasets
View(data_train_all)
View(data_test_all)

# Rattle is used to explore the data visually.
Rattle()

# Run Boruta on dataset 'data_train' using 'class' column as reference,and
# trace results. Assign results to 'Bor.son' and print.
Boruta(class ~ ., data = data_full, doTrace = 2, pValue = 0.001, mcAdj = TRUE, 
       getImp = getImpRfZ, maxRuns = 300) -> Bor.son
print(Bor.son)

# Print and plot Boruta run stats.
stats <- attStats(Bor.son)
print(stats)
plot(normHits ~ meanZ, col = stats$decision, data = stats)

# Subset the dataset according to features selection results.
data_train_sub <- data_full[1:656, -c(6, 7, 10, 13, 14, 15, 17)]
data_test_sub <- data_full[657:960, -c(6, 7, 10, 13, 14, 15, 17)]

# Create just bathymetry and backscatter matrices.
data_train_bybs <- data_full[1:841, -c(3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)]
data_test_bybs <- data_full[842:1195, -c(3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)]

# Find optimal value 'k' for k-NN_1 model (feature subset).
a <- data_train_sub[1:9]
b <- data_train_sub[, 10]
knn1p <- tune.knn(a, b, k = 1:10, tunecontrol = tune.control(sampling = "cross", performances = TRUE, sampling.aggregate = mean))
summary(knn1p)
plot(knn1p)

# Find optimal value 'k' for k-NN_2 model (full feature set).
c <- data_train_all[1:15]
d <- data_train_all[, 16]
knn2p <- tune.knn(c, d, k = 1:10, tunecontrol = tune.control(sampling = "cross", performances = TRUE, sampling.aggregate = mean))
summary(knn2p)
plot(knn2p)

# Find optimal value 'k' for k-NN_3 model (primary features).
e <- data_train_bybs[1:2]
f <- data_train_bybs[, 3]
knn3p <- tune.knn(e, f, k = 1:10, tunecontrol = tune.control(sampling = "cross", performances = TRUE, sampling.aggregate = mean))
summary(knn3p)
plot(knn3p)

# Find optimal 'nodesize', 'mtry', and 'ntree' values for RF model (feature subset)
RF_1 <- tune.randomForest(class ~ ., data = data_train_sub, nodesize = 1:14, mtry = 1:5, ntree = 100:600, 
                          tunecontrol = tune.control(sampling = "cross", 
                          cross = 10, 
                          performances = TRUE, 
                          sampling.aggregate = mean))
summary(RF_1)
plot(RF_1)

# Find optimal 'nodesize', 'mtry', and 'ntree' values for RF model (full feature set)
RF_2 <- tune.randomForest(class ~ ., data = data_train_all, nodesize = 1:14, mtry = 1:5, ntree = 100:600, 
                          tunecontrol = tune.control(sampling = "cross", 
                          cross = 10, 
                          performances = TRUE, 
                          sampling.aggregate = mean))
summary(RF_2)
plot(RF_2)

# Find optimal 'nodesize', 'mtry', and 'ntree' values for RF model (primary features)
RF_3 <- tune.randomForest(class ~ ., data = data_train_bybs, nodesize = 1:14, mtry = 1:5, ntree = 100:600, 
                          tunecontrol = tune.control(sampling = "cross", 
                          cross = 10, 
                          performances = TRUE, 
                          sampling.aggregate = mean))
summary(RF_3)
plot(RF_3)

# Find optimal 'size' and 'decay' values for ANN model (feature subset).
nnet_1p <- tune.nnet(class ~ ., data = data_train_sub, size = 2^(1:5), decay = 10^(-5:-1), 
                     MaxNWts = 5000, tunecontrol = tune.control(sampling = "cross", 
                     cross = 10, 
                     performances = TRUE, 
                     sampling.aggregate = mean))
summary(nnet_1p)
plot(nnet_1p)

# Find optimal 'size' and 'decay' values for ANN model (full feature set).
nnet_2p <- tune.nnet(class ~ ., data = data_train_all, size = 2^(1:5), decay = 10^(-5:-1), 
                     MaxNWts = 5000, tunecontrol = tune.control(sampling = "cross", 
                     cross = 10, 
                     performances = TRUE, 
                     sampling.aggregate = mean))
summary(nnet_2p)
plot(nnet_2p)

# Find optimal 'size' and 'decay' values for ANN model (primary features).
nnet_3p <- tune.nnet(class ~ ., data = data_train_bybs, size = 2^(1:5), decay = 10^(-5:-1), 
                     MaxNWts = 5000, tunecontrol = tune.control(sampling = "cross", 
                     cross = 10, 
                     performances = TRUE, 
                     sampling.aggregate = mean))
summary(nnet_3p)
plot(nnet_3p)

# Find optimal 'gamma' and 'cost' values for SVM model (feature subset).
SVM_1 <- tune.svm(class ~ ., data = data_train_sub, gamma = 2^(-14:4), cost = 2^(1:14), 
                  tunecontrol = tune.control(sampling = "cross", sampling.aggregate = mean, 
                  nrepeat = 5, 
                  performances = TRUE, 
                  repeat.aggregate = mean))
summary(SVM_1)
plot(SVM_1)

# Find optimal 'gamma' and 'cost' values for SVM model (full feature set).
SVM_2 <- tune.svm(class ~ ., data = data_train_all, gamma = 2^(-14:4), cost = 2^(1:14),
                  tunecontrol = tune.control(sampling = "cross", sampling.aggregate = mean, 
                  nrepeat = 5, 
                  performances = TRUE, 
                  repeat.aggregate = mean)
)
summary(SVM_2)
plot(SVM_2)

# Find optimal 'gamma' and 'cost' values for SVM model (primary features).
SVM_3 <- tune.svm(class ~ ., data = data_train_bybs, gamma = 2^(-14:4),
                  cost = 2^(1:14), tunecontrol = tune.control(sampling = "cross", 
                  sampling.aggregate = mean, 
                  nrepeat = 5, 
                  performances = TRUE, 
                  repeat.aggregate = mean)
)
summary(SVM_3)
plot(SVM_3)

# knn_1 (feature subset)
train_in <- as.matrix(data_train_sub[1:13])
train_out <- data_train_sub[,14]
test_in <- as.matrix(data_test_sub[1:13])
pred_knn_1 <- knn(train_in, test_in, train_out, k = 1)
table(pred_knn_1, data_test_sub$class)

# knn_2 (full feature set)
train_in <- as.matrix(data_train_full[1:15])
train_out <- data_train_full[,16]
test_in <- as.matrix(data_test_full[1:15])
pred_knn_2 <- knn(train_in, test_in, train_out, k = 1)
table(pred_knn_2, data_test_full$class)

# knn_3 (primary features)
train_in <- as.matrix(data_train_bybs[1:2])
train_out <- data_train_bybs[,3]
test_in <- as.matrix(data_test_bybs[1:2])
pred_knn_3 <- knn(train_in, test_in, train_out, k = 5)
table(pred_knn_3, data_test_bybs$class)

# Build RF model and perform classification on test data (feature subset)
model_RF_1 <- randomForest(class ~ ., data = data_train_sub, 
                           method = "C-classification", 
                           probability = TRUE, nodesize = 3, ntree = 500)
pred_RF_1 <- predict(model_RF_1, data_test_sub, type = "response")
table(data_test_sub$class, pred_RF_1)

# Build RF model and perform classification on test data (full feature set)
model_RF_2 <- randomForest(class ~ ., data = data_train_full, 
                           method = "C-classification", 
                           probability = TRUE, nodesize = 3, mtry = 2, ntree = 500)
pred_RF_2 <- predict(model_RF_2, data_test_full, type = "response")
table(data_test_full$class, pred_RF_2)

# Build RF model and perform classification on test data (primary features)
model_RF_3 <- randomForest(class ~ ., data = data_train_bybs, nodesize = 5, mtry = 2, ntree = 500)
pred_RF_3 <- predict(model_RF_3, data_test_bybs, type = "response")
table(data_test_bybs$class, pred_RF_3)

# Build NB model and perform classification on test data (feature subset)
model_NB_1 <- naiveBayes(class ~ ., data = data_train_sub)
pred_NB_1 <- predict(model_NB_1, data_test_sub[,-10])
table(pred_NB_1, data_test_sub$class)

# Build NB model and perform classification on test data (full feature set)
model_NB_2 <- naiveBayes(class ~ ., data = data_train_all)
pred_NB_2 <- predict(model_NB_2, data_test_all[,-16])
table(pred_NB_2, data_test_all$class)

# Build NB model and perform classification on test data (primary features)
model_NB_3 <- naiveBayes(class ~ ., data = data_train_bybs)
pred_NB_3 <- predict(model_NB_3, data_test_bybs[,-3])
table(pred_NB_3, data_test_bybs$class)

# Build SVM model and perform classification on test data (feature subset)
model_SVM_1 <- svm(class ~ ., data = data_train_sub, kernel = "radial", probability = TRUE, 
                    gamma = 0.25, 
                    cost = 8, 
                    class.weights = c(coarse = 0.5, medium = 0.2, muddy = 0.3))

pred_SVM_1 <- predict(model_SVM_1, data_test_sub, probability = TRUE)
table(pred_SVM_1, data_test_sub$class)

# Build SVM model and perform classification on test data (full feature set)
model_SVM_2 <- svm(class ~ ., data = data_train_all, kernel = "radial", probability = TRUE, 
                    gamma = 0.03125, 
                    cost = 64, 
                    class.weights = c(coarse = 0.5, medium = 0.2, muddy = 0.3))

pred_SVM_2 <- predict(model_SVM_2, data_test_all, probability = TRUE)
table(pred_SVM_2, data_test_all$class)

# Build SVM model and perform classification on test data (primary features)
model_SVM_3 <- svm(class ~ ., data = data_train_bybs, kernel = "radial",
                   probability = TRUE, gamma = 32, cost = 1024, 
                   class.weights = c(coarse = 0.5, medium = 0.2, muddy = 0.3))

pred_SVM_3 <- predict(model_SVM_3, data_test_bybs, probability = TRUE)
table(pred_SVM_3, data_test_bybs$class)

# Build ANN model and perform classification on test data (feature subset)
data_sub <- data_full[, c(-6, -7, -10, -13, -14, -15, -16, -17)]
targets <- class.ind(data_full$class)

samp1 <- data_sub[1:656, ]
samp2 <- data_sub[657:960, ]

targ1 <- targets[1:656, ]
targ2 <- targets[657:960, ]
row.names(targ2) <- seq(nrow(targ2))

ir1 <- nnet(samp1, targ1, size = 32, decay = 0.001, maxit = 200,
            targ1.weights = c(coarse = 0.5, medium = 0.2, muddy = 0.3))
test.cl <- function(true, pred) {
  true <- max.col(true)
  cres <- max.col(pred)
  table(true, cres)
}
test.cl(targ2, predict(ir1, samp2))

# Build ANN model and perform classification on test data (full feature set)
data_full_2 <- data_full[, c(-16, -17)]
targets <- class.ind(data_full$class)

samp3 <- data_full_2[1:656, ]
samp4 <- data_full_2[657:960, ]
row.names(samp4) <- seq(nrow(samp4))

targ3 <- targets[1:656, ]
targ4 <- targets[657:960, ]
row.names(targ4) <- seq(nrow(targ4))

ir1 <- nnet(samp3, targ3, size = 8, decay = 0.00001, maxit = 200,
            targ3.weights = c(coarse = 0.5, medium = 0.2, muddy = 0.3))

test.cl <- function(true, pred) {
  true <- max.col(true)
  cres <- max.col(pred)
  table(true, cres)
}
test.cl(targ4, predict(ir1, samp4))

# Build ANN model and perform classification on test data (primary features)
data_bybs <- data_full[, 1:2]
targets <- class.ind(data_full$class)

samp5 <- data_bybs[1:656, ]
samp6 <- data_bybs[657:960, ]
row.names(samp6) <- seq(nrow(samp6))

targ5 <- targets[1:656, ]
targ6 <- targets[657:960, ]
row.names(targ6) <- seq(nrow(targ6))

ir1 <- nnet(samp5, targ5, size = 32, decay = 0.001, 
            targ5.weights = c('coarse' = 0.5, 'medium' = 0.2, 'muddy' = 0.3))
test.cl <- function(true, pred) {
  true <- max.col(true)
  cres <- max.col(pred)
  table(true, cres)
}
test.cl(targ6, predict(ir1, samp6))