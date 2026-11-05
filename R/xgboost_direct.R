# ==============================================================================
# XGBOOST (same hyper-parameter grid and 5-fold CV as the master script), run with
# the xgboost API directly. caret's "xgbTree" wrapper is incompatible with
# xgboost >= 2.x / 3.x (ALTREP error), so the grid search is done here by hand.
# ==============================================================================
suppressPackageStartupMessages({ library(xgboost); library(dplyr) })
feat_cols <- setdiff(names(train_data), "target")
X_tr <- as.matrix(train_data[, feat_cols]); y_tr <- as.numeric(train_data$target == "Profitable")
X_te <- as.matrix(test_data[, feat_cols]);  y_te <- as.numeric(test_data$target == "Profitable")
dtrain <- xgb.DMatrix(X_tr, label = y_tr); dtest <- xgb.DMatrix(X_te, label = y_te)

xgb_grid <- expand.grid(nrounds = c(50, 100), max_depth = c(3, 5), eta = c(0.05, 0.1),
                        gamma = 0, colsample_bytree = 0.8, min_child_weight = 1, subsample = 0.8)
set.seed(42)
folds <- caret::createFolds(train_data$target, k = 5)
xgb_grid$cv_auc <- NA_real_
for (i in seq_len(nrow(xgb_grid))) {
  g <- xgb_grid[i, ]
  cv <- xgb.cv(params = list(objective = "binary:logistic", eval_metric = "auc", max_depth = g$max_depth,
                             eta = g$eta, gamma = g$gamma, colsample_bytree = g$colsample_bytree,
                             min_child_weight = g$min_child_weight, subsample = g$subsample, nthread = 2),
               data = dtrain, nrounds = g$nrounds, folds = folds, verbose = 0)
  xgb_grid$cv_auc[i] <- tail(cv$evaluation_log$test_auc_mean, 1)
}
print(xgb_grid[order(-xgb_grid$cv_auc), ], row.names = FALSE)
best <- xgb_grid[which.max(xgb_grid$cv_auc), ]
set.seed(42)
fit_xgb_raw <- xgb.train(params = list(objective = "binary:logistic", max_depth = best$max_depth, eta = best$eta,
                         gamma = best$gamma, colsample_bytree = best$colsample_bytree,
                         min_child_weight = best$min_child_weight, subsample = best$subsample, nthread = 2),
                         data = dtrain, nrounds = best$nrounds, verbose = 0)
xgb_prob_test <- predict(fit_xgb_raw, dtest)                    # P(Profitable)
pred_xgb <- factor(ifelse(xgb_prob_test >= 0.5, "Profitable", "Unprofitable"), levels = levels(test_data$target))
cm_xgb <- caret::confusionMatrix(pred_xgb, test_data$target)
cat(sprintf("XGBoost Accuracy: %.2f%% | best CV ROC-AUC: %.3f\n", cm_xgb$overall["Accuracy"] * 100, best$cv_auc))
xgb_imp <- as.data.frame(xgb.importance(model = fit_xgb_raw))
print(xgb_imp)
