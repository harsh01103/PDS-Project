# ==============================================================================
# EXTRA RESULTS: ROC curves, confusion matrices and XGBoost row
# Run in the same R session AFTER master_ml_benchmark.R has been sourced, so the
# fitted objects (fit_*), predictions (pred_*) and test_data already exist.
# ==============================================================================
suppressPackageStartupMessages({ library(pROC); library(ggplot2); library(dplyr); library(tidyr) })

pos <- "Profitable"
# P(Profitable) from each model that produces probabilities
p_of <- list(
  "Logit"          = 1 - prob_logit,
  "Probit"         = 1 - prob_probit,
  "LDA"            = predict(fit_lda,  newdata = test_data)$posterior[, pos],
  "QDA"            = predict(fit_qda,  newdata = test_data)$posterior[, pos],
  "Naive Bayes"    = predict(fit_nb,   newdata = test_data, type = "raw")[, pos],
  "CART"           = predict(fit_dt,   newdata = test_data, type = "prob")[, pos],
  "Bagged Trees"   = predict(fit_bag,  newdata = test_data, type = "prob")[, pos],
  "Random Forest"  = predict(fit_rf,   newdata = test_data, type = "prob")[, pos],
  "MLP (ANN)"      = 1 - as.numeric(prob_ann),
  "XGBoost"        = xgb_prob_test
)
y <- test_data$target
rocs <- lapply(p_of, function(p) roc(y, as.numeric(p), levels = c("Unprofitable", pos), direction = "<", quiet = TRUE))
auc_tbl <- data.frame(Model = names(rocs), AUC = round(sapply(rocs, function(r) as.numeric(auc(r))), 4))
auc_tbl <- auc_tbl[order(-auc_tbl$AUC), ]
print(auc_tbl, row.names = FALSE)
write.csv(auc_tbl, "auc_table.csv", row.names = FALSE)

roc_df <- do.call(rbind, lapply(names(rocs), function(n) {
  r <- rocs[[n]]
  data.frame(Model = sprintf("%s (AUC %.3f)", n, as.numeric(auc(r))), fpr = 1 - r$specificities, tpr = r$sensitivities)
}))
p_roc <- ggplot(roc_df, aes(fpr, tpr, color = Model)) + geom_line(linewidth = 0.9) +
  geom_abline(linetype = "dashed", color = "grey50") +
  labs(title = "ROC curves on the out-of-time test set", x = "False positive rate", y = "True positive rate") +
  theme_minimal(base_size = 13) + theme(plot.title = element_text(face = "bold"), legend.position = "right")
ggsave("figs/res_roc_curves.png", p_roc, width = 10, height = 6, dpi = 200)

# Confusion matrices for six representative models
cms <- list("Probit" = pred_probit, "SVM (Linear)" = pred_svm_lin, "Logit" = pred_logit,
            "Random Forest" = pred_rf, "XGBoost" = pred_xgb, "CART" = pred_dt)
cm_df <- do.call(rbind, lapply(names(cms), function(n) {
  t <- as.data.frame(table(Predicted = cms[[n]], Actual = y)); t$Model <- n; t }))
cm_df$Model <- factor(cm_df$Model, levels = names(cms))
p_cm <- ggplot(cm_df, aes(Actual, Predicted, fill = Freq)) + geom_tile(color = "white") +
  geom_text(aes(label = Freq), size = 5, fontface = "bold") +
  scale_fill_gradient(low = "#eaf4f1", high = "#2a9d8f", guide = "none") +
  facet_wrap(~Model, ncol = 3) +
  labs(title = "Confusion matrices (94 test signals)", x = "Actual class", y = "Predicted class") +
  theme_minimal(base_size = 12) + theme(plot.title = element_text(face = "bold"),
                                       axis.text.x = element_text(angle = 20, hjust = 1))
ggsave("figs/res_confusion_matrices.png", p_cm, width = 10, height = 6.5, dpi = 200)

# XGBoost feature importance (gain share)
imp <- xgb_imp; imp$Overall <- 100 * imp$Gain / max(imp$Gain)
p_imp <- ggplot(imp, aes(reorder(Feature, Overall), Overall)) + geom_col(fill = "#2a9d8f", width = 0.65) +
  coord_flip() + labs(title = "XGBoost feature importance (gain)", x = NULL, y = "Relative importance (0-100)") +
  theme_minimal(base_size = 13) + theme(plot.title = element_text(face = "bold"))
ggsave("figs/res_xgb_importance.png", p_imp, width = 8, height = 4.5, dpi = 200)

# Final 15-model leaderboard
lb15 <- rbind(leaderboard, record_model(15, "Boosting", "XGBoost (5-fold CV grid search)", pred_xgb, y)) %>%
  arrange(desc(Accuracy), desc(Kappa))
write.csv(lb15, "leaderboard_15_models.csv", row.names = FALSE)
print(lb15[, c("Model", "Accuracy", "Kappa", "Precision", "Sensitivity", "Specificity")], row.names = FALSE)
cat("\nXGBoost best CV tune:\n"); print(best)

# 15-model accuracy chart
lb15$Model <- factor(lb15$Model, levels = rev(lb15$Model))
p_acc <- ggplot(lb15, aes(Model, Accuracy, fill = Category)) + geom_col(width = 0.72) + coord_flip() +
  geom_text(aes(label = sprintf("%.2f%%", Accuracy)), hjust = -0.1, size = 3.6, fontface = "bold") +
  ylim(0, 110) + scale_fill_brewer(palette = "Set2") +
  labs(title = "Out-of-time accuracy across 15 models", x = NULL, y = "Accuracy (%)", fill = "Model family") +
  theme_minimal(base_size = 12) + theme(plot.title = element_text(face = "bold"), legend.position = "bottom")
ggsave("figs/res_accuracy_15_models.png", p_acc, width = 10, height = 7, dpi = 200)
