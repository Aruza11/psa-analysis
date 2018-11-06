library(xgboost)

fit_xgb_auc <- function(train, param, setup) {
  ###
  # Cross validates each combination of parameters in param and returns best model
  # param is a list of xgboost parameters as vectors
  # train is formatted for xgboost input
  ###
  
  n_param = nrow(param)
  
  ## Allocate space for performance statistics (and set seeds)
  performance = data.frame(
    i_param = 1:n_param,
    seed = sample.int(10000, n_param),
    matrix(NA,nrow=2,ncol=5,
           dimnames=list(NULL,
                         c("iter","train_auc_mean","train_auc_std","test_auc_mean","test_auc_std"))))
  col_eval_log = 3:7 # Adjust manually. Column index in performance of evaluation_log output from xgb.cv
  
  cat("Training on",n_param,"sets of parameters.\n")
  
  ## Loop through the different parameters sets
  for (i_param in 1:n_param) {
    
    set.seed(performance$seed[i_param])
    
    mdcv = xgb.cv(data=train, 
                  params = list(param[i_param,])[[1]], 
                  nrounds = setup$nrounds,
                  nfold = setup$nfold,
                  verbose = FALSE, 
                  metrics = "auc",
                  eval_metric = "auc",
                  maximize=TRUE)
    
    performance[i_param,col_eval_log] = mdcv$evaluation_log[which.max(mdcv$evaluation_log$test_auc_mean),]
  }
  
  ## Train on best parameters using best number of rounds
  i_param_best = performance$i_param[which.max(performance$test_auc_mean)]
  
  print("Best parameters:")
  print(t(param[i_param_best,])) #Prints the best parameters
  
  set.seed(performance$seed[i_param_best])
  
  mdl_best = xgb.train(data=train, 
                       params=list(param[i_param_best,])[[1]], 
                       nrounds=performance$iter[i_param_best])
  
  return(list(mdl_best=mdl_best, performance=performance))
}