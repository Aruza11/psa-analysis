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
    matrix(NA,nrow(param),
           ncol=5,
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


fit_bart_auc <- function(train, param, setup) {
  ###
  # Cross validates each combination of parameters in param and returns best model
  # train MUST have response variable named "y"
  ###
  
  n_param = nrow(param)
  
  ## Allocate space for performance statistics (and set seeds)
  eval_varnames = c("train_auc_mean","train_auc_std","test_auc_mean","test_auc_std")
  
  performance = data.frame(
    i_param = 1:n_param,
    seed = sample.int(10000, n_param),
    matrix(NA,nrow=nrow(param),
           ncol=length(eval_varnames),
           dimnames=list(NULL,eval_varnames)))
  
  ## Divide into folds
  train_fold = train %>%
    modelr::crossv_kfold(k = setup[["nfold"]]) %>%
    transmute(X = map(train, ~ as.data.frame(select(as.data.frame(.x), -y))),
              y = map(train, ~ as.data.frame(.x)$y),
              X_test = map(test, ~ as.data.frame(select(as.data.frame(.x), -y))),
              y_test = map(test, ~ as.data.frame(.x)$y))
  
  cat("Training on",n_param,"sets of parameters.\n")
  
  ## Loop through the different parameters sets
  for (i_param in 1:n_param) {
    
    set.seed(performance$seed[i_param])
    
    # Train model on each fold
    res = train_fold %>%
      mutate(model = map2(X,y, ~ bartMachine(X=.x, 
                                             y=.y,
                                             num_trees = param[i_param,"num_trees"],
                                             alpha = param[i_param,"alpha"], 
                                             beta = param[i_param,"beta"], 
                                             k = param[i_param,"k"], 
                                             q = param[i_param,"q"], 
                                             nu = param[i_param,"nu"]
      ))) %>% 
      
      # Compute performance statistics
      mutate(auc_train = pmap_dbl(list(X,y,model), ~ pROC::auc(response=..2,predictor=predict(..3,new_data=..1))),
             auc_test = pmap_dbl(list(X_test,y_test,model), ~ pROC::auc(response=..2,predictor=predict(..3,new_data=..1)))
      ) %>%
      
      # Record performance fit statistics over folds
      summarize(train_auc_mean = mean(auc_train), # Order should match columns of performance
                train_auc_std = sd(auc_train),
                test_auc_mean = mean(auc_test),
                test_auc_std = sd(auc_train))
    
    performance[i_param,eval_varnames] = as.numeric(res[1,eval_varnames])
  }
  
  ## Train on best parameters using best number of rounds
  i_param_best = performance$i_param[which.max(performance$test_auc_mean)]
  
  print("Best parameters:")
  print(t(param[i_param_best,])) #Prints the best parameters
  
  set.seed(performance$seed[i_param_best])
  
  mdl_best= bartMachine(X=as.data.frame(select(train,-y)), 
                        y=train$y,
                        num_trees = param[i_param_best,"num_trees"],
                        alpha = param[i_param_best,"alpha"], 
                        beta = param[i_param_best,"beta"], 
                        k = param[i_param_best,"k"], 
                        q = param[i_param_best,"q"], 
                        nu = param[i_param_best,"nu"]
  )
  
  return(list(mdl_best=mdl_best, performance=performance))
}