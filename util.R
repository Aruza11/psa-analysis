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

fit_glm_auc <- function(train, setup) {
  ###
  # Trains glm, returns cross validated AUC on training set and test set
  # train MUST have response variable named "y"
  ###
  
  ## Allocate space for performance statistics
  eval_varnames = c("train_auc_mean","train_auc_std","test_auc_mean","test_auc_std")
  
  performance = data.frame(
    # seed = sample.int(10000, 1),
    matrix(NA,nrow=1,
           ncol=length(eval_varnames),
           dimnames=list(NULL,eval_varnames)))
  
  ## Divide into folds
  train_fold = train %>%
    modelr::crossv_kfold(k = setup[["nfold"]]) %>%
    transmute(X = map(train, ~ as.data.frame(select(as.data.frame(.x), -y))),
              y = map(train, ~ as.data.frame(.x)$y),
              X_test = map(test, ~ as.data.frame(select(as.data.frame(.x), -y))),
              y_test = map(test, ~ as.data.frame(.x)$y))
  
    # Train model on each fold
    res = train_fold %>%
      mutate(model = map2(X,y, ~ glm(.y ~., 
                                     family = binomial(link = 'logit'), 
                                     data = .x))) %>%
      # Compute performance statistics
      mutate(auc_train = pmap_dbl(list(X,y,model), ~ pROC::auc(response=..2,predictor=predict(..3,type='response',newdata=..1))),
             auc_test = pmap_dbl(list(X_test,y_test,model), ~ pROC::auc(response=..2,predictor=predict(..3,type='response',newdata=..1)))
      ) %>%
      
      # Record performance fit statistics over folds
      summarize(train_auc_mean = mean(auc_train), # Order should match columns of performance
                train_auc_std = sd(auc_train),
                test_auc_mean = mean(auc_test),
                test_auc_std = sd(auc_train))
    
  performance[1,eval_varnames] = as.numeric(res[1,eval_varnames])
  #trains glm on all data
  mdl_glm = glm(y ~ ., family=binomial(link='logit'), data=train)
  
  return(list(mdl_glm=mdl_glm, performance=performance))
}


fit_svm <- function(formula, train, param) {
  ###
  # Cross validates each combination of parameters in param and returns best model
  # param is a list of svm parameters as vectors
  # svm parameters are cost, epsilon, and gamma_scale (a scaling factor on the default gamma value)
  # train is formatted for xgboost input
  ###
  
  
  param_df = expand.grid(param) # Each row is a set of parameters to be cross validated
  n_param = nrow(param_df)
  
  ## Compute default gamma parameter
  gamma_default = 1/ncol(train)
  param_df = param_df %>%
    mutate(gamma = gamma_scale * gamma_default)
  
  ## Make sure only one type parameter
  if(length(param$type) == 1){
    
    if(str_detect(param$type,'regression')){
      reg_or_class = 'reg'
    } else if (str_detect(param$type,'classification')) {
      reg_or_class = 'class'
    }
  } else{
    stop('Can only handle one type parameter')
  }
  
  ## Allocate space for performance statistics (and set seeds)
  performance = rep(NA,n_param)
  
  cat("Training on",n_param,"sets of parameters.\n")
  
  ## Loop through the different parameters sets
  for (i_param in 1:n_param) {
    
    mdcv = suppressWarnings(e1071::svm(formula = formula, 
                                       data = train, 
                                       type = param$type,
                                       kernel = 'radial',
                                       gamma = param_df$gamma[i_param],
                                       epsilon = param_df$epsilon[i_param],
                                       cost = param_df$cost[i_param],
                                       cross = 5,
                                       scale = TRUE))
    
    if(reg_or_class == "reg"){
      performance[i_param] = mdcv$tot.MSE
    } else if (reg_or_class == "class") {
      performance[i_param] = mdcv$tot.accuracy
    }
    
  }
  
  ## Train on best parameters using best number of rounds
  if(reg_or_class == "reg"){
    i_param_best = which.min(performance)
  } else if (reg_or_class == "class") {
    i_param_best = which.max(performance)
  }
  
  print("Best parameters:")
  print(t(param_df[i_param_best,]))
  
  mdl_best = suppressWarnings(e1071::svm(formula = formula, 
                                         data = train, 
                                         type = param$type,
                                         kernel = 'radial',
                                         gamma = param_df$gamma[i_param_best],
                                         epsilon = param_df$epsilon[i_param_best],
                                         cost = param_df$cost[i_param_best],
                                         cross = 5,
                                         scale = TRUE))
  
  return(mdl_best)
}