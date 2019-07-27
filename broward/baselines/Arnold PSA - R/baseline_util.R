######################################################################################################

## Log out performances
log_performance <- function (out_model, log_path, model_name){
  out_model$performance$Model = model_name
  
  eval_varnames = c("train_auc_mean","train_auc_std","test_auc_mean","test_auc_std", "heldout_test_auc","heldout_test_acc")
  out_model$performance = out_model$performance %>% select(Model, eval_varnames)
  
  write.csv(out_model$performance, paste0(log_path, model_name, "_performance.csv"))
  #add functionality later for logging rocs
}







######################################################################################################

## XGBoost
fit_xgb_auc <- function(train, test,param, setup) {
    
    ###
    # Cross validates each combination of parameters in param and returns best model
    # param is a list of xgboost parameters as vectors
    # train is formatted for xgboost input
    ###
    
    n_param = nrow(param)
    eval_varnames = c("iter", "train_auc_mean","train_auc_std","test_auc_mean","test_auc_std", "heldout_test_auc","heldout_test_acc")
    cv_eval_varnames = eval_varnames[1:5]
    heldout_eval_varnames = eval_varnames[6:7]
    
    # Allocate space for performance statistics (and set seeds)
    performance = data.frame(
        i_param = 1:n_param,
        seed = sample.int(10000, n_param),
        matrix(NA,nrow(param),
               ncol=length(eval_varnames),
               dimnames=list(NULL, eval_varnames)))
    
    # cv_eval_inds = 3:7 # Adjust manually. Column index in performance of evaluation_log output from xgb.cv
    # heldout_eval_inds = 8:10
    
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
        performance[i_param, cv_eval_varnames] = mdcv$evaluation_log[which.max(mdcv$evaluation_log$test_auc_mean),]
    }
    
    ## Train on best parameters using best number of rounds
    i_param_best = performance$i_param[which.max(performance$test_auc_mean)]
    
    print("Best parameters:")
    print(t(param[i_param_best,])) #Prints the best parameters
    
    set.seed(performance$seed[i_param_best])
    
    mdl_best = xgb.train(data=train, 
                         params=list(param[i_param_best,])[[1]], 
                         nrounds=performance$iter[i_param_best])
    
    labels = test %>% select(recid_use) %>% unlist()%>%as.numeric() -1 #sub 1 b/c casting to numeric gives vecto of 1s and 2s
    preds = predict(mdl_best, newdata = test%>% select(-recid_use)%>%data.matrix() )
    
    
    #compute performance measures using ROCR package
    pred_obj = prediction(preds, labels)
    
    perf_roc = performance(pred_obj, "tpr", "fpr")
    perf_acc = performance(pred_obj, "acc")
    perf_auc = performance(pred_obj, "auc")
    
    acc = max(perf_acc@y.values[[1]])
    auc = perf_auc@y.values[[1]]
    
    best_performance = performance[i_param_best,]
    best_performance[1, heldout_eval_varnames] = c(auc, acc)
    
    return(list(mdl_best=mdl_best, performance=best_performance, roc = perf_roc))
}






######################################################################################################

## BART

fit_bart_auc <- function(train, test, param, setup) {
  ###
  # Cross validates each combination of parameters in param and returns best model
  # train MUST have response variable named "y"
  ###
  
  n_param = nrow(param)
  
  ## Allocate space for performance statistics (and set seeds)
  eval_varnames = c("train_auc_mean","train_auc_std","test_auc_mean","test_auc_std", "heldout_test_auc","heldout_test_acc")
  cv_eval_varnames = eval_varnames[1:4]
  heldout_eval_varnames = eval_varnames[5:6]
  
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
    
    performance[i_param,cv_eval_varnames] = as.numeric(res[1,cv_eval_varnames])
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

  labels = test %>% select(recid_use) %>% unlist() %>% as.numeric() -1
  preds = predict(mdl_best, new_data = as.data.frame(select(test,-recid_use)) )
  
  #compute performance measures using ROCR package
  pred_obj = prediction(preds, labels)
  
  perf_roc = performance(pred_obj, "tpr", "fpr")
  perf_acc = performance(pred_obj, "acc")
  perf_auc = performance(pred_obj, "auc")
  
  acc = max(perf_acc@y.values[[1]])
  auc = perf_auc@y.values[[1]]
  
  best_performance = performance[i_param_best,]
  best_performance[i_param_best,heldout_eval_varnames] = c(auc, acc)
  
  
  return(list(mdl_best=mdl_best, performance=best_performance, roc = perf_roc))
}







######################################################################################################

## Logistic Regression

fit_glm_auc <- function(train, test,setup) {

  ###
  # Trains glm, returns cross validated AUC on training set and test set
  # train MUST have response variable named "y"
  ###
  
  ## Allocate space for performance statistics
  eval_varnames = c("train_auc_mean","train_auc_std","test_auc_mean","test_auc_std", "heldout_test_auc","heldout_test_acc")
  cv_eval_varnames = eval_varnames[1:4]
  heldout_eval_varnames = eval_varnames[5:6]
  
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
    
  performance[1,cv_eval_varnames] = as.numeric(res[1,cv_eval_varnames])
  #trains glm on all data
  mdl_best = glm(y ~ ., family=binomial(link='logit'), data=train)
  #create ROC
  # par(pty = "s")
  labels = test %>% select(recid_use) %>% unlist() %>% as.numeric() - 1
  preds = predict(mdl_best, newdata = test %>%select(-recid_use),type = "response")
  
  #compute performance measures using ROCR package
  pred_obj = prediction(preds, labels)
  
  perf_roc = performance(pred_obj, "tpr", "fpr")
  perf_acc = performance(pred_obj, "acc")
  perf_auc = performance(pred_obj, "auc")
  
  acc = max(perf_acc@y.values[[1]])
  auc = perf_auc@y.values[[1]]
  
  performance[1,heldout_eval_varnames] = c(auc,acc)
  
  return(list(mdl_best=mdl_best, performance=performance, roc = perf_roc))
  
  # return(list(mdl_best=mdl_best, performance=performance, roc = roc))
}






######################################################################################################
## Lasso

fit_lasso_auc <- function(train, test, setup) {
  ###
  # Trains LASSO, returns cross validated AUC on training set and test set
  # train MUST have response variable named "y"
  ###
  
  ## Allocate space for performance statistics
  eval_varnames = c("train_auc_mean","train_auc_std","test_auc_mean","test_auc_std", "heldout_test_auc", "heldout_test_acc")
  cv_eval_varnames = eval_varnames[1:4]
  heldout_eval_varnames = eval_varnames[5:6]
  
  performance = data.frame(
    seed = sample.int(10000, 1),
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
  
  # cv =cv.glmnet()
  res = train_fold %>%
    mutate(model = map2(X,y, ~ glmnet(x=data.matrix(.x), y=as.numeric(data.matrix(.y)),
                                      alpha=1,
                                      family = "binomial")), 
           best_lambda = map2(X, y, ~cv.glmnet(x=data.matrix(.x), 
                                               y = as.numeric(data.matrix(.y)), 
                                               alpha = 1 )$lambda.min)
    ) %>%
    
    # Compute performance statistics
    #IMPLEMENT CROSS VALIDATION??
    mutate(auc_train = pmap_dbl(list(X,y,model,best_lambda), ~ pROC::auc(response=..2,
                                                                         predictor=as.numeric(predict(..3,
                                                                                                      type='response',
                                                                                                      newx=data.matrix(..1), 
                                                                                                      s = ..4)))),
           auc_test = pmap_dbl(list(X_test,y_test,model,best_lambda), ~ pROC::auc(response=..2,
                                                                                  predictor=as.numeric(predict(..3,
                                                                                                               type='response',
                                                                                                               newx=data.matrix(..1),
                                                                                                               s = ..4))))
    ) %>%
    summarize(train_auc_mean = mean(auc_train), # Order should match columns of performance
              train_auc_std = sd(auc_train),
              test_auc_mean = mean(auc_test),
              test_auc_std = sd(auc_train))
  
  performance[1,cv_eval_varnames] = as.numeric(res[1,cv_eval_varnames])
  
  #trains lasso on all data
  X = train%>% select(-y)%>%data.matrix()
  y = train%>% select(y) %>% data.matrix() %>% as.numeric
  mdl_best = glmnet(x=X,y=y, family="binomial", alpha=1)
  best_lambda = cv.glmnet(x=X, y=y, alpha = 1)$lambda.min
  
  labels = test %>% select(recid_use) %>% unlist()%>%as.numeric() -1 #sub 1 b/c casting to numeric gives vecto of 1s and 2s
  preds = as.numeric(predict(mdl_best,
                     type='response',
                     newx=test %>% select(-recid_use)%>%data.matrix(), 
                     s = best_lambda))
  #compute performance measures using ROCR package
  pred_obj = prediction(preds, labels)
  
  perf_roc = performance(pred_obj, "tpr", "fpr")
  perf_acc = performance(pred_obj, "acc")
  perf_auc = performance(pred_obj, "auc")
  
  acc = max(perf_acc@y.values[[1]])
  auc = perf_auc@y.values[[1]]
  
  performance[1,heldout_eval_varnames] = c(auc, acc)
  
  return(list(mdl_best=mdl_best, performance=performance, roc = perf_roc, acc = acc, auc = auc, best_lambda = best_lambda))
}


######################################################################################################
## Random Forest

fit_rf_auc <- function(train, test, setup) {
  ###
  # Trains random forest, returns cross validated AUC on training set and test set
  # train MUST have response variable named "y"
  ###
  
  ## Allocate space for performance statistics
  eval_varnames = c("train_auc_mean","train_auc_std","test_auc_mean","test_auc_std", "heldout_test_auc", "heldout_test_acc")
  cv_eval_varnames = eval_varnames[1:4]
  heldout_eval_varnames = eval_varnames[5:6]
  
  performance = data.frame(
    seed = setup[["seed"]],
    matrix(NA,nrow=1,
           ncol=length(eval_varnames),
           dimnames=list(NULL,eval_varnames)))
  
  
  train_fold = train %>%
    modelr::crossv_kfold(k = setup[["nfold"]]) %>%
    transmute(X = map(train, ~ as.data.frame(select(as.data.frame(.x), -y))),
              y = map(train, ~ as.data.frame(.x)$y),
              X_test = map(test, ~ as.data.frame(select(as.data.frame(.x), -y))),
              y_test = map(test, ~ as.data.frame(.x)$y))
  
  # Train model on each fold
  res = train_fold %>%
    mutate(model = map2(X,y, ~randomForest(formula = .y ~., 
                                           data = .x)))%>%
    # Compute performance statistics
    mutate(auc_train = pmap_dbl(list(X,y,model), ~  pROC::auc(response=..2,predictor=as.numeric(as.character(predict(..3, newdata=..1, type = "response"))))),
           auc_test = pmap_dbl(list(X_test,y_test,model), ~  pROC::auc(response=..2,predictor=as.numeric(as.character(predict(..3, newdata=..1, type = "response")))))
    ) %>%
    
    # Record performance fit statistics over folds
    summarize(train_auc_mean = mean(auc_train), # Order should match columns of performance
              train_auc_std = sd(auc_train),
              test_auc_mean = mean(auc_test),
              test_auc_std = sd(auc_train))
  
  performance[1,cv_eval_varnames] = as.numeric(res[1,cv_eval_varnames])
  
  #trains rf on all data and predicts on all data
  mdl_best = randomForest(formula = y ~ ., data=train)
  # X = train%>% select(-c(y))%>%data.matrix()
  # y = train%>% select(y) %>% data.matrix() %>% as.numeric
  
  labels = test %>% select(recid_use) %>% unlist()%>%as.numeric() -1 #sub 1 b/c casting to numeric gives vecto of 1s and 2s
  preds = predict(mdl_best, 
                  newdata=test%>% select(-recid_use)%>%data.matrix(), 
                  type = "prob") %>%
          as_data_frame()%>%
          select(`1`)%>%
          unlist()%>%
          as.numeric()
  
  #compute performance measures using ROCR package
  pred_obj = prediction(preds, labels)
  
  perf_roc = performance(pred_obj, "tpr", "fpr")
  perf_acc = performance(pred_obj, "acc")
  perf_auc = performance(pred_obj, "auc")
  
  acc = max(perf_acc@y.values[[1]])
  auc = perf_auc@y.values[[1]]

  performance[1,heldout_eval_varnames] = c(auc,acc)
  
  return(list(mdl_best=mdl_best, performance=performance, roc = perf_roc))
}



######################################################################################################
## CART

fit_cart_auc <- function(train, test, param, setup) {
  ###
  # Trains CART on each combination of parameters,
  # returns cross validated AUC on training set and test set
  # train MUST have response variable named "y"
  ###
  
  ## Allocate space for performance statistics
  n_param = nrow(param)
  eval_varnames = c("train_auc_mean","train_auc_std","test_auc_mean","test_auc_std", "heldout_test_auc", "heldout_test_acc")
  cv_eval_varnames = eval_varnames[1:4]
  heldout_eval_varnames = eval_varnames[5:6]
  
  performance = data.frame(
    i_param = 1:n_param,
    seed = sample.int(10000, n_param),
    matrix(NA,nrow=n_param,
           ncol=length(eval_varnames),
           dimnames=list(NULL,eval_varnames)))
  
  
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
      mutate(model = map2(X,y, ~rpart(formula = .y ~., 
                                      data = .x, method="class", 
                                      control=rpart.control(cp = param[i_param, "cp"]) )
                          ))%>%
      
      # Compute performance statistics
      mutate(auc_train = pmap_dbl(list(X,y,model), ~  pROC::auc(response=..2,predictor=as.numeric(as.character(predict(..3, newdata=..1, type = "class"))))),
             auc_test = pmap_dbl(list(X_test,y_test,model), ~  pROC::auc(response=..2,predictor=as.numeric(as.character(predict(..3, newdata=..1, type = "class")))))
      ) %>%
      
      # Record performance fit statistics over folds
      summarize(train_auc_mean = mean(auc_train), # Order should match columns of performance
                train_auc_std = sd(auc_train),
                test_auc_mean = mean(auc_test),
                test_auc_std = sd(auc_train))
    
    performance[i_param,cv_eval_varnames] = as.numeric(res[1,cv_eval_varnames])
  }
  
  #Train all data using best parameters
  i_param_best = performance$i_param[which.max(performance$test_auc_mean)]
  print("Best parameters:")
  print(t(param[i_param_best,])) #Prints the best parameters
  set.seed(performance$seed[i_param_best])
  
  mdl_best =rpart(formula = y ~., 
                  data = train, method="class", 
                  control=rpart.control(cp = param[i_param_best, "cp"]) )
  
  # X = train%>% select(-c(y))

  labels = test %>% select(recid_use) %>% unlist()%>%as.numeric() -1 #sub 1 b/c casting to numeric gives vecto of 1s and 2s
  
  preds = predict(mdl_best, 
                  newdata= test%>% select(-recid_use),  
                  type = "prob") %>%
    as_data_frame()%>%
    select(`1`)%>%
    unlist()%>%
    as.numeric()
  
  
  #compute performance measures using ROCR package
  pred_obj = prediction(preds, labels)
  
  perf_roc = performance(pred_obj, "tpr", "fpr")
  perf_acc = performance(pred_obj, "acc")
  perf_auc = performance(pred_obj, "auc")
  
  acc = max(perf_acc@y.values[[1]])
  auc = perf_auc@y.values[[1]]
  
  best_performance = performance[i_param_best,]
  best_performance[1, heldout_eval_varnames] = c(auc, acc)
  
  return(list(mdl_best=mdl_best, performance=best_performance, roc = perf_roc))
}






######################################################################################################
## SVM

fit_svm_auc <- function(train, test, param, setup) {
  ###
  # Trains CART on each combination of parameters,
  # returns cross validated AUC on training set and test set
  # train MUST have response variable named "y"
  ###
  
  ## Allocate space for performance statistics
  n_param = nrow(param)
  eval_varnames = c("train_auc_mean","train_auc_std","test_auc_mean","test_auc_std", "heldout_test_auc", "heldout_test_acc")
  cv_eval_varnames = eval_varnames[1:4]
  heldout_eval_varnames = eval_varnames[5:6]
  
  performance = data.frame(
    i_param = 1:n_param,
    seed = sample.int(10000, n_param),
    matrix(NA,nrow=n_param,
           ncol=length(eval_varnames),
           dimnames=list(NULL,eval_varnames)))
  
  
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
      mutate(model = map2(X,y, ~suppressWarnings(
                                e1071::svm(formula = .y ~., 
                                           data = .x, 
                                           type = param[i_param, 'type'], 
                                           kernel = 'radial', 
                                           gamma = param[i_param, 'gamma'], 
                                           epsilon = param[i_param, 'epsilon'], 
                                           cost = param[i_param,  'cost'], 
                                           cross = 5, 
                                           scale = TRUE, 
                                           probability = TRUE
      ))))%>%
      
      # Compute performance statistics
      mutate(auc_train = pmap_dbl(list(X,y,model), ~  pROC::auc(response=..2,predictor=as.numeric(as.character(predict(..3, newdata=..1, probability = F))))),
             auc_test = pmap_dbl(list(X_test,y_test,model), ~  pROC::auc(response=..2,predictor=as.numeric(as.character(predict(..3, newdata=..1, probability = F)))))
      ) %>%
      
      # Record performance fit statistics over folds
      summarize(train_auc_mean = mean(auc_train), # Order should match columns of performance
                train_auc_std = sd(auc_train),
                test_auc_mean = mean(auc_test),
                test_auc_std = sd(auc_train))
    
    performance[i_param,cv_eval_varnames] = as.numeric(res[1,cv_eval_varnames])
  }
  
  #Train all data using best parameters
  i_param_best = performance$i_param[which.max(performance$test_auc_mean)]
  print("Best parameters:")
  print(t(param[i_param_best,])) #Prints the best parameters
  set.seed(performance$seed[i_param_best])
  
  mdl_best =suppressWarnings(e1071::svm(formula = y~., 
                                        data = train, 
                                        type = param$type,
                                        kernel = 'radial',
                                        gamma = param[i_param_best, "gamma"],
                                        epsilon = param[i_param_best, "epsilon"],
                                        cost = param[i_param_best, "cost"],
                                        # cross = 5,
                                        scale = TRUE, 
                                        probability = TRUE))
  
  labels = test %>% select(recid_use) %>% unlist()%>%as.numeric() -1 #sub 1 b/c casting to numeric gives vecto of 1s and 2s
  
  preds = predict(mdl_best, 
                  newdata=test %>% select(-recid_use) %>% data.matrix(), 
                  probability = TRUE) %>%
    attr("prob")%>%
    as_data_frame()%>%
    select(`1`)%>%
    unlist()%>%
    as.numeric()
  
  
  #compute performance measures using ROCR package
  pred_obj = prediction(preds, labels)
  
  perf_roc = performance(pred_obj, "tpr", "fpr")
  perf_acc = performance(pred_obj, "acc")
  perf_auc = performance(pred_obj, "auc")
  
  acc = max(perf_acc@y.values[[1]])
  auc = perf_auc@y.values[[1]]

  best_performance = performance[i_param_best,]
  best_performance[1, heldout_eval_varnames] = c(auc, acc)
  
  
  return(list(mdl_best=mdl_best, performance=best_performance, roc = perf_roc))
}

