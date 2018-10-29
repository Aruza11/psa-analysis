compute_outcomes = function(person_id,screening_date,first_offense_date,current_offense_date,
                            arrest,charge,jail,prison,prob,people){
  
  out = list()
  
  # pmap coerces dates to numbers so convert back to date.
  first_offense_date = as_date(first_offense_date)
  screening_date = as_date(screening_date)
  current_offense_date = as_date(current_offense_date)
  
  ### ID information
  out$person_id = person_id
  out$screening_date = screening_date
  
  if(is.null(charge)) {
    out$recid = 0
    out$recid_violent = 0
    
  } else {
    
    # Sort charges in ascending order
    charge = charge %>% dplyr::arrange(offense_date)
    
    # General recidivism
    date_next_offense = charge$offense_date[1]
    years_next_offense = as.numeric(as.period(interval(screening_date,date_next_offense)), "years")
    out$recid = if_else(years_next_offense <= 2, 1, 0)
    
    # Violent recidivism
    date_next_offense_violent = filter(charge,is_violent==1)$offense_date[1]
    if(is.na(date_next_offense_violent)) {
      out$recid_violent = 0
    } else {
      years_next_offense_violent = as.numeric(as.period(interval(screening_date,date_next_offense_violent)), "years")
      out$recid_violent = if_else(years_next_offense_violent <= 2, 1, 0)
    }
  }
  
  return(out)
}


compute_features = function(person_id,screening_date,first_offense_date,current_offense_date,
                            arrest,charge,jail,prison,prob,people) {
  ### Computes features (e.g., number of priors) for each person_id/screening_date.
  
  # pmap coerces dates to numbers so convert back to date.
  first_offense_date = as_date(first_offense_date)
  screening_date = as_date(screening_date)
  current_offense_date = as_date(current_offense_date) 
  
  out = list()
  
  ### ID information
  out$person_id = person_id
  out$screening_date = screening_date
  
  ### Other features
  
  # Number of felonies
  out$p_felony_count_person = ifelse(is.null(charge), 0, sum(charge$is_felony, na.rm = TRUE))
  
  # Number of misdemeanors
  out$p_misdem_count_person  = ifelse(is.null(charge), 0, sum(charge$is_misdem, na.rm = TRUE))
  
  
  ### History of Violence
  
  #p_current_age: Age at screening date
  out$p_current_age = floor(as.numeric(as.period(interval(people$dob,screening_date)), "years"))
  
  #p_age_first_offense: Age at first offense 
  out$p_age_first_offense = floor(as.numeric(as.period(interval(people$dob,first_offense_date)), "years"))
  
  #p_juv_fel_count
  out$p_juv_fel_count = ifelse(is.null(people), 0, people$juv_fel_count)
  
  #p_felprop_violarrest
  out$p_felprop_violarrest = ifelse(is.null(charge), 0,sum(charge$is_felprop_violarrest, na.rm = TRUE))
  
  #p_murder_arrest
  out$p_murder_arrest = ifelse(is.null(charge), 0, sum(charge$is_murder, na.rm = TRUE))
  
  #p_felassault_arrest
  out$p_felassault_arrest = ifelse(is.null(charge), 0, sum(charge$is_felassault_arrest, na.rm = TRUE))
  
  #p_misdemassault_arrest
  out$p_misdemassault_arrest = ifelse(is.null(charge), 0, sum(charge$is_misdemassault_arrest, na.rm = TRUE))
  
  #p_famviol_arrest
  out$p_famviol_arrest = ifelse(is.null(charge), 0, sum(charge$is_family_violence, na.rm = TRUE))
  
  #p_sex_arrest
  out$p_sex_arrest = ifelse(is.null(charge), 0, sum(charge$is_sex_offense, na.rm = TRUE))
  
  #p_weapons_arrest
  out$p_weapons_arrest =  ifelse(is.null(charge), 0, sum(charge$is_weapons, na.rm = TRUE))
  
  ### History of Non-Compliance
  
  # Number of offenses while on probation
  out$p_n_on_probation = ifelse(is.null(charge) | is.null(prob), 0, count_on_probation(charge,prob))
  
  # Whether or not current offense was while on probation
  out$p_current_on_probation = ifelse(is.null(prob), 0, count_on_probation(data.frame(offense_date=current_offense_date),prob))
  
  # Number of times provation was violated or revoked
  out$p_prob_revoke =  ifelse(is.null(prob), 0, sum(prob$Description=="File Order Of Revocation Of Probation"))
  
  ### Criminal Involvement
  
  # Number of charges / arrests
  out$p_charge = ifelse(is.null(charge), 0, nrow(charge))
  out$p_arrest = ifelse(is.null(arrest), 0, nrow(arrest))
  
  # Number of times sentenced to jail/prison 30 days or more
  out$p_jail30 = ifelse(is.null(prison), 0, sum(jail$sentence_days >= 30, na.rm=TRUE))
  out$p_prison30 = ifelse(is.null(prison), 0, sum(prison$sentence_days >= 30, na.rm=TRUE))
  
  # Number of prison sentences
  out$p_prison =  ifelse(is.null(prison), 0, nrow(prison))
  
  # Number of times on probation
  out$p_probation =  ifelse(is.null(prob), 0, sum(prob$prob_event=="On", na.rm = TRUE))
  
  
  return(out)
}



count_on_probation = function(charge, prob){
  
  # Make sure prob is sorted in ascending order of EventDate
  
  u_charge = charge %>%
    group_by(offense_date) %>%
    summarize(count = n()) %>%
    mutate(rank = findInterval(as.numeric(offense_date), as.numeric(prob$EventDate)))  %>%
    group_by(rank) %>%
    mutate(
      event_before = ifelse(rank==0, NA, prob$prob_event[rank]),
      days_before = ifelse(rank==0, NA, floor(as.numeric(as.period(interval(prob$EventDate[rank],offense_date)), "days"))),
      event_after = ifelse(rank==nrow(prob), NA, prob$prob_event[rank+1]),
      days_after = ifelse(rank==nrow(prob),NA, floor(as.numeric(as.period(interval(offense_date, prob$EventDate[rank+1])), "days")))
    ) %>%
    mutate(is_on_probation = pmap(list(event_before, days_before, event_after, days_after), .f=classify_charge)) %>%
    unnest()
  
  return(sum(u_charge$count[u_charge$is_on_probation]))
}



classify_charge = function(event_before, days_before, event_after, days_after,
                           thresh_days_before=365, thresh_days_after=30) {
  
  if (is.na(event_before)) {
    # No events before
    if (event_after == "Off" & days_after <= thresh_days_after) {
      return(TRUE)
    }
    
  } else if (is.na(event_after)) {
    # No events after
    if (event_before == "On" & days_before <= thresh_days_before) {
      return(TRUE)
    }
  }
  
  else { # Neither event is NA
    
    if (event_before=="On" & event_after=="Off") {
      return(TRUE)
      
    } else if (event_before=="On" & days_before <= thresh_days_before & event_after=="On") {
      return(TRUE)
      
    } else if (event_before=="Off" & event_after=="Off" & days_after <= thresh_days_after) {
      return(TRUE)
    } 
  }
  return(FALSE)
}




compare_cols <- function(df) {
  # Fraction of non-missing rows that agree
  
  # Check data type
  df_class <- unique(unlist(sapply(df,class)))
  
  if (length(df_class)>1){
    stop("Variables not of same class")
  }
  
  # Only use non NA rows
  row_keep <- !(is.na(df[1]) | is.na(df[2]))
  
  if (any(row_keep)) {
    df_keep <- df[row_keep,]
    
    # Comparison is different depending on the class
    # In all cases diff is a number between 0 and 1
    # with smaller number indicating less difference
    
    if (df_class %in% c("integer","double")) {
      # Average absolute difference
      diff <- mean(abs(df_keep[,1] - df_keep[,2]))
      
      
    } else if(df_class %in% c("Date","POSIXct","POSIXt")) {
      # Average difference in days
      diff <- mean(as.numeric(as.period(interval(as.Date(df_keep[,1]), 
                                                 as.Date(df_keep[,2])), 
                                        "days"), "days"))
      
    } else if(df_class %in% "character") {
      # Fraction do not agree
      diff <- mean(df_keep[,1] != df_keep[,2])
      
    } else {
      stop("Variable class not recognized")
    }
  } else {
    diff <- 0
  }
  
  return(diff)
  
}


compare_df <- function(df1, df2, by){
  # Dark magic
  by <- enquo(by)
  
  # Compares two dataframes on specified columns
  out <- list()
  
  # Compare columns
  out[["cols_in1not2"]] <- setdiff(colnames(df1),colnames(df2))
  out[["cols_in2not1"]] <- setdiff(colnames(df2),colnames(df1))
  out[["cols_in1and2"]] <- intersect(colnames(df1),colnames(df2))
  
  # Subset on shared rows and select columns
  row_keep <- base::intersect(select(df1,!!by)[[1]],select(df2,!!by)[[1]])
  out[["n_rows_compare"]] <- length(row_keep)
  
  df1 <- df1 %>%
    filter((!!by) %in% row_keep) %>%
    arrange(!!by)
  
  df2 <- df2 %>%
    filter((!!by) %in% row_keep) %>%
    arrange(!!by)
  
  # Convert time objects to date only
  df1 <- df1 %>% mutate_if(is.POSIXt, funs(as.Date))
  df2 <- df2 %>% mutate_if(is.POSIXt, funs(as.Date))
  
  out[["diff"]] <- sapply(out[["cols_in1and2"]], function(x) 
    compare_cols(cbind(
      df1[x],
      df2[x]
    )))
  
  return(out)
}

days_between <- function(d1,d2) {
  as.numeric(as.period(interval(as.Date(d1), as.Date(d2)), "days"), "days")
}