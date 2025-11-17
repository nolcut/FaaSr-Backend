# function to help source the R files in the system 
faasr_source_r_files <- function(directory = "."){
  # find all R files in `directory` and source them into globql env
  r_files <- list.files(path = directory, pattern="\\.R$", recursive=TRUE, full.names=TRUE)
  for (rfile in r_files){
    bn <- basename(rfile)
    if (bn %in% c("r_func_entry.R", "r_user_func_entry.R", "r_func_helper.R", "http_wrappers.R")) next
    cat(paste0('{"faasr_source_r_files":"Sourcing R file ', bn, '"}\n'))
    tryCatch(
      expr = {
        # source func into global env 
        source(rfile, local = globalenv())
        cat(paste0('{"faasr_source_r_files":"Sourced ', bn, '"}\n'))
      },
      error = function(e){
        cat(paste0('{"faasr_source_r_files":"R file ', bn, ' has following source error: ', as.character(e), '"}\n'))
      }
    )
  }
}


# Run user function
faasr_run_user_function <- function(func_name, user_args){ 
  if (!exists(func_name, envir = globalenv(), inherits = FALSE)){
    err_msg <- paste0('{"faasr_user_function":"Cannot find function, ', func_name,', check the name and sources"}', "\n")
    message(err_msg)
    try(faasr_log(err_msg), silent = TRUE)
    faasr_exit()
  }

  user_function <- get(func_name, envir = globalenv(), inherits = FALSE)

  faasr_result <- tryCatch(
    expr = do.call(user_function, user_args),
    error = function(e){
      nat_err_msg <- paste0('"faasr_user_function":Errors in the user function - ', as.character(e))
      err_msg <- paste0('{"faasr_user_function":"Errors in the user function, ', func_name, ', check the log for the detail"}', "\n")
      # Log native error and session info for debugging
      try(faasr_log(nat_err_msg), silent = TRUE)
      try(faasr_log(paste(capture.output(sessionInfo()), collapse='\n')), silent = TRUE)
      message(err_msg)
      faasr_exit()
    }
  )

  return(faasr_result)
}



