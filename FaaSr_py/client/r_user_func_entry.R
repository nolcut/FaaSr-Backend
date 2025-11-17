.libPaths(c("/tmp/Rlibs", .libPaths()))

library("httr")
library("jsonlite")
source("r_client_stubs.R")
source("r_func_helper.R")

# Entry for R function process

args <- commandArgs(trailingOnly = TRUE)
func_name <- args[1]
user_args <- fromJSON(args[2])
invocation_id <- args[3]

# Source CRAN packages
cran_pkgs <- c()
if (length(args) >= 4) {
    tryCatch({
        cran_pkgs <- fromJSON(args[4])
    }, error=function(e){
        message(paste0('{"r_user_func_entry":"Could not parse CRAN package list: ', as.character(e), '"}'))
        cran_pkgs <- c()
    })
}
if (!is.null(cran_pkgs) && length(cran_pkgs) > 0) {
    for (pkg in cran_pkgs) {
        tryCatch({
            # library will error if package isn't installed
            suppressPackageStartupMessages(library(pkg, character.only = TRUE))
            cat(paste0('{"r_user_func_entry":"Loaded package ', pkg, '"}\n'))
        }, error = function(e) {
            # if package loading fails, exit
            cat(paste0('{"r_user_func_entry":"Missing required package or failed to load: ', pkg, " - ", as.character(e), '"}\n'))
            quit(status = 1, save = "no")
        })
    }
}

faasr_source_r_files(file.path("/tmp/functions", invocation_id))

# Execute User function
result <- faasr_run_user_function(func_name, user_args)

if (!is.logical(result)) {
    result <- NULL
}

faasr_return(result)
