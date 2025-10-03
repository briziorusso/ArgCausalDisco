runpipeexternal<-function(dat_loc, algo='log-weights') {
  #Function for running simple tests. These tests are run in the most
  #general model space allowing for cycles and latents(=bidirected arcs). Endless
  #number of parameter combinations exist for running the inference, see the plots
  #for some of the most relevant combinations for calling the pipeline function.
  #
  #algo     - defines the algorithm run.
  #     "log-weights" or 1 - the log weighting scheme [section 4.3 in the paper]
  #                          the fastest and most accurate method
  #     "hard-deps" or 2  - putting in deps as hard constraints and max.
  #                         the number of independiencies [section 4.1 in the paper]
  #     "constant-weights" or 3 - weight of 1 for both indeps and desp
  #                             [section 4.2 in the paper]
  
  
  ######### SET ALGORITHM PARAMETERS #########
  #using a single passive observational data set here
  exconf<-'passive'
  #and doing all tests

  #what are the different 
  if ( algo=='log-weights' || algo==1 ) {
    test<-"bayes"
    weight<-"log"
    p<-0.4 #roughly optimal for 500 samples of passively observed data (paper ROCs)
    alpha<-20   
    #another roughly optimal set of parameters
    #p<-0.08
    #alpha<-1.5
    solver<-"clingo"
    encode<-"new_wmaxsat.pl"
    restrict<-c()
  } else if ( algo == 'hard-deps' || algo==2) {
    test<-"classic"
    weight<-"constant"
    p<-0.001 #roughly optimal parameters for 500 samples
    solver<-"clingo"
    encode<-"new_maxindep.pl"
    restrict<-c()
  } else if ( algo == 'constant-weights' || algo==3) {
    test<-"classic"
    weight<-"constant"    
    p<-0.05 #quite optimal for constant weights
    encode<-"new_wmaxsat.pl"
    solver<-"clingo"
    restrict<-c()
  } else if ( algo == 'test only' || algo==4 ) {
    #note that for all different proposed tests (bayes,BIC,classic)
    #the test performances give out roughly the same ROC curve, due to
    #the fact that the inference is quite similar in the end. 
    #This does not mean that the probabilities are same, as ROC does not take 
    #into account the probabilities of the constraints.
    
    test<-"classic"
    weight<-"constant"
    solver<-"test"
    encode<-NA
    p<-0.05
    restrict<-c()
  }

  N<-5000  #total number of samples

  G_loc <- str_replace(dat_loc, 'data_', 'true_graph_')
  G <- read.csv(G_loc, header=FALSE, sep=",")
  stopifnot(nrow(G) == ncol(G))

  n <<- nrow(G)
  schedule <<- n-2

  ## auxiliary matrices, not used when data is given
  M<-list(G=array(0,c(n,n)),Ge=array(0,c(n,n)),Gs=array(0,c(n,n)))
  M$B<-M$G*matrix(runif(n*n,0.2,0.8),n,n)*matrix(sample(c(-1,1),n*n,replace=TRUE),n,n)  
  M$Ce<-diag( abs(1+0.1*rnorm(n)) )  
  M$G <- as.matrix(G)
  
  #run the inference
  start_time <- Sys.time()
  L<-pipeline(n=n,exconf=exconf,schedule=schedule,
            test=test, p=p, solver=solver, model=M, data_file=dat_loc,
            encode=encode, weight=weight, N=N, restrict=restrict)
  elapsed_time <- Sys.time() - start_time

  out_loc <- str_replace(dat_loc, 'data/', 'results/')
  out_loc <- str_replace(out_loc, 'data_', paste0('est_graph_',algo,'_'))  
  write.table(L$G, file = out_loc, sep = ',', row.names = FALSE, col.names = FALSE)

  elapsed <- as.numeric(elapsed_time, units = 'secs')
  elapsed_loc <- str_replace(out_loc, '.csv', '_time.csv')
  write.table(elapsed, file = elapsed_loc, sep = ',', row.names = FALSE, col.names = FALSE)
}

########## Example usage
# cd to the R directory within the ASPCR folder
# make sure you have clingo installed and activated via conda
# run R in the terminal
#### In R
# source("load.R")
# loud()
# library(stringr)
# Seeds <- c(7816, 3578, 2656, 2688, 2494, 183, 7977, 3199, 316, 8266, 1532, 1009, 2250, 4447, 2415,
#            3595, 4187, 4135, 5174, 5303, 3942, 2778, 2804, 2145, 150, 8823, 4669, 6131, 2617, 7166,
#            1428, 8328, 8289, 5978, 5511, 5864, 606, 1933, 6071, 5504, 5163, 2406, 304, 3391, 359, 9096, 3533, 4125, 3094, 6398)

# for (algo in c('log-weights', 'hard-deps', 'constant-weights', 'test only')) {
#   for (dataset in c('cancer',
#                     'earthquake',
#                     'survey'
#                   # ,'asia' ### asia is too big for this
#                     )){
#     for (seed in Seeds) {
#       dat_loc <- paste0('/vol/bitbucket/fr920/ArgCausalDisco/results/aspcr/data/data_', dataset, '_', seed, '.csv')
#       runpipeexternal(dat_loc, algo)
#     }
#   }
# }