# Parallel item-level MICE; run from reproducibility root after prepare_revision2.py.
# Each independent chain uses a recorded seed, so worker count does not affect results.
suppressPackageStartupMessages(library(mice))
suppressPackageStartupMessages(library(jsonlite))
p <- 'work/acer/revision2_private'
d <- read.csv(file.path(p,'mi_input.csv'), colClasses=c(ID='character'), check.names=FALSE)
ids <- d$ID; d$ID <- NULL
base <- fromJSON(file.path(p,'mi_columns.json'))
for(v in c('education','marital')) d[[v]] <- factor(d[[v]])
for(v in c('female','hukou','smoking','drinking',grep('^condition_',names(d),value=TRUE))) d[[v]] <- factor(d[[v]],levels=c(0,1))
method <- make.method(d);method[colSums(is.na(d))==0] <- ''
pred <- make.predictorMatrix(d);pred[method=='',] <- 0
set.seed(261008);chain_seeds <- sample.int(.Machine$integer.max,20)
cores <- as.integer(Sys.getenv('MI_CORES',as.character(max(1,min(10,parallel::detectCores()-2)))))
cores <- max(1,min(cores,20))
write_json(list(master_seed=261008,chain_seeds=chain_seeds,cores=cores),file.path(p,'parallel_configuration.json'),auto_unbox=TRUE,pretty=TRUE)
run_chain <- function(i){
 suppressPackageStartupMessages(library(mice))
 cat('Starting chain',i,'seed',chain_seeds[i],'\n')
 imp <- mice(d,m=1,maxit=20,method=method,predictorMatrix=pred,seed=chain_seeds[i],donors=5,printFlag=TRUE)
 saveRDS(imp,file.path(p,sprintf('chain_%02d.rds',i)))
 cat('Finished chain',i,'\n')
 TRUE
}
cl <- parallel::makePSOCKcluster(cores,outfile=file.path(p,'parallel_chains.log'))
parallel::clusterExport(cl,c('d','method','pred','chain_seeds','p','run_chain'),envir=environment())
tryCatch(parallel::parLapply(cl,1:20,run_chain),finally=parallel::stopCluster(cl))
chains <- lapply(1:20,function(i) readRDS(file.path(p,sprintf('chain_%02d.rds',i))))
imp <- Reduce(mice::ibind,chains)
logs <- do.call(rbind,lapply(1:20,function(i){x<-chains[[i]]$loggedEvents;if(is.null(x)) return(NULL);data.frame(chain=i,x)}))
saveRDS(imp,file.path(p,'imputation.rds'))
for(i in 1:20){z<-complete(imp,i);z<-data.frame(ID=ids,z[,base]);write.csv(z,file.path(p,sprintf('completed_%02d.csv',i)),row.names=FALSE,na='');cat('Completed',i,'\n')}
write.csv(logs,file.path(p,'logged_events.csv'),row.names=FALSE)
split_rhat <- function(x){
 n<-floor(nrow(x)/2);x<-cbind(x[1:n,,drop=FALSE],x[(nrow(x)-n+1):nrow(x),,drop=FALSE]);w<-mean(apply(x,2,var));b<-n*var(colMeans(x));if(w==0) return(if(b==0) 1 else Inf);sqrt(((n-1)*w/n+b/n)/w)
}
incomplete <- names(imp$method)[imp$method!='']
conv <- data.frame(variable=incomplete,split_rhat=sapply(incomplete,function(v) split_rhat(imp$chainMean[v,,])))
write.csv(conv,file.path(p,'convergence.csv'),row.names=FALSE)
write_json(list(R=R.version.string,mice=as.character(packageVersion('mice')),m=20,iterations=20,master_seed=261008,chain_seeds=chain_seeds,cores=cores,methods=as.list(imp$method),predictor_matrix=unname(imp$predictorMatrix),predictor_names=colnames(imp$predictorMatrix),logged_events=logs,chain_mean=imp$chainMean,chain_var=imp$chainVar),'work/acer/enhancements/expanded_mi_diagnostics.json',pretty=TRUE,auto_unbox=TRUE,na='null')

