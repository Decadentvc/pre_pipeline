args = commandArgs(trailingOnly=TRUE)

setwd("/home/autoprep")
df <- read.csv(paste("./resources/raw_results/", args[1],"/prototype_construction/discretize_features/summary/chi2tests/binary/order_test.csv", sep = ""), header = TRUE)
dr <- read.csv(paste("./resources/raw_results/", args[1],"/prototype_construction/discretize_rebalance/summary/chi2tests/binary/order_test.csv", sep = ""), header = TRUE)
fn <- read.csv(paste("./resources/raw_results/", args[1],"/prototype_construction/features_normalize/summary/chi2tests/binary/order_test.csv", sep = ""), header = TRUE)
fr <- read.csv(paste("./resources/raw_results/", args[1],"/prototype_construction/features_rebalance/summary/chi2tests/binary/order_test.csv", sep = ""), header = TRUE)

fnp <- pbinom(fn[4, 4],fn[4, 2],0.5,lower.tail=FALSE)
dfp <- pbinom(df[4, 3],df[4, 2],0.5,lower.tail=FALSE)
frp <- pbinom(fr[4, 3],fr[4, 2],0.5,lower.tail=FALSE)
drp <- pbinom(dr[4, 3],dr[4, 2],0.5,lower.tail=FALSE)

output <- data.frame("T1" = c("F", "D", "F", "D"),
                     "T2" = c("N", "F", "R", "R"),
                     "T1->T2" = c(fn[4, 3], df[4, 3], fr[4, 3], dr[4, 3]),
                     "T2->T1" = c(fn[4, 4], df[4, 4], fr[4, 4], dr[4, 4]),
                     "alpha" = c(0.05, 0.05, 0.05, 0.05),
                     "p-value" = c(fnp, dfp, frp, drp)
                    )

write.csv(output, paste(c("./resources/artifacts/", args[1], "/Table4.csv"), collapse = ""), row.names = FALSE)
