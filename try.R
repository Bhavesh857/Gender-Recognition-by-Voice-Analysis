#! /usr/bin/Rscript

#install.packages("tuneR")
#install.packages("seewave")


library(tuneR )
library(seewave)

#my_data <- read.delim("file1")
#print(my_data)
s <- paste(readLines("file1"), collapse=" ")
print(s)
r <- tuneR::readWave(s,from = 1, to = Inf, units = "seconds")

songspec <- seewave::spec(r, f = r@samp.rate, plot = FALSE)
analysis <- seewave::specprop(songspec, f = r@samp.rate, flim = c(0, 280/1000), plot = FALSE)

meanfreq <- analysis$mean/1000

sd <- analysis$sd/1000

Q25 <- analysis$Q25/1000

IQR <- analysis$IQR/1000

sp.ent <- analysis$sh

sfm <- analysis$sfm

ff <- seewave::fund(r, f = r@samp.rate, ovlp = 50, threshold = 5, 
                    fmax = 280, ylim=c(0, 280/1000), plot = FALSE, wl = 2048)[, 2]
meanfun<-mean(ff, na.rm = T)


p <- c(meanfun,IQR,Q25,sp.ent,sd,sfm,meanfreq)

sink("/home/pooja/Documents/ml/out")
cat(p)
sink()




