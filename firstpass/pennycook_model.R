#setwd("~/Dropbox/university/toys/pennycook2018") #tom's laptop

dat = read.csv("/home/tom/Dropbox/university/toys/pennycook2018/dprimedata_wide.csv")

m1 =lm(C ~ CRT, data=dat)
summary(m1)
visreg(m1)

#install.packages('visreg')
require(visreg)
fit <- lm(Ozone ~ Solar.R,data=airquality)
visreg(fit)