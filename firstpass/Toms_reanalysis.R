library(lme4)
library(lmerTest)
library(plyr)
library(tidyverse)
library(car)

#setwd("C:/Users/Psychology/Desktop/R")
setwd("/home/tom/t.stafford@sheffield.ac.uk/A_UNIVERSITY/toys/pennycook2018")
# hello Jenny https://github.com/jennybc/here_here

wide_data <- read.csv("dprimedata_wide.csv",header=TRUE, sep=",", na.strings="-999", dec=".", strip.white=TRUE)
wide_data$sub <- as.factor(wide_data$X)
#I create the long form here because the long form csv seemed to have a glitch (the CRT values were included in the dprime column)
long_data <- gather(wide_data, type, dprime, C:N, factor_key=TRUE) #this function comes from the tidyr library
long_data$sub <- as.factor(long_data$sub)
#type is factor
#dprime and CRT are numerical, I think that's okay

mean(wide_data$CRT)
mean(long_data$CRT) #these two give the same value, as they should

#apparently if one wants to look at interactions, type III ANOVAs are optimal, as explained at the link below
#https://mcfromnz.wordpress.com/2011/03/02/anova-type-iiiiii-ss-explained/ #
#"In general, if there is no significant interaction effect, then type II is more powerful, and follows the principle 
#of marginality. If interaction is present, then type II is inappropriate while type III can still be used, but results
#need to be interpreted with caution (in the presence of interactions, main effects are rarely interpretable)."
#the link also says that for type III to work, it is necessary to choose a contrasts setting that sums to zero, so
#dummy coding is suboptimal - I switch to effect coding here
contrasts(long_data$type) <- contr.sum

#the models:

mod1 <- lmer(dprime ~ 1 + type*CRT + (1|sub), data = long_data, verbose = 0, REML = F)
Anova(mod1)
Anova(mod1, type ="III")

#I tried to run a model with a random slope for type, too:
mod2 <- lmer(dprime ~ 1 + type*CRT + (1+type|sub), data = long_data, verbose = 0, REML = F)
#this, however, gives a surprising error message:
#"Error: number of observations (=2402) <= number of random effects (=2406) for term (1 + type | sub);
#the random-effects parameters and the residual variance (or scale parameter) are probably unidentifiable"
#based on this CrossValidated answer: https://stats.stackexchange.com/questions/193678/number-of-random-effects-is-not-correct-in-lmer-model
#this is because this is trying to fit two slopes and an intercept for each participant, so N * 3 parameters, but there is only
#N * 3 data points to begin with, so there's just no enough data to do that
#this I think goes back to the question of whether it even makes sense to have a mixed model

#based on a quick googling, this is what a mixed ANOVA should look like in R:

dprime_anova <- aov(dprime ~ type*CRT + Error(sub/type), data=long_data) #this gives a weird error message, probably because there's missing data
#at least according to this site (scroll to bottom): https://www.r-bloggers.com/two-way-anova-with-repeated-measures/
summary(dprime_anova)

#I remove incomplete cases here
complete_wide <- wide_data[complete.cases(wide_data), ]
complete_long  <- gather(complete_wide, type, dprime, C:N, factor_key=TRUE)

dprime_anova <- aov(dprime ~ type*CRT + Error(sub/type), data=complete_long) #the error message is gone
summary(dprime_anova)
