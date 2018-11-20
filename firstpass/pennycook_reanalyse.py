'''
reanalysis of

Pennycook, G., & Rand, D. G. (2018). Lazy, not biased: Susceptibility to partisan fake news is better explained by lack of reasoning than by motivated reasoning. Cognition.
https://www.sciencedirect.com/science/article/pii/S001002771830163X



Data from:

https://osf.io/f5dgh/

'''

import socket #to get host machine identity
import os # for joining paths and filenames sensibly
import pandas as pd #dataframes
import numpy as np #number functions
import matplotlib.pyplot as plt #plotting
import seaborn as sns
pd.set_option('use_inf_as_null', True) #this helps for calculating dprime if some values are inf

#test which machine we are on and set working directory
if 'tom' in socket.gethostname():
    os.chdir('/home/tom/t.stafford@sheffield.ac.uk/A_UNIVERSITY/toys/pennycook2018')
else:
    print("I don't know where I am! ")
    print("Maybe the script will run anyway...")


#load data
df1=pd.read_csv('Pennycook & Rand (Study 1).csv')

#our key variables
df1['CRT_Int']
df1['ClintonTrump']=='1'
df1['ClintonTrump']=='2'


#make sure variable we want to work with are loaded correctly

#fake items, by partisan bias
CFakes=df1[['Fake1_2','Fake2_2','Fake3_2','Fake4_2','Fake5_2']].apply(pd.to_numeric,errors='coerce')
LFakes=df1[['Fake6_2','Fake7_2','Fake8_2','Fake9_2','Fake10_2']].apply(pd.to_numeric,errors='coerce')
NFakes=df1[['Fake11_2','Fake12_2','Fake13_2','Fake14_2','Fake15_2']].apply(pd.to_numeric,errors='coerce')
#real items, by partisan bias
CReals=df1[['Real1_2','Real2_2','Real3_2','Real4_2','Real5_2']].apply(pd.to_numeric,errors='coerce')
LReals=df1[['Real6_2','Real7_2','Real8_2','Real9_2','Real10_2']].apply(pd.to_numeric,errors='coerce')
NReals=df1[['Real11_2','Real12_2','Real13_2','Real14_2','Real15_2']].apply(pd.to_numeric,errors='coerce')

#calculate proportion of False Alarms (fakes rated as accurate)
df1['CFalseAlarms']=(CFakes>2).sum(axis=1)/5
df1['LFalseAlarms']=(LFakes>2).sum(axis=1)/5
df1['NFalseAlarms']=(NFakes>2).sum(axis=1)/5

#calculate proportion of hits (real stories rated as accurate)
df1['CHits']=(CReals>2).sum(axis=1)/5
df1['LHits']=(LReals>2).sum(axis=1)/5
df1['NHits']=(NReals>2).sum(axis=1)/5


'''
interlude where we think about P&R's discernment scores

P&R (2018), p3 "truth discernment (average accuracy ratings of real news minus average
accuracy ratings of fake news)"

p4 "which was computed by subtracting z-scores for fake news (false alarms) from z-scores for real news (hits)"

'''

df1['Discernment']==df1['ZReal']-df1['ZFake'] # == True
len(df1['ZFake'].unique()) #42 unique values, from scores on 12 items
#how are Z scores calculated?
df1['ZFake'] # Z scores of Fake items

'''
795    0.412995
796    0.572527
797    2.008314
798    0.572527
799    0.093932
800   -1.022791
801    1.051123
'''

fakes=['Fake'+str(n+1)+'_2' for n in range(12)]
reals=['Real'+str(n+1)+'_2' for n in range(12)]


#also to think about: how are proper d' scores calculated
z=(df1[fakes].sum(axis=1))/12

#summing fakes and computing z scores on average accuracy of fakes, doesn't give the same z score
F=df1[fakes].sum(axis=1)
z1=(F-F.mean())/F.std()

#put it does correlate perfectly
np.corrcoef(z,z1)

z.mean() # = 1.083, so something is funny in the calculation of the z score, since it is offset from 1

#we can convert between them by shifting the mean and scaling

z2 = (z - z.mean())
ratio=z1[3]/z2[3]
z2 = z2*ratio

plt.clf()
plt.plot(z1,'x')
plt.plot(z2,'o')

'''
in SDT the Z function is the inverse of the cumulative normal distribution
how does this relate to z scores?
'''




''
(df1['L_Fake_Accurate']-df1['L_Fake_Accurate'].mean())/df1['L_Fake_Accurate'].std()

smallval=0.01 #correction for floor or ceiling proportions

'''
Functions for calculatin d prime (three congruency conditions, so three functions)

#there's a better way to do this, but i'm too tired to figure it out
'''
def dprimeCfunc(row):
    from scipy.stats import norm
    if row['CHits']==1:
        hits=1-smallval
    elif row['CHits']==0:
        hits=0+smallval
    else:
        hits=row['CHits']
    if row['CFalseAlarms']==0:
        FAs=0+smallval
    elif row['CFalseAlarms']==1:
        FAs=1-smallval
    else:
        FAs=row['CFalseAlarms']
    return norm.ppf(hits)-norm.ppf(FAs)
    
def dprimeLfunc(row):
    from scipy.stats import norm
    if row['LHits']==1:
        hits=1-smallval
    elif row['LHits']==0:
        hits=0+smallval
    else:
        hits=row['LHits']
    if row['LFalseAlarms']==0:
        FAs=0+smallval
    elif row['LFalseAlarms']==1:
        FAs=1-smallval
    else:
        FAs=row['LFalseAlarms']
    return norm.ppf(hits)-norm.ppf(FAs)

def dprimeNfunc(row):
    from scipy.stats import norm
    if row['NHits']==1:
        hits=1-smallval
    elif row['NHits']==0:
        hits=0+smallval
    else:
        hits=row['NHits']
    if row['NFalseAlarms']==0:
        FAs=0+smallval
    elif row['NFalseAlarms']==1:
        FAs=1-smallval
    else:
        FAs=row['NFalseAlarms']
    return norm.ppf(hits)-norm.ppf(FAs)

#calculate d prime scores
df1['dprimeC']=df1.apply(dprimeCfunc,axis=1)
df1['dprimeL']=df1.apply(dprimeLfunc,axis=1)
df1['dprimeN']=df1.apply(dprimeNfunc,axis=1)

#re-code according to participant partisan bias - Dems looking at C news | Reps looking at L news = congruent, 
df1['cong_dprime']=df1[(df1['ClintonTrump']=='1')]['dprimeC'].append(df1[(df1['ClintonTrump']=='2')]['dprimeL'])
df1['incong_dprime']=df1[(df1['ClintonTrump']=='1')]['dprimeL'].append(df1[(df1['ClintonTrump']=='2')]['dprimeC'])

'''
compare d' and discernment
'''


df1['dprimeL']
df1['dprimeC']
df1['dprimeN']
df1['L_Discernment']
df1['C_Discernment']
df1['N_Discernment']

plt.clf()
plt.plot(df1['N_Discernment'],df1['dprimeN'],'.')
plt.xlabel('discernment')
plt.ylabel('dprime')

plt.clf()
plt.plot(df1['dprimeN'],df1['N_Discernment'],'.')
plt.ylabel('discernment')
plt.xlabel('dprime')

np.corrcoef(df1['dprimeN'],df1['N_Discernment'])

'''
plot d prime heatmaps
'''


#plot parameters
y_lower=-1
y_upper=+5

#partisan bias congruent dprime * CRT
plt.clf()
ax = sns.jointplot(df1['CRT_Int'], df1['cong_dprime'], kind="kde")
plt.ylim([y_lower,y_upper])
plt.savefig('cong_drprime.png',bbox='tight')

#partisan bias incongruent dprime * CRT
plt.clf()
ax = sns.jointplot(df1['CRT_Int'], df1['incong_dprime'], kind="kde")
plt.ylim([y_lower,y_upper])
plt.savefig('incong_drprime.png',bbox='tight')

#partisan bias neutral dprime * CRT
plt.clf()
ax = sns.jointplot(df1['CRT_Int'], df1['dprimeN'], kind="kde")
plt.ylim([y_lower,y_upper])
plt.savefig('neutral_drprime.png',bbox='tight')


#Really it should be a regression, 

'''
export data for regression modelling
'''

df_wide=df1[['cong_dprime','incong_dprime','dprimeN','CRT_Int']]

df_wide.columns=['C','I','N','CRT']
df_wide.to_csv('dprimedata_wide.csv')

df_long=df_wide.stack().reset_index()
df_long.columns=['p','congruency','dprime']
df_long.to_csv('drpimedata_long.csv')


# make some simple regressions
# cribbing from here https://stats.stackexchange.com/questions/256050/linear-regression-from-parameters-with-standard-error-to-prediction-interval

df_wide=df_wide.dropna()


from scipy.optimize import curve_fit

def make_plot(x,y,name):

    f = lambda x, *p: np.polyval(p, x) #our underlying model
    p, cov = curve_fit(f, x, y, [1, 1]) #first order fit = linear regression


    xi = np.linspace(np.min(x), np.max(x), 100)
    ps = np.random.multivariate_normal(p, cov, 10000)
    ysample = np.asarray([f(xi, *pi) for pi in ps])
    lower = np.percentile(ysample, 2.5, axis=0)
    upper = np.percentile(ysample, 97.5, axis=0)

    # regression estimate line
    y_fit = np.poly1d(p)(xi)

    # plot
    plt.clf()
    #plt.plot(x, y, 'bo') #don't plot individual points
    plt.plot(xi, y_fit, 'r-')
    plt.plot(xi, lower, 'b--')
    plt.plot(xi, upper, 'b--')
    plt.savefig(name+'.png',bbox_inches='tight')



make_plot(df_wide['CRT'],df_wide['C'],'Congruent')
make_plot(df_wide['CRT'],df_wide['I'],'Incongruent')
make_plot(df_wide['CRT'],df_wide['N'],'Neutral')



f = lambda x, *p: np.polyval(p, x) #our underlying model
plt.clf()

for cond,condlabel,colour in zip(['C','I','N'],['Congruent','Incongruent','Neutral'],['green','blue','black']):

    x = df_wide['CRT']
    y = df_wide[cond]
    
    p, cov = curve_fit(f, x, y, [1, 1]) #first order fit = linear regression

    xi = np.linspace(np.min(x), np.max(x), 100)
    ps = np.random.multivariate_normal(p, cov, 10000) #assume that the fit (p, cov) represents a normal distribution and sample from it
    ysample = np.asarray([f(xi, *pi) for pi in ps])
    lower = np.percentile(ysample, 2.5, axis=0) # selecting the lower and upper 2.5% quantiles:
    upper = np.percentile(ysample, 97.5, axis=0)

    # regression estimate line
    y_fit = np.poly1d(p)(xi)
    # same as 
    # y_fit2 = xi*p[0]+p[1]
    # plot
    
    #plt.plot(x, y, 'bo') #don't plot individual points
    plt.plot(xi, y_fit, '-',color=colour,label=condlabel)
    plt.fill_between(xi, lower, upper, facecolor=colour, alpha=0.5)


plt.legend(loc=0)
plt.ylim([0,2.75])
plt.xlabel('Cognitive Reflection Test (CRT) score')
plt.ylabel('discrimination (d prime)')
plt.savefig('dprime_all.png',bbox_inches='tight')

df2=pd.read_csv('Pennycook & Rand (Study 2).csv')
df2['CRT_Int']

fakes=['Fake'+str(n+1)+'_2' for n in range(12)]
reals=['Real'+str(n+1)+'_2' for n in range(12)]

for f in fakes:
    df2[f]=pd.to_numeric(df2[f],errors='coerce')

for f in reals:
    df2[f]=pd.to_numeric(df2[f],errors='coerce')
    
df2['ClintonTrump']=df2['ClintonTrump'].astype(str)


df2['fakes_ac1_to_6']=(df2[fakes[:6]]>2).sum(axis=1)/6
df2['fakes_ac7_to_12']=(df2[fakes[6:]]>2).sum(axis=1)/6
df2['reals_ac1_to_6']=(df2[reals[:6]]>2).sum(axis=1)/6
df2['reals_ac7_to_12']=(df2[reals[6:]]>2).sum(axis=1)/6

(df2[fakes]>2).sum(axis=1).hist(bins=13, rwidth=0.85)
plt.xlabel('Fakes rated as accurate (out of 12)',fontsize=18)
plt.ylabel('Frequency',fontsize=18)
plt.xticks(np.arange(0.5,13,1),[str(a) for a in range(13)])
plt.savefig('s2_FA_freq.png',bbox_inches='tight')


def congsort_fakes(row):
    if row['ClintonTrump']=='2':
        return row['fakes_ac1_to_6']
    elif row['ClintonTrump']=='1':
        return row['fakes_ac7_to_12']

def congsort_reals(row):
    if row['ClintonTrump']=='2':
        return row['reals_ac1_to_6']
    elif row['ClintonTrump']=='1':
        return row['reals_ac7_to_12']

def incongsort_fakes(row):
    if row['ClintonTrump']=='1':
        return row['fakes_ac1_to_6']
    elif row['ClintonTrump']=='2':
        return row['fakes_ac7_to_12']

def incongsort_reals(row):
    if row['ClintonTrump']=='1':
        return row['reals_ac1_to_6']
    elif row['ClintonTrump']=='2':
        return row['reals_ac7_to_12']


df2['fakeFAs_PC']=df2.apply(congsort_fakes,axis=1)
df2['realhits_PC']=df2.apply(congsort_reals,axis=1)

df2['fakeFAs_nPC']=df2.apply(incongsort_fakes,axis=1)
df2['realhits_nPC']=df2.apply(incongsort_reals,axis=1)

#df2[['fakes_ac1_to_6','fakes_ac7_to_12','reals_ac1_to_6','reals_ac7_to_12']]
#df2[['fakeFAs_PC','realhits_PC','fakeFAs_nPC','realhits_nPC']]

def dprimeCfunc(row):
    from scipy.stats import norm
    if row['realhits_PC']==1:
        hits=1-smallval
    elif row['realhits_PC']==0:
        hits=0+smallval
    else:
        hits=row['realhits_PC']
    if row['fakeFAs_PC']==0:
        FAs=0+smallval
    elif row['fakeFAs_PC']==1:
        FAs=1-smallval
    else:
        FAs=row['fakeFAs_PC']
    return norm.ppf(hits)-norm.ppf(FAs)
    

def dprimeIfunc(row):
    from scipy.stats import norm
    if row['realhits_nPC']==1:
        hits=1-smallval
    elif row['realhits_nPC']==0:
        hits=0+smallval
    else:
        hits=row['realhits_nPC']
    if row['fakeFAs_nPC']==0:
        FAs=0+smallval
    elif row['fakeFAs_nPC']==1:
        FAs=1-smallval
    else:
        FAs=row['fakeFAs_nPC']
    return norm.ppf(hits)-norm.ppf(FAs)
    
df2['dprimeC']=df2.apply(dprimeCfunc,axis=1)
df2['dprimeI']=df2.apply(dprimeIfunc,axis=1)

df2['dprimeC'].hist()

df2.dropna(inplace=True)


f = lambda x, *p: np.polyval(p, x) #our underlying model
plt.clf()

for cond,condlabel,colour in zip(['dprimeC','dprimeI'],['Congruent','Incongruent'],['green','blue']):

    x = df2['CRT_Int']
    y = df2[cond]
    
    p, cov = curve_fit(f, x, y, [1, 1]) #first order fit = linear regression

    xi = np.linspace(np.min(x), np.max(x), 100)
    ps = np.random.multivariate_normal(p, cov, 10000) #assume that the fit (p, cov) represents a normal distribution and sample from it
    ysample = np.asarray([f(xi, *pi) for pi in ps])
    lower = np.percentile(ysample, 2.5, axis=0) # selecting the lower and upper 2.5% quantiles:
    upper = np.percentile(ysample, 97.5, axis=0)

    # regression estimate line
    y_fit = np.poly1d(p)(xi)
    # same as 
    # y_fit2 = xi*p[0]+p[1]
    # plot
    
    #plt.plot(x, y, 'bo') #don't plot individual points
    plt.plot(xi, y_fit, '-',color=colour,label=condlabel)
    plt.fill_between(xi, lower, upper, facecolor=colour, alpha=0.5)


plt.legend(loc=0)
plt.ylim([-0.4,1.4])
plt.xlabel('Cognitive Reflection Test (CRT) score')
plt.ylabel('discrimination (d prime)')
plt.savefig('s2_dprime_all.png',bbox_inches='tight')