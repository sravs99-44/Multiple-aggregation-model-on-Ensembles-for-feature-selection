import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from skfeature.function.similarity_based import fisher_score
from skfeature.function.similarity_based import reliefF
from skfeature.function.similarity_based import trace_ratio
from sklearn.feature_selection import SelectKBest
from sklearn import feature_selection as sk
from skfeature.function.information_theoretical_based import MRMR
from skfeature.function.information_theoretical_based import CIFE
from skfeature.function.information_theoretical_based import MIM
from skfeature.function.information_theoretical_based import MIFS
from skfeature.function.information_theoretical_based import JMI
from skfeature.function.information_theoretical_based import ICAP
from skfeature.function.information_theoretical_based import DISR
from skfeature.function.information_theoretical_based import CMIM
from sklearn.utils import resample
import rankaggregation as ra
agg = ra.RankAggregator()

def data(df):
	df_array = df.values
	X = df_array[:,:-1]
	Y = df_array[:,-1]
	return X,Y

def bootstrap(df):
	newdf=[]
	for i in range(6):
		boot = resample(df, replace=True, n_samples=100)
		newdf.append(boot.values)
	return newdf

def dispcolumns(df,ranks):
	m=((df.columns).tolist())
	n=[]
	for i in range(len(m)-1):
		n.append(m[ranks[i] - 1])
	return n 


def rankaggregate(rank):
    R=np.array(rank).sum(axis=0)
    k=sorted(R)
    z=[]
    for i in R:
        z.append(k.index(i)+1)
    R=[]
    for i in range(len(z)):
        R.append(z.index(i+1))
    return R

def samp(k):
    K = [str(i) for i in k]
    n=[]
    for i in range(len(k)):
        n.append(str(K.index(str(i))+1))
    return n

def relieF(data):
	rank=[]
	for i in range(6):
		X=data[i][:,:-1]
		Y=data[i][:,-1]
		score = reliefF.reliefF(X, Y)
		idx1 = reliefF.feature_ranking(score)
		idx = samp(idx1.tolist())
		rank.append(idx)
	m = agg.instant_runoff(rank)
	R = [int(i) for i in m]

	return R


def fisher(data):
	rank=[]
	for i in range(6):
		X=data[i][:,:-1]
		Y=data[i][:,-1]
		score =fisher_score.fisher_score(X, Y)
		idx1 = fisher_score.feature_ranking(score)
		idx = samp(idx1.tolist())
		rank.append(idx)
	R = rankaggregate(rank)
	return R

def mim(data):
	rank=[]
	for i in range(6):
		X=data[i][:,:-1]
		Y=data[i][:,-1]
		F,_,_= MIM.mim(X,Y)
		idx = samp(F[:-1].tolist())
		rank.append(idx)
	R = rankaggregate(rank)
	return R

def icap(data):
	rank=[]
	for i in range(6):
		X=data[i][:,:-1]
		Y=data[i][:,-1]
		F,_,_= ICAP.icap(X,Y)
		idx = samp(F[:-1].tolist())
		rank.append(idx)
	R = rankaggregate(rank)
	return R

def cmim(data):
	rank=[]
	for i in range(6):
		X=data[i][:,:-1]
		Y=data[i][:,-1]
		F,_,_= CMIM.cmim(X,Y)
		idx = samp(F[:-1].tolist())
		rank.append(idx)
	R = rankaggregate(rank)
	return R

def jmi(data):
	rank=[]
	for i in range(6):
		X=data[i][:,:-1]
		Y=data[i][:,-1]
		F,_,_= JMI.jmi(X,Y)
		idx = samp(F[:-1].tolist())
		rank.append(idx)
	R = rankaggregate(rank)
	return R

def disr(data):
	rank=[]
	for i in range(6):
		X=data[i][:,:-1]
		Y=data[i][:,-1]
		F,_,_= DISR.disr(X,Y)
		idx = samp(F[:-1].tolist())
		rank.append(idx)
	R = rankaggregate(rank)
	return R



