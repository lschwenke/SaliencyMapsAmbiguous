#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import numpy as np

from modules import helper
from scipy import stats
from scipy.stats.stats import pearsonr
from modules import saliencyHelper as sh

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

from modules import dataset_selecter as ds
from modules import pytorchTrain as pt
from modules import saliencyHelper as sh
from modules import helper
from modules import GCRPlus
from scipy.stats import rankdata
import msgpack


# 

# In[3]:


class config:
    def __init__(self, configName, hyperss, hypNames, test_sizes, topLevelss, toplevels, dataset, symbols, nrEmpty,andStack,orStack,xorStack,nrAnds,nrOrs,nrxor,trueIndexes,orOffSet,xorOffSet,redaundantIndexes,batch_size):
        self.configName=configName
        self.hyperss=hyperss
        self.hypNames=hypNames
        self.test_sizes=test_sizes
        self.topLevelss=topLevelss
        self.toplevels=toplevels
        self.dataset=dataset
        self.symbols=symbols
        self.nrEmpty=nrEmpty
        self.andStack=andStack
        self.orStack=orStack
        self.xorStack=xorStack
        self.nrAnds=nrAnds
        self.nrOrs=nrOrs
        self.nrxor=nrxor
        self.trueIndexes=trueIndexes
        self.orOffSet=orOffSet
        self.xorOffSet=xorOffSet
        self.redaundantIndexes=redaundantIndexes
        self.batch_size=batch_size

allConfigs = []

hyperss = [['Transformer', 500, 64, 0.5, True, True, 4 ,  2, 0 , 0, True]
    , ['Transformer', 500, 64, 0.5, True, True, 4 ,  2, 0 , 0.3, True]
    , ['Transformer', 500, 64, 0.5, True, True, 4 ,  2, 0 , 0, False]
    , ['Transformer', 500, 64, 0.5, True, True, 4 ,  2, 0 , 0.3, False]
    , ['CNN', 500, 8, 1, True , True , 0 ,  2, 0 , 0, False]
    , ['CNN', 500, 8, 1, False , True , 0 ,  2, 0 , 0, False]
    , ['CNN', 500, 8, 1, True , False , 0 ,  2, 0 , 0, False]
    , ['CNN', 500, 8, 1, False , False , 0 ,  2, 0 , 0, False]
    ]

hypNames= 'TC'

#NOTE: Change test_sizes for all configs to remove some of them from the results (not applicable for all prints)
test_sizes = [0,0.1]
topLevelss = [['or', 'and', 'xor'],['and'], ['or'],['xor']]
toplevels = ['or', 'and', 'xor']

dataset = 'Andor'
symbols = 2#4
nrEmpty = 2#2
andStack = 1
orStack = 1
xorStack = 1
nrAnds = 2#2
nrOrs = 2#2
nrxor = 2#2
trueIndexes = [1]#[1]#[1,3]
orOffSet = 0
xorOffSet = 0
redaundantIndexes = []
batch_size = 10 #100#500#10
configName = '2inBinary'

in2Config = config(configName, hyperss, hypNames, test_sizes, topLevelss, toplevels, dataset, symbols, nrEmpty,andStack,orStack,xorStack,nrAnds,nrOrs,nrxor,trueIndexes,orOffSet,xorOffSet,redaundantIndexes,batch_size)
allConfigs.append(in2Config)

hyperss = [['Transformer', 150, 32, 0.5, True, True, 4 ,  2, 0 , 0, True]
    , ['Transformer', 150, 32, 0.5, True, True, 4 ,  2, 0 , 0.3, True]
    , ['Transformer', 150, 32, 0.5, True, True, 4 ,  2, 0 , 0, False]
    , ['Transformer', 150, 32, 0.5, True, True, 4 ,  2, 0 , 0.3, False]
    , ['CNN', 100, 8, 1, True , True , 0 ,  2, 0 , 0, False]
    , ['CNN', 100, 8, 1, False , True , 0 ,  2, 0 , 0, False]
    , ['CNN', 100, 8, 1, True , False , 0 ,  2, 0 , 0, False]
    , ['CNN', 100, 8, 1, False , False , 0 ,  2, 0 , 0, False]
    ]

hypNames= 'TC'

test_sizes = [0,0.2]
topLevelss = [['or', 'and', 'xor'],['and'], ['or'],['xor']]
toplevels = ['or', 'and', 'xor']

dataset = 'Andor'
symbols = 4#4
nrEmpty = 2#2
andStack = 1
orStack = 1
xorStack = 1
nrAnds = 2#2
nrOrs = 2#2
nrxor = 2#2
trueIndexes = [1,3]#[1]#[1,3]
orOffSet = 0
xorOffSet = 0
redaundantIndexes = []
batch_size = 500 #100#500#10
configName = '2inQuaternary'

in2s4Config = config(configName, hyperss, hypNames, test_sizes, topLevelss, toplevels, dataset, symbols, nrEmpty,andStack,orStack,xorStack,nrAnds,nrOrs,nrxor,trueIndexes,orOffSet,xorOffSet,redaundantIndexes,batch_size)
allConfigs.append(in2s4Config)

hyperss = [['Transformer', 150, 32, 0.5, True, True, 4 ,  2, 0 , 0, True]
    , ['Transformer', 150, 32, 0.5, True, True, 4 ,  2, 0 , 0.3, True]
    , ['Transformer', 150, 32, 0.5, True, True, 4 ,  2, 0 , 0, False]
    , ['Transformer', 150, 32, 0.5, True, True, 4 ,  2, 0 , 0.3, False]
    , ['CNN', 100, 8, 1, True , True , 0 ,  2, 0 , 0, False]
    , ['CNN', 100, 8, 1, False , True , 0 ,  2, 0 , 0, False]
    , ['CNN', 100, 8, 1, True , False , 0 ,  2, 0 , 0, False]
    , ['CNN', 100, 8, 1, False , False , 0 ,  2, 0 , 0, False]
    ]

hypNames= 'TC'
configName = '3inBinary'

test_sizes = [0,0.2]
topLevelss = [['or', 'and', 'xor'],['and'], ['or'],['xor']]
toplevels = ['or', 'and', 'xor']

dataset = 'Andor'
symbols = 2#4
nrEmpty = 3#2
andStack = 1
orStack = 1
xorStack = 1
nrAnds = 3#2
nrOrs = 3#2
nrxor = 3#2
trueIndexes = [1]#[1]#[1,3]
orOffSet = 0
xorOffSet = 0
redaundantIndexes = []
batch_size = 100 #100#500#10

in3Config = config(configName, hyperss, hypNames, test_sizes, topLevelss, toplevels, dataset, symbols, nrEmpty,andStack,orStack,xorStack,nrAnds,nrOrs,nrxor,trueIndexes,orOffSet,xorOffSet,redaundantIndexes,batch_size)
allConfigs.append(in3Config)


hyperss = [['Transformer', 500, 64, 0.5, True, True, 4 ,  2, 0 , 0, True]
    , ['Transformer', 500, 64, 0.5, True, True, 4 ,  2, 0 , 0.3, True]
    , ['Transformer', 500, 64, 0.5, True, True, 4 ,  2, 0 , 0, False]
    , ['Transformer', 500, 64, 0.5, True, True, 4 ,  2, 0 , 0.3, False]
    , ['CNN', 500, 8, 1, True , True , 0 ,  2, 0 , 0, False]
    , ['CNN', 500, 8, 1, False , True , 0 ,  2, 0 , 0, False]
    , ['CNN', 500, 8, 1, True , False , 0 ,  2, 0 , 0, False]
    , ['CNN', 500, 8, 1, False , False , 0 ,  2, 0 , 0, False]
    ]

####################################################################################

hypNames= 'TC'
configName = '2inBinaryDoubleGateOR'

test_sizes = [0,0.1]
topLevelss = [['or', 'and', 'xor'],['and'], ['or'],['xor']]
toplevels = ['or', 'and', 'xor']

dataset = 'Andor'
symbols = 2#4
nrEmpty = 2#2
andStack = 0
orStack = 2
xorStack = 0
nrAnds = 0#2
nrOrs = 2#2
nrxor = 0#2
trueIndexes = [1]#[1]#[1,3]
orOffSet = 0
xorOffSet = 0
redaundantIndexes = []
batch_size = 10 #100#500#10

in2BinaryDoubleGateOR = config(configName, hyperss, hypNames, test_sizes, topLevelss, toplevels, dataset, symbols, nrEmpty,andStack,orStack,xorStack,nrAnds,nrOrs,nrxor,trueIndexes,orOffSet,xorOffSet,redaundantIndexes,batch_size)
allConfigs.append(in2BinaryDoubleGateOR)


hyperss = [['Transformer', 500, 64, 0.5, True, True, 4 ,  2, 0 , 0, True]
    , ['Transformer', 500, 64, 0.5, True, True, 4 ,  2, 0 , 0.3, True]
    , ['Transformer', 500, 64, 0.5, True, True, 4 ,  2, 0 , 0, False]
    , ['Transformer', 500, 64, 0.5, True, True, 4 ,  2, 0 , 0.3, False]
    , ['CNN', 500, 8, 1, True , True , 0 ,  2, 0 , 0, False]
    , ['CNN', 500, 8, 1, False , True , 0 ,  2, 0 , 0, False]
    , ['CNN', 500, 8, 1, True , False , 0 ,  2, 0 , 0, False]
    , ['CNN', 500, 8, 1, False , False , 0 ,  2, 0 , 0, False]
    ]


hypNames= 'TC'
configName = '2inBinaryDoubleGateAND'

test_sizes = [0,0.1]
topLevelss = [['or', 'and', 'xor'],['and'], ['or'],['xor']]
toplevels = ['or', 'and', 'xor']

dataset = 'Andor'
symbols = 2#4
nrEmpty = 2#2
andStack = 2
orStack = 0
xorStack = 0
nrAnds = 2#2
nrOrs = 0#2
nrxor = 0#2
trueIndexes = [1]#[1]#[1,3]
orOffSet = 0
xorOffSet = 0
redaundantIndexes = []
batch_size = 10 #100#500#10

in2BinaryDoubleGateAND = config(configName, hyperss, hypNames, test_sizes, topLevelss, toplevels, dataset, symbols, nrEmpty,andStack,orStack,xorStack,nrAnds,nrOrs,nrxor,trueIndexes,orOffSet,xorOffSet,redaundantIndexes,batch_size)
allConfigs.append(in2BinaryDoubleGateAND)

hyperss = [['Transformer', 500, 64, 0.5, True, True, 4 ,  2, 0 , 0, True]
    , ['Transformer', 500, 64, 0.5, True, True, 4 ,  2, 0 , 0.3, True]
    , ['Transformer', 500, 64, 0.5, True, True, 4 ,  2, 0 , 0, False]
    , ['Transformer', 500, 64, 0.5, True, True, 4 ,  2, 0 , 0.3, False]
    , ['CNN', 500, 8, 1, True , True , 0 ,  2, 0 , 0, False]
    , ['CNN', 500, 8, 1, False , True , 0 ,  2, 0 , 0, False]
    , ['CNN', 500, 8, 1, True , False , 0 ,  2, 0 , 0, False]
    , ['CNN', 500, 8, 1, False , False , 0 ,  2, 0 , 0, False]
    ]


hypNames= 'TC'
configName = '2inBinaryDoubleGateXOR'

test_sizes = [0,0.1]
topLevelss = [['or', 'and', 'xor'],['and'], ['or'],['xor']]
toplevels = ['or', 'and', 'xor']

dataset = 'Andor'
symbols = 2#4
nrEmpty = 2#2
andStack = 0
orStack = 0
xorStack = 2
nrAnds = 0#2
nrOrs = 0#2
nrxor = 2#2
trueIndexes = [1]#[1]#[1,3]
orOffSet = 0
xorOffSet = 0
redaundantIndexes = []
batch_size = 10 #100#500#10

in2BinaryDoubleGateXOR = config(configName, hyperss, hypNames, test_sizes, topLevelss, toplevels, dataset, symbols, nrEmpty,andStack,orStack,xorStack,nrAnds,nrOrs,nrxor,trueIndexes,orOffSet,xorOffSet,redaundantIndexes,batch_size)
allConfigs.append(in2BinaryDoubleGateXOR)


hyperss = [['Transformer', 500, 64, 0.5, True, True, 4 ,  2, 0 , 0, True]
    , ['Transformer', 500, 64, 0.5, True, True, 4 ,  2, 0 , 0.3, True]
    , ['Transformer', 500, 64, 0.5, True, True, 4 ,  2, 0 , 0, False]
    , ['Transformer', 500, 64, 0.5, True, True, 4 ,  2, 0 , 0.3, False]
    , ['CNN', 500, 8, 1, True , True , 0 ,  2, 0 , 0, False]
    , ['CNN', 500, 8, 1, False , True , 0 ,  2, 0 , 0, False]
    , ['CNN', 500, 8, 1, True , False , 0 ,  2, 0 , 0, False]
    , ['CNN', 500, 8, 1, False , False , 0 ,  2, 0 , 0, False]
    ]


hypNames= 'TC'
configName = 'BinarySingleGate'

test_sizes = [0,0.1]
topLevelss = [['or', 'and', 'xor'],['and'], ['or'],['xor']]
toplevels = ['or', 'and', 'xor']

dataset = 'Andor'
symbols = 2#4
nrEmpty = 3#2
andStack = 0
orStack = 3
xorStack = 0
nrAnds = 0#2
nrOrs = 1#2
nrxor = 0#2
trueIndexes = [1]#[1]#[1,3]
orOffSet = 0
xorOffSet = 0
redaundantIndexes = []
batch_size = 10 #100#500#10

binarySingleGate = config(configName, hyperss, hypNames, test_sizes, topLevelss, toplevels, dataset, symbols, nrEmpty,andStack,orStack,xorStack,nrAnds,nrOrs,nrxor,trueIndexes,orOffSet,xorOffSet,redaundantIndexes,batch_size)
allConfigs.append(binarySingleGate)




folderGeneral = './BilderFinal/general/'
if not os.path.exists(folderGeneral):
    os.makedirs(folderGeneral)


# In[4]:


accTreshold = 1 
splitInds = [1]



# In[5]:


def arrayToString(indexes):
    out = ""
    for i in indexes:
        out = out + ',' + str(i)
    return out



def autolabel(rects):
    for rect in rects:
        h = rect.get_height()
        ax.text(rect.get_x()+rect.get_width()/2., 1.05*h, '%d'%int(h),
                ha='center', va='bottom')


# In[ ]:





# In[6]:


# In[7]:


filteredResultsFolder = ['filteredResults']
kNames = ['LRP-Full', 'Rollout', 'LRP-Transformer',  'LRP-Transformer CLS', 'Attention', 'IntegratedGradients', 'DeepLift', 'KernelSHAP', 'GuidedGradCam', 'FeaturePermutation', 'Deconvolution','GradCAM++', 'GradCAM', 'IQ-SHAP']


testFacNames = ['Not Split Test','Split Test Set']
accTNames = ['Acc>0','Acc=100']
ks = ['LRP-full', 'LRP-rollout', 'LRP-transformer_attribution', 'LRP-transformer_attribution cls', 'Attention-.', 'captum-IntegratedGradients', 'captum-DeepLift', 'captum-KernelShap', 'captum-GuidedGradCam', 'captum-FeaturePermutation', 'captum-Deconvolution','PytGradCam-GradCAMPlusPlus', 'PytGradCam-GradCAM', 'IQ-SHAP-2OrderShap']
kIter = ks
modes = [0,2,3]

allReses = dict()
for rFolder in filteredResultsFolder:
    if rFolder not in allReses.keys():
        allReses[rFolder] = dict()


# In[ ]:





# In[8]:


topLevelNamesSmall = ['and','or', 'xor']
fig, axs1 = plt.subplots(nrows=len(allConfigs), ncols=3, sharex=True, sharey=True,
                                    figsize=(14, 10), layout="constrained")

fig2, axs2 = plt.subplots(nrows=len(allConfigs), ncols=3, sharex=True, sharey=True,
                                    figsize=(14, 8), layout="constrained")

allRes = dict()

numberBaseModelsFull = 0
numberOverallBaseModels = 0
numberModelsTrainedOverall = 0


for j, toplevel in enumerate(topLevelNamesSmall):
    for ci, config in enumerate(allConfigs):
        hyperss = config.hyperss
        hypNames= config.hypNames
        test_sizes = config.test_sizes

        dataset = config.dataset
        symbols = config.symbols
        nrEmpty = config.nrEmpty
        andStack = config.andStack
        orStack = config.orStack
        xorStack = config.xorStack
        nrAnds = config.nrAnds
        nrOrs = config.nrOrs
        nrxor = config.nrxor
        trueIndexes = config.trueIndexes
        orOffSet = config.orOffSet
        xorOffSet = config.xorOffSet
        redaundantIndexes = config.redaundantIndexes
        batch_size = config.batch_size
        configName = config.configName


        ax1 = axs1[ci, j]
        ax2 = axs2[ci, j]
        resultV1s = []
        resultV2s = []
        resultV3s = [[],[],[],[]]
        resultV4s = []


        for hypers in hyperss:

                for test_size in test_sizes:
                    modelType = hypers[0]
                    epochs = hypers[1]
                    dmodel = hypers[2]
                    dfff = hypers[3]
                    doSkip = hypers[4]
                    doBn = hypers[5]
                    header = hypers[6]
                    numOfLayers = hypers[7]
                    dropout = hypers[8]
                    att_dropout = hypers[9]
                    if modelType == 'Transformer':
                        doClsTocken = hypers[10]
                    else:
                        doClsTocken = False

                    dsName = str(dataset) +  ',l:' + str(toplevel) +',s:' +  str(symbols)+',e:' +  str(nrEmpty)+',a:' +  str(andStack)+',o:' +  str(orStack)+',x:' +  str(xorStack)+',na:' +  str(nrAnds)+',no:' +  str(nrOrs)+',nx:' +  str(nrxor)+',i:' +  arrayToString(trueIndexes)+',t:' +  str(test_size)+',oo:' +  str(orOffSet)+',xo:' +  str(xorOffSet) +',r:' + arrayToString(redaundantIndexes)


                    for rFolder in filteredResultsFolder:


                        saveName = pt.getWeightName(dsName, dataset, batch_size, epochs, numOfLayers, header, dmodel, dfff, dropout, att_dropout, doSkip, doBn, doClsTocken, learning = False, results = True, resultsPath=rFolder)

                        if os.path.isfile(saveName + '.pkl'):
                            print('FOUND: ' + saveName)
                            if saveName in allReses[rFolder].keys():
                                res = allReses[rFolder][saveName]
                            else:
                                try:
                                    results = helper.load_obj(saveName)
                                except Exception as e:
                                    print(e)
                                    continue

                                res = dict()
                                for index, vv in np.ndenumerate(results):
                                    res = vv

                                allReses[rFolder][saveName] = res
                        else:
                            print('NOT FOUND: ' + saveName)
                            continue

                        
                        name = 'performance'
                        v  = res[name]

                        
                        for i, val in enumerate(v):
                                numberOverallBaseModels += 1
                                if val >= 1:
                                    numberBaseModelsFull += 1
                                
                                resultV1s.append(val)
                                resultV2s.append(res['tree performance'][i])
                                for r in res['treeImportances'][i]:
                                    for k in range(nrAnds):
                                        resultV3s[0].append(res['treeImportances'][i][k])
                                    for k in range(nrAnds, nrAnds+nrOrs):
                                        resultV3s[1].append(res['treeImportances'][i][k])
                                    for k in range(nrAnds+nrOrs, nrAnds+nrOrs+nrxor):
                                        resultV3s[2].append(res['treeImportances'][i][k])
                                    for k in range(nrAnds+nrOrs+nrxor, nrAnds+nrOrs+nrxor+nrEmpty):
                                        resultV3s[3].append(res['treeImportances'][i][k])
 
                                if res['baseline'][0] < 0.5:
                                    baseline = 1-res['baseline'][0]
                                else:
                                    baseline = res['baseline'][0]
                                resultV4s.append(baseline)

        print(str(np.round(np.mean(resultV1s, axis=0),4)) + ' +- ' + str(np.round(np.std(resultV1s, axis=0),4)))
        print(str(np.round(np.mean(resultV2s, axis=0),4)) + ' +- ' + str(np.round(np.std(resultV2s, axis=0),4)))

        resultV3sM = []
        resultV3sS = []
        for vi, vr in enumerate(resultV3s):
            resultV3sM.append(np.mean(resultV3s[vi]))
            resultV3sS.append(np.std(resultV3s[vi]))
        print(str(np.round(resultV3sM,4)) + ' +- ' + str(np.round(resultV3sS,4)))
        
        print(str(np.round(np.mean(resultV4s, axis=0),4)) + ' +- ' + str(np.round(np.std(resultV4s, axis=0),4)))
        print('------')


        lables = ['Avg. Model Acc.', 'Avg Tree Acc.', 'Avg. Baseline Acc.']
        counts = [np.round(np.mean(resultV1s, axis=0),4),np.round(np.mean(resultV2s, axis=0),4),np.round(np.mean(resultV4s, axis=0),4)]
        e =  [np.round(np.std(resultV1s, axis=0),4),np.round(np.std(resultV2s, axis=0),4),np.round(np.std(resultV4s, axis=0),4)]

        ax1.bar(lables, counts, yerr=e, linestyle='None', capsize=4)

        ax1.set_ylabel('Acc.')
        ax1.set_title('Model Performance ' +toplevel.upper() + ' ' +configName)
        


        lables= []
        for i in range(andStack):
            lables.append('AND')
        for i in range(orStack):
            lables.append('OR'+str(i))
        for i in range(xorStack):
            lables.append('XOR')
        for i in range(1):
            lables.append('Baseline')
            
        counts = np.round(resultV3sM,4)
        e =  np.round(resultV3sS,4)

        lables = ['AND','OR','XOR','Irrelevant']
        ax2.bar(lables, counts, yerr=e, linestyle='None', capsize=4)
        ax2.tick_params(labelrotation=90)
        ax2.set_ylabel('Imp.')
        ax2.set_title('Tree Model Importance ' + toplevel.upper() + ' ' + configName)


specificFolder = folderGeneral + 'performance/'
if not os.path.exists(specificFolder):
    os.makedirs(specificFolder)
fig.tight_layout()
fig2.tight_layout()

fig.savefig(specificFolder + 'generalPerformancesGates.png', dpi = 300, bbox_inches = 'tight')
fig2.savefig(specificFolder + 'generalTreesImportancesGates.png', dpi = 300, bbox_inches = 'tight')
plt.show() 
plt.close()


# In[8]:




# In[9]:

import sys
#TODO different settings beachten?
names = ['generalInformationBelowBaseline', 'neededInformationBelowBaseline'] 
saveNames = ['GIB', 'NIB']
"""
infoCombis = {'AND': [[[0,1], ['BinarySingleGate', '2inBinaryDoubleGateAND'], ['and']]], 
                'OR': [[[0,1], ['BinarySingleGate', '2inBinaryDoubleGateOR'], ['or']]], 
                'XOR': [[[0,1], ['BinarySingleGate', '2inBinaryDoubleGateXOR'], ['xor']]], 
                'Complementary': [[[1], ['BinarySingleGate', '2inBinaryDoubleGateAND'], ['and']],[[0], ['BinarySingleGate', '2inBinaryDoubleGateOR'], ['or']]], 
                'Redundant': [[[1], ['BinarySingleGate', '2inBinaryDoubleGateOR'], ['or']],[[0], ['BinarySingleGate', '2inBinaryDoubleGateAND'], ['and']]], 
                'AND-OR': [[[0,1], ['BinarySingleGate', '2inBinaryDoubleGateAND', '2inBinaryDoubleGateOR'], ['and', 'or']]],
                'AND-XOR':[[[0,1], ['BinarySingleGate', '2inBinaryDoubleGateAND', '2inBinaryDoubleGateXOR'], ['and', 'xor']]],
                'OR-XOR': [[[0,1], ['BinarySingleGate', '2inBinaryDoubleGateOR', '2inBinaryDoubleGateXOR'], ['or', 'xor']]],
                'AND-OR-XOR': [[[0,1], ['BinarySingleGate', '2inBinaryDoubleGateAND', '2inBinaryDoubleGateOR', '2inBinaryDoubleGateXOR', '2inQuaternary', '2inBinary', '3inBinary'], ['or', 'xor', 'and']]]
            }
"""


infoCombis = {'AND': [[[0,1], ['BinarySingleGate', '2inBinaryDoubleGateAND'], ['and']]], 
                'OR': [[[0,1], ['BinarySingleGate', '2inBinaryDoubleGateOR'], ['or']]], 
                'XOR': [[[0,1], ['BinarySingleGate', '2inBinaryDoubleGateXOR'], ['xor']]], 
                'Complementary': [[[1], ['BinarySingleGate', '2inBinaryDoubleGateAND'], ['and']],[[0], ['BinarySingleGate', '2inBinaryDoubleGateOR'], ['or']]], 
                'Redundant': [[[1], ['BinarySingleGate', '2inBinaryDoubleGateOR'], ['or']],[[0], ['BinarySingleGate', '2inBinaryDoubleGateAND'], ['and']]], 
                'AND-OR': [[[0,1], ['2inBinaryDoubleGateAND'], ['or']],[[0,1], ['2inBinaryDoubleGateOR'], ['and']]],
                'AND-XOR':[[[0,1], ['2inBinaryDoubleGateAND'], ['xor']],[[0,1], ['2inBinaryDoubleGateXOR'], ['and']]],
                'OR-XOR': [[[0,1], ['2inBinaryDoubleGateOR'], ['xor']],[[0,1], ['2inBinaryDoubleGateXOR'], ['or']]],
                'AND-OR-XOR': [[[0,1], ['2inQuaternary', '2inBinary', '3inBinary'], ['or', 'xor', 'and']]]
            }
infoCombisS = {'AND': [[[0,1], ['BinarySingleGate', '2inBinaryDoubleGateAND'], ['and']]], 
                'OR': [[[0,1], ['BinarySingleGate', '2inBinaryDoubleGateOR'], ['or']]], 
                'XOR': [[[0,1], ['BinarySingleGate', '2inBinaryDoubleGateXOR'], ['xor']]], 
                'AND-OR': [[[0,1], ['2inBinaryDoubleGateAND'], ['or']],[[0,1], ['2inBinaryDoubleGateOR'], ['and']]],
                'AND-XOR':[[[0,1], ['2inBinaryDoubleGateAND'], ['xor']],[[0,1], ['2inBinaryDoubleGateXOR'], ['and']]],
                'OR-XOR': [[[0,1], ['2inBinaryDoubleGateOR'], ['xor']],[[0,1], ['2inBinaryDoubleGateXOR'], ['or']]],
                'AND-OR-XOR': [[[0,1], ['2inQuaternary', '2inBinary', '3inBinary'], ['or', 'xor', 'and']]]
            }


def checkKIC(kic, c, configName, toplevel):
    for option in infoCombis[kic]:
        if c in option[0] and configName in option[1] and toplevel in option[2]:
            return True
    return False

def getBestMode(combis, k, metric='neededInformationBelowBaseline'):
    modeVs = []
    for m in range(np.max(modes)+1):
        modeVs.append([])
        for kic  in infoCombisS.keys():
            for v in combis[kic][metric][k][m]:
                modeVs[m].append(v)

    for mi, mv in enumerate(modeVs):
        if len(mv) == 0:
            modeVs[mi] = sys.maxsize
        else:
            modeVs[mi] = np.nanmean(mv)
    return np.argmin(modeVs)#, np.min(m)

def getBestModeOld(innerCombis):
    m = []
    for mi in innerCombis:
        if len(mi) == 0:
            m.append(sys.maxsize)
        else:
            m.append(np.nanmean(mi))
    return np.argmin(m)#, np.min(m)

nibComis = dict()
for kic in infoCombis.keys():
    nibComis[kic] = dict()
    for n in names:
        nibComis[kic][n] = dict()
        for k in ks:
            nibComis[kic][n][k] = []
            for mode in range(np.max(modes)+1): 
                nibComis[kic][n][k].append([])



for cNr, config in enumerate(allConfigs):
    configName = config.configName
    hyperss = config.hyperss
    hypNames= config.hypNames
    test_sizes = config.test_sizes
    topLevelss = config.topLevelss
    dataset = config.dataset
    symbols = config.symbols
    nrEmpty = config.nrEmpty
    andStack = config.andStack
    orStack = config.orStack
    xorStack = config.xorStack
    nrAnds = config.nrAnds
    nrOrs = config.nrOrs
    nrxor = config.nrxor
    trueIndexes = config.trueIndexes
    orOffSet = config.orOffSet
    xorOffSet = config.xorOffSet
    redaundantIndexes = config.redaundantIndexes
    batch_size = config.batch_size
    for nami, name in enumerate(names):
        for m in modes:
            fig, axs = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True,
                                            figsize=(18, 6), layout="constrained")

            for c in [0,1]:
                for j, toplevels in enumerate(topLevelss[1:]):

                    ax = axs[c,j]

                    
                    width = 0
                    n_bars = 2
                    standardWidth = 0.8
                    bar_width = 0.8 / n_bars
                    barInd = 0
                    legend = []
                    rects = []
                    for accTHold in [0]:
                        for splitInd in splitInds:

                            

                            x_offset = (barInd - n_bars / 2) * bar_width + bar_width / 2
                            barInd+= 1
                            resultVFs = []
                            resultVSs = []
                            width += standardWidth
                            


                            for k in ks:
                                resultVs = []

                                if k in ['performance', 'tree performance', 'treeImportances', 'baseline']:
                                    continue

                                if k == 'LRP-transformer_attribution cls':
                                    k = 'LRP-transformer_attribution'
                                    takeCls = True
                                else:
                                    takeCls = False

                                for hypers in hyperss:
                                    if takeCls and not hypers[-1]:
                                        continue
                                    elif not takeCls and hypers[-1]:
                                        continue
                                    for toplevel in toplevels:
                                            test_size = test_sizes[splitInd]
                                            #for test_size in test_sizes:
                                            modelType = hypers[0]
                                            epochs = hypers[1]
                                            dmodel = hypers[2]
                                            dfff = hypers[3]
                                            doSkip = hypers[4]
                                            doBn = hypers[5]
                                            header = hypers[6]
                                            numOfLayers = hypers[7]
                                            dropout = hypers[8]
                                            att_dropout = hypers[9]
                                            if modelType == 'Transformer':
                                                doClsTocken = hypers[10]
                                            else:
                                                doClsTocken = False

                                            dsName = str(dataset) +  ',l:' + str(toplevel) +',s:' +  str(symbols)+',e:' +  str(nrEmpty)+',a:' +  str(andStack)+',o:' +  str(orStack)+',x:' +  str(xorStack)+',na:' +  str(nrAnds)+',no:' +  str(nrOrs)+',nx:' +  str(nrxor)+',i:' +  arrayToString(trueIndexes)+',t:' +  str(test_size)+',oo:' +  str(orOffSet)+',xo:' +  str(xorOffSet) +',r:' + arrayToString(redaundantIndexes)


                                            for rFolder in filteredResultsFolder:


                                                saveName = pt.getWeightName(dsName, dataset, batch_size, epochs, numOfLayers, header, dmodel, dfff, dropout, att_dropout, doSkip, doBn, doClsTocken, learning = False, results = True, resultsPath=rFolder)

                                                if os.path.isfile(saveName + '.pkl'):
                                                    if saveName in allReses[rFolder].keys():
                                                        res = allReses[rFolder][saveName]
                                                    else:
                                                        try:
                                                            results = helper.load_obj(saveName)
                                                        except:
                                                            continue

                                                        res = dict()
                                                        for index, vv in np.ndenumerate(results):
                                                            res = vv

                                                        allReses[rFolder][saveName] = res
                                                else:
                                                    print('NOT FOUND: ' + saveName)
                                                    continue
                                                

                                                if k not in res.keys():
                                                    continue
                                                v  = res[k][m][c][name]
                                                for i, val in enumerate(v):
                                                    if res[k]['performance'][i][i%5] < (accTreshold):
                                                            continue
                                                    resultVs.append(np.nanmean(val) * 100)
                                                

                                                    for kic in infoCombis.keys():
                                                        if checkKIC(kic, c, configName, toplevel):
                                                            if takeCls:
                                                                nibComis[kic][name]['LRP-transformer_attribution cls'][m].append(np.nanmean(val) * 100)
                                                            else:
                                                                nibComis[kic][name][k][m].append(np.nanmean(val) * 100)


                                                print(str(np.nanmean(resultVs)) + ' +- ' + str(np.nanstd(resultVs)))

                                resultVFs.append(np.nanmean(resultVs))


                                resultVSs.append(np.nanstd(resultVs))

                            lables= kNames
                                
                            counts = resultVFs
                            
                            counts = np.where(np.isnan(counts),0,counts)
                            e = resultVSs
                            e = np.where(np.isnan(e),0,e)

                            legend.append(testFacNames[splitInd] + '; ' + accTNames[accTHold])
                            ind = np.arange(len(counts))
                            rect = ax.bar(ind+x_offset , counts, bar_width, yerr=e, linestyle='None', capsize=3)
                            rects.append(rect)
                            ax.set_ylabel('Percent')
                            ax.set_xticks(ind)
                            ax.set_xticklabels(lables)
                            ax.set_title(saveNames[nami]+' Class ' + str(c) + '\n'+ topLevelNamesSmall[j].upper())
                            ax.tick_params(labelrotation=90)

                            ax.set_ylim(bottom=0, top=100)


            fig.legend(rects, labels=legend, 
                loc="upper right", bbox_to_anchor=(1.088, 0.85)) 
            fig.tight_layout()
            fig.suptitle(config.configName,fontsize=14)
            fig.subplots_adjust(top=0.84)
            specificFolder = folderGeneral + 'NIBnGIB/'
            if not os.path.exists(specificFolder):
                os.makedirs(specificFolder)
            fig.savefig(specificFolder + saveNames[nami] +' ALL ' + configName+ ' + mode' + str(m) + '.png', dpi = 300, bbox_inches = 'tight')

            plt.show()
            plt.close()



splitNames = ['Split Test Set; Acc=100', 'Not Split Test; Acc>=0']
for ni, n in enumerate(names):
    fig, axs = plt.subplots(nrows=1, ncols=len(infoCombis.keys()), sharex=False, sharey=True,
                            figsize=(25, 4), layout="constrained")

    specificFolder = folderGeneral + 'NIBnGIBRanking/'
    if not os.path.exists(specificFolder):
        os.makedirs(specificFolder)


    f = open(specificFolder+ n + "-rankingVs.txt", "a")
    f.write(n)
    f.write("\n")
    f.write("full 	" + "	".join(str(x) for x in kNames))
    f.write("\n")
    f.close()
    f = open(specificFolder+ n + "-ranking.txt", "a")
    f.write(n)
    f.write("\n")
    f.write("full 	" + "	".join(str(x) for x in kNames))
    f.close()
    
    for ki, kic in enumerate(infoCombis.keys()):
    
        #ax = axs[si, ki]
        ax = axs[ki]
        resultVs = []
        resultSs = []
        
        lables = []
        for ksi, k in enumerate(ks):
            data = nibComis[kic][n][k] 
            m = getBestMode(nibComis, k)
            lables.append(kNames[ksi])
            resultVs.append(np.nanmean(data[m]))
            resultSs.append(np.nanstd(data[m]))

        resultVs = np.array(resultVs)
        resultSs = np.array(resultSs)

        ranks = rankdata(resultVs, method='min')
        f = open(specificFolder+ n + "-rankingVs.txt", "a")
        f.write(kic + "	")
        f.write("	".join(str(x) for x in resultVs))
        f.write("\n")

        f.close()

        f = open(specificFolder+ n + "-ranking.txt", "a")
        f.write(kic + "	")
        f.write("	".join(str(x) for x in ranks))
        f.write("\n")
        f.close()
        
        sortI = np.argsort(resultVs)
        counts = resultVs[sortI]
        e = resultSs[sortI]
        lables = np.array(lables)[sortI]



        ind = np.arange(len(counts))
        rect = ax.bar(ind , counts, yerr=e, linestyle='None', capsize=3)
        if ki == 0:
            ax.set_ylabel('Percent')
        ax.set_xticks(ind)
        ax.set_xticklabels(lables)
        ax.set_title(kic)#)
        ax.tick_params(labelrotation=90)

        ax.set_ylim(bottom=0, top=100)
    
    f = open(specificFolder+ n + "-rankingVs.txt", "a")
    f.write("\n")
    f.close()

    f = open(specificFolder+ n + "-ranking.txt", "a")
    f.write("\n")
    f.close()

    



    fig.suptitle(saveNames[ni] + '-Balanced',fontsize=14)
    
    fig.savefig(specificFolder + n +'-ranking.png', dpi = 300, bbox_inches = 'tight')
    plt.show()
    plt.close()




# In[ ]:

nibComisF = dict()
for kic in infoCombisS.keys():
    nibComisF[kic] = dict()
    for n in names:
        nibComisF[kic][n] = dict()
        for k in ks:

            nibComisF[kic][n][k] = []
            for mode in range(np.max(modes)+1): 
                nibComisF[kic][n][k].append([])



for cNr, config in enumerate(allConfigs):
    configName = config.configName
    hyperss = config.hyperss
    hypNames= config.hypNames
    test_sizes = config.test_sizes
    topLevelss = config.topLevelss
    dataset = config.dataset
    symbols = config.symbols
    nrEmpty = config.nrEmpty
    andStack = config.andStack
    orStack = config.orStack
    xorStack = config.xorStack
    nrAnds = config.nrAnds
    nrOrs = config.nrOrs
    nrxor = config.nrxor
    trueIndexes = config.trueIndexes
    orOffSet = config.orOffSet
    xorOffSet = config.xorOffSet
    redaundantIndexes = config.redaundantIndexes
    batch_size = config.batch_size
    for nami, name in enumerate(names):
        for m in modes:
            fig, axs = plt.subplots(ncols=3, sharex=True, sharey=True,
                                            figsize=(18, 6), layout="constrained")

            for j, toplevels in enumerate(topLevelss[1:]):

                ax = axs[j]

                
                width = 0
                n_bars = 2
                standardWidth = 0.8
                bar_width = 0.8 / n_bars
                barInd = 0
                legend = []
                rects = []
                for accTHold in [0]:
                    for splitInd in splitInds:

                        
                        

                        x_offset = (barInd - n_bars / 2) * bar_width + bar_width / 2
                        barInd+= 1
                        resultVFs = []
                        resultVSs = []
                        width += standardWidth
                        


                        for k in ks:
                            resultVs = []

                            if k in ['performance', 'tree performance', 'treeImportances', 'baseline']:
                                continue

                            if k == 'LRP-transformer_attribution cls':
                                k = 'LRP-transformer_attribution'
                                takeCls = True
                            else:
                                takeCls = False

                            for hypers in hyperss:
                                if takeCls and not hypers[-1]:
                                    continue
                                elif not takeCls and hypers[-1]:
                                    continue
                                for toplevel in toplevels:
                                        test_size = test_sizes[splitInd]
                                        #for test_size in test_sizes:
                                        modelType = hypers[0]
                                        epochs = hypers[1]
                                        dmodel = hypers[2]
                                        dfff = hypers[3]
                                        doSkip = hypers[4]
                                        doBn = hypers[5]
                                        header = hypers[6]
                                        numOfLayers = hypers[7]
                                        dropout = hypers[8]
                                        att_dropout = hypers[9]
                                        if modelType == 'Transformer':
                                            doClsTocken = hypers[10]
                                        else:
                                            doClsTocken = False

                                        dsName = str(dataset) +  ',l:' + str(toplevel) +',s:' +  str(symbols)+',e:' +  str(nrEmpty)+',a:' +  str(andStack)+',o:' +  str(orStack)+',x:' +  str(xorStack)+',na:' +  str(nrAnds)+',no:' +  str(nrOrs)+',nx:' +  str(nrxor)+',i:' +  arrayToString(trueIndexes)+',t:' +  str(test_size)+',oo:' +  str(orOffSet)+',xo:' +  str(xorOffSet) +',r:' + arrayToString(redaundantIndexes)


                                        for rFolder in filteredResultsFolder:


                                            saveName = pt.getWeightName(dsName, dataset, batch_size, epochs, numOfLayers, header, dmodel, dfff, dropout, att_dropout, doSkip, doBn, doClsTocken, learning = False, results = True, resultsPath=rFolder)

                                            if os.path.isfile(saveName + '.pkl'):
                                                if saveName in allReses[rFolder].keys():
                                                    res = allReses[rFolder][saveName]
                                                else:
                                                    try:
                                                        results = helper.load_obj(saveName)
                                                    except:
                                                        continue

                                                    res = dict()
                                                    for index, vv in np.ndenumerate(results):
                                                        res = vv

                                                    allReses[rFolder][saveName] = res
                                            else:
                                                print('NOT FOUND: ' + saveName)
                                                continue
                                            

                                            if k not in res.keys():
                                                continue
                                            v  = res[k][m][name]
                                            for i, val in enumerate(v):
                                                if res[k]['performance'][i][i%5] < (accTreshold):
                                                        continue
                                                resultVs.append(np.nanmean(val) * 100)
                                            

                                                for kic in infoCombisS.keys():
                                                    if checkKIC(kic, 1, configName, toplevel):
                                                        if takeCls:
                                                            nibComisF[kic][name]['LRP-transformer_attribution cls'][m].append(np.nanmean(val) * 100)
                                                        else:
                                                            nibComisF[kic][name][k][m].append(np.nanmean(val) * 100)


                                            print(str(np.nanmean(resultVs)) + ' +- ' + str(np.nanstd(resultVs)))

                            resultVFs.append(np.nanmean(resultVs))


                            resultVSs.append(np.nanstd(resultVs))

                        lables= kNames
                            
                        counts = resultVFs
                        
                        counts = np.where(np.isnan(counts),0,counts)
                        e = resultVSs
                        e = np.where(np.isnan(e),0,e)

                        legend.append(testFacNames[splitInd] + '; ' + accTNames[accTHold])
                        ind = np.arange(len(counts))
                        rect = ax.bar(ind+x_offset , counts, bar_width, yerr=e, linestyle='None', capsize=3)
                        rects.append(rect)
                        ax.set_ylabel('Percent')
                        ax.set_xticks(ind)
                        ax.set_xticklabels(lables)
                        ax.set_title(saveNames[nami]+ '\n'+ topLevelNamesSmall[j].upper())
                        ax.tick_params(labelrotation=90)
  
                        ax.set_ylim(bottom=0, top=100)




            fig.legend(rects, labels=legend, 
                loc="upper right", bbox_to_anchor=(1.088, 0.85)) 
            fig.tight_layout()
            fig.suptitle(config.configName,fontsize=14)
            fig.subplots_adjust(top=0.84)
            specificFolder = folderGeneral + 'NIBnGIBFull/'
            if not os.path.exists(specificFolder):
                os.makedirs(specificFolder)
            fig.savefig(specificFolder + saveNames[nami] +' ALL ' + configName+ ' + mode' + str(m) + '.png', dpi = 300, bbox_inches = 'tight')

            plt.show()
            plt.close()



splitNames = ['Split Test Set; Acc=100', 'Not Split Test; Acc>=0']
for ni,n in enumerate(names):
    fig, axs = plt.subplots(nrows=1, ncols=len(infoCombisS.keys()), sharex=False, sharey=True,
                            figsize=(25, 4), layout="constrained")

    specificFolder = folderGeneral + 'NIBnGIBFullRanking/'
    if not os.path.exists(specificFolder):
        os.makedirs(specificFolder)

    
       
    f = open(specificFolder+ n + "-rankingVs.txt", "a")
    f.write(n)
    f.write("\n")
    f.write("full 	" + "	".join(str(x) for x in kNames))
    f.write("\n")
    f.close()
    f = open(specificFolder+ n + "-ranking.txt", "a")
    f.write(n)
    f.write("\n")
    f.write("full 	" + "	".join(str(x) for x in kNames))
    f.close()
    for ki, kic in enumerate(infoCombisS.keys()):
        ax = axs[ki]
        resultVs = []
        resultSs = []
        
        lables = []
        for ksi, k in enumerate(ks):
            data = nibComisF[kic][n][k] 
            m = getBestMode(nibComisF, k)
            lables.append(kNames[ksi])# + ' ' + str(m))
            resultVs.append(np.nanmean(data[m]))
            resultSs.append(np.nanstd(data[m]))

        resultVs = np.array(resultVs)
        resultSs = np.array(resultSs)

        ranks = rankdata(resultVs, method='min')
        f = open(specificFolder+ n + "-rankingVs.txt", "a")
        f.write(kic + "	")
        f.write("	".join(str(x) for x in resultVs))
        f.write("\n")
        f.close()

        f = open(specificFolder+ n + "-ranking.txt", "a")
        f.write(kic + "	")
        f.write("	".join(str(x) for x in ranks))
        f.write("\n")
        f.close()
        sortI = np.argsort(resultVs)
        counts = resultVs[sortI]
        e = resultSs[sortI]
        lables = np.array(lables)[sortI]



        ind = np.arange(len(counts))
        rect = ax.bar(ind , counts, yerr=e, linestyle='None', capsize=3)
        if ki == 0:
            ax.set_ylabel('Percent')
        ax.set_xticks(ind)
        ax.set_xticklabels(lables)
        ax.set_title(kic)#)
        ax.tick_params(labelrotation=90)

        ax.set_ylim(bottom=0, top=100)
    
    f = open(specificFolder+ n + "-rankingVs.txt", "a")
    f.write("\n")
    f.close()

    f = open(specificFolder+ n + "-ranking.txt", "a")
    f.write("\n")
    f.close()



    fig.suptitle(saveNames[ni] + '-Full',fontsize=14)
    
    fig.savefig(specificFolder + n +'-ranking.png', dpi = 300, bbox_inches = 'tight')
    plt.show()
    plt.close()



# In[61]:


#TODO different settings beachten?
names = ['acc']
saveNames = ['GCR Acc.']
gcrOptions = ['rMA', 'rMS',  '1DSum', '1DSum+'] # 'max', 'max+', 'average', 'average+', 'median', 'median+',
optionNames = ['GCR Fidelity FCAM','GCR Fidelity FCAM S','GCR Fidelity GTM S','GCR Fidelity GTM']


gcrComis = dict()
for kic in infoCombisS.keys():
    gcrComis[kic] = dict()
    for n in gcrOptions:
        gcrComis[kic][n] = dict()
        for k in ks:
            gcrComis[kic][n][k] = []
            for mode in range(np.max(modes)+1): 
                gcrComis[kic][n][k].append([])

gcrComisNum = dict()
for kic in infoCombis.keys():
    gcrComisNum[kic] = dict()
    for n in gcrOptions:
        gcrComisNum[kic][n] = dict()
        for k in ks:
            gcrComisNum[kic][n][k] = dict()
            for classes in [0,1]:
                gcrComisNum[kic][n][k][classes] = dict()
                for input in [-1, -0.3333, 0.3333, 1]:
                    gcrComisNum[kic][n][k][classes][input] = []
                    for mode in range(np.max(modes)+1): 
                        gcrComisNum[kic][n][k][classes][input].append([])

gcrComisScen = dict()
for kico in allConfigs:
    kic = kico.configName
    gcrComisScen[kic] = dict()
    for n in gcrOptions:
        gcrComisScen[kic][n] = dict()
        for k in ks:
            gcrComisScen[kic][n][k] = dict()
            for classes in [0,1]:
                gcrComisScen[kic][n][k][classes] = dict()
                for input in [-1, -0.3333, 0.3333, 1]:
                    gcrComisScen[kic][n][k][classes][input] = []
                    for mode in range(np.max(modes)+1): 
                        gcrComisScen[kic][n][k][classes][input].append([])


normalizers = GCRPlus.getAllReductionNames()


for cNr, config in enumerate(allConfigs):
    configName = config.configName
    hyperss = config.hyperss
    hypNames= config.hypNames
    test_sizes = config.test_sizes
    topLevelss = config.topLevelss
    dataset = config.dataset
    symbols = config.symbols
    nrEmpty = config.nrEmpty
    andStack = config.andStack
    orStack = config.orStack
    xorStack = config.xorStack
    nrAnds = config.nrAnds
    nrOrs = config.nrOrs
    nrxor = config.nrxor
    trueIndexes = config.trueIndexes
    orOffSet = config.orOffSet
    xorOffSet = config.xorOffSet
    redaundantIndexes = config.redaundantIndexes
    batch_size = config.batch_size

    valuesA = helper.getMapValues(symbols)
    for nami, name in enumerate(names):
        for m in modes:
            for normName in ['OverMax']:
                fig, axs = plt.subplots(nrows=len(gcrOptions), ncols=len(topLevelNamesSmall), sharex=True, sharey=True,
                                                figsize=(20, 8), layout="constrained")

                for ci, c in enumerate(gcrOptions):
                    for j, toplevels in enumerate(topLevelss[1:]):

                        ax = axs[ci,j]


                        width = 0
                        n_bars = 2
                        standardWidth = 0.8
                        bar_width = 0.8 / n_bars
                        barInd = 0
                        legend = []
                        rects = []
                        for accTHold in [0]:
                            for splitInd in splitInds:

                                x_offset = (barInd - n_bars / 2) * bar_width + bar_width / 2
                                barInd+= 1
                                resultVFs = []
                                resultVSs = []
                                width += standardWidth

                                for k in ks:
                                    resultVs = []


                                    if k == 'LRP-transformer_attribution cls':
                                        k = 'LRP-transformer_attribution'
                                        takeCls = True
                                    else:
                                        takeCls = False

                                    for hypers in hyperss:
                                        if takeCls and not hypers[-1]:
                                            continue
                                        elif not takeCls and hypers[-1]:
                                            continue
                                        for toplevel in toplevels:
                                                test_size = test_sizes[splitInd]
                                                modelType = hypers[0]
                                                epochs = hypers[1]
                                                dmodel = hypers[2]
                                                dfff = hypers[3]
                                                doSkip = hypers[4]
                                                doBn = hypers[5]
                                                header = hypers[6]
                                                numOfLayers = hypers[7]
                                                dropout = hypers[8]
                                                att_dropout = hypers[9]
                                                if modelType == 'Transformer':
                                                    doClsTocken = hypers[10]
                                                else:
                                                    doClsTocken = False

                                                dsName = str(dataset) +  ',l:' + str(toplevel) +',s:' +  str(symbols)+',e:' +  str(nrEmpty)+',a:' +  str(andStack)+',o:' +  str(orStack)+',x:' +  str(xorStack)+',na:' +  str(nrAnds)+',no:' +  str(nrOrs)+',nx:' +  str(nrxor)+',i:' +  arrayToString(trueIndexes)+',t:' +  str(test_size)+',oo:' +  str(orOffSet)+',xo:' +  str(xorOffSet) +',r:' + arrayToString(redaundantIndexes)


                                                for rFolder in filteredResultsFolder:


                                                    saveName = pt.getWeightName(dsName, dataset, batch_size, epochs, numOfLayers, header, dmodel, dfff, dropout, att_dropout, doSkip, doBn, doClsTocken, learning = False, results = True, resultsPath=rFolder)

                                                    if os.path.isfile(saveName + '.pkl'):
                                                        if saveName in allReses[rFolder].keys():
                                                            res = allReses[rFolder][saveName]
                                                        else:
                                                            try:
                                                                results = helper.load_obj(saveName)
                                                            except:
                                                                continue

                                                            res = dict()
                                                            for index, vv in np.ndenumerate(results):
                                                                res = vv

                                                            allReses[rFolder][saveName] = res
                                                    else:
                                                        continue

                                                    if k not in res['gcr'].keys():
                                                        continue
                                                    v  = res['gcr'][k][m][c][normName][name]

                                                    #print(v[0])
                                                    for i, val in enumerate(v):
                                                        
                                                        if res[k]['performance'][i][i%5] < (accTreshold ):
                                                                continue

                                                        resultVs.append(np.nanmean(val) * 100) 

                                                        
                                                        for kic in infoCombisS.keys():
                                                            if checkKIC(kic, 1, configName, toplevel):
                                                                if takeCls:
                                                                    gcrComis[kic][c]['LRP-transformer_attribution cls'][m].append(np.nanmean(val) * 100)
                                                                else:
                                                                    gcrComis[kic][c][k][m].append(np.nanmean(val) * 100)

                                                    v  = res['gcr'][k][m][c][normName]['gcr']
                                                    for i, val in enumerate(v):
                                                        
                                                        if res[k]['performance'][i][i%5] < (accTreshold ):
                                                                continue

                                                        
                                                        for kic in infoCombis.keys():
                                                            
                                                            for gcrC, gcrV in enumerate(val):
                                                                if checkKIC(kic, gcrC, configName, toplevel):
                                                                    for gcrIn, gcrInV in enumerate(valuesA):
                                                                        if takeCls:
                                                                            gcrComisNum[kic][c]['LRP-transformer_attribution cls'][gcrC][gcrInV][m].append(np.nanmean(val[gcrC][gcrIn]))
                                                                        else:
                                                                            gcrComisNum[kic][c][k][gcrC][gcrInV][m].append(np.nanmean(val[gcrC][gcrIn]))

                                                        for gcrC, gcrV in enumerate(val):
                                                            for gcrIn, gcrInV in enumerate(valuesA):
                                                                if takeCls:
                                                                    gcrComisScen[configName][c]['LRP-transformer_attribution cls'][gcrC][gcrInV][m].append(np.nanmean(val[gcrC][gcrIn]))
                                                                else:
                                                                    gcrComisScen[configName][c][k][gcrC][gcrInV][m].append(np.nanmean(val[gcrC][gcrIn]))

                                    resultVFs.append(np.nanmean(resultVs))
                                    resultVSs.append(np.nanstd(resultVs))

                                lables= kNames
                                    
                                counts = resultVFs
                                counts = np.where(np.isnan(counts),0,counts)
                                e = resultVSs
                                e = np.where(np.isnan(e),0,e)


                                legend.append(testFacNames[splitInd] + '; ' + accTNames[accTHold])
                                ind = np.arange(len(counts))
                                rect = ax.bar(ind+x_offset , counts, bar_width, yerr=e, linestyle='None', capsize=3)
                                rects.append(rect)
                                ax.set_ylabel('Percent')
                                ax.set_xticks(ind)
                                ax.set_xticklabels(lables)
                                ax.set_title(saveNames[nami]+' ' + str(c) + '\n'+ topLevelNamesSmall[j].upper())
                                ax.tick_params(labelrotation=90)
                                ax.set_ylim(bottom=0, top=100)

                fig.legend(rects, labels=legend, 
                    loc="upper right", bbox_to_anchor=(1.088, 0.85)) 
                fig.tight_layout()
                fig.suptitle(config.configName,fontsize=14)
                fig.subplots_adjust(top=0.84)
                specificFolder = folderGeneral + 'GCRAcc/'
                if not os.path.exists(specificFolder):
                    os.makedirs(specificFolder)
                fig.savefig(specificFolder + saveNames[nami] +' ALL ' + configName+ ' mode'+str(m)+' ' +normName + '.png', dpi = 300, bbox_inches = 'tight')

                plt.show()   
                plt.close()

splitNames = ['Split Test Set; Acc=100', 'Not Split Test; Acc>=0']
for ni,n in enumerate(gcrOptions):
    fig, axs = plt.subplots(nrows=1, ncols=len(infoCombisS.keys()), sharex=False, sharey=True,
                            figsize=(25, 4), layout="constrained")    
                            
    specificFolder = folderGeneral + 'GCRAccRanking/'
    if not os.path.exists(specificFolder):
        os.makedirs(specificFolder)

        
    f = open(specificFolder+ n + "-rankingVs.txt", "a")
    f.write(n)
    f.write("\n")
    f.write("full 	" + "	".join(str(x) for x in kNames))
    f.write("\n")
    f.close()
    f = open(specificFolder+ n + "-ranking.txt", "a")
    f.write(n)
    f.write("\n")
    f.write("full 	" + "	".join(str(x) for x in kNames))
    f.close()



    
    for ki, kic in enumerate(infoCombisS.keys()):
        ax = axs[ki]
        resultVs = []
        resultSs = []
        lables = []
        for ksi, k in enumerate(ks):
            data = gcrComis[kic][n][k] 
            m = getBestMode(nibComis, k)

            m2 = getBestMode(gcrComis, k, metric=n)

            f = open(specificFolder+ n + "-rankingAlternative.txt", "a")
            f.write(k)
            f.write("\n")
            f.write(str(m2))
            
            f.close()
            lables.append(kNames[ksi])# + ' ' + str(m))
            resultVs.append(np.nanmean(data[m]))
            resultSs.append(np.nanstd(data[m]))

        resultVs = np.array(resultVs)
        resultSs = np.array(resultSs)

        ranks = rankdata(resultVs, method='max')
        ranks = (np.array(ranks) - (len(ranks)+1)) * -1
        f = open(specificFolder+ n + "-rankingVs.txt", "a")
        f.write(kic + "	")
        f.write("	".join(str(x) for x in resultVs))
        f.write("\n")
        f.close()

        f = open(specificFolder+ n + "-ranking.txt", "a")
        f.write(kic + "	")
        f.write("	".join(str(x) for x in ranks))
        f.write("\n")
        f.close()

        sortI = np.argsort(resultVs)
        counts = resultVs[sortI]
        e = resultSs[sortI]
        lables = np.array(lables)[sortI]

        ind = np.arange(len(counts))
        rect = ax.bar(ind , counts, yerr=e, linestyle='None', capsize=3)
        if ki == 0:
            ax.set_ylabel('Percent')
        ax.set_xticks(ind)
        ax.set_xticklabels(lables)
        ax.set_title(kic)
        ax.tick_params(labelrotation=90)

        ax.set_ylim(bottom=0, top=100)
    f = open(specificFolder+ n + "-rankingVs.txt", "a")
    f.write("\n")
    f.close()

    f = open(specificFolder+ n + "-ranking.txt", "a")
    f.write("\n")
    f.close()
    fig.suptitle(optionNames[ni],fontsize=14)

    fig.savefig(specificFolder + n +'-ranking.png', dpi = 300, bbox_inches = 'tight')

    plt.show()
    plt.close()




for n in gcrOptions:

    specificFolder = folderGeneral + 'GCRValues/'
    if not os.path.exists(specificFolder):
        os.makedirs(specificFolder)

        


    
    for ki, kic in enumerate(infoCombis.keys()):
        f = open(specificFolder+ n + "-valuesKICs.txt", "a")
        f.write(n)
        f.write("\n")
        f.write("full -" + kic +  "	" + "	".join(str(x) for x in kNames))
        f.write("\n")
        f.close()

        for classes in [0,1]:
            for vi, v in enumerate([-1,-0.3333,0.3333,1]):

                resultVs = []
                resultSs = []
                lables = []
                for ksi, k in enumerate(ks):
                    data = gcrComisNum[kic][n][k][classes][v]
                    m = getBestMode(nibComis, k)
                    lables.append(kNames[ksi] + ' ' + str(m))
                    resultVs.append(np.nanmean(data[m]))
                    resultSs.append(np.nanstd(data[m]))

                resultVs = np.array(resultVs)
                resultSs = np.array(resultSs)

                f = open(specificFolder+ n + "-valuesKICs.txt", "a")
                f.write(str(classes) + "#" + str(v) + "	")
                f.write("	".join(str(x) for x in resultVs))
                f.write("\n")
                f.close()


        f = open(specificFolder+ n + "-valuesKICs.txt", "a")
        f.write("\n")
        f.close()


for n in gcrOptions:

    specificFolder = folderGeneral + 'GCRValues/'
    if not os.path.exists(specificFolder):
        os.makedirs(specificFolder)

        


    
    for kico in allConfigs:
        kic = kico.configName
        f = open(specificFolder+ n + "-valuesDatasets.txt", "a")
        f.write(n)
        f.write("\n")
        f.write("full -" + kic +  "	" + "	".join(str(x) for x in kNames))
        f.write("\n")
        f.close()

        for classes in [0,1]:
            for vi, v in enumerate([-1,-0.3333,0.3333,1]):

                resultVs = []
                resultSs = []
                lables = []
                for ksi, k in enumerate(ks):
                    data = gcrComisScen[kic][n][k][classes][v]
                    m = getBestMode(nibComis, k)
                    lables.append(kNames[ksi] + ' ' + str(m))
                    resultVs.append(np.nanmean(data[m]))
                    resultSs.append(np.nanstd(data[m]))

                resultVs = np.array(resultVs)
                resultSs = np.array(resultSs)

                f = open(specificFolder+ n + "-valuesDatasets.txt", "a")
                f.write(str(classes) + "#" + str(v) + "	")
                f.write("	".join(str(x) for x in resultVs))
                f.write("\n")
                f.close()


        f = open(specificFolder+ n + "-valuesDatasets.txt", "a")
        f.write("\n")
        f.close()


# In[ ]:





# In[24]:
names = ['acc']
saveNames = ['GCR Acc. Thresholds']
gcrOptions = ['rMA', 'rMS', '1DSum', '1DSum+'] #'max', 'max+', 'average', 'average+', 'median', 'median+', 
optionNames = ['GCR Fidelity FCAM','GCR Fidelity FCAM S','GCR Fidelity GTM S','GCR Fidelity GTM']
                                        

tgcrComis = dict()
for kic in infoCombisS.keys():
    tgcrComis[kic] = dict()
    for n in gcrOptions:
        tgcrComis[kic][n] = dict()
        for k in ks:

            tgcrComis[kic][n][k] = []
            for mode in range(np.max(modes)+1): 
                tgcrComis[kic][n][k].append([])


normalizers = GCRPlus.getAllReductionNames()
            

for cNr, config in enumerate(allConfigs):
    configName = config.configName
    hyperss = config.hyperss
    hypNames= config.hypNames
    test_sizes = config.test_sizes
    topLevelss = config.topLevelss
    dataset = config.dataset
    symbols = config.symbols
    nrEmpty = config.nrEmpty
    andStack = config.andStack
    orStack = config.orStack
    xorStack = config.xorStack
    nrAnds = config.nrAnds
    nrOrs = config.nrOrs
    nrxor = config.nrxor
    trueIndexes = config.trueIndexes
    orOffSet = config.orOffSet
    xorOffSet = config.xorOffSet
    redaundantIndexes = config.redaundantIndexes
    batch_size = config.batch_size
    for nami, name in enumerate(names):
        for m in modes:
            for normName in ['OverMax']:
                fig, axs = plt.subplots(nrows=len(gcrOptions), ncols=len(topLevelNamesSmall), sharex=True, sharey=True,
                                                figsize=(20, 8), layout="constrained")

                for ci, c in enumerate(gcrOptions):
                    for j, toplevels in enumerate(topLevelss[1:]):

                        ax = axs[ci,j]


                        width = 0
                        n_bars = 8
                        standardWidth = 0.9
                        bar_width = 0.9 / n_bars
                        barInd = 0
                        legend = []
                        rects = []
                        
                        for accTHold in [0]:
                            for splitInd in splitInds:
                                for t in  ['tbaseline', 't1.0','t0.8','t0.5']:

                                    x_offset = (barInd - n_bars / 2) * bar_width + bar_width / 2
                                    barInd+= 1
                                    resultVFs = []
                                    resultVSs = []
                                    width += standardWidth

                                    for k in ks:
                                        resultVs = []


                                        if k == 'LRP-transformer_attribution cls':
                                            k = 'LRP-transformer_attribution'
                                            takeCls = True
                                        else:
                                            takeCls = False

                                        for hypers in hyperss:
                                            if takeCls and not hypers[-1]:
                                                continue
                                            elif not takeCls and hypers[-1]:
                                                continue
                                            for toplevel in toplevels:
                                                    test_size = test_sizes[splitInd]
                                                    modelType = hypers[0]
                                                    epochs = hypers[1]
                                                    dmodel = hypers[2]
                                                    dfff = hypers[3]
                                                    doSkip = hypers[4]
                                                    doBn = hypers[5]
                                                    header = hypers[6]
                                                    numOfLayers = hypers[7]
                                                    dropout = hypers[8]
                                                    att_dropout = hypers[9]
                                                    if modelType == 'Transformer':
                                                        doClsTocken = hypers[10]
                                                    else:
                                                        doClsTocken = False

                                                    dsName = str(dataset) +  ',l:' + str(toplevel) +',s:' +  str(symbols)+',e:' +  str(nrEmpty)+',a:' +  str(andStack)+',o:' +  str(orStack)+',x:' +  str(xorStack)+',na:' +  str(nrAnds)+',no:' +  str(nrOrs)+',nx:' +  str(nrxor)+',i:' +  arrayToString(trueIndexes)+',t:' +  str(test_size)+',oo:' +  str(orOffSet)+',xo:' +  str(xorOffSet) +',r:' + arrayToString(redaundantIndexes)


                                                    for rFolder in filteredResultsFolder:


                                                        saveName = pt.getWeightName(dsName, dataset, batch_size, epochs, numOfLayers, header, dmodel, dfff, dropout, att_dropout, doSkip, doBn, doClsTocken, learning = False, results = True, resultsPath=rFolder)

                                                        if os.path.isfile(saveName + '.pkl'):
                                                            if saveName in allReses[rFolder].keys():
                                                                res = allReses[rFolder][saveName]
                                                            else:
                                                                try:
                                                                    results = helper.load_obj(saveName)
                                                                except:
                                                                    continue

                                                                res = dict()
                                                                for index, vv in np.ndenumerate(results):
                                                                    res = vv

                                                                allReses[rFolder][saveName] = res
                                                        else:
                                                            continue

                                                        if k not in res['gcr'].keys():
                                                            continue
                                                        v  = res['gcr'][k][m][t][c][normName][name]
                                                        cv  = res['gcr'][k][m][c][normName][name]

                                                        #print(v[0])
                                                        for i, val in enumerate(v):
                                                            
                                                            if res[k]['performance'][i][i%5] < (accTreshold ):
                                                                    continue

                                                            resultVs.append(np.nanmean(val) * 100) 

                                                            
                                                            for kic in infoCombisS.keys():
                                                                if checkKIC(kic, 1, configName, toplevel):
                                                                    if takeCls:
                                                                        tgcrComis[kic][c]['LRP-transformer_attribution cls'][m].append((np.nanmean(cv[i]) * 100) - (np.nanmean(val) * 100))
                                                                    else:
                                                                        tgcrComis[kic][c][k][m].append((np.nanmean(cv[i]) * 100) - (np.nanmean(val) * 100))

                                        resultVFs.append(np.nanmean(resultVs))
                                        resultVSs.append(np.nanstd(resultVs))

                                    lables= kNames
                                        
                                    counts = resultVFs
                                    counts = np.where(np.isnan(counts),0,counts)
                                    e = resultVSs
                                    e = np.where(np.isnan(e),0,e)


                                    legend.append(testFacNames[splitInd] + '; ' + accTNames[accTHold] + '; ' + t)
                                    ind = np.arange(len(counts))
                                    rect = ax.bar(ind+x_offset , counts, bar_width, yerr=e, linestyle='None', capsize=3)
                                    rects.append(rect)
                                    ax.set_ylabel('Percent')
                                    ax.set_xticks(ind)
                                    ax.set_xticklabels(lables)
                                    ax.set_title(saveNames[nami]+' ' + str(c) + '\n'+ topLevelNamesSmall[j].upper())
                                    ax.tick_params(labelrotation=90)
                                    ax.set_ylim(bottom=0, top=100)

                fig.legend(rects, labels=legend, 
                    loc="upper right", bbox_to_anchor=(1.088, 0.85)) 
                fig.tight_layout()
                fig.suptitle(config.configName,fontsize=14)
                fig.subplots_adjust(top=0.84)
                specificFolder = folderGeneral + 'TGCRAcc/'
                if not os.path.exists(specificFolder):
                    os.makedirs(specificFolder)
                fig.savefig(specificFolder + saveNames[nami] +' ALL ' + configName+ ' mode'+str(m)+ ' ' + normName + '.png', dpi = 300, bbox_inches = 'tight')

                plt.show()   
                plt.close()


splitNames = ['Split Test Set; Acc=100', 'Not Split Test; Acc>=0']
for ni,n in enumerate(gcrOptions):
    fig, axs = plt.subplots(nrows=1, ncols=len(infoCombisS.keys()), sharex=False, sharey=True,
                            figsize=(25, 4), layout="constrained")    
                            
    specificFolder = folderGeneral + 'tGCRAccRanking/'
    if not os.path.exists(specificFolder):
        os.makedirs(specificFolder)
    
        
    f = open(specificFolder+ n + "-rankingVs.txt", "a")
    f.write(n)
    f.write("\n")
    f.write("full 	" + "	".join(str(x) for x in kNames))
    f.write("\n")
    f.close()
    f = open(specificFolder+ n + "-ranking.txt", "a")
    f.write(n)
    f.write("\n")
    f.write("full 	" + "	".join(str(x) for x in kNames))
    f.close()
    for ki, kic in enumerate(infoCombisS.keys()):

        ax = axs[ki]
        resultVs = []
        resultSs = []
        lables = []
        for ksi, k in enumerate(ks):
            m = getBestMode(nibComis, k)
            data = tgcrComis[kic][n][k]
            
            lables.append(kNames[ksi])# + ' ' + str(m))
            resultVs.append(np.nanmean(data[m]))
            resultSs.append(np.nanstd(data[m]))

        resultVs = np.array(resultVs)
        resultSs = np.array(resultSs)

        ranks = rankdata(resultVs, method='max')
        ranks = (np.array(ranks) - (len(ranks)+1)) * -1

        f = open(specificFolder+ n + "-ranking.txt", "a")
        f.write(kic + ' ' + 'full')
        f.write("\n")
        f.write("   ".join(str(x) for x in ranks))
        f.write("\n")
        f.write("   ".join(str(x) for x in resultVs))
        f.write("\n")
        f.write("\n")
        f.close()

        sortI = np.argsort(resultVs)
        counts = resultVs[sortI]
        e = resultSs[sortI]
        lables = np.array(lables)[sortI]

        ind = np.arange(len(counts))
        rect = ax.bar(ind , counts, yerr=e, linestyle='None', capsize=3)
        if ki == 0:
            ax.set_ylabel('Percent')
        ax.set_xticks(ind)
        ax.set_xticklabels(lables)
        ax.set_title(kic)
        ax.tick_params(labelrotation=90)
        ax.set_ylim(top=100)
    fig.suptitle('t'+str(optionNames[ni]),fontsize=14)

    fig.savefig(specificFolder + n +'-ranking.png', dpi = 300, bbox_inches = 'tight')

    plt.show()
    plt.close()




# In[13]:




# In[ ]:

# In[12]:

names = ['avgCorrelation', 'andCorrelation','orCorrelation','xorCorrelation','irrelevantCorrelation', 'neighbourhoodCorrelation', 'toIrrelevantCorrelation','avgCorrelation+', 'andCorrelation+','orCorrelation+','xorCorrelation+','irrelevantCorrelation+', 'neighbourhoodCorrelation+', 'toIrrelevantCorrelation+', 'avgCorrelationToP', 'toIrrelevantCorrelationToP' ]
kNamesRnks = ['avgCorrelation', 'andCorrelation','orCorrelation','xorCorrelation','irrelevantCorrelation', 'neighbourhoodCorrelation', 'To Irrelevant Correlation','avgCorrelation+', 'andCorrelation+','orCorrelation+','xorCorrelation+','irrelevantCorrelation+', 'neighbourhoodCorrelation+', 'Significant To Irrelevant Correlation', 'avgCorrelationToP', 'Percent Significant']

corComis = dict()
for kic in infoCombisS.keys():
    corComis[kic] = dict()
    for n in names:
        corComis[kic][n] = dict()
        for k in ks:

            corComis[kic][n][k] = []
            for mode in range(np.max(modes)+1): 
                corComis[kic][n][k].append([])


for m in modes:
    for name in names:
        fig, axs = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True,
                                        figsize=(14, 6), layout="constrained")
        ax = axs
        width = 0
        n_bars = 4
        standardWidth = 0.9
        bar_width = standardWidth / n_bars
        barInd = 0
        legend = []
        rects = []
        for accTHold in [0]:
            for splitInd in splitInds:


                x_offset = (barInd - n_bars / 2) * bar_width + bar_width / 2
                barInd+= 1
                resultVFs = []
                resultVSs = []
                avgResults = []
                width += standardWidth

                for k in ks:

                    if k == 'LRP-transformer_attribution cls':
                        k = 'LRP-transformer_attribution'
                        takeCls = True
                    else:
                        takeCls = False

                    resultVs = []
                    avgCorrVs = []

                    for j, toplevels in enumerate(topLevelss[1:]):

                        for cNr, config in enumerate(allConfigs):
                            configName = config.configName
                            hyperss = config.hyperss
                            hypNames= config.hypNames
                            test_sizes = config.test_sizes
                            #topLevelss = config.topLevelss
                            dataset = config.dataset
                            symbols = config.symbols
                            nrEmpty = config.nrEmpty
                            andStack = config.andStack
                            orStack = config.orStack
                            xorStack = config.xorStack
                            nrAnds = config.nrAnds
                            nrOrs = config.nrOrs
                            nrxor = config.nrxor
                            trueIndexes = config.trueIndexes
                            orOffSet = config.orOffSet
                            xorOffSet = config.xorOffSet
                            redaundantIndexes = config.redaundantIndexes
                            batch_size = config.batch_size



                            if k in ['performance', 'tree performance', 'treeImportances', 'baseline']:
                                continue
                            
                            for hypers in hyperss:
                                if takeCls and not hypers[-1]:
                                    continue
                                elif not takeCls and hypers[-1]:
                                    continue
                                for toplevel in toplevels:
                                        test_size = test_sizes[splitInd]
                                        modelType = hypers[0]
                                        epochs = hypers[1]
                                        dmodel = hypers[2]
                                        dfff = hypers[3]
                                        doSkip = hypers[4]
                                        doBn = hypers[5]
                                        header = hypers[6]
                                        numOfLayers = hypers[7]
                                        dropout = hypers[8]
                                        att_dropout = hypers[9]
                                        if modelType == 'Transformer':
                                            doClsTocken = hypers[10]
                                        else:
                                            doClsTocken = False

                                        dsName = str(dataset) +  ',l:' + str(toplevel) +',s:' +  str(symbols)+',e:' +  str(nrEmpty)+',a:' +  str(andStack)+',o:' +  str(orStack)+',x:' +  str(xorStack)+',na:' +  str(nrAnds)+',no:' +  str(nrOrs)+',nx:' +  str(nrxor)+',i:' +  arrayToString(trueIndexes)+',t:' +  str(test_size)+',oo:' +  str(orOffSet)+',xo:' +  str(xorOffSet) +',r:' + arrayToString(redaundantIndexes)

                                        
                                        
                                        for rFolder in filteredResultsFolder:

                                            saveName = pt.getWeightName(dsName, dataset, batch_size, epochs, numOfLayers, header, dmodel, dfff, dropout, att_dropout, doSkip, doBn, doClsTocken, learning = False, results = True, resultsPath=rFolder)

                                            if os.path.isfile(saveName + '.pkl'):
                                                if saveName in allReses[rFolder].keys():
                                                    res = allReses[rFolder][saveName]
                                                else:
                                                    try:
                                                        results = helper.load_obj(saveName)
                                                    except:
                                                        continue

                                                    res = dict()
                                                    for index, vv in np.ndenumerate(results):
                                                        res = vv

                                                    allReses[rFolder][saveName] = res
                                            else:
                                                print('NOT FOUND: ' + saveName)
                                                continue


                                            if k not in res.keys():
                                                continue
                                            v  = res[k][m][name]

                                            sd = np.array(v)
                                            do2DData = False
                                            if len(sd.shape) > 2:
                                                do2DData = True
                                            if do2DData:
                                                v = np.sum(sd, axis=1)#-1

                                            for i, val in enumerate(v):
                                                
                                                if res[k]['performance'][i][i%5] < (accTreshold ):
                                                    continue
                                                if isinstance(val, float):
                                                    resultVs.append(val)

                                                    
                                                    for kic in infoCombisS.keys():
                                                        if checkKIC(kic, 1, configName, toplevel):
                                                            if takeCls:

                                                                corComis[kic][name]['LRP-transformer_attribution cls'][m].append(val)
                                                            else:
                                                                corComis[kic][name][k][m].append(val)
                                                else:
                                                    for valV in val:
                                                        resultVs.append(np.nanmean(valV))
                                                        
                                                        for kic in infoCombisS.keys():
                                                            if checkKIC(kic, 1, configName, toplevel):
                                                                if takeCls:

                                                                    corComis[kic][name]['LRP-transformer_attribution cls'][m].append(np.nanmean(valV))
                                                                else:
                                                                    corComis[kic][name][k][m].append(np.nanmean(valV))

                                            for i, val in enumerate(res[k][m]['avgCorrelation'] ):
                                                avgCorrVs.append(val)


                    resultVFs.append(np.nanmean(resultVs))
                    resultVSs.append(np.nanstd(resultVs))
                    avgResults.append(np.nanmean(avgCorrVs))

                
                fruits= kNames
                    
                counts = resultVFs
                counts = np.where(np.isnan(counts),0,counts)
                e =  resultVSs
                e = np.where(np.isnan(e),0,e)


                legend.append(testFacNames[splitInd] + '; ' + accTNames[accTHold])
                ind = np.arange(len(counts))
                rect = ax.bar(ind+x_offset , counts, bar_width, yerr=e, linestyle='None', capsize=3)
                rects.append(rect)


                ax.set_ylabel('Correlation')
                ax.set_xticks(ind)
                ax.set_xticklabels(fruits)
                ax.set_title(name +' '+ toplevels[0] +' ' + configName)
                ax.tick_params(labelrotation=90)

        fig.legend(rects, labels=legend, 
            loc="upper right", bbox_to_anchor=(1.095, 0.85)) 
        fig.tight_layout()
        fig.suptitle(name,fontsize=14)
        fig.subplots_adjust(top=0.84)
        specificFolder = folderGeneral + 'correlation/'
        if not os.path.exists(specificFolder):
            os.makedirs(specificFolder)
        fig.savefig(specificFolder + name + ' mode' +str(m) +'-fullAvg100.png', dpi = 300, bbox_inches = 'tight')
        plt.show() 
        plt.close()

splitNames = ['Split Test Set; Acc=100', 'Not Split Test; Acc>=0']
for ni, n in enumerate(names):
    fig, axs = plt.subplots(nrows=1, ncols=len(infoCombisS.keys()), sharex=False, sharey=True,
                            figsize=(25, 4), layout="constrained")    
                            
    specificFolder = folderGeneral + 'CorrelationRanking/'
    if not os.path.exists(specificFolder):
        os.makedirs(specificFolder)
    
    
    f = open(specificFolder+ n + "-rankingVs.txt", "a")
    f.write(n)
    f.write("\n")
    f.write("full 	" + "	".join(str(x) for x in kNames))
    f.write("\n")
    f.close()
    f = open(specificFolder+ n + "-ranking.txt", "a")
    f.write(n)
    f.write("\n")
    f.write("full 	" + "	".join(str(x) for x in kNames))
    f.close()
    for ki, kic in enumerate(infoCombisS.keys()):

        ax = axs[ki]
        resultVs = []
        resultSs = []
        lables = []
        for ksi, k in enumerate(ks):
            data = corComis[kic][n][k] 
            m = getBestMode(nibComis, k)
            lables.append(kNames[ksi])# + ' ' + str(m))
            resultVs.append(np.nanmean(data[m]))
            resultSs.append(np.nanstd(data[m]))

        resultVs = np.array(resultVs)
        resultSs = np.array(resultSs)

        ranks = rankdata(resultVs, method='max')
        ranks = (np.array(ranks) - (len(ranks)+1)) * -1

        f = open(specificFolder+ n + "-rankingVs.txt", "a")
        f.write(kic + "	")
        f.write("	".join(str(x) for x in resultVs))
        f.write("\n")
        f.close()

        f = open(specificFolder+ n + "-ranking.txt", "a")
        f.write(kic + "	")
        f.write("	".join(str(x) for x in ranks))
        f.write("\n")
        f.close()
        
        sortI = np.argsort(resultVs)
        counts = resultVs[sortI]
        e = resultSs[sortI]
        lables = np.array(lables)[sortI]

        ind = np.arange(len(counts))
        rect = ax.bar(ind , counts, yerr=e, linestyle='None', capsize=3)
        if ki == 0:
            ax.set_ylabel('Percent')
        ax.set_xticks(ind)
        ax.set_xticklabels(lables)
        ax.set_title(kic)#)
        ax.tick_params(labelrotation=90)

        ax.set_ylim(bottom=0, top=1)
        f = open(specificFolder+ n + "-rankingVs.txt", "a")
        f.write("\n")
        f.close()

        f = open(specificFolder+ n + "-ranking.txt", "a")
        f.write("\n")
        f.close()
    fig.suptitle(kNamesRnks[ni],fontsize=14)


    fig.savefig(specificFolder + n +'-ranking.png', dpi = 300, bbox_inches = 'tight')

    plt.show()
    plt.close()



# In[12]:



names = ['avgCorrelation', 'andCorrelation','orCorrelation','xorCorrelation','irrelevantCorrelation', 'neighbourhoodCorrelation', 'toIrrelevantCorrelation','avgCorrelation+', 'andCorrelation+','orCorrelation+','xorCorrelation+','irrelevantCorrelation+', 'neighbourhoodCorrelation+', 'toIrrelevantCorrelation+', 'avgCorrelationToP', 'toIrrelevantCorrelationToP']
for m in modes:
    for name in names:
        fig, axs = plt.subplots(nrows=len(allConfigs), ncols=3, sharex=True, sharey=True,
                                        figsize=(22, 12), layout="constrained")

        for j, toplevels in enumerate(topLevelss[1:]):

            for cNr, config in enumerate(allConfigs):
                configName = config.configName
                hyperss = config.hyperss
                hypNames= config.hypNames
                test_sizes = config.test_sizes
                dataset = config.dataset
                symbols = config.symbols
                nrEmpty = config.nrEmpty
                andStack = config.andStack
                orStack = config.orStack
                xorStack = config.xorStack
                nrAnds = config.nrAnds
                nrOrs = config.nrOrs
                nrxor = config.nrxor
                trueIndexes = config.trueIndexes
                orOffSet = config.orOffSet
                xorOffSet = config.xorOffSet
                redaundantIndexes = config.redaundantIndexes
                batch_size = config.batch_size

                ax = axs[cNr, j]
                width = 0
                n_bars = 2
                standardWidth = 0.9
                bar_width = standardWidth / n_bars
                barInd = 0
                legend = []
                rects = []
                for accTHold in [0]:
                    for splitInd in splitInds:

                    

                        x_offset = (barInd - n_bars / 2) * bar_width + bar_width / 2
                        barInd+= 1
                        resultVFs = []
                        resultVSs = []
                        avgResults = []
                        width += standardWidth

                        for k in ks:

                            resultVs = []
                            avgCorrVs = []

                            if k in ['performance', 'tree performance', 'treeImportances', 'baseline']:
                                continue

                            if k == 'LRP-transformer_attribution cls':
                                k = 'LRP-transformer_attribution'
                                takeCls = True
                            else:
                                takeCls = False
                            
                            for hypers in hyperss:
                                if takeCls and not hypers[-1]:
                                    continue
                                elif not takeCls and hypers[-1]:
                                    continue
                                for toplevel in toplevels:
                                        test_size = test_sizes[splitInd]
                                        modelType = hypers[0]
                                        epochs = hypers[1]
                                        dmodel = hypers[2]
                                        dfff = hypers[3]
                                        doSkip = hypers[4]
                                        doBn = hypers[5]
                                        header = hypers[6]
                                        numOfLayers = hypers[7]
                                        dropout = hypers[8]
                                        att_dropout = hypers[9]
                                        if modelType == 'Transformer':
                                            doClsTocken = hypers[10]
                                        else:
                                            doClsTocken = False

                                        dsName = str(dataset) +  ',l:' + str(toplevel) +',s:' +  str(symbols)+',e:' +  str(nrEmpty)+',a:' +  str(andStack)+',o:' +  str(orStack)+',x:' +  str(xorStack)+',na:' +  str(nrAnds)+',no:' +  str(nrOrs)+',nx:' +  str(nrxor)+',i:' +  arrayToString(trueIndexes)+',t:' +  str(test_size)+',oo:' +  str(orOffSet)+',xo:' +  str(xorOffSet) +',r:' + arrayToString(redaundantIndexes)

                                        
                                        
                                        for rFolder in filteredResultsFolder:

                                            saveName = pt.getWeightName(dsName, dataset, batch_size, epochs, numOfLayers, header, dmodel, dfff, dropout, att_dropout, doSkip, doBn, doClsTocken, learning = False, results = True, resultsPath=rFolder)

                                            if os.path.isfile(saveName + '.pkl'):
                                                if saveName in allReses[rFolder].keys():
                                                    res = allReses[rFolder][saveName]
                                                else:
                                                    try:
                                                        results = helper.load_obj(saveName)
                                                    except:
                                                        continue

                                                    res = dict()
                                                    for index, vv in np.ndenumerate(results):
                                                        res = vv

                                                    allReses[rFolder][saveName] = res
                                            else:
                                                continue


                                            if k not in res.keys():
                                                continue
                                            v  = res[k][m][name]
                                            sd = np.array(v)
                                            do2DData = False
                                            if len(sd.shape) > 2:
                                                do2DData = True
                                            if do2DData:
                                                v = np.sum(sd, axis=1)-1
                                            for i, val in enumerate(v):
                                                if res[k]['performance'][i][i%5] < (accTreshold ):
                                                    continue
                                                if isinstance(val, float):
                                                    resultVs.append(val)
                                                else:
                                                    for valV in val:
                                                        resultVs.append(np.nanmean(valV))
                                            for i, val in enumerate(res[k][m]['avgCorrelation'] ):
                                                avgCorrVs.append(val)
                                                

                            resultVFs.append(np.nanmean(resultVs))
                            resultVSs.append(np.nanstd(resultVs))
                            avgResults.append(np.nanmean(avgCorrVs))

                        if name == 'avgCorrelation':
                            resultVFs = np.array(avgResults)
                        fruits= kNames
                            
                        counts = resultVFs
                        counts = np.where(np.isnan(counts),0,counts)
                        e =  resultVSs
                        e = np.where(np.isnan(e),0,e)

                        


                        legend.append(testFacNames[splitInd] + '; ' + accTNames[accTHold])
                        ind = np.arange(len(counts))
                        rect = ax.bar(ind+x_offset , counts, bar_width, yerr=e, linestyle='None', capsize=3)
                        rects.append(rect)



                        ax.set_ylabel('Correlation')
                        ax.set_xticks(ind)
                        ax.set_xticklabels(fruits)
                        ax.set_title(name +' '+ toplevels[0] +' ' + configName)
                        ax.tick_params(labelrotation=90)

        fig.legend(rects, labels=legend, 
            loc="upper right", bbox_to_anchor=(1.095, 0.85)) 
        fig.tight_layout()
        fig.suptitle(name,fontsize=14)
        fig.subplots_adjust(top=0.84)
        specificFolder = folderGeneral + 'correlation/'
        if not os.path.exists(specificFolder):
            os.makedirs(specificFolder)
        fig.savefig(specificFolder + name + ' mode' +str(m) +'-full100.png', dpi = 300, bbox_inches = 'tight')
        plt.show() 
        plt.close()


# In[13]:


for m in modes:
    for ki, k in enumerate(ks):     
        fig, axs1 = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True,
                                        figsize=(12, 6), layout="constrained")       

        if k == 'LRP-transformer_attribution cls':
            k = 'LRP-transformer_attribution'
            takeCls = True
        else:
            takeCls = False


        for c in [0,1]:
            for j, toplevels in enumerate(topLevelss[1:]):

                ax = axs1[c,j]
                width = 0
                n_bars = 2
                standardWidth = 0.9
                bar_width = standardWidth / n_bars
                barInd = 0
                legend = []
                rects = []
                for accTHold in [0]:
                    for splitInd in splitInds:


                        x_offset = (barInd - n_bars / 2) * bar_width + bar_width / 2
                        barInd+= 1
                        resultVs = []
                        fruits= []

                        resultVs.append([])
                        fruits.append('AND ')
                        resultVs.append([])
                        fruits.append('OR ')
                        resultVs.append([])
                        fruits.append('XOR ')
                        width += standardWidth

                        if k in ['performance', 'tree performance', 'treeImportances', 'baseline']:
                            continue


                                

                        for cNr, config in enumerate(allConfigs):
                            configName = config.configName
                            hyperss = config.hyperss
                            hypNames= config.hypNames
                            test_sizes = config.test_sizes
                            dataset = config.dataset
                            symbols = config.symbols
                            nrEmpty = config.nrEmpty
                            andStack = config.andStack
                            orStack = config.orStack
                            xorStack = config.xorStack
                            nrAnds = config.nrAnds
                            nrOrs = config.nrOrs
                            nrxor = config.nrxor
                            trueIndexes = config.trueIndexes
                            orOffSet = config.orOffSet
                            xorOffSet = config.xorOffSet
                            redaundantIndexes = config.redaundantIndexes
                            batch_size = config.batch_size

                            if configName not in ['2inBinary', '3inBinary', '2inQuaternary']:
                                continue
                            


                            for hypers in hyperss:
                                if takeCls and not hypers[-1]:
                                    continue
                                elif not takeCls and hypers[-1]:
                                    continue
                                for toplevel in toplevels:
                                    test_size = test_sizes[splitInd]
                                    modelType = hypers[0]
                                    epochs = hypers[1]
                                    dmodel = hypers[2]
                                    dfff = hypers[3]
                                    doSkip = hypers[4]
                                    doBn = hypers[5]
                                    header = hypers[6]
                                    numOfLayers = hypers[7]
                                    dropout = hypers[8]
                                    att_dropout = hypers[9]
                                    if modelType == 'Transformer':
                                        doClsTocken = hypers[10]
                                    else:
                                        doClsTocken = False

                                    dsName = str(dataset) +  ',l:' + str(toplevel) +',s:' +  str(symbols)+',e:' +  str(nrEmpty)+',a:' +  str(andStack)+',o:' +  str(orStack)+',x:' +  str(xorStack)+',na:' +  str(nrAnds)+',no:' +  str(nrOrs)+',nx:' +  str(nrxor)+',i:' +  arrayToString(trueIndexes)+',t:' +  str(test_size)+',oo:' +  str(orOffSet)+',xo:' +  str(xorOffSet) +',r:' + arrayToString(redaundantIndexes)

                                    for rFolder in filteredResultsFolder:
                                        saveName = pt.getWeightName(dsName, dataset, batch_size, epochs, numOfLayers, header, dmodel, dfff, dropout, att_dropout, doSkip, doBn, doClsTocken, learning = False, results = True, resultsPath=rFolder)
                                        
                                        if os.path.isfile(saveName + '.pkl'):
                                            if saveName in allReses[rFolder].keys():
                                                res = allReses[rFolder][saveName]
                                            else:
                                                try:
                                                    results = helper.load_obj(saveName)
                                                except:
                                                    continue

                                                res = dict()
                                                for index, vv in np.ndenumerate(results):
                                                    res = vv

                                                allReses[rFolder][saveName] = res
                                        else:
                                            continue


                                        if k not in res.keys():
                                            continue

                                        
                                        v  = res[k][m][c]['neededInformationStackBroken']
                                        for i, val in enumerate(v):
                                            if res[k]['performance'][i][i%5] < (accTreshold ):
                                                continue
                                            if isinstance(val, int) or isinstance(val, float):
                                                continue 
                                            
                                            viCount = 0
                                            for vstack in range(andStack):
                                                resultVs[0].append((val[viCount]))
                                                viCount+=1
                                            for vstack in range(orStack):
                                                resultVs[1].append(val[viCount])
                                                viCount+=1
                                            for vstack in range(xorStack):
                                                resultVs[2].append(val[viCount])
                                                viCount+=1
                        
                        if len(resultVs) == 0:
                            continue



                        counts = np.array([np.mean(vs) for vs in resultVs])
                        e =  np.array([np.std(vs) for vs in resultVs])

                        if testFacNames[splitInd] + '; ' + accTNames[accTHold] not in legend:
                            legend.append(testFacNames[splitInd] + '; ' + accTNames[accTHold])                        
                        ind = np.arange(len(counts))
                        rect = ax.bar(ind+x_offset , counts, bar_width, yerr=e, linestyle='None', capsize=3)
                        rects.append(rect)
                        ax.set_ylabel('Percent')
                        ax.set_xticks(ind)
                        ax.set_xticklabels(fruits)
                        ax.tick_params(labelrotation=90)
                        ax.set_title(kNames[ki] + '\nNeeded Rem ' + topLevelNamesSmall[j] + ' ' + str(c))

        fig.legend(rects, labels=legend, 
            loc="upper right", bbox_to_anchor=(1.088, 0.85)) 
        fig.tight_layout()
        fig.suptitle( kNames[ki],fontsize=14)
        fig.subplots_adjust(top=0.81)

        specificFolder = folderGeneral + 'NIBperStack/'
        if not os.path.exists(specificFolder):
            os.makedirs(specificFolder)
        fig.savefig(specificFolder + kNames[ki] + ' mode' + str(m) + ' NIB fault per stackAll.png', dpi = 300, bbox_inches = 'tight')
        plt.show() 
        plt.close()


# In[14]:




# In[15]:

for m in modes:
    for c, k in enumerate(ks):

        if k == 'LRP-transformer_attribution cls':
            k = 'LRP-transformer_attribution'
            takeCls = True
        else:
            takeCls = False

        plots = []

        fig, axs1 = plt.subplots(nrows=2, ncols=len(topLevelss[1:]), sharex=True, sharey=True,
                                        figsize=(19, symbols*3), layout="constrained")
        for j, toplevels in enumerate(topLevelss[1:]):
            for ini in range(2):
                ax1 = axs1[ini,j]
                width = 0
                n_bars = 4
                standardWidth = 0.8
                bar_width = standardWidth / n_bars
                barInd = 0
                legend = []
                rects = []
                for ti, t in enumerate(['t1.0', 't0.8', 't0.5', 'tbaseline']):


                    
                    x_offset = (barInd - n_bars / 2) * bar_width + bar_width / 2
                    barInd+= 1
                    width += standardWidth

                    for accTHold in [0]:
                        for splitInd in splitInds:


                            resultVs = []
                            fruits= []
                            resultVs.append([])
                            fruits.append('AND')
                            resultVs.append([])
                            fruits.append('OR')

                            resultVs.append([])
                            fruits.append('XOR')
                            
                            resultVs.append([])
                            fruits.append('Empty')
                            if k in ['performance', 'tree performance', 'treeImportances']:
                                continue



                            for cNr, config in enumerate(allConfigs):
                                configName = config.configName
                                hyperss = config.hyperss
                                hypNames= config.hypNames
                                test_sizes = config.test_sizes
                                dataset = config.dataset
                                symbols = config.symbols
                                nrEmpty = config.nrEmpty
                                andStack = config.andStack
                                orStack = config.orStack
                                xorStack = config.xorStack
                                nrAnds = config.nrAnds
                                nrOrs = config.nrOrs
                                nrxor = config.nrxor
                                trueIndexes = config.trueIndexes
                                orOffSet = config.orOffSet
                                xorOffSet = config.xorOffSet
                                redaundantIndexes = config.redaundantIndexes
                                batch_size = config.batch_size

                                
                                valuesA = helper.getMapValues(symbols)

                                if configName not in ['2inBinary', '3inBinary', '2inQuaternary']:
                                    continue

                                for hypers in hyperss:
                                    if takeCls and not hypers[-1]:
                                        continue
                                    elif not takeCls and hypers[-1]:
                                        continue
                                    for toplevel in toplevels:
                                            test_size = test_sizes[splitInd]
                                            modelType = hypers[0]
                                            epochs = hypers[1]
                                            dmodel = hypers[2]
                                            dfff = hypers[3]
                                            doSkip = hypers[4]
                                            doBn = hypers[5]
                                            header = hypers[6]
                                            numOfLayers = hypers[7]
                                            dropout = hypers[8]
                                            att_dropout = hypers[9]
                                            if modelType == 'Transformer':
                                                doClsTocken = hypers[10]
                                            else:
                                                doClsTocken = False

                                            dsName = str(dataset) +  ',l:' + str(toplevel) +',s:' +  str(symbols)+',e:' +  str(nrEmpty)+',a:' +  str(andStack)+',o:' +  str(orStack)+',x:' +  str(xorStack)+',na:' +  str(nrAnds)+',no:' +  str(nrOrs)+',nx:' +  str(nrxor)+',i:' +  arrayToString(trueIndexes)+',t:' +  str(test_size)+',oo:' +  str(orOffSet)+',xo:' +  str(xorOffSet) +',r:' + arrayToString(redaundantIndexes)

                                            for rFolder in filteredResultsFolder:
                                                saveName = pt.getWeightName(dsName, dataset, batch_size, epochs, numOfLayers, header, dmodel, dfff, dropout, att_dropout, doSkip, doBn, doClsTocken, learning = False, results = True, resultsPath=rFolder)
                                                
                                                if os.path.isfile(saveName + '.pkl'):
                                                    if saveName in allReses[rFolder].keys():
                                                        res = allReses[rFolder][saveName]
                                                    else:
                                                        try:
                                                            results = helper.load_obj(saveName)
                                                        except:
                                                            continue

                                                        res = dict()
                                                        for index, vv in np.ndenumerate(results):
                                                            res = vv

                                                        allReses[rFolder][saveName] = res
                                                else:
                                                    #print('NOT FOUND: ' + saveName)
                                                    continue


                                                if k not in res.keys():
                                                    continue

                                                if t not in res[k][m].keys():
                                                    continue

                                                

                                                
                                                v = res[k][m][t]['RemovalChancePInputValue']
                                                for u in valuesA:
                                                    if ini == 1 and u not in trueIndexes:
                                                        continue
                                                    elif ini == 0 and u in trueIndexes:
                                                        continue

                                                    if u not in v.keys():
                                                        continue

                                                    
                                                    for vi, val in enumerate(v[u]):
                                                        if res[k]['performance'][vi][vi%5] < (accTreshold ):
                                                            continue
                                                        if isinstance(val, int) or isinstance(val, float):
                                                            continue 
                                                        curLen = 0
                                                        for i in range(nrAnds * andStack):
                                                            resultVs[0].append(val[curLen])
                                                            curLen += 1
                                                        for i in range(nrOrs * orStack):
                                                            resultVs[1].append(val[curLen])
                                                            curLen += 1
                                                        for i in range(nrxor * xorStack):
                                                            resultVs[2].append(val[curLen])
                                                            curLen += 1
                                                        for i in range(nrEmpty):
                                                            resultVs[3].append(val[curLen])
                                                            curLen += 1







                            if len(resultVs) == 0:
                                counts = np.zeros(len(fruits))
                            else:
                                counts = np.array([np.mean(vs) for vs in resultVs])
                            e =  np.array([np.std(vs) for vs in resultVs])
                                
                            colors = ['tab:blue', 'orange','g', 'tab:red']
                            colorN =colors[ti]
                            
                            lableName = testFacNames[splitInd] + '; ' + accTNames[accTHold]       + '; ' + t                      
                            if lableName not in legend:
                                legend.append(lableName)
                            ind = np.arange(len(fruits))
                            if splitInd == 1 and accTHold == 1:
                                p = ax1.errorbar(ind+x_offset, counts, marker='d', yerr=e,label=lableName, color=colorN, linestyle="none")
                            elif splitInd == 1 and accTHold == 0:
                                p = ax1.errorbar(ind+x_offset, counts, marker='v', yerr=e,label=lableName, color=colorN, linestyle="none")
                            elif splitInd == 0 and accTHold == 1:
                                p = ax1.errorbar(ind+x_offset, counts, marker='<', yerr=e,label=lableName, color=colorN, linestyle="none")
                            elif splitInd == 0 and accTHold == 0:
                                p = ax1.errorbar(ind+x_offset, counts, marker='s', yerr=e,label=lableName, color=colorN, linestyle="none")
                            rects.append(p)

                            ax1.set_ylabel('Removal Chance')
                            ax1.set_xticks(ind)
                            ax1.set_xticklabels(fruits)
                            ax1.set_title(kNames[c] +' ' + topLevelNamesSmall[j] + ' InputValue ' + str(ini))
                            ax1.tick_params(labelrotation=90)

            
            fig.legend(rects, labels=legend, 
                loc="upper right", bbox_to_anchor=(1.118, 0.85)) 
            fig.suptitle(config.configName + ' ' + kNames[c],fontsize=14)

            fig.tight_layout()
            specificFolder = folderGeneral + 'RemoveChancePerInput/'
            if not os.path.exists(specificFolder):
                os.makedirs(specificFolder)
            fig.savefig(specificFolder + kNames[c] + ' mode' + str(m) + '-RemovalChancePInputValue.png', dpi = 300, bbox_inches = 'tight')
            plt.show() 
            plt.close()


# In[16]:


names = ['lasa acc', 'lasa red', 'treeScores', 'LogicalAcc', 'LogicalAccDiff', 'LogicalAccStatistics', 'LogicalAccStatisticsDiff']
saveNames = ['Avg. Retrained Model Acc.', 'Avg. Masking Data', 'treeScores', 'Logical Acc.', 'Avg. Logical Acc. Diff.', 'LogicalAccStatistics', 'Avg. Statistical Logical Acc. Diff.']


#TODO noch nicht done!!!!
lasaComis = dict()
for kic in infoCombisS.keys():
    lasaComis[kic] = dict()
    for n in names:
        lasaComis[kic][n] = dict()
        for k in ks:
            lasaComis[kic][n][k] = dict()
            for tm in ['t0.5', 't0.8','t1.0','tbaseline']:
                lasaComis[kic][n][k][tm] = []
                for mode in range(np.max(modes)+1): 
                    lasaComis[kic][n][k][tm].append([])

for m in modes:
    for nami, name in enumerate(names):
        

        diff = False
        if name[-4:] == 'Diff':
            name = name[:-4]
            diff = True

        fig, axs = plt.subplots(nrows=len(allConfigs), ncols=3, sharex=True, sharey=False,
                                        figsize=(24, 20), layout="constrained")
        for cNr, config in enumerate(allConfigs):
            configName = config.configName
            hyperss = config.hyperss
            hypNames= config.hypNames
            test_sizes = config.test_sizes
            topLevelss = config.topLevelss
            dataset = config.dataset
            symbols = config.symbols
            nrEmpty = config.nrEmpty
            andStack = config.andStack
            orStack = config.orStack
            xorStack = config.xorStack
            nrAnds = config.nrAnds
            nrOrs = config.nrOrs
            nrxor = config.nrxor
            trueIndexes = config.trueIndexes
            orOffSet = config.orOffSet
            xorOffSet = config.xorOffSet
            redaundantIndexes = config.redaundantIndexes
            batch_size = config.batch_size

            for j, toplevels in enumerate(topLevelss[1:]):

                    ax = axs[cNr, j]

                    
                    width = 0
                    n_bars = 4
                    standardWidth = 0.9
                    bar_width = 0.8 / n_bars
                    barInd = 0
                    legend = []
                    rects = []
                    for ti, t in enumerate(['t0.5', 't0.8','t1.0','tbaseline']):
                        

                            x_offset = (barInd - n_bars / 2) * bar_width + bar_width / 2
                            barInd+= 1
                            resultVFs = []
                            resultVSs = []
                            width += standardWidth
                            


                            for k in ks:
                                resultVs = []

                                if k in ['performance', 'tree performance', 'treeImportances', 'baseline']:
                                    continue

                                if k == 'LRP-transformer_attribution cls':
                                    k = 'LRP-transformer_attribution'
                                    takeCls = True
                                else:
                                    takeCls = False

                                for hypers in hyperss:
                                    if takeCls and not hypers[-1]:
                                        continue
                                    elif not takeCls and hypers[-1]:
                                        continue
                                    for toplevel in toplevels:
                                        for test_size in test_sizes:
                                            modelType = hypers[0]
                                            epochs = hypers[1]
                                            dmodel = hypers[2]
                                            dfff = hypers[3]
                                            doSkip = hypers[4]
                                            doBn = hypers[5]
                                            header = hypers[6]
                                            numOfLayers = hypers[7]
                                            dropout = hypers[8]
                                            att_dropout = hypers[9]
                                            if modelType == 'Transformer':
                                                doClsTocken = hypers[10]
                                            else:
                                                doClsTocken = False

                                            dsName = str(dataset) +  ',l:' + str(toplevel) +',s:' +  str(symbols)+',e:' +  str(nrEmpty)+',a:' +  str(andStack)+',o:' +  str(orStack)+',x:' +  str(xorStack)+',na:' +  str(nrAnds)+',no:' +  str(nrOrs)+',nx:' +  str(nrxor)+',i:' +  arrayToString(trueIndexes)+',t:' +  str(test_size)+',oo:' +  str(orOffSet)+',xo:' +  str(xorOffSet) +',r:' + arrayToString(redaundantIndexes)

                                            for rFolder in filteredResultsFolder:


                                                saveName = pt.getWeightName(dsName, dataset, batch_size, epochs, numOfLayers, header, dmodel, dfff, dropout, att_dropout, doSkip, doBn, doClsTocken, learning = False, results = True, resultsPath=rFolder)

                                                if os.path.isfile(saveName + '.pkl'):
                                                    if saveName in allReses[rFolder].keys():
                                                        res = allReses[rFolder][saveName]
                                                    else:
                                                        try:
                                                            results = helper.load_obj(saveName)
                                                        except:
                                                            continue

                                                        res = dict()
                                                        for index, vv in np.ndenumerate(results):
                                                            res = vv

                                                        allReses[rFolder][saveName] = res
                                                else:
                                                    continue
                                                if k not in res.keys():
                                                    continue
                                                if t not in res[k][m].keys():
                                                    continue
                                                v  = res[k][m][t][name]

                                                for i, val in enumerate(v):

                                                    numberModelsTrainedOverall += 1

                                                    
                                                    if diff:
                                                        resultVs.append((res[k][m][t]['lasa acc'][i] - np.nanmean(val))*100)
                                                        for kic in infoCombisS.keys():
                                                            if checkKIC(kic, 1, configName, toplevel):
                                                                if takeCls:
                                                                    lasaComis[kic][name+'Diff']['LRP-transformer_attribution cls'][t][m].append((res[k][m][t]['lasa acc'][i] - np.nanmean(val))*100)
                                                                else:
                                                                    lasaComis[kic][name+'Diff'][k][t][m].append((res[k][m][t]['lasa acc'][i] - np.nanmean(val))*100)

                                                    else:
                                                        resultVs.append(np.nanmean(val)*100) 

                                                        for kic in infoCombisS.keys():
                                                            if checkKIC(kic, 1, configName, toplevel):
                                                                if takeCls:
                                                                    lasaComis[kic][name]['LRP-transformer_attribution cls'][t][m].append(np.nanmean(val) * 100)
                                                                else:
                                                                    lasaComis[kic][name][k][t][m].append(np.nanmean(val) * 100)


                                resultVFs.append(np.nanmean(resultVs))
                                resultVSs.append(np.nanstd(resultVs))

                            lables= kNames
                                
                            counts = resultVFs
                            counts = np.where(np.isnan(counts),0,counts)
                            e = resultVSs
                            e = np.where(np.isnan(e),0,e)

                            legend.append(t)
                            ind = np.arange(len(counts))
                            rect = ax.bar(ind+x_offset , counts, bar_width, yerr=e, linestyle='None', capsize=3)
                            rects.append(rect)
                            ax.set_ylabel('Percent')
                            ax.set_xticks(ind)
                            ax.set_xticklabels(lables)
                            ax.set_title(configName + '\n'+ topLevelNamesSmall[j].upper())
                            ax.tick_params(labelrotation=90)


        fig.legend(rects, labels=legend, 
            loc="upper right", bbox_to_anchor=(1.088, 0.85)) 
        fig.tight_layout()
        fig.suptitle(name,fontsize=14)
        fig.subplots_adjust(top=0.84)
        specificFolder = folderGeneral + 'SingularThresholdPath/'
        if not os.path.exists(specificFolder):
            os.makedirs(specificFolder)
        fig.savefig(specificFolder + saveNames[nami] + ' mode' + str(m) + '.png', dpi = 300, bbox_inches = 'tight')

        plt.show() 
        plt.close()



splitNames = ['tbaseline']
for ni, n in enumerate(names):
    fig, axs = plt.subplots(nrows=1, ncols=len(infoCombisS.keys()), sharex=False, sharey=True,
                            figsize=(24, 4), layout="constrained")    
                            
    specificFolder = folderGeneral + 'LasaRanking/'
    if not os.path.exists(specificFolder):
        os.makedirs(specificFolder)
    
    for si, t in enumerate(['tbaseline']):
        f = open(specificFolder+ n + "-rankingVs.txt", "a")
        f.write(n)
        f.write("\n")
        f.write("full 	" + "	".join(str(x) for x in kNames))
        f.write("\n")
        f.close()
        f = open(specificFolder+ n + "-ranking.txt", "a")
        f.write(n)
        f.write("\n")
        f.write("full 	" + "	".join(str(x) for x in kNames))
        f.close()
        for ki, kic in enumerate(infoCombisS.keys()):

            ax = axs[ki]
            resultVs = []
            resultSs = []
            lables = []
            
            for ksi, k in enumerate(ks):
                data = lasaComis[kic][n][k][t] 
                m = getBestMode(nibComis, k)
                lables.append(kNames[ksi])# + ' ' + str(m))
                resultVs.append(np.nanmean(data[m]))
                resultSs.append(np.nanstd(data[m]))

            resultVs = np.array(resultVs)
            resultSs = np.array(resultSs)

            ranks = rankdata(resultVs, method='min')
            f = open(specificFolder+ n + "-rankingVs.txt", "a")
            f.write(kic + "	")
            f.write("	".join(str(x) for x in resultVs))
            f.write("\n")
            f.close()

            f = open(specificFolder+ n + "-ranking.txt", "a")
            f.write(kic + "	")
            f.write("	".join(str(x) for x in ranks))
            f.write("\n")
            f.close()

            sortI = np.argsort(resultVs)
            counts = resultVs[sortI]
            e = resultSs[sortI]
            lables = np.array(lables)[sortI]

            ind = np.arange(len(counts))
            rect = ax.bar(ind , counts, yerr=e, linestyle='None', capsize=3)
            if ki ==0:
                ax.set_ylabel('Percent')
            ax.set_xticks(ind)
            ax.set_xticklabels(lables)
            ax.set_title(kic)#)
            ax.tick_params(labelrotation=90)

            ax.set_ylim(top=100)
        
        f = open(specificFolder+ n + "-rankingVs.txt", "a")
        f.write("\n")
        f.close()

        f = open(specificFolder+ n + "-ranking.txt", "a")
        f.write("\n")
        f.close()
    fig.suptitle(saveNames[ni],fontsize=14)

    fig.savefig(specificFolder + n +'-ranking.png', dpi = 300, bbox_inches = 'tight')

    plt.show()
    plt.close()

# In[17]:


names = ['Tendeny mean', 'Tendeny median', 'Tendeny 1s', 'NegativeValueRemoval', 'PositiveValueRemoval']
saveNames = ['Tendency mean', 'Tendency median', 'Tendency 1s', 'Negative Value Removal', 'Positive Value Removal']

for m in modes:
    for nami, name in enumerate(names):
        for cNr, config in enumerate(allConfigs):
            configName = config.configName
            hyperss = config.hyperss
            hypNames= config.hypNames
            test_sizes = config.test_sizes
            topLevelss = config.topLevelss
            dataset = config.dataset
            symbols = config.symbols
            nrEmpty = config.nrEmpty
            andStack = config.andStack
            orStack = config.orStack
            xorStack = config.xorStack
            nrAnds = config.nrAnds
            nrOrs = config.nrOrs
            nrxor = config.nrxor
            trueIndexes = config.trueIndexes
            orOffSet = config.orOffSet
            xorOffSet = config.xorOffSet
            redaundantIndexes = config.redaundantIndexes
            batch_size = config.batch_size
        
            fig, axs = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True,
                                            figsize=(22, 6), layout="constrained")

            for c in [0,1]:
                for j, toplevels in enumerate(topLevelss[1:]):

                    ax = axs[c,j]

                    
                    width = 0
                    n_bars = 4
                    standardWidth = 0.9
                    bar_width = 0.8 / n_bars
                    barInd = 0
                    legend = []
                    rects = []
                    for ti, t in enumerate(['t0.5', 't0.8','t1.0','tbaseline']):
                        

                            x_offset = (barInd - n_bars / 2) * bar_width + bar_width / 2
                            barInd+= 1
                            resultVFs = []
                            resultVSs = []
                            width += standardWidth
                            


                            for k in ks:
                                resultVs = []

                                if k in ['performance', 'tree performance', 'treeImportances', 'baseline']:
                                    continue

                                if k == 'LRP-transformer_attribution cls':
                                    k = 'LRP-transformer_attribution'
                                    takeCls = True
                                else:
                                    takeCls = False

                                for hypers in hyperss:
                                    if takeCls and not hypers[-1]:
                                        continue
                                    elif not takeCls and hypers[-1]:
                                        continue
                                    for toplevel in toplevels:
                                        for test_size in test_sizes:
                                            modelType = hypers[0]
                                            epochs = hypers[1]
                                            dmodel = hypers[2]
                                            dfff = hypers[3]
                                            doSkip = hypers[4]
                                            doBn = hypers[5]
                                            header = hypers[6]
                                            numOfLayers = hypers[7]
                                            dropout = hypers[8]
                                            att_dropout = hypers[9]
                                            if modelType == 'Transformer':
                                                doClsTocken = hypers[10]
                                            else:
                                                doClsTocken = False

                                            dsName = str(dataset) +  ',l:' + str(toplevel) +',s:' +  str(symbols)+',e:' +  str(nrEmpty)+',a:' +  str(andStack)+',o:' +  str(orStack)+',x:' +  str(xorStack)+',na:' +  str(nrAnds)+',no:' +  str(nrOrs)+',nx:' +  str(nrxor)+',i:' +  arrayToString(trueIndexes)+',t:' +  str(test_size)+',oo:' +  str(orOffSet)+',xo:' +  str(xorOffSet) +',r:' + arrayToString(redaundantIndexes)

                                            for rFolder in filteredResultsFolder:


                                                saveName = pt.getWeightName(dsName, dataset, batch_size, epochs, numOfLayers, header, dmodel, dfff, dropout, att_dropout, doSkip, doBn, doClsTocken, learning = False, results = True, resultsPath=rFolder)

                                                if os.path.isfile(saveName + '.pkl'):
                                                    if saveName in allReses[rFolder].keys():
                                                        res = allReses[rFolder][saveName]
                                                    else:
                                                        try:
                                                            results = helper.load_obj(saveName)
                                                        except:
                                                            continue

                                                        res = dict()
                                                        for index, vv in np.ndenumerate(results):
                                                            res = vv

                                                        allReses[rFolder][saveName] = res
                                                else:
                                                    #print('NOT FOUND: ' + saveName)
                                                    continue
                                                if k not in res.keys():
                                                    continue
                                                if t not in res[k][m].keys():
                                                    continue
                                                v  = res[k][m][t][c][name]
                                                for i, val in enumerate(v):
                                                    
                                                    resultVs.append(np.nanmean(val)*100) 

                                resultVFs.append(np.nanmean(resultVs))
                                resultVSs.append(np.nanstd(resultVs))

                            lables= kNames
                                
                            counts = resultVFs
                            counts = np.where(np.isnan(counts),0,counts)
                            e = resultVSs
                            e = np.where(np.isnan(e),0,e)

                            legend.append(t)
                            ind = np.arange(len(counts))
                            rect = ax.bar(ind+x_offset , counts, bar_width, yerr=e, linestyle='None', capsize=3)
                            rects.append(rect)
                            ax.set_ylabel('Percent')
                            ax.set_xticks(ind)
                            ax.set_xticklabels(lables)
                            ax.set_title(saveNames[nami]+' Class ' + str(c) + '\n'+ topLevelNamesSmall[j].upper())
                            ax.tick_params(labelrotation=90)
                            ax.set_ylim(bottom=0, top=100)



            fig.legend(rects, labels=legend, 
                loc="upper right", bbox_to_anchor=(1.088, 0.85)) 
            fig.tight_layout()
            fig.suptitle(config.configName,fontsize=14)
            fig.subplots_adjust(top=0.84)
            specificFolder = folderGeneral + 'TendencyPerClass/'
            if not os.path.exists(specificFolder):
                os.makedirs(specificFolder)
            fig.savefig(specificFolder + saveNames[nami] + ' mode' + str(m) + ' ALL ' + configName+'.png', dpi = 300, bbox_inches = 'tight')

            plt.show() 
            plt.close()







# In[18]:





# In[19]:


tables = ['DoubleAssigmentTruthTableMinPercent', 'DoubleAssigmentFullPercent']
tabNames = ['Minimal-DCA', 'Full-DCA']

dcaComis = dict()
for kic in infoCombisS.keys():
    dcaComis[kic] = dict()
    for n in tables:
        dcaComis[kic][n] = dict()
        for k in ks:

            dcaComis[kic][n][k] = []
            for mode in range(np.max(modes)+1): 
                dcaComis[kic][n][k].append([])

for m in modes:
    for tabi, tab in enumerate(tables):
        
        fig, axs1 = plt.subplots(nrows=len(allConfigs), ncols=len(topLevelss[1:]), sharex=True, sharey=False,
                                            figsize=(24, 16), layout="constrained")
        plots = []
        lables = []
        width = 0
        if tab == "DoubleAssigmentFullPercent":
            n_bars = 3
        else:
            n_bars = 4
        standardWidth = 0.8
        bar_width = 0.8 / n_bars
        
        legend = []

        fullMincounts = [[],[],[],[],[],[],[],[],[],[],[],[],[]]
        barInd = 0

        for i, t in enumerate(['t1.0', 't0.8', 't0.5', 'tbaseline']):

            if t == 'tbaseline' and tab == 'DoubleAssigmentFullPercent':
                continue

            
            x_offset = (barInd - n_bars / 2) * bar_width + bar_width / 2
            barInd+= 1
            width += standardWidth
            #NOTE control plots here
            for testFac in [0,1]: # 0 = all; 1 = splitted data
                for accT in [0]: # 0= no acc threshold; 1= only 100% data
                    
                    #NOTE comment in or out to control plots
                    #if testFac != accT:
                    #    continue


                    for conI, config in enumerate(allConfigs):

                        hyperss = config.hyperss
                        configName = config.configName
                        hypNames= config.hypNames
                        test_size = config.test_sizes[testFac]
                        topLevelss = config.topLevelss
                        toplevels = config.toplevels
                        dataset = config.dataset
                        symbols = config.symbols
                        nrEmpty = config.nrEmpty
                        andStack = config.andStack
                        orStack = config.orStack
                        xorStack = config.xorStack
                        nrAnds = config.nrAnds
                        nrOrs = config.nrOrs
                        nrxor = config.nrxor
                        trueIndexes = config.trueIndexes
                        orOffSet = config.orOffSet
                        xorOffSet = config.xorOffSet
                        redaundantIndexes = config.redaundantIndexes
                        batch_size = config.batch_size
                        
                        stackNumbers = andStack * nrAnds + orStack * nrOrs + xorStack * nrxor
                        datasetSize = symbols**(andStack * nrAnds + orStack * nrOrs + xorStack * nrxor + nrEmpty)
                        
                        

                        for j, toplevels in enumerate(topLevelss[1:]):

                            ax1 = axs1[conI,j]
                            resultVs = []

                            
                            for c, k in enumerate(ks):
                                resultVs.append([])
                                if k in ['performance', 'tree performance', 'treeImportances']:
                                    continue

                                if k == 'LRP-transformer_attribution cls':
                                    k = 'LRP-transformer_attribution'
                                    takeCls = True
                                else:
                                    takeCls = False

                                for hypers in hyperss:
                                    if takeCls and not hypers[-1]:
                                        continue
                                    elif not takeCls and hypers[-1]:
                                        continue
                                    for toplevel in toplevels:
                                            if test_size == 0:
                                                test_size = int(0)
                                            modelType = hypers[0]
                                            epochs = hypers[1]
                                            dmodel = hypers[2]
                                            dfff = hypers[3]
                                            doSkip = hypers[4]
                                            doBn = hypers[5]
                                            header = hypers[6]
                                            numOfLayers = hypers[7]
                                            dropout = hypers[8]
                                            att_dropout = hypers[9]
                                            if modelType == 'Transformer':
                                                doClsTocken = hypers[10]
                                            else:
                                                doClsTocken = False

                                            dsName = str(dataset) +  ',l:' + str(toplevel) +',s:' +  str(symbols)+',e:' +  str(nrEmpty)+',a:' +  str(andStack)+',o:' +  str(orStack)+',x:' +  str(xorStack)+',na:' +  str(nrAnds)+',no:' +  str(nrOrs)+',nx:' +  str(nrxor)+',i:' +  arrayToString(trueIndexes)+',t:' +  str(test_size)+',oo:' +  str(orOffSet)+',xo:' +  str(xorOffSet) +',r:' + arrayToString(redaundantIndexes)

                                            for rFolder in filteredResultsFolder:

                                                saveName = pt.getWeightName(dsName, dataset, batch_size, epochs, numOfLayers, header, dmodel, dfff, dropout, att_dropout, doSkip, doBn, doClsTocken, learning = False, results = True, resultsPath=rFolder)

                                                if os.path.isfile(saveName + '.pkl'):
                                                    if saveName in allReses[rFolder].keys():
                                                        res = allReses[rFolder][saveName]
                                                    else:
                                                        try:
                                                            results = helper.load_obj(saveName)
                                                        except:
                                                            continue

                                                        res = dict()
                                                        for index, vv in np.ndenumerate(results):
                                                            res = vv

                                                        allReses[rFolder][saveName] = res
                                                else:
                                                    #print('NOT FOUND: ' + saveName)
                                                    continue
                                                
                                                if k not in res.keys():
                                                    continue
                                                if t not in res[k][m].keys(): 
                                                    continue
                                                for logic in res[k][m][t][tab].keys():
                                                    v = res[k][m][t][tab][logic]
                                                    for valII, val in enumerate(v):
                                                        if val == -1:
                                                            continue
                                                        if res[k]['performance'][valII][valII%5] < accT:
                                                            continue
                                                        #dFactor = datasetSize * test_size
                                                        #if dFactor == 0:
                                                        #    dFactor = datasetSize

                                                        resultVs[c].append(val*100)
                                                        #if t == 'tbaseline':
                                                        for kic in infoCombisS.keys():
                                                            if checkKIC(kic, 1, configName, toplevel):
                                                                if takeCls:
                                                                    dcaComis[kic][tab]['LRP-transformer_attribution cls'][m].append(val * 100)
                                                                else:
                                                                    dcaComis[kic][tab][k][m].append(val * 100)
                                
                            counts = []
                            e = []
                                
                            for vu in range(len(resultVs)):
                                counts.append(np.nanmean(np.array(resultVs[vu])))
                                e.append(np.nanstd(np.array(resultVs[vu])))


                            colors = ['tab:blue', 'orange','g', 'tab:red']
                            colorN =colors[i]
                            lableName = testFacNames[testFac] + '; ' + str(accTNames[accT]) + '; ' + t
                            if lableName not in legend:
                                legend.append(lableName)

                            ind = np.arange(len(counts))

                            if testFac == 1 and accT == 1:
                                p = ax1.errorbar(ind+x_offset, counts, marker='d', yerr=e,label=lableName, color=colorN, linestyle="none")
                            elif testFac == 1 and accT == 0:
                                p = ax1.errorbar(ind+x_offset, counts, marker='v', yerr=e,label=lableName, color=colorN, linestyle="none")
                            elif testFac == 0 and accT == 1:
                                p = ax1.errorbar(ind+x_offset, counts, marker='<', yerr=e,label=lableName, color=colorN, linestyle="none")
                            elif testFac == 0 and accT == 0:
                                p = ax1.errorbar(ind+x_offset, counts, marker='s', yerr=e,label=lableName, color=colorN, linestyle="none")

                            ax1.set_xticks(ind)
                            ax1.set_xticklabels(kNames)

                            plots.append(p)
                            ax1.tick_params(labelrotation=90)
                            ax1.set_ylim(bottom=0)
                
                            if j == 0 and conI==1:
                                ax1.set_ylabel(tabNames[tabi])
                            if conI == 0:
                                ax1.set_title(topLevelNamesSmall[j].upper() +'\n' + config.configName)
                            else:
                                ax1.set_title(config.configName)

        fig.legend(plots, labels=legend, 
                loc="upper right", bbox_to_anchor=(1.07, 0.75)) 

        specificFolder = folderGeneral + 'DCA/'
        if not os.path.exists(specificFolder):
            os.makedirs(specificFolder)
        fig.tight_layout()
        fig.savefig(specificFolder + ' ' + tabNames[tabi] + ' mode' + str(m) + 'OverConfigsAll.png', dpi = 300, bbox_inches = 'tight')
        plt.show() 
        plt.close()


splitNames = ['Split Test Set; Acc=100', 'Not Split Test; Acc>=0']
for ni, n in enumerate(tables):
    fig, axs = plt.subplots(nrows=1, ncols=len(infoCombisS.keys()), sharex=False, sharey=True,
                            figsize=(25, 4), layout="constrained")    
                            
    specificFolder = folderGeneral + 'DCARanking/'
    if not os.path.exists(specificFolder):
        os.makedirs(specificFolder)

    
    f = open(specificFolder+ n + "-rankingVs.txt", "a")
    f.write(n)
    f.write("\n")
    f.write("full 	" + "	".join(str(x) for x in kNames))
    f.write("\n")
    f.close()
    f = open(specificFolder+ n + "-ranking.txt", "a")
    f.write(n)
    f.write("\n")
    f.write("full 	" + "	".join(str(x) for x in kNames))
    f.close()
    
    for kici, kic in enumerate(infoCombisS.keys()):
    
        ax = axs[kici]
        resultVs = []
        resultSs = []

        lables = []
        
        for ki, k in enumerate(ks):

            data = dcaComis[kic][n][k] 

            m = getBestMode(nibComis, k)
            lables.append(kNames[ki])# + ' ' + str(m))
            resultVs.append(np.nanmean(data[m]))
            resultSs.append(np.nanstd(data[m]))

        resultVs = np.array(resultVs)
        resultSs = np.array(resultSs)

        ranks = rankdata(resultVs, method='min')
        f = open(specificFolder+ n + "-rankingVs.txt", "a")
        f.write(kic + "	")
        f.write("	".join(str(x) for x in resultVs))
        f.write("\n")
        f.close()

        f = open(specificFolder+ n + "-ranking.txt", "a")
        f.write(kic + "	")
        f.write("	".join(str(x) for x in ranks))
        f.write("\n")
        f.close()

        sortI = np.argsort(resultVs)
        counts = resultVs[sortI]
        e = resultSs[sortI]

        lables = np.array(lables)[sortI]
            

        ind = np.arange(len(counts))
        rect = ax.bar(ind , counts, yerr=e, linestyle='None', capsize=3)
        if kici == 0:
            ax.set_ylabel('Percent')
        ax.set_xticks(ind)
        ax.set_xticklabels(lables)
        ax.set_title(kic)#)
        ax.tick_params(labelrotation=90)
        #for tick in ax.get_xticklabels():
        #    tick.set_rotation(45)
        #    tick.set_ha('right')
        ax.set_ylim(bottom=0, top=100)
        
        f = open(specificFolder+ n + "-rankingVs.txt", "a")
        f.write("\n")
        f.close()

        f = open(specificFolder+ n + "-ranking.txt", "a")
        f.write("\n")
        f.close()
    fig.suptitle(tabNames[ni],fontsize=14)

    fig.savefig(specificFolder + n +'-ranking.png', dpi = 300, bbox_inches = 'tight')

    plt.show()
    plt.close()

# In[20]:


specificFolder = folderGeneral + 'numberOfExperiments/'
if not os.path.exists(specificFolder):
    os.makedirs(specificFolder)
f = open(specificFolder+ "experimentNumbers.txt", "a")
f.write("---------")
f.write("\n")
f.write("numberBaseModelsFull")
f.write("\n")
f.write(str(numberBaseModelsFull))
f.write("\n")
f.write("numberOverallBaseModels")
f.write("\n")
f.write(str(numberOverallBaseModels))
f.write("\n")
f.write("numberModelsTrainedOverall")
f.write("\n")
f.write(str(numberModelsTrainedOverall))
f.write("\n")

f.close()
