import numpy as np
from collections import defaultdict
from modules import helper
from sklearn import metrics
from datetime import datetime
from joblib import Parallel, delayed

import itertools


#nestest dict for saves
def nested_dict(n, type):
    if n == 1:
        return defaultdict(type)
    else:
        return defaultdict(lambda: nested_dict(n-1, type))

def getddList():
    return defaultdict(list)

def getddlv2():
    return defaultdict(getddList)

def nested_dict_static():
    return defaultdict(getddlv2)

def makeGTM(attentionQ, x_train1, y_train1, num_of_classes, valuesA, reducePredictions=True, addOne=True, threshold=-1, doThreshold=False, doFidelity = False):
    #TODO check if data has right shape!
    data_att = attentionQ[0][0]
    if reducePredictions:
        if addOne:
            predictions = np.argmax(y_train1,axis=1) +1
        else:
            predictions = np.argmax(y_train1,axis=1) 
    else:
        predictions = y_train1

    if addOne:
        endPoint = num_of_classes+1
    else:
        endPoint = num_of_classes

    rMA = nested_dict_static()
    rM = nested_dict_static()

    for lable in range(0,endPoint):
            for toL in valuesA:
                if len(rMA[lable]['x'][toL]) is 0:
                    rMA[lable]['x'][toL] = np.zeros((len(data_att)))
                    rM[lable]['x'][toL] = np.zeros((len(data_att)))

                if len(rMA[lable]['xAvg'][toL]) is 0:
                    rMA[lable]['xAvg'][toL] = np.zeros((len(data_att)))

    x_train1s = np.array(x_train1).squeeze()

    if doThreshold:
            maxHeat = np.average(data_att)
            borderHeat = maxHeat/threshold
    else:
        borderHeat = 0

    for di, data_att in enumerate(attentionQ[0]):
        X_ori = x_train1s[di]
        predictionsI = predictions[di]
        for i in range(len(data_att)):
            if X_ori[i] not in valuesA:
                continue
            if data_att[i] != 0 and validataHeat(data_att[i], borderHeat, doFidelity):
                label = predictionsI
                rM[label]['x'][X_ori[i]][i] += 1 
                rMA[label]['x'][X_ori[i]][i] += data_att[i]

    for label in rMA.keys():
        for fromV in valuesA:
            for i in range(len(data_att)):
                if rM[label]['x'][fromV][i] != 0:
                    rMA[label]['xAvg'][fromV][i] = data_att[i]/rM[label]['x'][fromV][i]
    return rMA, rM

#create groundwork for all types of GCRs
def makeAttention(attentionQ, x_train1, y_train1, order, step1, step2, num_of_classes, valuesA, reductionName="MixedClasses", doThreshold=False, doFidelity = False, doMax=False, doPenalty=False, threshold = -1, penaltyMode = 'entropy', reducePredictions=True, addOne=True, do3DData=True, do2DData=False):

    methodStart = datetime.now()

    #predicted lables
    #attentionQ = outSax[9]
    
    if(order == 'lh'):
        axis1 = 0
        axis2 = 1
    elif(order == 'hl'):
        axis1 = 2
        axis2 = 0
    else:
        axis1 = 0
        axis2 = 1


    if do3DData:
        attentionQ[1] = helper.doCombiStep(step1, attentionQ[2], axis1)
        attentionQ[1] = helper.doCombiStep(step2, attentionQ[1], axis2)
        attentionQ[0] = helper.doCombiStep(step2, attentionQ[1], 1) 
    elif do2DData:
        attentionQ[1] = attentionQ[2]
        attentionQ[0] = helper.doCombiStep(step2, attentionQ[1], 1)
    else:
        attentionN = []
        for a in attentionQ[2]:
            attentionN.append([])
            for b in range(len(a)):
                #newAdd = []
                #for c in range(len(a)):
                #    newAdd.append(a[b])
                #attentionN[-1].append(newAdd)
                attentionN[-1].append(a)
        attentionQ[1] = np.array(attentionN)
        attentionQ[0] = attentionQ[2]
    

        
        #attentionN = []
        #for i, a in enumerate(attentionQ[2]):   
        #    
        #    c=np.transpose(np.array(a))
        #    for b in range(len(c)):
        #        attentionN.append(a * c)
        #attentionQ[1] = np.array(attentionN)


        #attentionN = []
        #for a in attentionQ[2]:
        #    attentionN.append([])
        #    for b in range(len(a)):
        #        attentionN[-1].append(a.squeeze())
        #
        #attentionN = np.array(attentionN)

        #print(attentionN[0])
        #print(attentionN[0,:,0].shape)
        #print(attentionQ[2][0].shape)

        #for i, a in enumerate(attentionQ[2]):   
        #    
        #    c=np.transpose(np.array(a).squeeze())
        #    for b in range(len(c)):
        #        attentionN[i,:,b] = attentionN[i,:,b] * c
        #attentionQ[1] = np.array(attentionN)


    #true lables
    if reducePredictions:
        if addOne:
            predictions = np.argmax(y_train1,axis=1) +1
        else:
            predictions = np.argmax(y_train1,axis=1) 
    else:
        predictions = y_train1

    #position counter
    rM = nested_dict_static()
    #attention sum at each point
    rMS = nested_dict_static()
    #relative average at each point + more side combinations
    rMA = nested_dict_static()
    #penalty buffer
    rMP = nested_dict_static()
    
    labelSet = set(predictions)
    laCount = dict()
    entropyDic = dict()
    entropyDicRelativeMult = dict()
    entropyDicRelativeDiv = dict()
    countingConstant = (1 * (num_of_classes + 1))
    if doPenalty:
        for la in labelSet:
            laCount[la] = predictions.tolist().count(la)
            part = laCount[la] / len(predictions)
            entropy = -(part * np.log(part))
            entropyDic[la] = entropy
            entropyDicRelativeMult[la] = entropy * num_of_classes
            entropyDicRelativeDiv[la] = (1+num_of_classes)/entropy
            #entropyDicRelativeMult[la] = entropy# * num_of_classes
            #entropyDicRelativeDiv[la] = (1+num_of_classes)/entropy
            #entropyDicRelativeMult[la] = entropy#/ num_of_classes
            #entropyDicRelativeDiv[la] = (1+num_of_classes) * entropy
    
    data_att = attentionQ[1][0]

    print("starting default dict" + str(datetime.now() - methodStart))

    #print(data_att.shape)
    if addOne:
        endPoint = num_of_classes+1
    else:
        endPoint = num_of_classes
    for lable in range(0,endPoint):
        

            for toL in valuesA:

                for fromL in valuesA:
                    if(len(rM[lable][fromL][toL]) is 0):
                        rM[lable][fromL][toL] = np.zeros((len(data_att), len(data_att[0])))
                        rMS[lable][fromL][toL] = np.zeros((len(data_att), len(data_att[0])))
                        rMP[str(lable)+"pen"][fromL][toL] = np.zeros((len(data_att), len(data_att[0])))
                        rMA[lable][fromL][toL] = np.zeros((len(data_att), len(data_att[0])))

                if len(rMA[lable]['x'][toL]) is 0:
                    rMA[lable]['x'][toL] = np.zeros((len(data_att), len(data_att[0])))
                    rMP[str(lable)+"pen"]['x'][toL] = np.zeros((len(data_att), len(data_att[0])))

                if len(rMA[lable]['xAvg'][toL]) is 0:
                    rMA[lable]['xAvg'][toL] = np.zeros((len(data_att), len(data_att[0])))

    #put together all train attention to from symbol x to symbol y representation
    z = 0
    x_train1s = np.array(x_train1).squeeze()

    if doThreshold:
        if doMax:
            maxHeat = np.max(data_att)
            borderHeat = maxHeat / threshold
        else:
            maxHeat = np.average(data_att)
            borderHeat = maxHeat/threshold# maxHeat/1.6
        # if predictions[index] == 1:
        #     borderHeat = maxHeat/1.7
        # if predictions[index] == 2:
        #     borderHeat = 0 
        # borderHeat2 = maxHeat/1.65
    else:
        borderHeat = 0
    
    print("starting attention adding" + str(datetime.now() - methodStart))

    #not parallel
    #for index in range(len(attentionQ[1])):
    #    sumAttention(index, attentionQ[1][index], x_train1s[index], borderHeat, doFidelity, rM, rMP, rMA, rMS, predictions[index], doPenalty, penaltyMode)
    Parallel(n_jobs=1, require='sharedmem', prefer="threads")(delayed(sumAttention)(index, attentionQ[1][index], x_train1s[index], valuesA, borderHeat, doFidelity, rM, rMP, rMA, rMS, predictions[index], doPenalty, penaltyMode) for index in range(len(attentionQ[1]))) #avgScore
    
    #not parallel
    #for index in range(len(attentionQ[1])):
    #    sumAttention(index, attentionQ, x_train1s, borderHeat, doFidelity, rM, rMP, rMA, rMS, predictions, doPenalty, penaltyMode)

    print("starting making relative attention" + str(datetime.now() - methodStart))

    #not parallel
    #for lable in rMA.keys():
    #    relativeAttentionMaking(lable, valuesA, data_att, rM, rMS, rMA, rMP, doPenalty, penaltyMode, entropyDicRelativeDiv, entropyDicRelativeMult, laCount, countingConstant)
    
    Parallel(n_jobs=1, require='sharedmem', prefer="threads")(delayed(relativeAttentionMaking)(lable, valuesA, data_att, rM, rMS, rMA, rMP, doPenalty, penaltyMode, entropyDicRelativeDiv, entropyDicRelativeMult, laCount, countingConstant) for lable in rMA.keys())

    rMA2, rM2 = makeGTM(attentionQ, x_train1, y_train1, num_of_classes, valuesA, reducePredictions=reducePredictions, addOne=addOne, threshold=-threshold, doThreshold=doThreshold, doFidelity = doFidelity)
    for lable in rMA2.keys():
        rMA[lable]['1DSum'] = rMA2[lable]['x']
        rMA[lable]['1DSum+'] = rMA2[lable]['xAvg']

    if reductionName == "DiffNormalize":
        rMS = diffNormalizeFCAM(rMS,valuesA, doOVerMax=True)
        rMA = diffNormalizeFCAM(rMA,valuesA, doOVerMax=True)
    elif reductionName=="OverMax":
        rMS = scaleFCAMOverMax(rMS,valuesA,globalMax=False)
        rMA = scaleFCAMOverMax(rMA,valuesA,globalMax=False)
    elif reductionName=="PerClass":
        rMS = scaleFCAMPerClass(rMS,valuesA)
        rMA = scaleFCAMPerClass(rMA,valuesA)
    elif reductionName=="BetweenClasses":
        rMS = scaleFCAMBetweenClasses(rMS,valuesA,globalMax=True)
        rMA = scaleFCAMBetweenClasses(rMA,valuesA,globalMax=True)
    elif reductionName=="MixedClasses":
        rMS = scaleFCAMMixClasses(rMS,valuesA,globalMax=False)
        rMA = scaleFCAMMixClasses(rMA,valuesA,globalMax=False)

    print('done' + str(datetime.now() - methodStart))


    return rMA, rMS, rM

def getAllReductionNames():
    #return ['DiffNormalize', 'OverMax', 'PerClass', 'BetweenClasses', 'MixedClasses']
    return ['OverMax','DiffNormalize','PerClass', 'BetweenClasses', 'MixedClasses']

def getAllNeededReductionNames():
    #return ['DiffNormalize', 'OverMax', 'PerClass', 'BetweenClasses', 'MixedClasses']
    #return ['OverMax','PerClass', 'BetweenClasses', 'MixedClasses', 'MixedClasses2']
    return ['PerClass', 'BetweenClasses', 'MixedClasses']


# relativate the summed scores
def relativeAttentionMaking(lable, valuesA, data_att, rM, rMS, rMA, rMP, doPenalty, penaltyMode, entropyDicRelativeDiv, entropyDicRelativeMult, laCount, countingConstant, doAgregation1D=False):
    #minV, maxV = getMinMax(rMS, lable)
    for toL in valuesA:

        for fromL in valuesA:
            #rMS[lable][fromL][toL] = (rMS[lable][fromL][toL] - minV) / (maxV - minV)
            #X_scaled = X_std * (1 - 0) + 0
            for i in range(len(data_att)):
                for j in range(len(data_att[i])): 
                    #FCAM r. average
                    if rM[lable][fromL][toL][i][j] > 0:
                        aAdder = 0
                        if doPenalty:
                            if(penaltyMode == "entropy"):
                                entropyRelativeDiv = entropyDicRelativeDiv[lable]
                                divver = rMP[str(lable)+"pen"][fromL][toL][i][j] / float(rM[lable][fromL][toL][i][j])
                                aAdder = entropyRelativeDiv * divver
                                adder = entropyRelativeDiv * rMP[str(lable)+"pen"][fromL][toL][i][j]

                                entropyRelativeMult = entropyDicRelativeMult[lable]
                                aSubber = divver / entropyRelativeMult # divver * entropyRelativeMult when mult is only : entropy
                                subber = rMP[str(lable)+"pen"][fromL][toL][i][j] / entropyRelativeMult

                                
                                rMS[lable][fromL][toL][i][j] += adder
                                rMA[lable]['x'][toL][i][j] += adder
                                for innerLable in rMA.keys():
                                    rMS[innerLable][fromL][toL][i][j] -= subber
                                    rMA[innerLable][fromL][toL][i][j] -= aSubber
                                    rMA[innerLable]['x'][toL][i][j] -= subber
                                    rMA[innerLable]['xAvg'][toL][i][j] -= aSubber
                                    
                            else:
                                divPart = rMP[str(lable)+"pen"][fromL][toL][i][j]/ laCount[lable]
                                adder =  countingConstant * divPart
                                aAdder = adder/float(rM[lable][fromL][toL][i][j])

                                aSubber = divPart / float(rM[lable][fromL][toL][i][j])
                                subber = divPart

                                rMS[lable][fromL][toL][i][j] += adder
                                rMA[lable]['x'][toL][i][j] += adder
                                for innerLable in rMA.keys():
                                    rMS[innerLable][fromL][toL][i][j] -= subber
                                    rMA[innerLable][fromL][toL][i][j] -= aSubber
                                    rMA[innerLable]['x'][toL][i][j] -= subber
                                    rMA[innerLable]['xAvg'][toL][i][j] -= aSubber
                        else:
                            aAdder = rMS[lable][fromL][toL][i][j] / float(rM[lable][fromL][toL][i][j])
                        
                        rMA[lable][fromL][toL][i][j] += aAdder
                        rMA[lable]['xAvg'][toL][i][j] += aAdder
                        rMA[lable]['xAvg'][toL][i][j] += aAdder

                    

                        
        if doAgregation1D:
            #GTM max of sum                
            rMA[lable]['max'][toL] = np.max(rMA[lable]['x'][toL], axis=0) 
            #GTM average of sum         
            rMA[lable]['average'][toL] = np.mean(rMA[lable]['x'][toL], axis=0) 
            #GTM median of sum         
            rMA[lable]['median'][toL] = np.median(rMA[lable]['x'][toL], axis=0) 
            #GTM max of r.average          
            rMA[lable]['max+'][toL] = np.max(rMA[lable]['xAvg'][toL], axis=0)  
            #GTM average of r.average          
            rMA[lable]['average+'][toL] = np.mean(rMA[lable]['xAvg'][toL], axis=0)
            #GTM median of r.average         
            rMA[lable]['median+'][toL] = np.median(rMA[lable]['xAvg'][toL], axis=0)


#making attention dicts and np arrays (not supporting penatly yet)
def fastMakeAttention(attentionQ, index_trains, y_train1, combis, ranges, order, step1, step2, num_of_classes, valuesA, mode=0, reductionName="MixedClasses",ignoreMaskedValue=False, doFidelity = False, doMax=False, doPenalty=False, threshold = -1, penaltyMode = 'entropy', reducePredictions=True, addOne=False, do3DData=True, do2DData=False, doAgregation1D=False):

    methodStart = datetime.now()
    print('Making GCR')
    if(order == 'lh'):
        axis1 = 0
        axis2 = 1
    elif(order == 'hl'):
        axis1 = 2
        axis2 = 0
    else:
        axis1 = 0
        axis2 = 1


    if do3DData:
        print("starting combiSteps"+ str(datetime.now() - methodStart))
        attentionQ[1] = helper.doCombiStep(step1, attentionQ[2].flatten(), axis1)
        attentionQ[1] = helper.doCombiStep(step2, attentionQ[1], axis2)
        attentionQ[0] = helper.doCombiStep(step2, attentionQ[1], 1) 
        print("finished combiSteps" + str(datetime.now() - methodStart))    
    elif do2DData:
        attentionQ[1] = attentionQ[2]
        attentionQ[0] = helper.doCombiStep(step2, attentionQ[1], 1)
    else:
        print('Start 1d')
        attentionN = []
        for a in attentionQ[2]:
            attentionN.append([])
            for b in range(len(a)):
                attentionN[-1].append(a)
        attentionQ[1] = np.array(attentionN)
        attentionQ[0] = attentionQ[2]
        print('End 1d')


    #true lables
    if reducePredictions:
        if addOne:
            predictions = np.argmax(y_train1,axis=1) +1
        else:
            predictions = np.argmax(y_train1,axis=1) 
    else:
        predictions = y_train1

    data_att = attentionQ[1][0]

    rMS = np.zeros([num_of_classes, len(valuesA), len(valuesA), len(data_att),len(data_att)])
    rMA = np.zeros([num_of_classes, len(valuesA), len(valuesA), len(data_att),len(data_att)])
    rM = rMS.copy()



    #lables müssten alle lables sein zum entsprechenden index!
    #indexes müsste ne range sein für alle Einträge! (und data_Att entsprechend fullAttention)
    #Mode 0: take values as they are
    #Mode 1: Ignore 0 values as they represent not occuring
    #Mode 2: Ignore everything <= 0 as it is not relevant for the class
    #Mode 3: Use negative values as abs values!
    #Mode 5: Aktive the individual thresholdclassFullAttFast

    if mode == 0:
        #rMS[combis[:,0],combis[:,1],combis[:,2],ranges[:,1],ranges[:,2]] += attentionQ[1][ranges[:,0],ranges[:,1],ranges[:,2]]
        np.add.at(rMS, [combis[:,0],combis[:,1],combis[:,2],ranges[:,1],ranges[:,2]],attentionQ[1][ranges[:,0],ranges[:,1],ranges[:,2]])
        np.add.at(rM, [combis[:,0],combis[:,1],combis[:,2],ranges[:,1],ranges[:,2]],1)
        #rM[combis[:,0],combis[:,1],combis[:,2],ranges[:,1],ranges[:,2]] += 1
    elif mode == 1:
        attentionAdd = attentionQ[1][ranges[:,0],ranges[:,1],ranges[:,2]]
        attentionCon = attentionAdd != 0
        rmAdd = np.ones(attentionAdd.shape)
        #rMS[combis[:,0],combis[:,1],combis[:,2],ranges[:,1],ranges[:,2]] += attentionAdd * attentionCon
        #rM[combis[:,0],combis[:,1],combis[:,2],ranges[:,1],ranges[:,2]] += rmAdd * attentionCon
        np.add.at(rMS, [combis[:,0],combis[:,1],combis[:,2],ranges[:,1],ranges[:,2]],attentionAdd * attentionCon)
        np.add.at(rM, [combis[:,0],combis[:,1],combis[:,2],ranges[:,1],ranges[:,2]],rmAdd*attentionCon)

    elif mode == 2:
        attentionAdd = attentionQ[1][ranges[:,0],ranges[:,1],ranges[:,2]]
        attentionCon = attentionAdd >= 0
        rmAdd = np.ones(attentionAdd.shape)
        np.add.at(rMS, [combis[:,0],combis[:,1],combis[:,2],ranges[:,1],ranges[:,2]],attentionAdd * attentionCon)
        np.add.at(rM, [combis[:,0],combis[:,1],combis[:,2],ranges[:,1],ranges[:,2]],rmAdd*attentionCon)
    elif mode==3:
        attentionQc = np.absolute(attentionQ[1])
        np.add.at(rMS, [combis[:,0],combis[:,1],combis[:,2],ranges[:,1],ranges[:,2]],attentionQc[ranges[:,0],ranges[:,1],ranges[:,2]])
        np.add.at(rM, [combis[:,0],combis[:,1],combis[:,2],ranges[:,1],ranges[:,2]],1)

    elif mode==4:
        attentionAdd = attentionQ[1][ranges[:,0],ranges[:,1],ranges[:,2]]
        attentionCon = attentionAdd >= threshold
        rmAdd = np.ones(attentionAdd.shape)
        np.add.at(rMS, [combis[:,0],combis[:,1],combis[:,2],ranges[:,1],ranges[:,2]],attentionAdd * attentionCon)
        np.add.at(rM, [combis[:,0],combis[:,1],combis[:,2],ranges[:,1],ranges[:,2]],rmAdd*attentionCon)
    notZero = np.where(rM != 0)

    
    rMA[notZero] = rMS[notZero]/rM[notZero]



    gtms = dict()
    if doAgregation1D:
        gtms['max'] = np.zeros([num_of_classes, len(valuesA), len(data_att)])
        gtms['average'] = np.zeros([num_of_classes, len(valuesA), len(data_att)])
        gtms['median'] = np.zeros([num_of_classes, len(valuesA), len(data_att)])
        gtms['max+'] = np.zeros([num_of_classes, len(valuesA), len(data_att)])
        gtms['average+'] = np.zeros([num_of_classes, len(valuesA), len(data_att)])
        gtms['median+'] = np.zeros([num_of_classes, len(valuesA), len(data_att)])

    gtms['1DSum'] = np.zeros([num_of_classes, len(valuesA), len(data_att)])
    gtms['1DSum+'] = np.zeros([num_of_classes, len(valuesA), len(data_att)])
    rMGTM = np.zeros([num_of_classes, len(valuesA), len(data_att)])

    if doAgregation1D:
        gtms['max'] = np.max(rMS, axis=(1,3))
        gtms['average'] = np.average(rMS, axis=(1,3))
        gtms['median'] =np.median(rMS, axis=(1,3))
        rMC = np.sum(rM, axis=(1,3))
        notZeroC = np.where(rMC != 0)
        gtms['max+'][notZeroC] = gtms['max'][notZeroC]/rMC[notZeroC]
        gtms['average+'][notZeroC] = gtms['average'][notZeroC]/rMC[notZeroC]
        gtms['median+'][notZeroC] = gtms['median'][notZeroC]/rMC[notZeroC]


    combisGTM = [list(itertools.product([predictions[i]],s)) for i, s in enumerate(index_trains)]
    combisGTM = np.concatenate(np.array(combisGTM))
    rangesSmall = np.array(list(itertools.product(range(len(y_train1)),range(len(data_att)))))


    np.add.at(gtms['1DSum'], [combisGTM[:,0], combisGTM[:,1],rangesSmall[:,1]], attentionQ[0][rangesSmall[:,0],rangesSmall[:,1]])
    np.add.at(rMGTM,[combisGTM[:,0], combisGTM[:,1], rangesSmall[:,1]], 1)
    notZero = np.where(rMGTM != 0)
    gtms['1DSum+'][notZero] = gtms['1DSum'][notZero]/rMGTM[notZero]

    """
    if ignoreMaskedValue:
        rMA[:,:,-1,:,:] = 0
        rMS[:,:,-1,:,:] = 0
        rM[:,:,:-1,:,:] = 0
        rMA[:,-1,:,:,:] = 0
        rMS[:,-1,:,:,:] = 0
        rM[:,-1,:,:,:] = 0
        for k in gtms.keys():
            gtms[k][:,-1,:] = 0
            rMGTM[:,-1,:] = 0
    """
    if ignoreMaskedValue:
        rMA = rMA[:,:-1,:-1,:,:] 
        rMS = rMS[:,:-1,:-1,:,:] 
        rM = rM[:,:-1,:-1,:,:] 

        for k in gtms.keys():
            gtms[k] = gtms[k][:,:-1,:]
        rMGTM = rMGTM[:,:-1,:]

    if reductionName == "DiffNormalize":
        rMS = diffNormalizeFCAMNP(rMS, doOVerMax=True)
        rMA = diffNormalizeFCAMNP(rMA, doOVerMax=True)
    elif reductionName=="OverMax":
        rMS = scaleFCAMOverMaxNP(rMS,globalMax=False)
        rMA = scaleFCAMOverMaxNP(rMA,globalMax=False)
    elif reductionName=="PerClass":
        rMS = scaleFCAMPerClassNP(rMS)
        rMA = scaleFCAMPerClassNP(rMA)
    elif reductionName=="BetweenClasses":
        rMS = scaleFCAMBetweenClassesNP(rMS)
        rMA = scaleFCAMBetweenClassesNP(rMA)
    elif reductionName=="MixedClasses":
        rMS = scaleFCAMMixClassesNP(rMS)
        rMA = scaleFCAMMixClassesNP(rMA)
    elif reductionName=="MixedClasses2":
        rMS = scaleFCAMMixClassesNP(rMS, scalingMin=True)
        rMA = scaleFCAMMixClassesNP(rMA, scalingMin=True)


    for rst in gtms.keys():
        if reductionName == "DiffNormalize":
            gtms[rst] = diffNormalizeGTMNP(gtms[rst],doOVerMax=True)
        elif reductionName=="OverMax":
            gtms[rst] = scaleGTMOverMaxNP(gtms[rst],globalMax=False)
        elif reductionName=="PerClass":
            gtms[rst] = scaleGTMPerClassNP(gtms[rst])
        elif reductionName=="BetweenClasses":
            gtms[rst] = scaleGTMBetweenClassesNP(gtms[rst])
        elif reductionName=="MixedClasses":
            gtms[rst] = scaleGTMMixClassesNP(gtms[rst])



    #NOTE penalties are not supported atm

    return rMA, rMS, rM, gtms, rMGTM

# sum attention values together into gcr format
def sumAttention(index, data_att, data_word, valuesA, borderHeat, doFidelity, rM, rMP, rMA, rMS, predictionsI, doPenalty, penaltyMode, retry = 0, doAvgScore=False): #avgScore
    X_ori = data_word

    for i in range(len(data_att)):
        if data_word[i] not in valuesA:
            continue
        for j in range(len(data_att[i])):
            if data_word[j] not in valuesA:
                continue
            if True or (data_att[i][j] != 0 and validataHeat(data_att[i][j], borderHeat, doFidelity)):
                label = predictionsI
                rM[label][X_ori[i]][X_ori[j]][i][j] += 1 
                
                if doPenalty:

                    if(penaltyMode == 'entropy'):
                        rMP[str(label)+"pen"][X_ori[i]][X_ori[j]][i][j] += data_att[i][j]

                    else:
                        rMP[str(label)+"pen"][X_ori[i]][X_ori[j]][i][j] += data_att[i][j]

                else:
                    rMS[label][X_ori[i]][X_ori[j]][i][j] += data_att[i][j]
                    rMA[label]['x'][X_ori[j]][i][j] += data_att[i][j]
            #if data_att[i][j] > borderHeat2:
            #    #sum FCAM
            #    rMS[predictions[index]][X_ori[i]][X_ori[j]][i][j] += data_att[i][j] / 1.5
            #    #CRCAM Sum
            #    rMA[predictions[index]]['x'][X_ori[j]][i][j] += data_att[i][j] / 1.5
            
        

# sum class scores
# thresholds müssen sortiert sein
def sumLabelScores(saliency,label, indList, rMG, combis, ranges, rM, useThreshold=False, thresholds=[0], useRM=True, globalT=True):
    """
    
    indListDel = list(range(lenTrial))
    indListDefault = []
    defaultV = 0
    #l1 = rMG[label]
    trial = trial.squeeze()
    #rMGc = rMG.copy()
    
    #print(rMGKeys)
    for fromVi in indList:
        if trial[fromVi] not in rM[label].keys():
            indListDel.remove(fromVi)
        elif useRM and rM != None and rM[label][trial[fromVi]][trial[fromVi]][fromVi][fromVi] == 0:
            indListDefault.append(fromVi)
            indListDel.remove(fromVi)

    lenTrialSq = lenTrial*(lenTrial-len(indListDel)) 
    defaultV = 0
    for ind in indListDefault:
        if trial[ind] in rM[label].keys():
            defaultV = rMG[label,index_Trial[ind],index_Trial[ind],ind,ind]
            break
    """

    if useThreshold:
        tlne = len(thresholds)

        tScores = np.zeros(tlne)

        if globalT:
            thresholdVs = thresholds
        else:
            for ti, t in enumerate(thresholds):

                thresholdVs = np.zeros(tlne)

                trialMin = np.min(saliency)
                trialMax = np.max(saliency)
                thresholdVs[ti] = (trialMax-trialMin) * t + trialMin

    	
        #trialy = enumerate(np.array(trial)[indListDefault])
        #trialy = enumerate(trial)

        #for ti, t in enumerate(thresholds):
        #    tScores[ti] = np.sum([l1[fromV][toV][fromVi][toVi] if l1[fromV][toV][fromVi][toVi] >= t else 0 for toVi, toV in trialy for fromVi, fromV in trialy])
        return sumScoresThresholds(rMG, label, combis, ranges, tScores, thresholdVs)
        """
        for fromVi in indListDel:
            #fromV = trial[fromVi]
            #l2 = l1[fromV]
            for toVi in indListDel:
                #toV = trial[toVi]
                value = rMG[label][rMGKeys[trial[fromVi]]][rMGKeys[trial[toVi]]][fromVi][toVi]#l2[toV][fromVi][toVi]
                if value is 0:
                    continue
                for ti, t in enumerate(thresholdVs):
                    if value >= t:
                        tScores[ti] += value
                    else:
                        break
                #lableScore += l1[trial[fromVi]][trial[toVi]][fromVi][toVi]
        if useRM and defaultV > 0:
            for ti, tValues in enumerate(thresholdVs):
                if defaultV >= tValues:
                    tScores[ti] += defaultV * (lenTrial**2 -1)

        return (tScores, label)"""
    else:
        return sumScores(rMG, label, combis, ranges)
        """
        lableScore = 0
        for fromVi in indListDel:
            #fromV = trial[fromVi]
            #l2 = l1[fromV]
            for toVi in indListDel:
                #toV = trial[toVi]
                #value = l2[toV][fromVi][toVi]
                lableScore += rMG[label][rMGKeys[trial[fromVi]]][rMGKeys[trial[toVi]]][fromVi][toVi]#value
                #lableScore += l1[trial[fromVi]][trial[toVi]][fromVi][toVi]
        if useRM:
            lableScore += defaultV * (lenTrial**2 -1)
        return (lableScore, label)
        """

# sum class scores for penalty GCRs
def sumLabelScoresPenalty(trial, label, indList, rMG):
    lableScore = 0
    l1 = rMG[label]

    trial = trial.squeeze()

    for fromVi in indList:
        #fromV = trial[fromVi]
        #l2 = l1[trial[fromVi]]
        for toVi in indList:
            #toV = trial[toVi]
            #value = l1[trial[fromVi]][trial[toVi]][fromVi][toVi]
            lableScore += l1[trial[fromVi]][trial[toVi]][fromVi][toVi]
    return (lableScore, label)

#@jit(nopython=True)
def sumScores(rMG, label, combis, ranges):

    
    #print(rMG.shape)
    lableScore = np.sum(rMG[label,combis[:,0],combis[:,1],ranges[:,0],ranges[:,1]])
    #lableScore = (rMG[label][index_Trial[fromVi]][index_Trial[toVi]][fromVi][toVi] for fromVi in indListDel for toVi in indListDel)

    #lableScore = 0
    #for fromVi in indListDel:
        #fromV = trial[fromVi]
        #l2 = l1[fromV]
    #    for toVi in indListDel:
            #toV = trial[toVi]
            #value = l2[toV][fromVi][toVi]
    #        lableScore += rMG[label][index_Trial[fromVi]][index_Trial[toVi]][fromVi][toVi]#value
            #lableScore += l1[trial[fromVi]][trial[toVi]][fromVi][toVi]
    #lableScore = np.sum(list(lableScore))
    return (lableScore, label)

import itertools as it
#@jit(nopython=True)
def sumScoresThresholds(rMG, label, combis, ranges, tScores, thresholdVs):
        #(rMG[label][index_Trial[fromVi]][index_Trial[toVi]][fromVi][toVi] if rMG[label][index_Trial[fromVi]][index_Trial[toVi]][fromVi][toVi] >= t else 0 for fromVi in indListDel for toVi in indListDel)
        #lableScore = (rMG[label][index_Trial[fromVi]][index_Trial[toVi]][fromVi][toVi] for fromVi in indListDel for toVi in indListDel)
        #for ti, t in enumerate(thresholdVs):
        #    tScores[ti] = np.sum(list(it.filterfalse(lambda x : x<t, lableScore)))
        #for fromVi in indListDel:
            #fromV = trial[fromVi]
            #l2 = l1[fromV]
        #    for toVi in indListDel:
                #toV = trial[toVi]
        #        value = rMG[label][index_Trial[fromVi]][index_Trial[toVi]][fromVi][toVi]#l2[toV][fromVi][toVi]
        #        if value is 0:
        #            continue
        #        for ti, t in enumerate(thresholdVs):
        #            if value >= t:
        #                tScores[ti] += value
        #            else:
        #                break
                #lableScore += l1[trial[fromVi]][trial[toVi]][fromVi][toVi]
        lableScore = rMG[label,combis[:,0],combis[:,1],ranges[:,0],ranges[:,1]]
        for ti, t in enumerate(thresholdVs):
            tindex = np.where(lableScore >= t)
            tScores[ti] = np.sum(lableScore[tindex])

        return (tScores,label)


#classify using a GCR
def doFullCLassify(saliency, ylabel, rMG,index_Trial, ranges, rM, minScores, doPenalty=False,  useThreshold=False, thresholds=[0], globalT=True, useRM=True):
    lableScores = dict()
    indList = range(len(saliency))
    combis = np.array(list(itertools.product(index_Trial,index_Trial)))

    if doPenalty:
        # not parallel
        #answers = []    
        #for lable in rMG.keys():
        #    answers.append(sumLabelScoresPenalty(trial,lable, indList, rMG))
        print('TODO need update')
        #answers = Parallel(n_jobs=1, prefer="threads")(delayed(sumLabelScoresPenalty)(trial,lable, indList, rMG) for lable in range(len(rMG)))

    else:
        answers = Parallel(n_jobs=1, prefer="threads")(delayed(sumLabelScores)(saliency,lable, indList, rMG, combis, ranges, rM, useRM=useRM, useThreshold=useThreshold, thresholds=thresholds, globalT=globalT) for lable in range(len(rMG)))
    
    for lableScore, label in answers:
        lableScores[label] = lableScore

    #get final score
    #for lable in rMG.keys():
    #    if maxScores[lable] > 0:
    #        lableScores[lable] = lableScores[lable]/maxScores[lable]

    if not useThreshold or len(thresholds) == 1:
        biggestLable = None
        biggestValue = np.max(list(lableScores.values()))
        minResult = np.max(list(minScores.values()))
        #print(minScores)
        for k in lableScores.keys():
            if lableScores[k] == biggestValue and minScores[k] <= minResult:
                minResult = minScores[k]
                biggestLable = k

        #biggestLable = list(lableScores.keys())[np.argmax(list(lableScores.values()))]
        #biggestValue = lableScores[biggestLable]
        boolResult = biggestLable == ylabel

        return lableScores, boolResult, biggestLable, biggestValue, ylabel
    else:
        biggestLables = []
        boolResults = []
        biggestValues = []
        ylabels = []
        newLableScores = []
        minResult = np.max(list(minScores.values()))

        for ti in range(len(thresholds)):
            biggestLable = None
            newValueScores = np.array(list(lableScores.values()))[:,ti]
            biggestValue = np.max(newValueScores)
            biggestIndex = np.argmax(newValueScores)
            minResultT = minResult
            
            for k in lableScores.keys():
                if lableScores[k][ti] == biggestValue and minScores[k] <= minResultT:
                    minResultT = minScores[k]
                    biggestLable = k
            if biggestLable == None:
                print(lableScores)
                print(biggestIndex)
                print(round(float(lableScores[list(lableScores.keys())[biggestIndex]][ti]),8))
                print(biggestValue)
                print(round(float(biggestValue),8))
                print(newValueScores)
                print(minScores)
                print(minResultT)

                #biggestLable = list(lableScores.keys())[biggestIndex]
                #minResult = minScores[biggestLable]

            #biggestLable = list(lableScores.keys())[np.argmax(list(lableScores.values()))]
            #biggestValue = lableScores[biggestLable]
            boolResult = biggestLable == ylabel
            biggestLables.append(biggestLable)
            boolResults.append(boolResult)
            biggestValues.append(biggestValue)
            ylabels.append(ylabel)
            newLableScores.append(newValueScores)

        return newLableScores, boolResults, biggestLables, biggestValues, ylabels


def calcFCAMMaxScoreNP(rMG, sumUp=True):
    #sum maximum score
    methodStart = datetime.now()
    print("starting maxing FCAM" + str(datetime.now() - methodStart))
    maxScores = np.max(rMG, axis=(1,2))
    if sumUp:
        maxScores = np.sum(maxScores, axis=(1,2))
    np.around(maxScores, 5)

    print('done summing max FCAM' + str(datetime.now() - methodStart))
    return maxScores

def calcFCAMMinScoreNP(rMG, sumUp=True):
    #sum maximum score
    methodStart = datetime.now()
    print("starting mining FCAM" + str(datetime.now() - methodStart))
    maxScores = np.min(rMG, axis=(1,2))
    if sumUp:
        maxScores = np.sum(maxScores, axis=(1,2))
    np.around(maxScores, 5)

    print('done summing min FCAM' + str(datetime.now() - methodStart))
    return maxScores

def calcFCAMMaxScore(rMG, inputKeys, sumUp=True):
    #sum maximum score
    methodStart = datetime.now()
    maxScores = dict()
    print("starting maxing FCAM" + str(datetime.now() - methodStart))
    for lable in rMG.keys():
        maxis = []
        for fromV in inputKeys:
            #print('starting maxing ' + str(lable) + ' progress: ' + str(fromV) + '/' + str(len(inputKeys)))
            maxis.append(np.max(list(rMG[lable][float(fromV)].values()), axis=0))
        if len(maxis) is not 0:
            maxScore =  np.max(maxis, axis=0)
            #if maxScore == 0:
            #    maxScores[lable]  = -1
            #else:
            if sumUp:
                maxScore = np.sum(maxScore)
            maxScores[lable]  = np.around(maxScore, 5)
    print('done summing max FCAM' + str(datetime.now() - methodStart))
    return maxScores


def calcFCAMMinScore(rMG, inputKeys, sumUp=True):
    #sum maximum score
    methodStart = datetime.now()
    maxScores = dict()
    print("starting maxing FCAM" + str(datetime.now() - methodStart))
    for lable in rMG.keys():
        maxis = []
        #print(lable)
        for fromV in inputKeys:
            #print('starting maxing ' + str(lable) + ' progress: ' + str(fromV) + '/' + str(len(inputKeys)))
            maxis.append(np.min(list(rMG[lable][float(fromV)].values()), axis=0))
            maxScore =  np.min(maxis, axis=0)
            if sumUp:
                maxScore = np.sum(maxScore)
            maxScores[lable]  = np.around(maxScore, 5)


    print('done summing max FCAM' + str(datetime.now() - methodStart))
    return maxScores

def calcFCAMMinValue(rMG, inputKeys):
    #sum maximum score
    methodStart = datetime.now()
    maxScores = dict()
    print("starting maxing FCAM" + str(datetime.now() - methodStart))
    for lable in rMG.keys():
        maxis = []
        #print(lable)
        for fromV in inputKeys:
            #print('starting maxing ' + str(lable) + ' progress: ' + str(fromV) + '/' + str(len(inputKeys)))
            maxis.append(np.min(list(rMG[lable][float(fromV)].values()), axis=0))
        if len(maxis) is not 0:
            maxScore =  np.min(maxis)
            #if maxScore == 0:
            #    maxScores[lable]  = -1
            #else:
            maxScores[lable]  = maxScore
        else:
            maxScores[lable] = 0

    print('done summing max FCAM' + str(datetime.now() - methodStart))
    return maxScores


def scaleFCAMPerClassNP(rMG):
    minValues = calcFCAMMinScoreNP(rMG, sumUp=False)
    minValue = calcFCAMMinScoreNP(rMG, sumUp=True)
    maxValues = calcFCAMMaxScoreNP(rMG, sumUp=False)
    maxValue = calcFCAMMaxScoreNP(rMG, sumUp=True)

    div = maxValues - minValues

    for label in range(len(rMG)):
        if maxValue[label] == 0:
            maxValue[label] = len(rMG[0,0,0])**2


        nullDiv = np.where(div[label] == 0)
        minValues[label][nullDiv] = 0

        div[label][nullDiv] = maxValue[[label]]
        maxValues[label][nullDiv] = maxValue[[label]]

    rMG = ((rMG - minValues[:,None,None,:,:]) / div[:,None,None,:,:]) * (maxValues/maxValue[:,None,None])[:,None,None,:,:]

    return rMG



def scaleFCAMBetweenClassesNP(rMG):
    
    minValuesTemp = calcFCAMMinScoreNP(rMG, sumUp=False)
    maxValuesTemp = calcFCAMMaxScoreNP(rMG, sumUp=False)
    minValues = np.min(minValuesTemp, axis=0)
    maxValues = np.max(maxValuesTemp, axis=0)

    maxValue = np.sum(maxValues)
    minValue = np.sum(minValues)



    if maxValue == 0:
        maxValue = len(rMG[0,0,0])**2

   
    div = maxValues - minValues
    nullDiv = np.where(div == 0)
    minValues[nullDiv] = 0

    div[nullDiv] = maxValue
    maxValues[nullDiv] = maxValue

    rMG = ((rMG - minValues[None,None,None,:,:]) / div[None,None,None,:,:]) * (maxValues/maxValue)[None,None,None,:,:]

    return rMG

def scaleFCAMMixClassesNP(rMG, globalMax=False, scalingMin=False):
    rMG = scaleFCAMOverMaxNP(rMG)

    minValuesTemp = calcFCAMMinScoreNP(rMG, sumUp=False)
    maxValuesTemp = calcFCAMMaxScoreNP(rMG, sumUp=False)
    sumMaxValues = calcFCAMMaxScoreNP(rMG)
    sumMinValues = np.min(calcFCAMMinScoreNP(rMG))



    minValues = np.min(minValuesTemp, axis=0)
    maxValues = np.max(maxValuesTemp, axis=0)

    if globalMax:
        div = maxValues - minValues
        nullDiv = np.where(div == 0)
        minValues[nullDiv] = 0

        div[nullDiv] = 1
        maxValues[nullDiv] = 1
        rMG = ((rMG - minValues[None,None,None,:,:]) / div[None,None,None,:,:]) * maxValues[None,None,None,:,:]
    else:

        if not scalingMin:
            div = maxValuesTemp - minValues[None,:,:]

            nullDiv = np.where(div == 0)
            minValuesTemp = np.zeros(minValuesTemp.shape) + minValues[None,:,:]
            minValuesTemp[nullDiv] = 0


            div[nullDiv] = 1
            maxValuesTemp[nullDiv] = 1
            rMG = ((rMG - minValuesTemp[:,None,None,:,:]) / div[:,None,None,:,:]) * maxValuesTemp[:,None,None,:,:]
        else:
            div = 1 - sumMinValues
            if div == 0:
                div = len(rMG[0,0,0])**2
            minValuesTemp = np.zeros(minValuesTemp.shape) + minValues[None,:,:]

            rMG = (rMG - minValuesTemp[:,None,None,:,:])  / div #* maxValuesTemp[:,None,None,:,:]


    return rMG

def scaleFCAMOverMaxNP(rMG, globalMax=False):

    maxValues = calcFCAMMaxScoreNP(rMG)
    maxValue = np.max(maxValues)

    if globalMax:
        if maxValue == 0:
            maxValue = len(rMG[0,0,0])**2
        rMG = rMG / maxValue
    else:
        for mi in range(len(maxValues)):
            if maxValues[mi]== 0:
                maxValues[mi] = len(rMG[0,0,0])**2
            rMG[mi] = rMG[mi] / maxValues[mi]

    return rMG


def diffNormalizeFCAMNP(rMG, doOVerMax=True):
    if doOVerMax:
        rMG = scaleFCAMOverMaxNP(rMG, globalMax=False)

    keynLen = len(rMG)-1
    backUpValues = []
    #rMG = rMG - (rMG/keynLen)

    for label in range(len(rMG)):
        backUpValues.append(rMG[label]/keynLen)
    
    for label in range(len(rMG)):

        for label2 in range(len(rMG)):
            if label != label2:
                rMG[label] = rMG[label]-backUpValues[label2]

    return rMG




def scaleFCAMPerClass(rMG, inputKeys, forceScaleOnlyMinusValues=False):
    minValues = calcFCAMMinScore(rMG, inputKeys, sumUp=False)
    minValue = calcFCAMMinScore(rMG, inputKeys, sumUp=True)
    maxValues = calcFCAMMaxScore(rMG, inputKeys, sumUp=False)
    maxValue = calcFCAMMaxScore(rMG, inputKeys, sumUp=True)

    for label in rMG.keys():
        if maxValue[label] == 0:
            maxValue[label] = len(rMG[label][inputKeys[0]][inputKeys[0]])**2
        for toL in inputKeys:
            for fromL in inputKeys:
                for i in range(len(rMG[label][fromL][toL])):
                    for j in range(len(rMG[label][fromL][toL][i])):
                        if forceScaleOnlyMinusValues and rMG[label][fromL][toL][i][j] > 0:
                            rMG[label][fromL][toL][i][j] = (rMG[label][fromL][toL][i][j] / maxValue[label])                       
                        else:
                            if maxValue[label] == 0:
                                maxValue[label] = len(rMG[label][fromL][toL]) ** 2
                            div = (maxValues[label][i][j] - minValues[label][i][j])
                            if div != 0:
                                rMG[label][fromL][toL][i][j] = ((rMG[label][fromL][toL][i][j] - minValues[label][i][j]) / div) * (maxValues[label][i][j]/maxValue[label])
                            else:
                                rMG[label][fromL][toL][i][j] = (rMG[label][fromL][toL][i][j] / maxValue[label])


    return rMG


def scaleFCAMBetweenClasses(rMG, inputKeys, forceScaleOnlyMinusValues=False, globalMax=True):
    
    minValuesTemp = calcFCAMMinScore(rMG, inputKeys, sumUp=False)
    maxValuesTemp = calcFCAMMaxScore(rMG, inputKeys, sumUp=False)
    minValues = np.min(list(minValuesTemp.values()), axis=0)
    maxValues = np.max(list(maxValuesTemp.values()), axis=0)
    if globalMax:
        maxValue = np.sum(maxValues)
        minValue = np.sum(minValues)

    else:
        maxValue = calcFCAMMaxScore(rMG, inputKeys, sumUp=True)
        minValue = calcFCAMMinScore(rMG, inputKeys, sumUp=True)


    for label in rMG.keys():
        if globalMax:
            if maxValue == 0:
                maxValue = len(rMG[label][inputKeys[0]][inputKeys[0]])**2
        else:
            if maxValue[label] == 0:
                maxValue[label] = len(rMG[label][inputKeys[0]][inputKeys[0]])**2
        for toL in inputKeys:
            for fromL in inputKeys:
                for i in range(len(rMG[label][fromL][toL])):
                    for j in range(len(rMG[label][fromL][toL][i])):
                        if forceScaleOnlyMinusValues and rMG[label][fromL][toL][i][j] > 0:
                            if globalMax:
                                    rMG[label][fromL][toL][i][j] = (rMG[label][fromL][toL][i][j] /maxValue)
                            else:
                                    rMG[label][fromL][toL][i][j] = (rMG[label][fromL][toL][i][j] /maxValue[label])
                        else:   
                            div = (maxValues[i][j] - minValues[i][j])
                            if div != 0:
                                if globalMax:
                                    rMG[label][fromL][toL][i][j] = ((rMG[label][fromL][toL][i][j] - minValues[i][j]) / div) * (maxValues[i][j]/maxValue) 
                                else:
                                    
                                    
                                    rMG[label][fromL][toL][i][j] = ((rMG[label][fromL][toL][i][j] - minValues[i][j]) / div) * (maxValues[i][j]/maxValue[label])

                                # für maxValues würde sprechen, dass beim Kürzen das halt genau aufgeht!
                            elif globalMax:
                                rMG[label][fromL][toL][i][j] = (rMG[label][fromL][toL][i][j] / maxValue)
                            else:
                                rMG[label][fromL][toL][i][j] = (rMG[label][fromL][toL][i][j] / maxValue[label])
    return rMG

def scaleFCAMMixClasses(rMG, inputKeys, globalMax=False):
    rMG = scaleFCAMOverMax(rMG, inputKeys)

    minValuesTemp = calcFCAMMinScore(rMG, inputKeys, sumUp=False)
    maxValuesTemp = calcFCAMMaxScore(rMG, inputKeys, sumUp=False)
    minValues = np.min(list(minValuesTemp.values()), axis=0)
    maxValues = np.max(list(maxValuesTemp.values()), axis=0)
    
    
    fullMaxScores = calcFCAMMaxScore(rMG, inputKeys)
    fullMinScores = calcFCAMMinScore(rMG, inputKeys)

    for label in rMG.keys():
        for toL in inputKeys:
            for fromL in inputKeys:
                for i in range(len(rMG[label][fromL][toL])):
                    for j in range(len(rMG[label][fromL][toL][i])):
                        if globalMax:
                            div = (maxValues[i][j] - minValues[i][j])
                            if div != 0:
                                rMG[label][fromL][toL][i][j] = ((rMG[label][fromL][toL][i][j] - minValues[i][j]) / div) * maxValues[i][j] 
                        else:
                            div = (maxValuesTemp[label][i][j] - minValues[i][j])
                            if div != 0:
                                rMG[label][fromL][toL][i][j] = ((rMG[label][fromL][toL][i][j] - minValues[i][j]) / div) * maxValuesTemp[label][i][j]
                            

                        


    return rMG

def scaleFCAMOverMax(rMG, inputKeys, globalMax=False):

    maxScores = calcFCAMMaxScore(rMG, inputKeys)
    maxScore = np.max(list(maxScores.values()))
    #print(maxScores)
    #print('############## BEFORE')
    for label in rMG.keys():
        if globalMax:
            if maxScore == 0:
                maxScore = len(rMG[label][inputKeys[0]][inputKeys[0]])**2
        else:
            if maxScores[label] == 0:
                maxScores[label] = len(rMG[label][inputKeys[0]][inputKeys[0]])**2
        for toL in inputKeys:
            for fromL in inputKeys:
                for i in range(len(rMG[label][fromL][toL])):
                    for j in range(len(rMG[label][fromL][toL][i])):
                        if globalMax:
                            rMG[label][fromL][toL][i][j] = (rMG[label][fromL][toL][i][j]) /  maxScore
                        else:
                            rMG[label][fromL][toL][i][j] = (rMG[label][fromL][toL][i][j]) /  maxScores[label]
    maxScores = calcFCAMMaxScore(rMG, inputKeys)
    #print(maxScores)
    #print('############## NOW')

    return rMG


def diffNormalizeFCAM(rMG, inputKeys, doOVerMax=True):
    if doOVerMax:
        maxScores = calcFCAMMaxScore(rMG, inputKeys)
        for label in rMG.keys():
            if maxScores[label] == 0:
                maxScores[label] = len(rMG[label][inputKeys[0]][inputKeys[0]])**2
            for toL in inputKeys:
                for fromL in inputKeys:
                    for i in range(len(rMG[label][fromL][toL])):
                        for j in range(len(rMG[label][fromL][toL][i])): 
                            rMG[label][fromL][toL][i][j] = (rMG[label][fromL][toL][i][j]) /  maxScores[label]

    keynLen = len(rMG.keys())-1
    minValue = np.zeros(len(rMG.keys()))
    labelSome = list(rMG.keys())[0]
    for toL in inputKeys:
        for fromL in inputKeys:
            for i in range(len(rMG[labelSome][fromL][toL])):
                for j in range(len(rMG[labelSome][fromL][toL][i])):
                    backUpValues = np.zeros(len(rMG.keys()))
                    for label in rMG.keys():
                            backUpValues[label] = rMG[label][fromL][toL][i][j]/keynLen
                    for label in rMG.keys():
                        for label2 in rMG.keys():
                            if label != label2:
                                rMG[label][fromL][toL][i][j] = rMG[label][fromL][toL][i][j]  - backUpValues[label2]
                                if rMG[label][fromL][toL][i][j] < minValue[label]:
                                    minValue[label] = rMG[label][fromL][toL][i][j]

    return rMG

def gtmToNumpy(rMG, reduceString):
    newRMG = list(rMG.values())
    for ni in range(len(newRMG)):
        newRMG[ni] = list(rMG[ni][reduceString].values())
    return np.array(newRMG)
    
def gcrToNumpy(rMG, valuesA):
    newRMG = list(rMG.values())
    for ni in range(len(newRMG)):
        newRMG[ni] = list(newRMG[ni].values())[:len(valuesA)]
        for nj in range(len(newRMG[ni])):
            newRMG[ni][nj] = list(newRMG[ni][nj].values())
            #for nk in range(len(newRMG[ni][nj])):
            #    newRMG[ni][nj][nk] = list(newRMG[ni][nj][nk].values())
            #del newRMG[ni][v2]
    #print(np.array(np.array(newRMG)[:,0:2]))
    return np.array(newRMG)

def classFullAttFast(rMG, ix_test, ranges, combis, predictions, rM, minScores, valuesA, doPenalty=False, useThreshold=False, thresholdPercents=[0], useRM=True, rMA=True):
    methodStart = datetime.now()
    print("starting classify FCAM" + str(datetime.now() - methodStart))

    if len(thresholdPercents) == 0:
        thresholdPercents = [0]
        globalT=True
        


    lowestScore = np.min(rMG)
    highestScore = np.max(rMG)
    thresholds = np.zeros(len(thresholdPercents))

    for ti, tp in enumerate(thresholdPercents):
        thresholds[ti] = (highestScore-lowestScore) * tp + lowestScore




    nrTrials = len(ix_test)
    resultss = []
    predictResultss = []
    biggestScoress = []
    allLableScoress= []
    accs = []
    predicsions =[]
    recalls=[]
    f1s=[]
    confidenceAccs=[]


    lableScore = rMG[:,combis[:,0].reshape(nrTrials,-1),combis[:,1].reshape(nrTrials,-1),ranges[:,0], ranges[:,1]]
    
    for ti, t in enumerate(thresholds):
        if not useThreshold:
            lableScore = lableScore
        else:
            tindex = lableScore >= t
            lableScore = lableScore * tindex
       
            



        scores = np.sum(lableScore, axis=2)
        allLableScores = scores.swapaxes(0,1)
                
        biggestScores = np.max(scores, axis=0)


        mask = allLableScores == biggestScores[:,None]

        minArg = np.argsort(minScores) [::-1]
        minScoresMask = minScores == np.min(minScores)

        ranking = minArg.argsort() +1 

        predictResults = np.argmax(mask * ranking, axis=1)


        results = predictResults == predictions

        acc = metrics.accuracy_score(predictResults, predictions)
        predicsion = metrics.precision_score(predictResults, predictions, average='macro')
        recall = metrics.recall_score(predictResults, predictions, average='macro')
        f1= metrics.f1_score(predictResults, predictions, average='macro')

        confidenceAcc = helper.confidenceGCR2(biggestScores, results, 10)
        
        print('FCAM results t' + str(t) + ' : ' + str(acc))

        accs.append(acc)
        predicsions.append(predicsion)
        recalls.append(recall)
        f1s.append(f1)
        confidenceAccs.append(confidenceAcc)
        allLableScoress.append(allLableScores)
        biggestScoress.append(biggestScores)
        predictResultss.append(predictResults)
        resultss.append(results)
    print('processes end FCAM' + str(datetime.now() - methodStart))


    return [accs, predicsions, recalls, f1s], [allLableScoress, biggestScoress], resultss, predictResultss, confidenceAccs
        


#validate the full coherence matrices 
def classFullAtt(rMG, saliencies, ix_test, ranges, y_testy, rM, minScores, valuesA, doPenalty=False, useThreshold=False, thresholdPercents=[0], globalT=True, useRM=True):
    methodStart = datetime.now()

    print("starting classify FCAM" + str(datetime.now() - methodStart))


    #sum normal score
    # create a list to keep all processes
    processes = []
    
    # create a list to keep connections
    parent_connections = []
    #tupels = list(zip(range(len(x_test[0])), range(len(x_test[0]))))

    print('processes start FCAM' + str(datetime.now() - methodStart))
    thresholds = np.zeros(len(thresholdPercents))

    if useThreshold:

        for ti, tp in enumerate(thresholdPercents):
            if globalT:
                lowestScore = np.min(list(calcFCAMMinScore(rMG, valuesA, sumUp=False).values()))
                highestScore = np.max(list(calcFCAMMaxScore(rMG, valuesA, sumUp=False).values()))
                thresholds[ti]= (highestScore-lowestScore) * tp + lowestScore
            else:
                thresholds[ti] = tp



    rMGList = gcrToNumpy(rMG, valuesA)
    


    answers = []
    #with parallel_backend("loky", inner_max_num_threads=2):
    answers = Parallel(n_jobs=1, prefer="threads")(delayed(doFullCLassify)(saliencies[ti], y_testy[ti], rMGList, ix_test[ti], ranges, rM, minScores, doPenalty=doPenalty, useRM=useRM, useThreshold=useThreshold, thresholds=thresholds, globalT=globalT) for ti in range(len(ix_test)))
    #for ti in range(len(x_test)):
    #    answers.append(doFullCLassify(x_test[ti], y_testy[ti], rMG, maxScores, rM, doPenalty=doPenalty))

    #for ti in range(len(x_test)):
    #    trial = x_test[ti]
    #    #print('starting trial ' + str(ti) + '/' + str(len(x_test)))
    #    parent_conn, child_conn = mp.Pipe()
    #    parent_connections.append(parent_conn)
        
    #    process = mp.Process(target=doFullCLassify, args=(trial, y_testy[ti], tupels, rMG, maxScores, child_conn))
    #    processes.append(process)

    print('processes end FCAM' + str(datetime.now() - methodStart))
        
    # start all processes
    #for process in processes:
    #    process.start()
        
    # make sure that all processes have finished
    #for process in processes:
    #    process.join()
        
    if not useThreshold or len(thresholds) == 1:
        results = []
        predictResults = []
        biggestScores = []
        allLableScores = []

        asynLabels = []
        for ans in answers:
            #ans = parent_connection.recv()
            results.append(ans[1])
            predictResults.append(ans[2])
            biggestScores.append(ans[3])
            allLableScores.append(ans[0])
            asynLabels.append(ans[4])
        print("FCAM results :" + str(sum(results)/len(results)))
        

        print('start results FCAM' + str(datetime.now() - methodStart))

        acc = metrics.accuracy_score(predictResults, asynLabels)
        predicsion = metrics.precision_score(predictResults, asynLabels, average='macro')
        recall = metrics.recall_score(predictResults, asynLabels, average='macro')
        f1= metrics.f1_score(predictResults, asynLabels, average='macro')

        confidenceAcc = helper.confidenceGCR2(biggestScores, results, 10)

        print('done results FCAM' + str(datetime.now() - methodStart))

        return [acc, predicsion, recall, f1], [allLableScores, biggestScores], results, predictResults, confidenceAcc

    else:
        results = []
        predictResults = []
        biggestScores = []
        allLableScores = []
        asynLabels = []
        accs = []
        predicsions =[]
        recalls=[]
        f1s=[]
        confidenceAccs=[]
        for ti in range(len(thresholds)):
            results.append([])
            predictResults.append([])
            biggestScores.append([])
            allLableScores.append([])
            asynLabels.append([])
            for ans in answers:
                #ans = parent_connection.recv()
                results[ti].append(ans[1][ti])
                predictResults[ti].append(ans[2][ti])
                biggestScores[ti].append(ans[3][ti])
                allLableScores[ti].append(ans[0][ti])
                asynLabels[ti].append(ans[4][ti])
            print("FCAM results :" + str(thresholds[ti]) + ' : '+ str(sum(results[ti])/len(results[ti])))
            

            acc = metrics.accuracy_score(predictResults[ti], asynLabels[ti])
            predicsion = metrics.precision_score(predictResults[ti], asynLabels[ti], average='macro')
            recall = metrics.recall_score(predictResults[ti], asynLabels[ti], average='macro')
            f1= metrics.f1_score(predictResults[ti], asynLabels[ti], average='macro')

            confidenceAcc = helper.confidenceGCR2(biggestScores, results, 10)

            accs.append(acc)
            predicsions.append(predicsion)
            recalls.append(recall)
            f1s.append(f1)
            confidenceAccs.append(confidenceAcc)


        return [accs, predicsions, recalls, f1s], [allLableScores, biggestScores], results, predictResults, confidenceAccs
    
def calcGTMMinScore(rMA, reduceString, sumUp=True):
    maxScores = dict()
    for lable in rMA.keys():
        maxScores[lable] =  np.min(list(rMA[lable][reduceString].values()), axis=0)
        if sumUp:
            maxScores[lable] =  np.sum(maxScores[lable])

    return maxScores
    
def calcGTMMaxScore(rMA, reduceString, sumUp=True):
    maxScores = dict()
    for lable in rMA.keys():
        maxScores[lable] =  np.max(list(rMA[lable][reduceString].values()), axis=0)
        if sumUp:
            maxScores[lable] =  np.sum(maxScores[lable])    
    return maxScores

def gtmReductionStrings():
    return ['1DSum', '1DSum+'] #['max','max+','average','average+','median','median+', '1DSum', '1DSum+']


#validate the GTM
def calcFullAbstractAttentionFast(gtm, reduceString, ix_test, minScores, ranges, predictions, rM, useThreshold=False, thresholdPercents=[0]):
    methodStart = datetime.now() 
    results = []
    predictResults = []
    biggestScores = []
    allLableScores = []
    
    #all possible implemented reductions
    #reduceStrings = gtmReductionStrings()
    #reduceString = reduceStrings[reductionInt]
    print("starting classify GTM " + reduceString  + " " +str(datetime.now() - methodStart))

    #if divScore != None:
    #    maxScores = calcGTMMaxScore(rMA, reduceString)
    #else:
    #    maxScores = divScore

    rMG = gtm[reduceString]

    #HERE TODO
    if len(thresholdPercents) == 0:
        thresholdPercents = [0]
        globalT=True
        
    thresholds = np.zeros(len(thresholdPercents))


    lowestScore = np.min(rMG)
    highestScore = np.max(rMG)

    for ti, tp in enumerate(thresholdPercents):
        thresholds[ti] = (highestScore-lowestScore) * tp + lowestScore


    resultss = []
    predictResultss = []
    biggestScoress = []
    allLableScoress= []
    accs = []
    predicsions =[]
    recalls=[]
    f1s=[]
    confidenceAccs=[]
    print(rMG.shape)
    lableScore = rMG[:,ix_test,ranges]

    
    for ti, t in enumerate(thresholds):
        if not useThreshold:
            lableScore = lableScore
        else:
            tindex = lableScore >= t
            lableScore = lableScore * tindex
            
        scores = np.sum(lableScore, axis=2)

        allLableScores = scores.swapaxes(0,1)
                
        biggestScores = np.max(scores, axis=0)


        mask = allLableScores == biggestScores[:,None]

        minArg = np.argsort(minScores) [::-1]
        minScoresMask = minScores == np.min(minScores)

        ranking = minArg.argsort() +1 

        predictResults = np.argmax(mask * ranking, axis=1)


        results = predictResults == predictions

        acc = metrics.accuracy_score(predictResults, predictions)
        predicsion = metrics.precision_score(predictResults, predictions, average='macro')
        recall = metrics.recall_score(predictResults, predictions, average='macro')
        f1= metrics.f1_score(predictResults, predictions, average='macro')

        confidenceAcc = helper.confidenceGCR2(biggestScores, results, 10)
        
        print('GTM '+  reduceString + ' results t' + str(t) + ' : ' + str(acc))

        #if not useThreshold or len(thresholds) == 1:
        #    return [acc, predicsion, recall, f1], [allLableScores, biggestScores], results, predictResults, confidenceAcc

        accs.append(acc)
        predicsions.append(predicsion)
        recalls.append(recall)
        f1s.append(f1)
        confidenceAccs.append(confidenceAcc)
        allLableScoress.append(allLableScores)
        biggestScoress.append(biggestScores)
        predictResultss.append(predictResults)
        resultss.append(results)
    print('processes end GTM' + str(datetime.now() - methodStart))

    return [accs, predicsions, recalls, f1s], [allLableScoress, biggestScoress], resultss, predictResultss, confidenceAccs
        




#validate the GTM
def calcFullAbstractAttention(rMA, saliencies, reduceString, ix_test, y_testy, dataLen, valuesA, reductionName="MixedClasses", useThreshold=False, thresholdPercents=[0], globalT=True):
    methodStart = datetime.now() 
    results = []
    predictResults = []
    biggestScores = []
    allLableScores = []
    
    #all possible implemented reductions
    #reduceStrings = gtmReductionStrings()
    #reduceString = reduceStrings[reductionInt]
    print("starting classify GTM " + reduceString  + " " +str(datetime.now() - methodStart))

    #if divScore != None:
    #    maxScores = calcGTMMaxScore(rMA, reduceString)
    #else:
    #    maxScores = divScore

    if reductionName == "DiffNormalize":
        rMA = diffNormalizeGTM(rMA,reduceString, doOVerMax=True)
    elif reductionName=="OverMax":
        rMA = scaleGTMOverMax(rMA,reduceString,globalMax=False)
    elif reductionName=="PerClass":
        rMA = scaleGTMPerClass(rMA,reduceString)
    elif reductionName=="BetweenClasses":
        rMA = scaleGTMBetweenClasses(rMA,reduceString,globalMax=True)
    elif reductionName=="MixedClasses":
        rMA = scaleGTMMixClasses(rMA,reduceString,globalMax=False)

    thresholds = np.zeros(len(thresholdPercents))

    if useThreshold:

        for ti, tp in enumerate(thresholdPercents):
            if globalT:
                lowestScore = np.min(list(calcFCAMMinScore(rMA, valuesA, sumUp=False).values()))
                highestScore = np.max(list(calcFCAMMaxScore(rMA, valuesA, sumUp=False).values()))
                thresholds[ti]= (highestScore-lowestScore) * tp + lowestScore
            else:
                thresholds[ti] = tp


    minScores = calcGTMMinScore(rMA, reduceString)

    numRMA = gtmToNumpy(rMA, reduceString)

    print('processes start GTM' + str(datetime.now() - methodStart))
    
    
    answers = []
    answers = Parallel(n_jobs=1, prefer="threads")(delayed(classifyGTM)(saliencies[ti], y_testy[ti], numRMA, ix_test[ti], dataLen, minScores, useThreshold=useThreshold, thresholds=thresholds, globalT=globalT) for ti in range(len(ix_test)))

    print('processes end GTM' + str(datetime.now() - methodStart))
        
    if not useThreshold or len(thresholds) == 1:
        results = []
        predictResults = []
        biggestScores = []
        allLableScores = []

        asynLabels = []
        for ans in answers:
            #ans = parent_connection.recv()
            results.append(ans[1])
            predictResults.append(ans[2])
            biggestScores.append(ans[3])
            allLableScores.append(ans[0])
            asynLabels.append(ans[4])
        print("GTM results " +reduceString+ ":"+ str(sum(results)/len(results)))
        

        print('start results GTM' + str(datetime.now() - methodStart))

        acc = metrics.accuracy_score(predictResults, asynLabels)
        predicsion = metrics.precision_score(predictResults, asynLabels, average='macro')
        recall = metrics.recall_score(predictResults, asynLabels, average='macro')
        f1= metrics.f1_score(predictResults, asynLabels, average='macro')

        confidenceAcc = helper.confidenceGCR2(biggestScores, results, 10)

        print('done results GTM' + str(datetime.now() - methodStart))

        return [acc, predicsion, recall, f1], [allLableScores, biggestScores], results, predictResults, confidenceAcc

    else:
        results = []
        predictResults = []
        biggestScores = []
        allLableScores = []
        asynLabels = []
        accs = []
        predicsions =[]
        recalls=[]
        f1s=[]
        confidenceAccs=[]
        for ti in range(len(thresholds)):
            results.append([])
            predictResults.append([])
            biggestScores.append([])
            allLableScores.append([])
            asynLabels.append([])
            for ans in answers:
                #ans = parent_connection.recv()
                results[ti].append(ans[1][ti])
                predictResults[ti].append(ans[2][ti])
                biggestScores[ti].append(ans[3][ti])
                allLableScores[ti].append(ans[0][ti])
                asynLabels[ti].append(ans[4][ti])
            print("GTM results "+ reduceString +' ' + str(thresholds[ti]) + ' : '+ str(sum(results[ti])/len(results[ti])))

            acc = metrics.accuracy_score(predictResults[ti], asynLabels[ti])
            predicsion = metrics.precision_score(predictResults[ti], asynLabels[ti], average='macro')
            recall = metrics.recall_score(predictResults[ti], asynLabels[ti], average='macro')
            f1= metrics.f1_score(predictResults[ti], asynLabels[ti], average='macro')

            confidenceAcc = helper.confidenceGCR2(biggestScores, results, 10)

            accs.append(acc)
            predicsions.append(predicsion)
            recalls.append(recall)
            f1s.append(f1)
            confidenceAccs.append(confidenceAcc)

        return [accs, predicsions, recalls, f1s], [allLableScores, biggestScores], results, predictResults, confidenceAccs





def classifyGTM(saliency, ylabel, rMGList, index_Trial, dataLen, minScores, useThreshold=False, thresholds=[0], globalT=True):
    methodStart = datetime.now() 
    lableScores = dict()

    if useThreshold:
        tlne = len(thresholds)

        tScores = np.zeros(tlne)

        if globalT:
            thresholdVs = thresholds
        else:
            for ti, t in enumerate(thresholds):

                thresholdVs = np.zeros(tlne)

                trialMin = np.min(saliency)
                trialMax = np.max(saliency)
                thresholdVs[ti] = (trialMax-trialMin) * t + trialMin
                
        lableScores = dict()
        for label in range(len(rMGList)):
            lableScores[label] = np.zeros(len(thresholdVs))
            lableScore = rMGList[label,index_Trial,dataLen]
            for ti, t in enumerate(thresholdVs):
                tindex = np.where(lableScore >= t)
                lableScores[label][ti] = np.sum(lableScore[tindex])

    else:

        lableScores = dict()
        for label in range(len(rMGList)):
            lableScores[label] = np.sum(rMGList[label,index_Trial,dataLen])

    
    if not useThreshold or len(thresholds) == 1:
        biggestLable = None
        biggestValue = np.max(list(lableScores.values()))
        minResult = np.max(list(minScores.values()))
        #print(minScores)
        for k in lableScores.keys():
            if lableScores[k] == biggestValue and minScores[k] <= minResult:
                minResult = minScores[k]
                biggestLable = k

        #biggestLable = list(lableScores.keys())[np.argmax(list(lableScores.values()))]
        #biggestValue = lableScores[biggestLable]
        boolResult = biggestLable == ylabel

        return lableScores, boolResult, biggestLable, biggestValue, ylabel
    else:
        biggestLables = []
        boolResults = []
        biggestValues = []
        ylabels = []
        newLableScores = []
        minResult = np.max(list(minScores.values()))

        for ti in range(len(thresholds)):
            biggestLable = None
            newValueScores = np.array(list(lableScores.values()))[:,ti]
            biggestValue = np.max(newValueScores)
            biggestIndex = np.argmax(newValueScores)
            minResultT = minResult
            
            for k in lableScores.keys():
                if lableScores[k][ti] == biggestValue and minScores[k] <= minResultT:
                    minResultT = minScores[k]
                    biggestLable = k
            if biggestLable == None:
                print(lableScores)
                print(biggestIndex)
                print(round(float(lableScores[list(lableScores.keys())[biggestIndex]][ti]),8))
                print(biggestValue)
                print(round(float(biggestValue),8))
                print(newValueScores)
                print(minScores)
                print(minResultT)

            boolResult = biggestLable == ylabel
            biggestLables.append(biggestLable)
            boolResults.append(boolResult)
            biggestValues.append(biggestValue)
            ylabels.append(ylabel)
            newLableScores.append(newValueScores)

        return newLableScores, boolResults, biggestLables, biggestValues, ylabels

def validataHeat(value, heat, doFidelity):
    if doFidelity:
        return value <= heat
    else:
        return value > heat

def scaleGTMPerClass(rMG,reduceString, forceScaleOnlyMinusValues=False):
    minValues = calcGTMMinScore(rMG, reduceString, sumUp=False)
    minValue = calcGTMMinScore(rMG, reduceString, sumUp=True)
    maxValues = calcGTMMaxScore(rMG, reduceString, sumUp=False)
    maxValue = calcGTMMaxScore(rMG, reduceString, sumUp=True)

    for label in rMG.keys():
        if maxValue[label] == 0:
            maxValue[label] = len(rMG[label][reduceString][list(rMG[label][reduceString].keys())[0]])
        for fromL in rMG[label][reduceString].keys():
                for i in range(len(rMG[label][reduceString][fromL])):
                        if forceScaleOnlyMinusValues and rMG[label][reduceString][fromL][i] > 0:
                            rMG[label][reduceString][fromL][i] = (rMG[label][reduceString][fromL][i] / maxValue[label])                                
                        else:
                            if maxValue[label] == 0:
                                maxValue[label] = len(rMG[label][reduceString][fromL]) ** 2
                            div = (maxValues[label][i] - minValues[label][i])
                            if div != 0:
                                rMG[label][reduceString][fromL][i] = ((rMG[label][reduceString][fromL][i] - minValues[label][i]) / div) * (maxValues[label][i]/maxValue[label])
                            else:
                                rMG[label][reduceString][fromL][i] = (rMG[label][reduceString][fromL][i] / maxValue[label])


    return rMG


def scaleGTMBetweenClasses(rMG, reduceString, forceScaleOnlyMinusValues=False, globalMax=True):
    
    minValuesTemp = calcGTMMinScore(rMG, reduceString, sumUp=False)
    maxValuesTemp = calcGTMMaxScore(rMG, reduceString, sumUp=False)
    minValues = np.min(list(minValuesTemp.values()), axis=0)
    maxValues = np.max(list(maxValuesTemp.values()), axis=0)
    if globalMax:
        maxValue = np.sum(maxValues)
        minValue = np.sum(minValues)
    else:
        maxValue = calcGTMMaxScore(rMG, reduceString, sumUp=True)
        minValue = calcGTMMinScore(rMG, reduceString, sumUp=True)

    for label in rMG.keys():
            if globalMax:
                if maxValue == 0:
                    maxValue = len(rMG[label][reduceString][list(rMG[label][reduceString].keys())[0]])
            else:
                if maxValue[label] == 0:
                    maxValue[label] = len(rMG[label][reduceString][list(rMG[label][reduceString].keys())[0]])
            for fromL in rMG[label][reduceString].keys():
                for i in range(len(rMG[label][reduceString][fromL])):
                        if forceScaleOnlyMinusValues and rMG[label][reduceString][fromL][i] > 0:
                            if globalMax:
                                rMG[label][reduceString][fromL][i] = (rMG[label][reduceString][fromL][i] /maxValue)
                            else:
                                rMG[label][reduceString][fromL][i] = (rMG[label][reduceString][fromL][i] /maxValue[label])
                        else:   
                            div = (maxValues[i] - minValues[i])
                            if div != 0:
                                if globalMax:
                                    rMG[label][reduceString][fromL][i] = ((rMG[label][reduceString][fromL][i] - minValues[i]) / div) * (maxValues[i]/maxValue) 
                                else:                                   
                                    rMG[label][reduceString][fromL][i] = ((rMG[label][reduceString][fromL][i] - minValues[i]) / div) * (maxValues[i]/maxValue[label])

                                # für maxValues würde sprechen, dass beim Kürzen das halt genau aufgeht!
                            elif globalMax:
                                rMG[label][reduceString][fromL][i]= (rMG[label][reduceString][fromL][i] / maxValue)
                            else:
                                rMG[label][reduceString][fromL][i]= (rMG[label][reduceString][fromL][i] / maxValue[label])

    return rMG

def scaleGTMMixClasses(rMG, reduceString, globalMax=False):
    rMG = scaleGTMOverMax(rMG, reduceString)

    minValuesTemp = calcGTMMinScore(rMG, reduceString, sumUp=False)
    maxValuesTemp = calcGTMMaxScore(rMG, reduceString, sumUp=False)
    minValues = np.min(list(minValuesTemp.values()), axis=0)
    maxValues = np.max(list(maxValuesTemp.values()), axis=0)
    #if globalMax:
    #    maxValue = np.sum(maxValues, axis=0)
    #    minValue = np.max(minValues, axis=0)
    #else:
    #    maxValue = calcGTMMaxScore(rMG, reduceString, sumUp=True)
    #    minValue = calcGTMMinScore(rMG, reduceString, sumUp=True)

    for label in rMG.keys():
            for fromL in rMG[label][reduceString].keys():
                for i in range(len(rMG[label][reduceString][fromL])):
                        div = (maxValues[i] - minValues[i])
                        if div != 0:
                            if globalMax:
                                rMG[label][reduceString][fromL][i] = ((rMG[label][reduceString][fromL][i] - minValues[i]) / div) * maxValues[i]
                            else:
                                rMG[label][reduceString][fromL][i] = ((rMG[label][reduceString][fromL][i] - minValues[i]) / div) * maxValuesTemp[label][i]


    return rMG

def scaleGTMOverMax(rMG, reduceString, globalMax=False):

    maxScores = calcGTMMaxScore(rMG, reduceString)
    maxScore = np.max(list(maxScores.values()))

    for label in rMG.keys():
            if globalMax:
                if maxScore == 0:
                    maxScore = len(rMG[label][reduceString][list(rMG[label][reduceString].keys())[0]])
            else:
                if maxScores[label] == 0:
                    maxScores[label] = len(rMG[label][reduceString][list(rMG[label][reduceString].keys())[0]])
            for fromL in rMG[label][reduceString].keys():
                for i in range(len(rMG[label][reduceString][fromL])):
                        if globalMax:
                            rMG[label][reduceString][fromL][i] = (rMG[label][reduceString][fromL][i]) /  maxScore
                        else:
                            rMG[label][reduceString][fromL][i] = (rMG[label][reduceString][fromL][i]) /  maxScores[label]
    maxScores = calcGTMMaxScore(rMG, reduceString)

    return rMG


def diffNormalizeGTM(rMG, reduceString, doOVerMax=True):
    if doOVerMax:
        maxScores = calcGTMMaxScoreNP(rMG, reduceString)
        for label in rMG.keys():
                if maxScores[label] == 0:
                    maxScores[label] = len(rMG[label][reduceString][list(rMG[label][reduceString].keys())[0]])
                for fromL in rMG[label][reduceString].keys():
                    for i in range(len(rMG[label][reduceString][fromL])):
                            rMG[label][reduceString][fromL][i] = (rMG[label][reduceString][fromL][i]) /  maxScores[label]

    keynLen = len(rMG.keys())-1
    minValue = np.zeros(len(rMG.keys()))
    labelSome = list(rMG.keys())[0]
    for fromL in rMG[labelSome][reduceString].keys():
            for i in range(len(rMG[labelSome][reduceString][fromL])):
                    backUpValues = np.zeros(len(rMG.keys()))
                    for label in rMG.keys():
                            backUpValues[label] = rMG[label][reduceString][fromL][i]/keynLen
                    for label in rMG.keys():
                        for label2 in rMG.keys():
                            if label != label2:
                                rMG[label][reduceString][fromL][i] = rMG[label][reduceString][fromL][i] - backUpValues[label2]
                                if rMG[label][reduceString][fromL][i] < minValue[label]:
                                    minValue[label] = rMG[label][reduceString][fromL][i]

    return rMG


def calcGTMMaxScoreNP(rMG, sumUp=True):
    #sum maximum score
    methodStart = datetime.now()
    maxScores = np.max(rMG, axis=(1))
    if sumUp:
        maxScores = np.sum(maxScores, axis=(1))
    np.around(maxScores, 5)

    return maxScores

def calcGTMMinScoreNP(rMG, sumUp=True):
    #sum maximum score
    methodStart = datetime.now()
    maxScores = np.min(rMG, axis=(1))
    if sumUp:
        maxScores = np.sum(maxScores, axis=(1))
    np.around(maxScores, 5)

    return maxScores

def scaleGTMPerClassNP(rMG):
    minValues = calcGTMMinScoreNP(rMG, sumUp=False)
    minValue = calcGTMMinScoreNP(rMG, sumUp=True)
    maxValues = calcGTMMaxScoreNP(rMG, sumUp=False)
    maxValue = calcGTMMaxScoreNP(rMG, sumUp=True)

    div = maxValues - minValues

    for label in range(len(rMG)):
        if maxValue[label] == 0:
            maxValue[label] = len(rMG[0,0])**2


        nullDiv = np.where(div[label] == 0)
        minValues[label][nullDiv] = 0

        div[label][nullDiv] = maxValue[[label]]
        maxValues[label][nullDiv] = maxValue[[label]]

    rMG = ((rMG - minValues[:,None,:]) / div[:,None,:]) * (maxValues/maxValue[:,None])[:,None,:]

    return rMG



def scaleGTMBetweenClassesNP(rMG):
    
    minValuesTemp = calcGTMMinScoreNP(rMG, sumUp=False)
    maxValuesTemp = calcGTMMaxScoreNP(rMG, sumUp=False)
    minValues = np.min(minValuesTemp, axis=0)
    maxValues = np.max(maxValuesTemp, axis=0)

    maxValue = np.sum(maxValues)
    minValue = np.sum(minValues)



    if maxValue == 0:
        maxValue = len(rMG[0,0])**2

   
    div = maxValues - minValues
    nullDiv = np.where(div == 0)
    minValues[nullDiv] = 0

    div[nullDiv] = maxValue
    maxValues[nullDiv] = maxValue

    rMG = ((rMG - minValues[None,None,:]) / div[None,None,:]) * (maxValues/maxValue)[None,None,:]

    return rMG

def scaleGTMMixClassesNP(rMG, globalMax=False):
    rMG = scaleGTMOverMaxNP(rMG)

    minValuesTemp = calcGTMMinScoreNP(rMG, sumUp=False)
    maxValuesTemp = calcGTMMaxScoreNP(rMG, sumUp=False)
    minValues = np.min(minValuesTemp, axis=0)
    maxValues = np.max(maxValuesTemp, axis=0)

    if globalMax:
        div = maxValues - minValues
        nullDiv = np.where(div == 0)
        minValues[nullDiv] = 0

        div[nullDiv] = 1
        maxValues[nullDiv] = 1
        rMG = ((rMG - minValues[None,None,:]) / div[None,None,:]) * maxValues[None,None,:]
    else:
        div = maxValuesTemp - minValues[None,:]
        nullDiv = np.where(div == 0)
        minValuesTemp = np.zeros(minValuesTemp.shape) + minValues[None,:]
        minValuesTemp[nullDiv] = 0

        div[nullDiv] = 1
        maxValuesTemp[nullDiv] = 1

        rMG = ((rMG - minValuesTemp[:,None,:]) / div[:,None,:]) * maxValuesTemp[:,None,:]



    return rMG

def scaleGTMOverMaxNP(rMG, globalMax=False):

    maxValues = calcGTMMaxScoreNP(rMG)
    maxValue = np.max(maxValues)

    if globalMax:
        if maxValue == 0:
            maxValue = len(rMG[0,0])**2
        rMG = rMG / maxValue
    else:
        for mi in range(len(maxValues)):
            if maxValues[mi]== 0:
                maxValues[mi] = len(rMG[0,0])**2
            rMG[mi] = rMG[mi] / maxValues[mi]

    return rMG


def diffNormalizeGTMNP(rMG, doOVerMax=True):
    if doOVerMax:
        rMG = scaleGTMOverMaxNP(rMG, globalMax=False)

    keynLen = len(rMG)-1
    backUpValues = []
    #rMG = rMG - (rMG/keynLen)

    for label in range(len(rMG)):
        backUpValues.append(rMG[label]/keynLen)
    
    for label in range(len(rMG)):

        for label2 in range(len(rMG)):
            if label != label2:
                rMG[label] = rMG[label]-backUpValues[label2]

    return rMG


