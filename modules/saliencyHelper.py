from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, XGradCAM, EigenCAM, EigenGradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from captum.attr import IntegratedGradients, Saliency,DeepLift,InputXGradient,GuidedBackprop,GuidedGradCam,FeatureAblation,KernelShap,Deconvolution,FeaturePermutation
import torch
import numpy as np
from modules import helper
import shap
from modules import pytorchTrain as pt
from modules import GCRPlus
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn import metrics
from modules.games import games
from modules.approximators import SHAPIQEstimator


def getRelpropSaliency(device, data, model, method=None, outputO = None, batchSize=5000):
        outRel = None
        for batch_start in range(0, len(data), batchSize):
            batchEnd = batch_start + batchSize      
            input_ids = torch.from_numpy(data[batch_start:batchEnd]).to(device) 
            input_ids.requires_grad_()
            output = model(input_ids)

            if outputO is None:
                outputOut = output.cpu().data.numpy()#[0]
                index = np.argmax(outputOut, axis=-1)
                one_hot = np.zeros((outputOut.shape[0], outputOut.shape[-1]), dtype=np.float32)
                for h in range(len(one_hot)):
                    one_hot[h, index[h]] = 1
            else:
                outputOut = outputO[batch_start:batchEnd]
                index = np.argmax(outputOut, axis=-1)
                one_hot = outputOut

            one_hot_vector = one_hot
            one_hot = torch.from_numpy(one_hot).requires_grad_(True)
            one_hot = torch.sum(one_hot.cuda() * output)

            model.zero_grad()
            one_hot.backward(retain_graph=True)
            one_hot.shape
            kwargs = {"alpha": 1}

            if method:
                outRelB = model.relprop(torch.tensor(one_hot_vector).to(input_ids.device), method=method, **kwargs).cpu().detach().numpy()
            else:
                outRelB = model.relprop(torch.tensor(one_hot_vector).to(input_ids.device) , **kwargs).cpu().detach().numpy()
            
            if outRel is None:
                outRel = outRelB
            else:
                outRel = np.vstack([outRel, outRelB])
        return outRel

def reshape_transform(tensor):
    result = tensor.reshape(tensor.size(0), tensor.size(1), 1, tensor.size(2))
    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def interpret_dataset(sal, data, targets, package='PytGradCam', smooth=False):
    if package == 'captum':
        attributions_ig = sal.attribute(data, target=targets)
        return attributions_ig.cpu().squeeze().detach().numpy()
    elif package == 'PytGradCam':
        grayscale_cam = sal(input_tensor=data, targets=targets, eigen_smooth=smooth)

        grayscale_cam = grayscale_cam#[0, :]
        return grayscale_cam


def splitSaliency(num_of_classes, outRel, targets, do3DData=False, axis1= 2, axis2=0, axis3=1, op1='max',op2='sum',op3='max'):

    if do3DData:
        outRel = helper.doCombiStep(op1, outRel, axis1)
        outRel = helper.doCombiStep(op2, outRel, axis2) 
        outRel = helper.doCombiStep(op3, outRel, axis3) 
    avgOut = []
    argsOut = []
    for c in range(num_of_classes):
        argsVC = np.argwhere(np.argmax(targets, axis=1)==c).flatten()
        if len(outRel[argsVC]) == 1:
            meanC = np.mean(outRel[argsVC].squeeze(), axis=0) 
        else:
            meanC = outRel[argsVC].squeeze()
        avgOut.append(meanC)
        argsOut.append(argsVC)

    return avgOut, argsOut

def mapSaliency(output, num_of_classes, outRel, y_train, outVal, y_val, outTest, y_test, do3DData=False, axis1= 2, axis2=0, axis3=1, op1='max',op2='sum',op3='max'):
    cTrain, argTrain = splitSaliency(num_of_classes, outRel, y_train, do3DData=do3DData, axis1=axis1, axis2=axis2, axis3=axis3, op1=op1, op2=op2, op3=op3)
    caseVal, argVal= splitSaliency(num_of_classes, outVal, y_val, do3DData=do3DData, axis1=axis1, axis2=axis2, axis3=axis3, op1=op1, op2=op2, op3=op3)
    cTest, argTest= splitSaliency(num_of_classes, outTest, y_test, do3DData=do3DData, axis1=axis1, axis2=axis2, axis3=axis3, op1=op1, op2=op2, op3=op3)
    if  'ClassTrain' not in output:
        output['ClassTrain']= []
        output['ArgClassTrain']= []
        output['ClassVal']= []
        output['ArgClassVal']= []  
        output['ClassTest']= []
        output['ArgClassTest']= []
    output['ClassTrain'].append(cTrain)
    output['ArgClassTrain'].append(argTrain) 
    output['ClassVal'].append(caseVal)
    output['ArgClassVal'].append(argVal)     
    output['ClassTest'].append(cTest)
    output['ArgClassTest'].append(argTest)

def splitPerValue(valueSaliency, saliency, x_train1, nrSymbols):
    inputIds = x_train1.squeeze()
    sValues = saliency.squeeze()
    symbolA = helper.getMapValues(nrSymbols)
    symbolA = np.array(symbolA)

    for a in symbolA:
        if a not in valueSaliency.keys():
            valueSaliency[a] = []
            for l in range(len(inputIds[0])):
                valueSaliency[a].append([])

    for n in range(len(inputIds)): 
        for k in range(len(inputIds[n])):
            valueSaliency[round(float(inputIds[n][k]), 4)][k].append(sValues[n][k]) 

    return valueSaliency

def splitPerValueAndClass(valueSaliency, saliency, x_train1, nrSymbols, numerOfClasses, lables):
    targets = np.argmax(lables, axis=1)
    inputIds = x_train1.squeeze()
    sValues = saliency.squeeze()
    symbolA = helper.getMapValues(nrSymbols)
    symbolA = np.array(symbolA)

    for c in range(numerOfClasses):
        if c not in valueSaliency.keys():
            valueSaliency[c] = dict()
        for a in symbolA:
            if a not in valueSaliency[c].keys():
                valueSaliency[c][a] = []
                for l in range(len(inputIds[0])):
                    valueSaliency[c][a].append([])

    for n in range(len(inputIds)): 
        for k in range(len(inputIds[n])):
            valueSaliency[targets[n]][round(float(inputIds[n][k]), 4)][k].append(sValues[n][k]) 

    return valueSaliency
        
def reduceMap(saliencyMap, do3DData=False, do3D2Step=False, do3rdStep=False, axis1= 2, axis2=0, axis3=1, op1='max',op2='sum',op3='sum'):#NOTE op3 is sum
    saliencyMapReturn = saliencyMap.copy()
    if do3DData:
        saliencyMapReturn = helper.doCombiStep(op1, saliencyMapReturn, axis1)
        saliencyMapReturn = helper.doCombiStep(op2, saliencyMapReturn, axis2) 
        saliencyMapReturn = helper.doCombiStep(op3, saliencyMapReturn, axis3) 
    elif do3D2Step:
        saliencyMapReturn = helper.doCombiStep(op1, saliencyMapReturn, axis1)
        saliencyMapReturn = helper.doCombiStep(op2, saliencyMapReturn, axis2) 
    elif do3rdStep:
        saliencyMapReturn = helper.doCombiStep(op3, saliencyMapReturn, axis3) 

    return saliencyMapReturn

def splitSaliencyPerStack(saliencyStacks, saliency, x_train1, newTrain, y_train1, num_of_classes, nrSymbols, andStack, orStack, xorStack, nrAnds, nrOrs, nrxor, orOffSet, xorOffSet, trueIndexes, do3DData=False, do3rdStep=False):
    preds = y_train1
    targets = np.argmax(preds, axis=1)
    inputIds = x_train1.squeeze()  
    symbolA = helper.getMapValues(nrSymbols)
    symbolA = np.array(symbolA)
    trueSymbols = symbolA[trueIndexes]

    saliency1Ds = reduceMap(saliency, do3DData=do3DData, do3rdStep=do3rdStep)

    if 'and' not in saliencyStacks.keys():
        saliencyStacks['and'] = dict()
        saliencyStacks['or'] = dict()
        saliencyStacks['xor'] = dict()
        for c in range(num_of_classes):
            saliencyStacks['and c' + str(c)] = dict()
            saliencyStacks['or c'+ str(c)] = dict()
            saliencyStacks['xor c'+ str(c)] = dict()
            saliencyStacks['and rank c' + str(c)] = dict()
            saliencyStacks['or rank c'+ str(c)] = dict()
            saliencyStacks['xor rank c'+ str(c)] = dict()
    

    for n in range(len(inputIds)):
        for j in range(andStack):
            combi = str(newTrain[n,nrAnds*j: nrAnds*(j+1)])
            saliencyVs = saliency[n,nrAnds*j: nrAnds*(j+1)]
            saliency1Vs = saliency1Ds[n,nrAnds*j: nrAnds*(j+1)]
            if combi not in saliencyStacks['and'].keys():
                saliencyStacks['and'][combi] = []
                for c in range(num_of_classes):
                    saliencyStacks['and c' + str(c)][combi] = []
                    saliencyStacks['and rank c'+ str(c)][combi] = []
            saliencyStacks['and'][combi].append(saliencyVs)
            saliencyStacks['and c'+str(targets[n])][combi].append(saliencyVs)
            temp = saliency1Vs.argsort()

            ranks = np.empty_like(temp)

            ranks[temp] = np.arange(len(saliency1Vs))
            saliencyStacks['and rank c'+str(targets[n])][combi].append(ranks)

            
        maxAnds = (nrAnds * andStack)

        for j in range(orStack):
            combi = str(newTrain[n,maxAnds + (nrOrs *j - orOffSet * j): maxAnds+(nrOrs *(j+1) - orOffSet * (j+1))])
            saliencyVs = saliency[n,maxAnds + (nrOrs *j - orOffSet * j): maxAnds+(nrOrs *(j+1) - orOffSet * (j+1))]
            saliency1Vs = saliency1Ds[n,maxAnds + (nrOrs *j - orOffSet * j): maxAnds+(nrOrs *(j+1) - orOffSet * (j+1))]
            if combi not in saliencyStacks['or'].keys():
                saliencyStacks['or'][combi] = []
                for c in range(num_of_classes):
                    saliencyStacks['or c'+ str(c)][combi] = []
                    saliencyStacks['or rank c'+ str(c)][combi] = []
            saliencyStacks['or'][combi].append(saliencyVs)
            saliencyStacks['or c'+str(targets[n])][combi].append(saliencyVs)
            temp = saliency1Vs.argsort()

            ranks = np.empty_like(temp)

            ranks[temp] = np.arange(len(saliency1Vs))
            saliencyStacks['or rank c'+str(targets[n])][combi].append(ranks)

        maxOrs = nrOrs * orStack - orOffSet * orStack

        for j in range(xorStack):
            combi = str(newTrain[n,maxAnds+maxOrs+(nrxor *j - xorOffSet * j): maxAnds+maxOrs+(nrxor *(j+1) - xorOffSet * (j+1))])
            saliencyVs = saliency[n,maxAnds+maxOrs+(nrxor *j - xorOffSet * j): maxAnds+maxOrs+(nrxor *(j+1) - xorOffSet * (j+1))]
            saliency1Vs = saliency1Ds[n,maxAnds+maxOrs+(nrxor *j - xorOffSet * j): maxAnds+maxOrs+(nrxor *(j+1) - xorOffSet * (j+1))]
            if combi not in saliencyStacks['xor'].keys():
                saliencyStacks['xor'][combi] = []
                for c in range(num_of_classes):
                    saliencyStacks['xor c'+ str(c)][combi] = []
                    saliencyStacks['xor rank c'+ str(c)][combi] = []


            saliencyStacks['xor'][combi].append(saliencyVs)
            saliencyStacks['xor c'+str(targets[n])][combi].append(saliencyVs)
            temp = saliency1Vs.argsort()

            ranks = np.empty_like(temp)

            ranks[temp] = np.arange(len(saliency1Vs))
            saliencyStacks['xor rank c'+str(targets[n])][combi].append(ranks)
            
            
    return saliencyStacks

#show how often irrelevant data is more important than any must have inforamtion
def getStackImportanceBreak(nrEmpty, saliencys, x_train1, classes, y_train1, nrSymbols, andStack, orStack, xorStack, nrAnds, nrOrs, nrxor, orOffSet, xorOffSet, trueIndexes, topLevel):
    targets = np.argmax(y_train1, axis=1)
    
    inputIds = x_train1#.squeeze()            
    symbolA = helper.getMapValues(nrSymbols)
    symbolA = np.array(symbolA)
    trueSymbols = symbolA[trueIndexes]       

    brokenImportance = dict()
    brokenStackImportance = dict()

    for c in classes: 
        brokenImportance[int(c)] =  0
        brokenImportance[int(c)] =  0
        brokenStackImportance[int(c)] =  np.zeros(andStack + orStack + xorStack)
        brokenStackImportance[int(c)] =  np.zeros(andStack + orStack + xorStack)
       
    for n in range(len(inputIds)):
        baseline = np.max(saliencys[n][-1* nrEmpty:])
        broken = False
        if targets[n] == 1 and topLevel == 'and':
            for j in range(andStack):
                for k in range((nrAnds*j), nrAnds*(j+1)):
                    if saliencys[n][k] < baseline:
                        brokenStackImportance[targets[n]][j] += 1
                        broken = True
                        break
            maxAnds = (nrAnds * andStack)
            for j in range(orStack):
                maxOr = -1
                orK = 0
                for k in range(maxAnds + (nrOrs *j - orOffSet * j),maxAnds+(nrOrs *(j+1) - orOffSet * (j+1))):
                    if round(float(inputIds[n][k]), 4) in trueSymbols:
                        if saliencys[n][k] > maxOr:
                            maxOr = saliencys[n][k]
                            orK = k
                if maxOr < baseline:
                    brokenStackImportance[targets[n]][j+ andStack] += 1
                    broken = True
            maxOrs = nrOrs * orStack - orOffSet * orStack

            for j in range(xorStack):
                for k in range(maxAnds+maxOrs+(nrxor *j - xorOffSet * j),maxAnds+maxOrs+(nrxor *(j+1) - xorOffSet * (j+1))):
                    if saliencys[n][k] < baseline:
                        broken = True
                        brokenStackImportance[targets[n]][j+ andStack + orStack] += 1
                        break

        elif targets[n] == 0 and topLevel == 'and': #One of the breakpoint conditions must be met!!
            strongestBreak = -1
            breakK = 0
            for j in range(andStack):
                andT = True
                for k in range((nrAnds*j), nrAnds*(j+1)):
                    if round(float(inputIds[n][k]), 4) not in trueSymbols:
                        andT = False
                        if saliencys[n][k] > strongestBreak:
                            strongestBreak = saliencys[n][k]
                            breakK = j
            maxAnds = (nrAnds * andStack)
            for j in range(orStack):
                orT = False
                for k in range(maxAnds + (nrOrs *j - orOffSet * j),maxAnds+(nrOrs *(j+1) - orOffSet * (j+1))):
                    if round(float(inputIds[n][k]), 4) in trueSymbols:
                        orT = True
                        break
                if not orT: #Take min because one is enough to break it!
                    minOr = np.min(saliencys[n][maxAnds + (nrOrs *j - orOffSet * j) : maxAnds+(nrOrs *(j+1) - orOffSet * (j+1))])
                    if minOr > strongestBreak:
                        strongestBreak = minOr
                        breakK = j + andStack
            maxOrs = nrOrs * orStack - orOffSet * orStack

            for j in range(xorStack): #Take min because one is enough to break it!
                xorT = False
                all0s = True
                ones = []
                for k in range(maxAnds+maxOrs+(nrxor *j - xorOffSet * j),maxAnds+maxOrs+(nrxor *(j+1) - xorOffSet * (j+1))):
                    if round(float(inputIds[n][k]), 4) in trueSymbols:
                        ones.append(saliencys[n][k])
                        if all0s:
                            xorT = True
                            all0s = False
                        elif xorT == True:
                            xorT = False
                            
                if not xorT:
                    if all0s:
                        minXOr =np.min(saliencys[n][maxAnds+maxOrs+(nrxor *j - xorOffSet * j) : maxAnds+maxOrs+(nrxor *(j+1) - xorOffSet * (j+1))])
                        if minXOr > strongestBreak:
                            strongestBreak = minXOr
                            breakK = j + andStack + orStack
                    else:
                        maxKs = np.argsort(ones)
                        maxk2 = maxKs[-2]
                        if ones[maxk2] > strongestBreak:
                            strongestBreak = ones[maxk2]
                            breakK = j + andStack + orStack

            if strongestBreak < baseline:
                broken = True
                brokenStackImportance[targets[n]][breakK] += 1



        elif targets[n] == 0 and topLevel == 'or':
            for j in range(andStack):
                maxAnd = -1
                for k in range((nrAnds*j), nrAnds*(j+1)):
                    if round(float(inputIds[n][k]), 4) not in trueSymbols:
                        if saliencys[n][k] > maxAnd:
                            maxAnd = saliencys[n][k]
                if maxAnd < baseline:
                    brokenStackImportance[targets[n]][j] += 1
                    broken = True
            maxAnds = (nrAnds * andStack)
            for j in range(orStack):
                minOr = np.min(saliencys[n][maxAnds + (nrOrs *j - orOffSet * j) : maxAnds+(nrOrs *(j+1) - orOffSet * (j+1))])
                if minOr < baseline:
                    brokenStackImportance[targets[n]][j+ andStack] += 1
                    broken = True
            maxOrs = nrOrs * orStack - orOffSet * orStack

            for j in range(xorStack):
                all0s = True
                ones = []
                for k in range(maxAnds+maxOrs+(nrxor *j - xorOffSet * j),maxAnds+maxOrs+(nrxor *(j+1) - xorOffSet * (j+1))):
                    if round(float(inputIds[n][k]), 4) in trueSymbols:
                        all0s = False
                        ones.append(saliencys[n][k])
                        
                if all0s:
                    minXOr =np.min(saliencys[n][maxAnds+maxOrs+(nrxor *j - xorOffSet * j) : maxAnds+maxOrs+(nrxor *(j+1) - xorOffSet * (j+1))])
                    if minXOr < baseline:
                        broken = True
                        brokenStackImportance[targets[n]][j+ andStack + orStack] += 1
                        
                else:
                    maxKs = np.argsort(ones)
                    maxk2 = maxKs[-2]
                    if ones[maxk2] < baseline:
                        broken = True
                        brokenStackImportance[targets[n]][j+ andStack + orStack] += 1
                        

        elif targets[n] == 1 and topLevel == 'or': #One of the breakpoint conditions must be met!!
            strongestBreak = -1
            breakK = 0
            for j in range(andStack):
                andT = True
                for k in range((nrAnds*j), nrAnds*(j+1)):
                    if round(float(inputIds[n][k]), 4) not in trueSymbols:
                        andT = False
                        break
                if andT:
                    andnMin = np.min(saliencys[n][(nrAnds*j): nrAnds*(j+1)])
                    if andnMin > strongestBreak:
                        strongestBreak = andnMin
                        breakK = j
            maxAnds = (nrAnds * andStack)
            for j in range(orStack):
                biggestOr = -1
                orT = False
                for k in range(maxAnds + (nrOrs *j - orOffSet * j),maxAnds+(nrOrs *(j+1) - orOffSet * (j+1))):
                    if round(float(inputIds[n][k]), 4) in trueSymbols:
                        orT = True
                        if saliencys[n][k] > biggestOr:
                            biggestOr = saliencys[n][k]
                if orT: #Take min because one is enough to break it!
                    if biggestOr > strongestBreak:
                        strongestBreak = biggestOr
                        breakK = j + andStack
            maxOrs = nrOrs * orStack - orOffSet * orStack

            for j in range(xorStack): #Take min because one is enough to break it!
                xorT = False
                for k in range(maxAnds+maxOrs+(nrxor *j - xorOffSet * j),maxAnds+maxOrs+(nrxor *(j+1) - xorOffSet * (j+1))):
                    if round(float(inputIds[n][k]), 4) in trueSymbols:
                        if xorT == True:
                            xorT = False
                            break
                        else:
                            xorT = True
                if xorT:
                    minXOr = np.min(saliencys[n][maxAnds+maxOrs+(nrxor *j - xorOffSet * j) : maxAnds+maxOrs+(nrxor *(j+1) - xorOffSet * (j+1))])
                    if minXOr > strongestBreak:
                        strongestBreak = minXOr
                        breakK = j + andStack + orStack

            if strongestBreak < baseline:
                broken = True
                brokenStackImportance[targets[n]][breakK] += 1


        elif targets[n] == 0 and topLevel == 'xor':
            all0 = True
            maxK1 = -1
            maxK2 = -1
            maxV = -1
            maxV2 = -1
            xorMin = []
            for j in range(andStack):
                andT = True
                andMax= 0 
                for k in range((nrAnds*j), nrAnds*(j+1)):
                    if round(float(inputIds[n][k]), 4) not in trueSymbols:
                        andT = False
                        if all0:
                            if saliencys[n][k] > andMax:
                                andMax = saliencys[n][k]
                        else:
                            break
                        
                if andT:
                    all0 = False
                    andMax = np.min(saliencys[n][(nrAnds*j): nrAnds*(j+1)])

                    if andMax > maxV:
                        maxV2 = maxV
                        maxV = andMax
                        maxK2 = maxK1
                        maxK1 = j
                    elif andMax > maxV2:
                        maxV2 = andMax
                        maxK2 = j
                else:
                    xorMin.append(andMax)
                
            maxAnds = (nrAnds * andStack)
            for j in range(orStack):
                orMax = -1
                orT = False
                for k in range(maxAnds + (nrOrs *j - orOffSet * j),maxAnds+(nrOrs *(j+1) - orOffSet * (j+1))):
                    if round(float(inputIds[n][k]), 4) in trueSymbols:
                        orT = True
                        if saliencys[n][k] > orMax:
                            orMax = saliencys[n][k]

                if orT:
                    all0 = False
                    if orMax > maxV:
                        maxV2 = maxV
                        maxV = orMax
                        maxK2 = maxK1
                        maxK1 = j + andStack
                    elif orMax > maxV2:
                        maxV2 = orMax
                        maxK2 = j + andStack
                else:
                    orMax = np.min(saliencys[n][maxAnds + (nrOrs *j - orOffSet * j): maxAnds+(nrOrs *(j+1) - orOffSet * (j+1))])
                    xorMin.append(orMax)

            maxOrs = nrOrs * orStack - orOffSet * orStack


            for j in range(xorStack): #Take min because one is enough to break it!
                xorT = False
                internHighest = -1
                internNdHighest = -1
                all0Intern = True

                for k in range(maxAnds+maxOrs+(nrxor *j - xorOffSet * j),maxAnds+maxOrs+(nrxor *(j+1) - xorOffSet * (j+1))):
                    if round(float(inputIds[n][k]), 4) in trueSymbols:
                        if saliencys[n][k] > internHighest:
                            internHighest = internNdHighest
                            internHighest = saliencys[n][k]
                        elif saliencys[n][k] > internNdHighest:
                            internNdHighest = saliencys[n][k]

                        if all0Intern:
                            all0Intern = False
                            xorT = True
                        elif xorT == True:
                            xorT = False


                if xorT:
                    all0 = False
                    minXOr = np.min(saliencys[n][maxAnds+maxOrs+(nrxor *j - xorOffSet * j) : maxAnds+maxOrs+(nrxor *(j+1) - xorOffSet * (j+1))])
                    if minXOr > maxV:
                        maxV2 = maxV
                        maxV = minXOr
                        maxK2 = maxK1
                        maxK1 = j + andStack + orStack
                    elif minXOr > maxV2:
                        maxV2 = minXOr
                        maxK2 = j + andStack + orStack
                else:
                    
                    if all0Intern:
                        minXOr = np.min(saliencys[n][maxAnds+maxOrs+(nrxor *j - xorOffSet * j) : maxAnds+maxOrs+(nrxor *(j+1) - xorOffSet * (j+1))])
                    else:
                        minXOr = internNdHighest
                    xorMin.append(minXOr)




            if all0:
                for j, v in enumerate(xorMin):
                    if v < baseline:
                        brokenStackImportance[targets[n]][j] += 1
                        broken = True

            else:
                if maxV2 < baseline:
                    brokenStackImportance[targets[n]][maxK2] += 1
                    brokenStackImportance[targets[n]][maxK1] += 1
                    broken = True


        elif targets[n] == 1 and topLevel == 'xor': #One of the breakpoint conditions must be met!!
            for j in range(andStack):
                andMax = -1
                andT = True
                for k in range((nrAnds*j), nrAnds*(j+1)):
                    if round(float(inputIds[n][k]), 4) not in trueSymbols:
                        andT = False
                        if saliencys[n][k] > andMax:
                            andMax = saliencys[n][k]
                if andT:
                    andMax = np.min(saliencys[n][(nrAnds*j): nrAnds*(j+1)])

                if andMax < baseline:
                    brokenStackImportance[targets[n]][j] += 1
                    broken = True
                    
                
            maxAnds = (nrAnds * andStack)
            for j in range(orStack):
                orMax = -1
                orT = False
                for k in range(maxAnds + (nrOrs *j - orOffSet * j),maxAnds+(nrOrs *(j+1) - orOffSet * (j+1))):
                    if round(float(inputIds[n][k]), 4) in trueSymbols:
                        orT = True
                        if saliencys[n][k] > orMax:
                            orMax = saliencys[n][k]
                if not orT: #Take min because one is enough to break it!
                    orMax = np.min(saliencys[n][maxAnds + (nrOrs *j - orOffSet * j): maxAnds+(nrOrs *(j+1) - orOffSet * (j+1))])
                
                if orMax < baseline:
                    brokenStackImportance[targets[n]][j + andStack] += 1
                    broken = True
                    
            maxOrs = nrOrs * orStack - orOffSet * orStack


            for j in range(xorStack): #Take min because one is enough to break it!
                xorT = False
                all0Intern = True
                ones = []
                for k in range(maxAnds+maxOrs+(nrxor *j - xorOffSet * j),maxAnds+maxOrs+(nrxor *(j+1) - xorOffSet * (j+1))):
                    if round(float(inputIds[n][k]), 4) in trueSymbols:
                        ones.append(saliencys[n][k])
                        if all0Intern:
                            all0Intern= False
                            xorT = True
                        elif xorT:
                            xorT = False

                if xorT or all0Intern:
                    minXOr = np.min(saliencys[n][maxAnds+maxOrs+(nrxor *j - xorOffSet * j) : maxAnds+maxOrs+(nrxor *(j+1) - xorOffSet * (j+1))])
                else:
                    maxKs = np.argsort(ones)
                    maxk2 = maxKs[-2]
                    minXOr = ones[maxk2]

                if minXOr < baseline:
                    brokenStackImportance[targets[n]][j + andStack + orStack] += 1
                    broken = True
                    

        if broken:
            brokenImportance[targets[n]] += 1

    flatBroken = 0

    for k in brokenImportance.keys():
        if brokenImportance[k] == 0:
            brokenStackImportance[k] = 0
        else:
            brokenStackImportance[k] = brokenStackImportance[k]/brokenImportance[k]
        flatBroken += brokenImportance[k] 
        brokenImportance[k] = brokenImportance[k]/np.sum(targets == k)

    flatBroken = flatBroken / len(targets)
    return brokenImportance, brokenStackImportance, flatBroken

#gives how often inputs with information is below the irrelevant data
def getPredictionSaliency(nrEmpty, saliencys, x_train1, classes, y_train1, nrSymbols, andStack, orStack, xorStack, nrAnds, nrOrs, nrxor, orOffSet, xorOffSet, trueIndexes):
    targets = np.argmax(y_train1, axis=1)
    
    inputIds = x_train1.squeeze()            
    symbolA = helper.getMapValues(nrSymbols)
    symbolA = np.array(symbolA)
    trueSymbols = symbolA[trueIndexes]       

    wrongImportanceMeaning = dict() 
    countImportanceMeaning = dict()
    wrongGIB = 0
    for c in classes:
        wrongImportanceMeaning[int(c)] =  np.zeros(len(x_train1[0]))
        countImportanceMeaning[int(c)] =  np.zeros(len(x_train1[0]))
       
            
    for n in range(len(inputIds)):
        baseline = np.max(saliencys[n][-1* nrEmpty:])
        if targets[n] == 0:
            for j in range(andStack):
                for k in range((nrAnds*j), nrAnds*(j+1)):
                    if round(float(inputIds[n][k]), 4) not in trueSymbols:
                        if saliencys[n][k] < baseline:
                            wrongImportanceMeaning[targets[n]][k] += 1
                        countImportanceMeaning[targets[n]][k] += 1
        if targets[n] == 1:
            for j in range(andStack):
                for k in range((nrAnds*j), nrAnds*(j+1)):
                    if round(float(inputIds[n][k]), 4) in trueSymbols:
                        if saliencys[n][k] < baseline:
                            wrongImportanceMeaning[targets[n]][k] += 1
                        countImportanceMeaning[targets[n]][k] += 1

        maxAnds = (nrAnds * andStack)


        if targets[n] == 1:
            for j in range(orStack):
                for k in range(maxAnds + (nrOrs *j - orOffSet * j),maxAnds+(nrOrs *(j+1) - orOffSet * (j+1))):
                    if round(float(inputIds[n][k]), 4) in trueSymbols:
                        if saliencys[n][k] < baseline:
                            wrongImportanceMeaning[targets[n]][k] += 1
                        countImportanceMeaning[targets[n]][k] += 1
                        
        if targets[n] == 0:
            for j in range(orStack):
                for k in range(maxAnds + (nrOrs *j - orOffSet * j),maxAnds+(nrOrs *(j+1) - orOffSet * (j+1))):
                    if round(float(inputIds[n][k]), 4) not in trueSymbols:
                        if saliencys[n][k] < baseline:
                            wrongImportanceMeaning[targets[n]][k] += 1
                        countImportanceMeaning[targets[n]][k] += 1

        maxOrs = nrOrs * orStack - orOffSet * orStack


        for j in range(xorStack):
            all0 = True
            highest = 0
            ndHighest = 0
            k1 = -1
            k2 = -1
            if targets[n] == 0:
                for k in range(maxAnds+maxOrs+(nrxor *j - xorOffSet * j),maxAnds+maxOrs+(nrxor *(j+1) - xorOffSet * (j+1))):
                    if round(float(inputIds[n][k]), 4) not in trueSymbols:
                        all0 = False
                        break
                    if saliencys[n][k] > highest:
                        ndHighest = highest
                        highest=saliencys[n][k]
                        k2 = k1
                        k1 = k
                    elif saliencys[n][k] > ndHighest:
                        ndHighest = saliencys[n][k]
                        k2 = k

                if not all0: #NOTE assumption -1 could be ignored if two 1s are seen
                    if saliencys[n][k1] < baseline:
                        wrongImportanceMeaning[targets[n]][k1] += 1
                    countImportanceMeaning[targets[n]][k1] += 1  
                    if saliencys[n][k2] < baseline:
                        wrongImportanceMeaning[targets[n]][k2] += 1
                    countImportanceMeaning[targets[n]][k2] += 1       

            for k in range(maxAnds+maxOrs+(nrxor *j - xorOffSet * j),maxAnds+maxOrs+(nrxor *(j+1) - xorOffSet * (j+1))):
                if targets[n] == 1:
                    if saliencys[n][k] < baseline:
                        wrongImportanceMeaning[targets[n]][k] += 1
                    countImportanceMeaning[targets[n]][k] += 1       
                elif targets[n] == 0: 
                    if all0:
                        if saliencys[n][k] < baseline:
                            wrongImportanceMeaning[targets[n]][k] += 1
                        countImportanceMeaning[targets[n]][k] += 1
                    else:
                        break      

    return wrongImportanceMeaning, countImportanceMeaning

def getPredictionMaps(device, model, x_train1, newTrain, y_train1, num_of_classes, toplevel, nrSymbols, andStack, orStack, xorStack, nrAnds, nrOrs, nrxor, orOffSet, xorOffSet, trueIndexes, val_batch_size=1000, batch_size=1000):
    preds = y_train1
    
    targets = np.argmax(preds, axis=1)
    valueMap = dict()
    conValueMap = dict()
    classConValueMap = dict()
    minConValueMap = dict()
    inputIds = x_train1#.squeeze()            
    for c in range(num_of_classes):
        
        for j in range(andStack):
            b = []
            for i, a in enumerate(newTrain[:,nrAnds*j: nrAnds*(j+1)]):
                if targets[i] == c and -2 in a:
                        b.append(a)
            if 'and'+str(j) not in valueMap.keys():
                valueMap['and'+str(j)] = dict()
                conValueMap['and'+str(j)] = dict()
                minConValueMap['and'+str(j)] = dict()
                classConValueMap['and'+str(j)] = dict()
            valueMap['and'+str(j)][c] =  np.unique(np.array(b), axis=0)
            conValueMap['and'+str(j)][c] = []
            minConValueMap['and'+str(j)][c] = []
            classConValueMap['and'+str(j)][c] = dict()
            for c2 in range(num_of_classes):
                classConValueMap['and'+str(j)][c][c2] = []
            
        maxAnds = (nrAnds * andStack)

        for j in range(orStack):
            b = []
            for i, a in enumerate(newTrain[:,maxAnds + (nrOrs *j - orOffSet): maxAnds+(nrOrs *(j+1) - orOffSet)]):
                if targets[i] == c and -2 in a:
                    b.append(a)
            if 'or'+str(j) not in valueMap.keys():
                valueMap['or'+str(j)] = dict()
                conValueMap['or'+str(j)] = dict()
                minConValueMap['or'+str(j)] = dict()
                classConValueMap['or'+str(j)] = dict()
            valueMap['or'+str(j)][c] =  np.unique(np.array(b), axis=0)
            conValueMap['or'+str(j)][c] = []
            minConValueMap['or'+str(j)][c] = []
            classConValueMap['or'+str(j)][c] = dict()
            for c2 in range(num_of_classes):
                classConValueMap['or'+str(j)][c][c2] = []

        maxOrs = nrOrs * orStack - orOffSet * orStack
        for j in range(xorStack):
            b = []
            for i, a in enumerate(newTrain[:,maxAnds+maxOrs+(nrxor *j - xorOffSet * j): maxAnds+maxOrs+(nrxor *(j+1) - xorOffSet *(j+1))]):
                if targets[i] == c and -2 in a:
                    b.append(a)
            if 'xor'+str(j) not in valueMap.keys():
                valueMap['xor'+str(j)] = dict()
                conValueMap['xor'+str(j)] = dict()
                minConValueMap['xor'+str(j)] = dict()
                classConValueMap['xor'+str(j)] = dict()
            valueMap['xor'+str(j)][c] =  np.unique(np.array(b), axis=0)
            conValueMap['xor'+str(j)][c] = []
            minConValueMap['xor'+str(j)][c] = []
            classConValueMap['xor'+str(j)][c] = dict()
            for c2 in range(num_of_classes):
                classConValueMap['xor'+str(j)][c][c2] = []
            
            
    symbolA = helper.getMapValues(nrSymbols)
    symbolA = np.array(symbolA)
    trueSymbols = symbolA[trueIndexes]       
            
    for n in range(len(inputIds)):
        andTs = []
        for j in range(andStack):
            andT = True
            for k in range((nrAnds*j), nrAnds*(j+1)):
                if round(float(inputIds[n][k]), 4) not in trueSymbols:
                    andT = False
                    break
            if nrAnds == 0:
                andT = True
            if andT:
                #conValueMap['and'+str(j)][targets[n]].append(newTrain[n,nrAnds*j: nrAnds*(j+1)])
                conValueMap['and'+str(j)][1].append(newTrain[n,nrAnds*j: nrAnds*(j+1)])
                classConValueMap['and'+str(j)][1][targets[n]].append(newTrain[n,nrAnds*j: nrAnds*(j+1)])
            else:
                conValueMap['and'+str(j)][0].append(newTrain[n,nrAnds*j: nrAnds*(j+1)])
                classConValueMap['and'+str(j)][0][targets[n]].append(newTrain[n,nrAnds*j: nrAnds*(j+1)])

            
            andTs.append(andT)
        maxAnds = (nrAnds * andStack)

        orTs = []
        for j in range(orStack):
            orT = False
            for k in range(maxAnds + (nrOrs *j - orOffSet * j),maxAnds+(nrOrs *(j+1) - orOffSet * (j+1))):
                if round(float(inputIds[n][k]), 4) in trueSymbols:
                    orT = True
                    break
            if nrOrs == 0:
                orT = True
            if orT:
                conValueMap['or'+str(j)][1].append(newTrain[n,maxAnds + (nrOrs *j - orOffSet * j): maxAnds+(nrOrs *(j+1) - orOffSet * (j+1))])
                classConValueMap['or'+str(j)][1][targets[n]].append(newTrain[n,maxAnds + (nrOrs *j - orOffSet * j): maxAnds+(nrOrs *(j+1) - orOffSet * (j+1))])

            else:
                conValueMap['or'+str(j)][0].append(newTrain[n,maxAnds + (nrOrs *j - orOffSet * j): maxAnds+(nrOrs *(j+1) - orOffSet * (j+1))])
                classConValueMap['or'+str(j)][0][targets[n]].append(newTrain[n,maxAnds + (nrOrs *j - orOffSet * j): maxAnds+(nrOrs *(j+1) - orOffSet * (j+1))])


            orTs.append(orT)
        maxOrs = nrOrs * orStack - orOffSet * orStack

        xorTs = []
        for j in range(xorStack):
            xorT = False
            for k in range(maxAnds+maxOrs+(nrxor *j - xorOffSet * j),maxAnds+maxOrs+(nrxor *(j+1) - xorOffSet * (j+1))):
                if round(float(inputIds[n][k]), 4) in trueSymbols:
                    if xorT == True:
                        xorT = False
                        break
                    else:
                        xorT = True
            if nrxor == 0:
                xorT = True
            if xorT:
                conValueMap['xor'+str(j)][1].append(newTrain[n,maxAnds+maxOrs+(nrxor *j - xorOffSet * j): maxAnds+maxOrs+(nrxor *(j+1) - xorOffSet * (j+1))])
                classConValueMap['xor'+str(j)][1][targets[n]].append(newTrain[n,maxAnds+maxOrs+(nrxor *j - xorOffSet * j): maxAnds+maxOrs+(nrxor *(j+1) - xorOffSet * (j+1))])

            else: 
                conValueMap['xor'+str(j)][0].append(newTrain[n,maxAnds+maxOrs+(nrxor *j - xorOffSet * j): maxAnds+maxOrs+(nrxor *(j+1) - xorOffSet * (j+1))])
                classConValueMap['xor'+str(j)][0][targets[n]].append(newTrain[n,maxAnds+maxOrs+(nrxor *j - xorOffSet * j): maxAnds+maxOrs+(nrxor *(j+1) - xorOffSet * (j+1))])

            xorTs.append(xorT)
         
        if (toplevel == 'and' and targets[n] == 1) or (toplevel == 'or' and targets[n] == 0):
            for k3 in conValueMap.keys():
                for j in range(len(andTs)):
                        minConValueMap['and'+str(j)][targets[n]].append(conValueMap['and'+str(j)][andTs[j]][-1])
                for j in range(len(orTs)):
                        minConValueMap['or'+str(j)][targets[n]].append(conValueMap['or'+str(j)][orTs[j]][-1])
                for j in range(len(xorTs)):
                        minConValueMap['xor'+str(j)][targets[n]].append(conValueMap['xor'+str(j)][xorTs[j]][-1])
        elif (toplevel == 'and' and targets[n] == 0) or (toplevel == 'or' and targets[n] == 1):
            if (np.sum(andTs == targets[n]) + np.sum(orTs == targets[n]) + np.sum(xorTs == targets[n])) == 1:
                for j in range(len(andTs)):
                    if andTs[j] == targets[n]:
                        minConValueMap['and'+str(j)][targets[n]].append(conValueMap['and'+str(j)][andTs[j]][-1])
                for j in range(len(orTs)):
                    if orTs[j] == targets[n]:
                        minConValueMap['or'+str(j)][targets[n]].append(conValueMap['or'+str(j)][orTs[j]][-1])
                for j in range(len(xorTs)):
                    if xorTs[j] == targets[n]:
                        minConValueMap['xor'+str(j)][targets[n]].append(conValueMap['xor'+str(j)][xorTs[j]][-1])
        elif (toplevel == 'xor' and targets[n] == 1):
            for j in range(len(andTs)):
                    minConValueMap['and'+str(j)][andTs[j]].append(conValueMap['and'+str(j)][andTs[j]][-1])
            for j in range(len(orTs)):
                    minConValueMap['or'+str(j)][orTs[j]].append(conValueMap['or'+str(j)][orTs[j]][-1])
            for j in range(len(xorTs)):
                    minConValueMap['xor'+str(j)][xorTs[j]].append(conValueMap['xor'+str(j)][xorTs[j]][-1])
        elif (toplevel == 'xor' and targets[n] == 0):
            if (np.sum(andTs == targets[n]) + np.sum(orTs == targets[n]) + np.sum(xorTs == targets[n])) == (nrAnds + nrOrs + nrxor):
                for j in range(len(andTs)):
                    minConValueMap['and'+str(j)][targets[n]].append(conValueMap['and'+str(j)][andTs[j]][-1])
                for j in range(len(orTs)):
                    minConValueMap['or'+str(j)][targets[n]].append(conValueMap['or'+str(j)][orTs[j]][-1])
                for j in range(len(xorTs)):
                    minConValueMap['xor'+str(j)][targets[n]].append(conValueMap['xor'+str(j)][xorTs[j]][-1])
            elif(np.sum(andTs == 1) + np.sum(orTs == 1) + np.sum(xorTs == 1)) == 2:
                for j in range(len(andTs)):
                    if andTs[j] == 1:
                        minConValueMap['and'+str(j)][1].append(conValueMap['and'+str(j)][andTs[j]][-1])
                for j in range(len(orTs)):
                    if orTs[j] == 1:
                        minConValueMap['or'+str(j)][1].append(conValueMap['or'+str(j)][orTs[j]][-1])
                for j in range(len(xorTs)):
                    if xorTs[j] == 1:
                        minConValueMap['xor'+str(j)][1].append(conValueMap['xor'+str(j)][xorTs[j]][-1])


        
            
            
                   

    return valueMap, conValueMap, classConValueMap, minConValueMap


def fillOutTruthTable(data, target, nrSymbols, topLevel, andTables, orTables, xorTables, nrAnds = 2, nrOrs = 2, nrxor = 2, trueIndexes=[3,1], orOffSet=0, xorOffSet=0, alsoTopLevel=False):
    predCur = []
    for s in data:

        andTs = []
        for j, andTable in enumerate(andTables):
            combi = str(s[(nrAnds*j): nrAnds*(j+1)].squeeze())
            if combi in andTable.keys():
                andT = andTable[combi]
            else:
                andT = -1
            andTs.append(andT)
        maxAnds = (nrAnds * len(andTables))

        orTs = []
        for j, orTable in enumerate(orTables):

            combi = str(s[maxAnds + (nrOrs *j - orOffSet * j):maxAnds+(nrOrs *(j+1) - orOffSet * (j+1))].squeeze())
            if combi in orTable.keys():
                orT = orTable[combi]
            else:
                orT = -1
            orTs.append(orT)

        maxOrs = nrOrs * len(orTables) - orOffSet * len(orTables)

        xorTs = []
        for j, xorTable in enumerate(xorTables):
            combi = str(s[maxAnds+maxOrs+(nrxor *j - xorOffSet * j):maxAnds+maxOrs+(nrxor *(j+1) - xorOffSet * (j+1))].squeeze())
            if combi in xorTable.keys():
                xorT = xorTable[combi]
            else:
                xorT = -1
            xorTs.append(xorT)

        andTs = np.array(andTs)
        orTs = np.array(orTs)
        xorTs = np.array(xorTs)

        #TODO complex toplevels
        
        if topLevel == 'and':
            if (np.sum(andTs == 0) + np.sum(orTs == 0) + np.sum(xorTs == 0)) > 0:
                    predCur.append(0)
            elif (np.sum(andTs == -1) + np.sum(orTs==-1) + np.sum(xorTs==-1)) > 0:
                predCur.append(-1)
            else:
                predCur.append(1)
        elif topLevel == 'or':
            if (np.sum([np.sum(andTs == 1), np.sum(orTs == 1), np.sum(xorTs == 1)]) >= 1):
                predCur.append(1)
            elif (np.sum(andTs == -1) + np.sum(orTs==-1) + np.sum(xorTs==-1)) > 0:
                predCur.append(-1)
            else:
                predCur.append(0)
        elif topLevel == 'xor':
            if (np.sum(andTs == 1) + np.sum(orTs == 1) + np.sum(xorTs == 1) == 2):
                predCur.append(0)
            elif (np.sum(andTs == -1) + np.sum(orTs==-1) + np.sum(xorTs==-1)) > 0:
                predCur.append(-1)
            elif (np.sum(andTs == 1) + np.sum(orTs == 1) + np.sum(xorTs == 1) == 1):
                predCur.append(1)
            else:
                predCur.append(0)
        else:
                raise ValueError("Need a valid top level")

    acc = accuracy_score(np.array(predCur),target.argmax(axis=1))
    return acc, predCur

def makeMaxPredictions(newData, lables, symbolCount, topLevel, trueIndexes, andStack, orStack, xorStack, nrAnds, nrOrs, nrxor, orOffSet, xorOffSet):
    inputData = np.squeeze(newData)
    inputLable = lables
    index = np.zeros(len(inputData[0]))
    indexBest = []


    
    logicMax, logicPred = logicAcc(inputData, inputLable, symbolCount, topLevel, trueIndexes=trueIndexes, andStack = andStack, orStack = orStack, xorStack = xorStack, nrAnds = nrAnds, nrOrs = nrOrs, nrxor = nrxor, orOffSet=orOffSet, xorOffSet=xorOffSet)

    andTable, fillinAnds, orTable, fillinOrs, xorTable, fillinXors = createTruthTable(index, inputData, inputLable, symbolCount, topLevel, trueIndexes=trueIndexes, andStack = andStack, orStack = orStack, xorStack = xorStack, nrAnds = nrAnds, nrOrs = nrOrs, nrxor = nrxor, orOffSet=orOffSet, xorOffSet=xorOffSet)

    #TODO top level?
    andTables = []
    orTables = []
    xorTables = []

    
    for s in range(andStack):
        andTables.append(andTable.copy())
    for s in range(orStack):
        orTables.append(orTable.copy())
    for s in range(xorStack):
        xorTables.append(xorTable.copy())

    for s in range(andStack):
        for f in fillinAnds:
            accP = []
            for v in [0,1]:
                andTables[s][str(f)] = v
                accV, _ = fillOutTruthTable(inputData, inputLable, symbolCount, topLevel, andTables, orTables, xorTables, nrAnds = nrAnds, nrOrs = nrOrs, nrxor = nrxor, trueIndexes=trueIndexes, orOffSet=orOffSet, xorOffSet=xorOffSet, alsoTopLevel=False)
                accP.append(accV)
            andTables[s][str(f)] = np.argmax(accP)

    for s in range(orStack):
        for f in fillinOrs:
            accP = []
            for v in [0,1]:
                orTables[s][str(f)] = v
                accV, _ = fillOutTruthTable(inputData, inputLable, symbolCount, topLevel, andTables, orTables, xorTables, nrAnds = nrAnds, nrOrs = nrOrs, nrxor = nrxor, trueIndexes=trueIndexes, orOffSet=orOffSet, xorOffSet=xorOffSet, alsoTopLevel=False)
                accP.append(accV)
            orTables[s][str(f)] = np.argmax(accP)

    for s in range(xorStack):
        for f in fillinXors:
            accP = []
            for v in [0,1]:
                xorTables[s][str(f)] = v
                accV, _ = fillOutTruthTable(inputData, inputLable, symbolCount, topLevel, andTables, orTables, xorTables, nrAnds = nrAnds, nrOrs = nrOrs, nrxor = nrxor, trueIndexes=trueIndexes, orOffSet=orOffSet, xorOffSet=xorOffSet, alsoTopLevel=False)
                accP.append(accV)
            xorTables[s][str(f)] = np.argmax(accP)

    finalAcc, finalPred = fillOutTruthTable(inputData, inputLable, symbolCount, topLevel, andTables, orTables, xorTables, nrAnds = nrAnds, nrOrs = nrOrs, nrxor = nrxor, trueIndexes=trueIndexes, orOffSet=orOffSet, xorOffSet=xorOffSet, alsoTopLevel=False)

    baselineAcc = accuracy_score(np.zeros(len(inputLable)), inputLable.argmax(axis=1)) #High baseline dosnt matter, because we trained unbiased

    return finalAcc, finalPred, baselineAcc, logicMax, logicPred

def createTruthTable(index, data, target, nrSymbols, topLevel, andStack = 1, orStack = 1, xorStack = 1, nrAnds = 2, nrOrs = 2, nrxor = 2, trueIndexes=[3,1], orOffSet=0, xorOffSet=0):
    predCur = []
    symbolA = helper.getMapValues(nrSymbols)
    symbolA = np.array(symbolA)
    trueSymbols = symbolA[trueIndexes]
    falseSymbols = np.delete(symbolA, trueIndexes)
    symbols = nrSymbols+1
    symbolB = []
    for a in symbolA:
        symbolB.append(a)
    symbolB.append(-2)
    symbolB=np.array(symbolB)

    andTable = dict()
    fillinAnds = []
    size = pow(symbols, nrAnds)
    for i in range(size):
        combi = np.zeros(nrAnds)
        p = i
        for j in range(nrAnds):
            inV = p % symbols
            combi[j] = symbolB[inV]
            p = int(p/symbols)
        
        andT = 1
        for k in combi:
            if k == -2 and (andT == 1 or andT == -1):
                andT = -1
            elif round(float(k), 4) not in trueSymbols:
                andT = 0
                break
        
        if andT != -1:
            andTable[str(combi)] = andT
        else:
            fillinAnds.append(str(combi))


    orTable = dict()
    fillinOrs = []
    size = pow(symbols, nrOrs)
    for i in range(size):
        combi = np.zeros(nrOrs)
        p = i
        for j in range(nrOrs):
            inV = p % symbols
            combi[j] = symbolB[inV]
            p = int(p/symbols)
        
        orT = 0
        for k in combi:
            if round(float(k), 4) in trueSymbols:
                orT = 1
                break
            elif k == -2 and orT==0:
                orT = -1
        
        if orT != -1:
            orTable[str(combi)] = orT
        else:
            fillinOrs.append(str(combi))

    xorTable = dict()
    fillinXors = []
    size = pow(symbols, nrxor)
    for i in range(size):
        combi = np.zeros(nrxor)
        p = i
        for j in range(nrxor):
            inV = p % symbols
            combi[j] = symbolB[inV]
            p = int(p/symbols)
        
        xorT = 0
        broken = False
        for k in combi:
            if round(float(k), 4) in trueSymbols:
                if xorT == 1:
                    xorT = 0
                    broken = False
                    break
                else:
                    xorT = 1
            elif k == -2:
                broken = True
        if broken:
            xorT = -1
        
        if xorT != -1:
            xorTable[str(combi)] = xorT
        else:
            fillinXors.append(str(combi))

    return andTable, fillinAnds, orTable, fillinOrs, xorTable, fillinXors

def logicAccGuess(data, target, nrSymbols, topLevel, andStack = 1, orStack = 1, xorStack = 1, nrAnds = 2, nrOrs = 2, nrxor = 2, trueIndexes=[3,1], orOffSet=0, xorOffSet=0):
    predCur = []
    symbolA = helper.getMapValues(nrSymbols)
    symbolA = np.array(symbolA)
    trueSymbols = symbolA[trueIndexes]
    falseSymbols = np.delete(symbolA, trueIndexes)

    for s in data:

        andTs = []
        for j in range(andStack):
            maskCOunt = 0
            andT = 1
            for k in range((nrAnds*j), nrAnds*(j+1)):
                if s[k] == -2 and (andT == 1 or andT == -1):
                    andT = -1
                    maskCOunt += 1
                elif round(float(s[k]), 4) not in trueSymbols:
                    andT = 0
                    break
            if nrAnds == 0:
                andT = 1
            if maskCOunt >= 2:
                andT = 0
            andTs.append(andT)
        maxAnds = (nrAnds * andStack)

        orTs = []
        for j in range(orStack):
            orT = 0
            maskCOunt = 0
            for k in range(maxAnds + (nrOrs *j - orOffSet * j),maxAnds+(nrOrs *(j+1) - orOffSet * (j+1))):
                if s[k] == -2:
                    maskCOunt += 1
                if round(float(s[k]), 4) in trueSymbols:
                    orT = 1
                    break
                elif s[k] == -2 and orT==0:
                    orT = -1
            if nrOrs == 0:
                orT = 1
            if maskCOunt >= 2:
                orT = 1
            orTs.append(orT)
        maxOrs = nrOrs * orStack - orOffSet * orStack

        xorTs = []
        for j in range(xorStack):
            xorT = 0
            maskCOunt = 0
            broken = False
            for k in range(maxAnds+maxOrs+(nrxor *j - xorOffSet * j),maxAnds+maxOrs+(nrxor *(j+1) - xorOffSet * (j+1))):
                if round(float(s[k]), 4) in trueSymbols:
                    if xorT == 1:
                        xorT = 0
                        broken = False
                        break
                    else:
                        xorT = 1
                elif s[k] == -2:
                    broken = True
                    maskCOunt += 1
            if nrxor == 0:
                xorT = 1
            if broken:
                if maskCOunt >= 3:
                    xorT = 0
                else:
                    xorT = -1
            xorTs.append(xorT)

        andTs = np.array(andTs)
        orTs = np.array(orTs)
        xorTs = np.array(xorTs)

        if topLevel == 'complex':
            if (np.sum(andTs) + np.sum(orTs) + np.sum(xorTs) == andStack+orStack+xorStack):
                predCur.append(3)
            elif (np.sum(andTs == 1) + np.sum(orTs == 1) + np.sum(xorTs == 1) == 1):
                predCur.append(2)
            elif (np.sum([np.sum(andTs == 1), np.sum(orTs == 1), np.sum(xorTs == 1)]) >= 1):
                predCur.append(1)
            else:
                predCur.append(0)
        if topLevel == 'subAnd':
            if (np.sum(andTs) + np.sum(orTs) + np.sum(xorTs) == andStack+orStack+xorStack):
                predCur.append(4)
            elif (np.sum(andTs == 1) == andStack):
                predCur.append(3)
            elif (np.sum(xorTs == 1) == xorStack):
                predCur.append(2)
            elif (np.sum(orTs == 1) == orStack):
                predCur.append(1)
            else:
                predCur.append(0)
        else:
            if topLevel == 'and':
                if (np.sum(andTs == 0) + np.sum(orTs == 0) + np.sum(xorTs == 0)) > 0:
                     predCur.append(0)
                elif (np.sum(andTs == -1) + np.sum(orTs == -1) + np.sum(xorTs == -1)) >= 2:
                    predCur.append(0)
                elif (np.sum(andTs == -1) + np.sum(orTs==-1) + np.sum(xorTs==-1)) > 0:
                    predCur.append(-1)
                else:
                    predCur.append(1)
            elif topLevel == 'or':
                if (np.sum([np.sum(andTs == 1), np.sum(orTs == 1), np.sum(xorTs == 1)]) >= 1):
                    predCur.append(1)
                elif (np.sum(andTs == -1) + np.sum(orTs == -1) + np.sum(xorTs == -1)) >= 2:
                    predCur.append(1)
                elif (np.sum(andTs == -1) + np.sum(orTs==-1) + np.sum(xorTs==-1)) > 0:
                    predCur.append(-1)
                else:
                    predCur.append(0)
            elif topLevel == 'xor':
                if (np.sum(andTs == 1) + np.sum(orTs == 1) + np.sum(xorTs == 1) == 2):
                    predCur.append(0)
                elif (np.sum(andTs == -1) + np.sum(orTs == -1) + np.sum(xorTs == -1)) >= 3:
                    predCur.append(0)
                elif (np.sum(andTs == -1) + np.sum(orTs==-1) + np.sum(xorTs==-1)) > 0:
                    predCur.append(-1)
                elif (np.sum(andTs == 1) + np.sum(orTs == 1) + np.sum(xorTs == 1) == 1):
                    predCur.append(1)
                else:
                    predCur.append(0)
            else:
                    raise ValueError("Need a valid top level")

    acc = accuracy_score(np.array(predCur),target.argmax(axis=1))
    return acc, predCur

def logicAcc(data, target, nrSymbols, topLevel, andStack = 1, orStack = 1, xorStack = 1, nrAnds = 2, nrOrs = 2, nrxor = 2, trueIndexes=[3,1], orOffSet=0, xorOffSet=0):
    predCur = []
    symbolA = helper.getMapValues(nrSymbols)
    symbolA = np.array(symbolA)
    trueSymbols = symbolA[trueIndexes]
    falseSymbols = np.delete(symbolA, trueIndexes)

    for s in data:

        andTs = []
        for j in range(andStack):
            andT = 1
            for k in range((nrAnds*j), nrAnds*(j+1)):
                if s[k] == -2 and (andT == 1 or andT == -1):
                    andT = -1
                elif round(float(s[k]), 4) not in trueSymbols:
                    andT = 0
                    break
            if nrAnds == 0:
                andT = 1
            andTs.append(andT)
        maxAnds = (nrAnds * andStack)

        orTs = []
        for j in range(orStack):
            orT = 0
            for k in range(maxAnds + (nrOrs *j - orOffSet * j),maxAnds+(nrOrs *(j+1) - orOffSet * (j+1))):
                if round(float(s[k]), 4) in trueSymbols:
                    orT = 1
                    break
                elif s[k] == -2 and orT==0:
                    orT = -1
            if nrOrs == 0:
                orT = 1
            orTs.append(orT)
        maxOrs = nrOrs * orStack - orOffSet * orStack

        xorTs = []
        for j in range(xorStack):
            xorT = 0
            broken = False
            for k in range(maxAnds+maxOrs+(nrxor *j - xorOffSet * j),maxAnds+maxOrs+(nrxor *(j+1) - xorOffSet * (j+1))):
                if round(float(s[k]), 4) in trueSymbols:
                    if xorT == 1:
                        xorT = 0
                        broken = False
                        break
                    else:
                        xorT = 1
                elif s[k] == -2:
                    broken = True
            if nrxor == 0:
                xorT = 1
            if broken:
                xorT = -1
            xorTs.append(xorT)

        andTs = np.array(andTs)
        orTs = np.array(orTs)
        xorTs = np.array(xorTs)

        if topLevel == 'complex':
            if (np.sum(andTs) + np.sum(orTs) + np.sum(xorTs) == andStack+orStack+xorStack):
                predCur.append(3)
            elif (np.sum(andTs == 1) + np.sum(orTs == 1) + np.sum(xorTs == 1) == 1):
                predCur.append(2)
            elif (np.sum([np.sum(andTs == 1), np.sum(orTs == 1), np.sum(xorTs == 1)]) >= 1):
                predCur.append(1)
            else:
                predCur.append(0)
        if topLevel == 'subAnd':
            if (np.sum(andTs) + np.sum(orTs) + np.sum(xorTs) == andStack+orStack+xorStack):
                predCur.append(4)
            elif (np.sum(andTs == 1) == andStack):
                predCur.append(3)
            elif (np.sum(xorTs == 1) == xorStack):
                predCur.append(2)
            elif (np.sum(orTs == 1) == orStack):
                predCur.append(1)
            else:
                predCur.append(0)
        else:
            if topLevel == 'and':
                if (np.sum(andTs == 0) + np.sum(orTs == 0) + np.sum(xorTs == 0)) > 0:
                     predCur.append(0)
                elif (np.sum(andTs == -1) + np.sum(orTs==-1) + np.sum(xorTs==-1)) > 0:
                    predCur.append(-1)
                else:
                    predCur.append(1)
            elif topLevel == 'or':
                if (np.sum([np.sum(andTs == 1), np.sum(orTs == 1), np.sum(xorTs == 1)]) >= 1):
                    predCur.append(1)
                elif (np.sum(andTs == -1) + np.sum(orTs==-1) + np.sum(xorTs==-1)) > 0:
                    predCur.append(-1)
                else:
                    predCur.append(0)
            elif topLevel == 'xor':
                if (np.sum(andTs == 1) + np.sum(orTs == 1) + np.sum(xorTs == 1) == 2):
                    predCur.append(0)
                elif (np.sum(andTs == -1) + np.sum(orTs==-1) + np.sum(xorTs==-1)) > 0:
                    predCur.append(-1)
                elif (np.sum(andTs == 1) + np.sum(orTs == 1) + np.sum(xorTs == 1) == 1):
                    predCur.append(1)
                else:
                    predCur.append(0)
            else:
                    raise ValueError("Need a valid top level")

    acc = accuracy_score(np.array(predCur),target.argmax(axis=1))
    return acc, predCur


def fillOutMultiInNOut(index, data, target, nrSymbols, topLevel, andStack = 1, orStack = 1, xorStack = 1, nrAnds = 2, nrOrs = 2, nrxor = 2, trueIndexes=[3,1], orOffSet=0, xorOffSet=0):
    predCur = []
    symbolA = helper.getMapValues(nrSymbols)
    symbolA = np.array(symbolA)
    trueSymbols = symbolA[trueIndexes]
    falseSymbols = np.delete(symbolA, trueIndexes)

    for s in data:

        andTs = []
        for j in range(andStack):
            andT = True
            for k in range((nrAnds*j), nrAnds*(j+1)):
                if s[k] == -2 and index[k] == 1:
                    continue
                elif round(float(s[k]), 4) not in trueSymbols:
                    andT = False
                    break
                
            if nrAnds == 0:
                andT = True
            andTs.append(andT)
        maxAnds = (nrAnds * andStack)

        orTs = []
        for j in range(orStack):
            orT = False
            for k in range(maxAnds + (nrOrs *j - orOffSet * j),maxAnds+(nrOrs *(j+1) - orOffSet * (j+1))):
                if round(float(s[k]), 4) in trueSymbols:
                    orT = True
                    break
                elif s[k] == -2 and index[k] == 1:
                    andT = True
                    break
            if nrOrs == 0:
                orT = True
            orTs.append(orT)
        maxOrs = nrOrs * orStack - orOffSet * orStack 

        xorTs = []
        for j in range(xorStack):
            xorT = False
            for k in range(maxAnds+maxOrs+(nrxor *j - xorOffSet * j),maxAnds+maxOrs+(nrxor *(j+1) - xorOffSet * (j+1))):
                if round(float(s[k]), 4) in trueSymbols:
                    if xorT == True:
                        xorT = False
                        break
                    else:
                        xorT = True
                elif s[k] == -2 and index[k] == 1:
                    if xorT == True:
                        xorT = False
                        break
                    else:
                        xorT = True
            if nrxor == 0:
                xorT = True
            xorTs.append(xorT)

        andTs = np.array(andTs)
        orTs = np.array(orTs)
        xorTs = np.array(xorTs)

        if topLevel == 'complex':
            if (np.sum(andTs) + np.sum(orTs) + np.sum(xorTs) == andStack+orStack+xorStack):
                predCur.append(3)
            elif (np.sum(andTs) + np.sum(orTs) + np.sum(xorTs) == 1):
                predCur.append(2)
            elif (np.sum(andTs) + np.sum(orTs) + np.sum(xorTs) >= 1):
                predCur.append(1)
            else:
                predCur.append(0)
        if topLevel == 'subAnd':
            if (np.sum(andTs) + np.sum(orTs) + np.sum(xorTs) == andStack+orStack+xorStack):
                predCur.append(4)
            elif (np.sum(andTs) == andStack):
                predCur.append(3)
            elif (np.sum(xorTs) == xorStack):
                predCur.append(2)
            elif (np.sum(orTs) == orStack):
                predCur.append(1)
            else:
                predCur.append(0)
        else:
            if topLevel == 'and':
                if (np.sum(andTs) + np.sum(orTs) + np.sum(xorTs) == andStack+orStack+xorStack):
                    predCur.append(1)
                else:
                    predCur.append(0)
            elif topLevel == 'or':
                if (np.sum(andTs) + np.sum(orTs) + np.sum(xorTs) >= 1):
                    predCur.append(1)
                else:
                    predCur.append(0)
            elif topLevel == 'xor':
                if (np.sum(andTs) + np.sum(orTs) + np.sum(xorTs) == 1):
                    predCur.append(1)
                else:
                    predCur.append(0)
            else:
                    raise ValueError("Need a valid top level")

    acc = accuracy_score(np.array(predCur),target.argmax(axis=1))
    return acc, predCur

        

def getAttention(device, model, x_train1, y_train1, val_batch_size=1000, batch_size=1000, doClsTocken=False):
        attention = pt.validate_model(device, model, x_train1, y_train1, val_batch_size, batch_size, '', output_attentions=True)
        fullAttention = []
        for n in range(model.num_hidden_layers):
            fullAttention.append([])
        for a in attention:
                for layers in range(len(list(a))):
                        fullAttention[layers] = fullAttention[layers] + list(a[layers])


        def makecpu(inputList):
                if (isinstance(inputList, list)):
                        for a in range(len(inputList)):
                                inputList[a] = np.array(makecpu(inputList[a].cpu()))
                        return np.array(inputList)
                elif (isinstance(inputList, tuple)):
                        for a in range(len(inputList)):
                                inputList[a] = np.array(makecpu(inputList[a].cpu()))
                        return np.array(inputList)
                elif (isinstance(inputList, torch.Tensor)):
                        return inputList.cpu()
                else:
                        return inputList.cpu()

        for a in range(len(fullAttention)):
            fullAttention[a]= np.array(makecpu(fullAttention[a]))
        
        if doClsTocken:
            fullAttention = np.array(fullAttention)[:, :, :, 1:, 1:]
        else:
            fullAttention = np.array(fullAttention)
        return fullAttention

def buildSHAP2D(approx_value, n_sii_order, shapley_extractor_sii):
    n_shapley_values = shapley_extractor_sii.transform_interactions_in_n_shapley(interaction_values=approx_value, n=n_sii_order, reduce_one_dimension=False)
    for i, v in enumerate(n_shapley_values[1]):
        n_shapley_values[2][i][i] = v
    return n_shapley_values[2]

def getIQSHAP(data, model, modelType):
    metaGame = games.DLMetaGame(model, data, modelType)

    interaction_order = 2
    budget = 2**7
    outShap = []
    for l in range(len(data)):
        game = games.DLGame(
            meta_game= metaGame,
            data_index=l
        )
        game_name = game.game_name
        game_fun = game.set_call
        n = game.n
        N = set(range(n))
        shapley_extractor_sii = SHAPIQEstimator(
            N=N,
            order=interaction_order,
            interaction_type="SII",
            top_order=False
        )
        approx_value = shapley_extractor_sii.compute_interactions_from_budget(
            game=game.set_call,
            budget=budget,
            pairing=False
        )
        shapResults = buildSHAP2D(approx_value, interaction_order, shapley_extractor_sii)
        outShap.append(shapResults)
    return outShap

def getSaliencyMap(outMap, saveKey, device, numberOfLables, modelType: str, method: str, submethod: str, model, x_train, x_val, x_test, y_train, y_val, y_test, smooth=False, batches=True, batchSize=50, doClassBased=True, doClsTocken=False):
    outTrain = []
    outVal = []
    outTest = []
    print('methods:')
    print(method)
    print(submethod)

    if not batches:
        batchSize = len(y_train)
    
    if method == 'LRP':
            outTrain = getRelpropSaliency(device, x_train, model, method=submethod, batchSize=batchSize)
            outVal = getRelpropSaliency(device, x_val, model, method=submethod, batchSize=batchSize)
            outTest = getRelpropSaliency(device, x_test, model, method=submethod, batchSize=batchSize)
            for lable in range(numberOfLables):
                targets = np.zeros((x_train.shape[0], numberOfLables), dtype=np.float32)
                for t in range(len(targets)):
                    targets[t, lable] = 1
                outTrainC = getRelpropSaliency(device, x_train, model, method=submethod, outputO=targets, batchSize=batchSize)
                targets = np.zeros((x_val.shape[0], numberOfLables), dtype=np.float32)
                for t in range(len(targets)):
                    targets[t, lable] = 1
                outValC = getRelpropSaliency(device, x_val, model, method=submethod, outputO=targets, batchSize=batchSize)
                targets = np.zeros((x_test.shape[0], numberOfLables), dtype=np.float32)
                for t in range(len(targets)):
                    targets[t, lable] = 1
                outTestC = getRelpropSaliency(device, x_test, model, method=submethod, outputO=targets, batchSize=batchSize)
                outMap['classes'][str(lable)][saveKey + 'Train'].append(outTrainC)
                outMap['classes'][str(lable)][saveKey + 'Val'].append(outValC)
                outMap['classes'][str(lable)][saveKey + 'Test'].append(outTestC)
    elif method == 'Random':
        x_trainf = x_train.squeeze()
        x_valf = x_val.squeeze()
        x_testf = x_test.squeeze()
        outTrain = np.random.randn(x_trainf.shape[0], x_trainf.shape[1], x_trainf.shape[1])
        outVal = np.random.randn(x_valf.shape[0], x_valf.shape[1], x_valf.shape[1])
        outTest = np.random.randn(x_testf.shape[0], x_testf.shape[1], x_testf.shape[1])
    elif method == 'IQ-SHAP':
        if submethod == "2OrderShap":
            outTrain = getIQSHAP(x_train, model,modelType)
            outVal = getIQSHAP(x_val, model,modelType)
            outTest = getIQSHAP(x_test, model,modelType)
    elif method ==  'captum':
            if submethod == "IntegratedGradients":
                    lig = IntegratedGradients(model)
                    batchSize = 1
            elif submethod == "Saliency":
                    lig = Saliency(model)
            elif submethod == "DeepLift":
                    lig = DeepLift(model)
            elif submethod == "KernelShap":
                    lig = KernelShap(model)
            elif submethod == "InputXGradient":
                    lig = InputXGradient(model)
            elif submethod == "GuidedBackprop":
                    lig = GuidedBackprop(model)
            elif submethod == "GuidedGradCam":
                    if modelType == "Transformer":
                        lig = GuidedGradCam(model, layer=model.encoder.layer[-1])
                    elif modelType == "CNN":
                        lig = GuidedGradCam(model, layer=model.lastConv)
                    else:
                        raise ValueError("Not a valid model type for gradcam")
            elif submethod == "FeatureAblation":
                    lig = FeatureAblation(model)
            elif submethod == "FeaturePermutation":
                    lig = FeaturePermutation(model)
            elif submethod == "Deconvolution":
                    lig = Deconvolution(model)
            else:
                    raise ValueError("Not a valid captum submethod")

            if doClassBased:
                maxGoal = numberOfLables
            else:
                maxGoal = 0
            for lable in range(-1, maxGoal):
                outTrainA = None
                outValA = None
                outTestA = None

                for batch_start in range(0, len(y_train), batchSize):
                    batchEnd = batch_start + batchSize
                    if lable == -1:           
                        targets = torch.from_numpy(y_train[batch_start:batchEnd].argmax(axis=1)).to(device) 
                    else:
                        targets = torch.from_numpy(np.zeros((len(y_train[batch_start:batchEnd])), dtype=np.int64) + lable).to(device) 
                    input_ids = torch.from_numpy(x_train[batch_start:batchEnd]).to(device) 

                    outTrainB = interpret_dataset(lig, input_ids, targets, package=method, smooth=smooth)
                    if outTrainA is None:
                        outTrainA = outTrainB
                    else:
                        outTrainA = np.vstack([outTrainA,outTrainB])

                for batch_start in range(0, len(y_val), batchSize):
                    batchEnd = batch_start + batchSize
                    if lable == -1:           
                        targets = torch.from_numpy(y_val[batch_start:batchEnd].argmax(axis=1)).to(device) 
                    else:
                        targets = torch.from_numpy(np.zeros((len(y_val[batch_start:batchEnd])), dtype=np.int64) + lable).to(device) 
                    input_ids = torch.from_numpy(x_val[batch_start:batchEnd]).to(device) 
                    outValB = interpret_dataset(lig, input_ids, targets, package=method, smooth=smooth)
                    if outValA is None:
                        outValA = outValB
                    else:
                        outValA = np.vstack([outValA,outValB])


                for batch_start in range(0, len(y_test), batchSize):
                    batchEnd = batch_start + batchSize
                    if lable == -1:           
                        targets = torch.from_numpy(y_test[batch_start:batchEnd].argmax(axis=1)).to(device) 
                    else:
                        targets = torch.from_numpy(np.zeros((len(y_test[batch_start:batchEnd])), dtype=np.int64) + lable).to(device) 
                    input_ids = torch.from_numpy(x_test[batch_start:batchEnd]).to(device) 
                    outTestb = interpret_dataset(lig, input_ids, targets, package=method, smooth=smooth)
                    if outTestA is None:
                        outTestA = outTestb
                    else:
                        outTestA = np.vstack([outTestA, outTestb])
                if lable == -1:
                    outTrain = outTrainA
                    outVal = outValA
                    outTest = outTestA
                else:
                    outMap['classes'][str(lable)][saveKey + 'Train'].append(outTrainA)
                    outMap['classes'][str(lable)][saveKey + 'Val'].append(outValA)
                    outMap['classes'][str(lable)][saveKey + 'Test'].append(outTestA)

    elif method ==  'PytGradCam':
            if modelType == "Transformer":
                layer=model.encoder.layer[-1]
                reshapeMethod = reshape_transform
            elif modelType == "CNN":
                layer=model.lastConv
                reshapeMethod = None
            else:
                raise ValueError("Not a valid model type for PytGradCam")

            if submethod == "EigenCAM":
                    lig = EigenCAM(model, target_layers=[layer], use_cuda=True, reshape_transform=reshapeMethod)
            elif submethod == "GradCAMPlusPlus":
                    lig = GradCAMPlusPlus(model, target_layers=[layer], use_cuda=True, reshape_transform=reshapeMethod)
            elif submethod == "XGradCAM":
                    lig = XGradCAM(model, target_layers=[layer], use_cuda=True, reshape_transform=reshapeMethod)
            elif submethod == "GradCAM":
                    lig = GradCAM(model, target_layers=[layer], use_cuda=True, reshape_transform=reshapeMethod)
            elif submethod == "EigenGradCAM":
                    lig = EigenGradCAM(model, target_layers=[layer], use_cuda=True, reshape_transform=reshapeMethod)

            else:
                    raise ValueError("Not a valid PytGradCam submethod")

            if not batches:
                batchSize = len(y_train)

            for lable in range(-1, numberOfLables):

                outTrainA = None
                outValA = None
                outTestA = None
                for batch_start in range(0, len(y_train), batchSize):
                    batchEnd = batch_start + batchSize
                    input_ids = torch.from_numpy(x_train[batch_start:batchEnd]).to(device) 

                    target_categories = torch.from_numpy(y_train[batch_start:batchEnd].argmax(axis=1)).to(device) 
                    if lable != -1:           

                        target_categories = target_categories.squeeze()
                        targets = [ClassifierOutputTarget(category) for category in target_categories]

                        outTrainB = interpret_dataset(lig, input_ids, targets, package=method, smooth=smooth)
                        if outTrainA is None:
                            outTrainA = outTrainB
                    else:
                        outTrainA = np.vstack([outTrainA,outTrainB])

                for batch_start in range(0, len(y_val), batchSize):
                    batchEnd = batch_start + batchSize
                    target_categories = torch.from_numpy(y_val[batch_start:batchEnd].argmax(axis=1)).to(device) 
                    if lable != -1:           
                        target_categories = target_categories * 0 + lable

                    target_categories = target_categories.squeeze()
                    targets = [ClassifierOutputTarget(category) for category in target_categories]

                    input_ids = torch.from_numpy(x_val[batch_start:batchEnd]).to(device) 
                    
                    outValB = interpret_dataset(lig, input_ids, targets, package=method, smooth=smooth)
                    if outValA is None:
                        outValA = outValB
                    else:
                        outValA = np.vstack([outValA,outValB])

                for batch_start in range(0, len(y_test), batchSize):
                    batchEnd = batch_start + batchSize
                    target_categories = torch.from_numpy(y_test[batch_start:batchEnd].argmax(axis=1)).to(device) 
                    if lable != -1:           
                        target_categories = target_categories * 0 + lable

                    target_categories = target_categories.squeeze()
                    targets = [ClassifierOutputTarget(category) for category in target_categories]

                    input_ids = torch.from_numpy(x_test[batch_start:batchEnd]).to(device) 

                    outTestb = interpret_dataset(lig, input_ids, targets, package=method, smooth=smooth)
                    if outTestA is None:
                        outTestA = outTestb
                    else:
                        outTestA = np.vstack([outTestA, outTestb])
                if lable == -1:
                    outTrain = outTrainA
                    outVal = outValA
                    outTest = outTestA
                else:
                    outMap['classes'][str(lable)][saveKey + 'Train'].append(outTrainA)
                    outMap['classes'][str(lable)][saveKey + 'Val'].append(outValA)
                    outMap['classes'][str(lable)][saveKey + 'Test'].append(outTestA)

    elif method ==  'SHAP':
            explainer = shap.TreeExplainer(model, x_train)
            outTrain = explainer.shap_values(x_train)
            
            explainer = shap.TreeExplainer(model, x_val)
            outVal = explainer.shap_values(x_val)
            
            explainer = shap.TreeExplainer(model, x_test)
            outTest = explainer.shap_values(x_test)
            
    elif method == 'Attention':
            outTrain = getAttention(device, model, x_train, y_train, val_batch_size=batchSize, batch_size=batchSize, doClsTocken=doClsTocken)

            outVal = getAttention(device,model, x_val, y_val, val_batch_size=batchSize, batch_size=batchSize, doClsTocken=doClsTocken)

            outTest = getAttention(device, model, x_test, y_test, val_batch_size=batchSize, batch_size=batchSize, doClsTocken=doClsTocken)

    else:
        print('unknown saliency method: ' + method)
        raise Exception('Unknown Saliency Method: ' + method)

    outTrain = np.array(outTrain).squeeze()
    outVal = np.array(outVal).squeeze()
    outTest = np.array(outTest).squeeze()

    outMap[saveKey + 'Train'].append(outTrain)
    outMap[saveKey + 'Val'].append(outVal)
    outMap[saveKey + 'Test'].append(outTest)
    outMap['means'][saveKey + 'Train'].append(np.mean(outTrain.squeeze(), axis=1))
    outMap['means'][saveKey + 'Val'].append(np.mean(outVal.squeeze(), axis=1))
    outMap['means'][saveKey + 'Test'].append(np.mean(outTest.squeeze(), axis=1))

    do3DData = False
    do2DData = False
    if len(outTrain.shape) > 3:
        do3DData = False
        outTrain = reduceMap(outTrain, do3D2Step=True, do3rdStep=False)
        outVal = reduceMap(outVal, do3D2Step=True, do3rdStep=False)
        outTest = reduceMap(outTest, do3D2Step=True, do3rdStep=False)
        do2DData=True
    elif len(outTrain.shape) > 2:
        do2DData = True


    return outTrain, outVal, outTest, do3DData, do2DData
    
def transformMod(sd, mode):
    if mode == 2:
        sd[sd < 0] = 0 
    elif mode == 3:
        sd = np.absolute(sd)
    return sd

def doLasaAuc3DGCR(resultDict, fullAttention, testAttention, traindata, trainLables, testdata, testLables, num_of_classes, symbolsCount, trainCombis, ranges, rangesSmall, testCombis, gtmRange, saliencyCombis, reductionName="MixedClasses", mode=0, threshold=-1, percentSteps=20, doMax=False, doPenalty=False, penaltyMode="entropy", addOne=False, useRM = True, order = 'lh', step1 = 'sum', step2 = 'sum', do3DData=True, do2DData=False, doGTM=True):   



        fullAttentionM = transformMod(fullAttention, mode)
        saliencyCombisM = transformMod(saliencyCombis, mode)

        ignoreMaskedValue = True
        addMaskedValue = True
        doMetrics=False
        #mode = 0

        for rk in resultDict.keys():
            resultDict[rk][reductionName]['trainGlobal'].append([[],[],[],[],[],[],[],[]])
            resultDict[rk][reductionName]['trainLocal'].append([[],[],[],[],[],[],[],[]])
            resultDict[rk][reductionName]['fullGlobal'].append([[],[],[],[],[],[],[],[]])
            resultDict[rk][reductionName]['fullLocal'].append([[],[],[],[],[],[],[],[]])
            resultDict[rk][reductionName]['trainGlobalR'].append([])
            resultDict[rk][reductionName]['trainLocalR'].append([])
            resultDict[rk][reductionName]['fullGlobalR'].append([])
            resultDict[rk][reductionName]['fullLocalR'].append([])


        
        for thresholdFactor in (np.array(list(range(0, 100, percentSteps))) /100):
            newTrainGlobal, newTrainGlobalR,newTrainCombisGlobal = doSimpleLasaReduction(fullAttentionM, traindata, thresholdFactor, trainCombis, saliencyCombisM, maskValue=symbolsCount, doFidelity=False, do3DData=do3DData, do3rdStep=do2DData, globalT=True)
            newTrainLocal, newTrainLocalR,newTrainCombisLocal = doSimpleLasaReduction(fullAttentionM, traindata, thresholdFactor, trainCombis, saliencyCombisM, maskValue=symbolsCount,doFidelity=False, do3DData=do3DData, do3rdStep=do2DData, globalT=False)


            trainGlobal= do3DGCR(fullAttention, newTrainGlobal, trainLables, testdata, testLables, num_of_classes, symbolsCount, np.array(newTrainCombisGlobal), ranges, rangesSmall, testCombis, gtmRange, mode=mode, reductionName=reductionName, doMetrics=doMetrics, threshold=threshold, addMaskedValue=addMaskedValue, ignoreMaskedValue=ignoreMaskedValue, doMax=doMax, doPenalty=doPenalty, penaltyMode=penaltyMode,useRM=useRM, addOne=addOne, order = order, step1 = step1, step2 = step2, do3DData=do3DData, do2DData=do2DData,doGTM=doGTM)
            trainLocal = do3DGCR(fullAttention, newTrainLocal, trainLables, testdata, testLables, num_of_classes, symbolsCount,  np.array(newTrainCombisLocal), ranges, rangesSmall, testCombis, gtmRange, mode=mode, reductionName=reductionName, doMetrics=doMetrics, threshold=threshold, addMaskedValue=addMaskedValue, ignoreMaskedValue=ignoreMaskedValue, doMax=doMax, doPenalty=doPenalty, penaltyMode=penaltyMode, useRM=useRM,addOne=addOne, order = order, step1 = step1, step2 = step2, do3DData=do3DData, do2DData=do2DData,doGTM=doGTM)


            resultDict['rMS'][reductionName]['trainGlobalR'][-1].append(np.mean(newTrainGlobalR))
            resultDict['rMS'][reductionName]['trainLocalR'][-1].append(np.mean(newTrainLocalR))
            for vi, v in enumerate(trainGlobal[0][0]):
                resultDict['rMS'][reductionName]['trainGlobal'][-1][vi].append(v)
                resultDict['rMS'][reductionName]['trainLocal'][-1][vi].append(trainLocal[0][0][vi])

            resultDict['rMA'][reductionName]['trainGlobalR'][-1].append(np.mean(newTrainGlobalR))
            resultDict['rMA'][reductionName]['trainLocalR'][-1].append(np.mean(newTrainLocalR))


            for vi, v in enumerate(trainGlobal[1][0]):
                resultDict['rMA'][reductionName]['trainGlobal'][-1][vi].append(v)
                resultDict['rMA'][reductionName]['trainLocal'][-1][vi].append(trainLocal[1][0][vi])


            for k in GCRPlus.gtmReductionStrings():
                resultDict[k][reductionName]['trainGlobalR'][-1].append(np.mean(newTrainGlobalR))
                resultDict[k][reductionName]['trainLocalR'][-1].append(np.mean(newTrainLocalR))

                for vi, v in enumerate(trainGlobal[2][k]['performance']):
                    resultDict[k][reductionName]['trainGlobal'][-1][vi].append(v)
                    resultDict[k][reductionName]['trainLocal'][-1][vi].append(trainLocal[2][k]['performance'][vi])


            if thresholdFactor == 0:
                resultDict['rMS'][reductionName]['acc'].append(trainGlobal[0][0][0])
                resultDict['rMS'][reductionName]['predicsion'].append(trainGlobal[0][0][1])
                resultDict['rMS'][reductionName]['recall'].append(trainGlobal[0][0][2])
                resultDict['rMS'][reductionName]['f1'].append(trainGlobal[0][0][3])
                resultDict['rMS'][reductionName]['confidence'].append(metrics.auc(range(len(trainGlobal[0][4])), trainGlobal[0][4]))

                resultDict['rMA'][reductionName]['acc'].append(trainGlobal[1][0][0])
                resultDict['rMA'][reductionName]['predicsion'].append(trainGlobal[1][0][1])
                resultDict['rMA'][reductionName]['recall'].append( trainGlobal[1][0][2])
                resultDict['rMA'][reductionName]['f1'].append(trainGlobal[1][0][3])

                resultDict['rMA'][reductionName]['confidence'].append(metrics.auc(range(len(trainGlobal[1][4])), trainGlobal[1][4]))

                for gtmAbstact in GCRPlus.gtmReductionStrings():
                    resultDict[gtmAbstact][reductionName]['acc'].append(trainGlobal[2][gtmAbstact]['performance'][0])
                    resultDict[gtmAbstact][reductionName]['predicsion'].append(trainGlobal[2][gtmAbstact]['performance'][1])
                    resultDict[gtmAbstact][reductionName]['recall'].append(trainGlobal[2][gtmAbstact]['performance'][2])
                    resultDict[gtmAbstact][reductionName]['f1'].append(trainGlobal[2][gtmAbstact]['performance'][3])
                    resultDict[gtmAbstact][reductionName]['confidence'].append(metrics.auc(range(len(trainGlobal[2][gtmAbstact]['confidence'])), trainGlobal[2][gtmAbstact]['confidence']) )




def do3DGCR(fullAttention, ix_train, trainLables, ix_test, testLables, num_of_classes, symbolsCount, trainCombis, ranges, rangesSmall, testCombis, gtmRange, reductionName="MixedClasses", mode=0, threshold=-1, doMetrics=False, addMaskedValue=False,  ignoreMaskedValue=False, doMax=False, doPenalty=False, penaltyMode="entropy", addOne=False, useRM = True, order = 'lh', step1 = 'sum', step2 = 'sum', do3DData=True, do2DData=False, doGTM=True):    

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
        fullAttention = helper.doCombiStep(step1, fullAttention, axis1)
        fullAttention = helper.doCombiStep(step2, fullAttention, axis2)
        do3DData = False
        do2DData = True

    valuesA = helper.getMapValues(symbolsCount)
    if addMaskedValue:
        valuesA.append(-2)
    attentionQ = [[],[],np.array(fullAttention).squeeze()]
    print("Attention shape")
    print(np.array(fullAttention.squeeze()).shape)
    print(do3DData)
    print(do2DData)



    trainCombis = np.array(trainCombis)



    predictions = np.argmax(testLables,axis=1)
    gtmAbstractions = GCRPlus.gtmReductionStrings()# ['max','max+','average','average+','median','median+']

    print('start making attention')

    rMA, rMS, rM, gtms, gtmRM = GCRPlus.fastMakeAttention(attentionQ, ix_train, trainLables, trainCombis, ranges, order, step1, step2, num_of_classes, valuesA,mode=mode, threshold=threshold, ignoreMaskedValue=ignoreMaskedValue, reductionName=reductionName, doMax=doMax, doPenalty=doPenalty, penaltyMode=penaltyMode, addOne=False, do3DData=do3DData, do2DData=do2DData)



    minScoresA = GCRPlus.calcFCAMMinScoreNP(rMA)
    minScoresS = GCRPlus.calcFCAMMinScoreNP(rMS)

    rMAAUCglobal = GCRPlus.classFullAttFast(rMA, ix_test, rangesSmall, testCombis, predictions, rM, minScoresA, valuesA, useRM=useRM, doPenalty=False, rMA=True)
    rMSAUCglobal = GCRPlus.classFullAttFast(rMS, ix_test, rangesSmall, testCombis, predictions, rM, minScoresS, valuesA, useRM=useRM, doPenalty=False, rMA=False)

    
    gcrSFcamOutA = [rMAAUCglobal[0][0][0],rMAAUCglobal[0][1][0],rMAAUCglobal[0][2][0],rMAAUCglobal[0][3][0]],[rMAAUCglobal[1][0][0],rMAAUCglobal[1][1][0]], rMAAUCglobal[2][0],rMAAUCglobal[3][0],rMAAUCglobal[4][0]

    gcrSFcamOutS = [rMSAUCglobal[0][0][0],rMSAUCglobal[0][1][0],rMSAUCglobal[0][2][0],rMSAUCglobal[0][3][0]],[rMSAUCglobal[1][0][0],rMSAUCglobal[1][1][0]], rMSAUCglobal[2][0],rMSAUCglobal[3][0],rMSAUCglobal[4][0]


    gcrGTMOut = dict()
    if doGTM:
        for i,e  in enumerate(gtmAbstractions):
            gcrGTMOut[e] = dict()
            minScoresGTM = GCRPlus.calcGTMMinScoreNP(gtms[e])

            gtmOut =  GCRPlus.calcFullAbstractAttentionFast(gtms, e, ix_test, minScoresGTM, gtmRange, predictions, gtmRM)


            gcrGTMOut[e]['performance'] = [gtmOut[0][0][0],gtmOut[0][1][0],gtmOut[0][2][0],gtmOut[0][3][0]]
            gcrGTMOut[e]['confidence']  = gtmOut[4][0]

    if doMetrics:
        return gcrSFcamOutS, gcrSFcamOutA, gcrGTMOut, gtms, rM, rMA, rMS
    else: 
        return gcrSFcamOutS, gcrSFcamOutA, gcrGTMOut, gtms, rM, rMA, rMS

def do2DGCR(fullAttention, traindata, trainLables, testdata, testLables, num_of_classes, symbolsCount, do3DData=False, do3rdStep=False): 
    fullAttention = reduceMap(fullAttention, do3DData=do3DData, do3rdStep=do3rdStep)   
    valuesA = helper.getMapValues(symbolsCount)
    gtmS = dict()
    gtmA = dict()
    rm = dict()
    for lable in range(num_of_classes):
        gtmS[lable] = dict()
        gtmA[lable] = dict()
        rm[lable] = dict()
        for symbol in valuesA:
            gtmS[lable][symbol] = np.zeros(len(fullAttention[0]))
            gtmA[lable][symbol] = np.zeros(len(fullAttention[0]))
            rm[lable][symbol] = np.zeros(len(fullAttention[0]))

    traindataS = traindata.squeeze()
    testdata = testdata.squeeze()
    for i, y in enumerate(trainLables):
        for j, x in enumerate(traindataS[i]):
            gtmS[y][x][j] += fullAttention[i][j]
            rm[y][x][j] += +1

    for lable in range(num_of_classes):
        for symbol in valuesA:
            gtmA[lable][symbol] =  np.nan_to_num(gtmS[lable][symbol]/rm[lable][symbol])


    [rA, _,_,_], _, _,_,_ = evalGTM(gtmA, symbolsCount, testdata, testLables)
    [rS, _,_,_], _, _,_,_ = evalGTM(gtmS, symbolsCount, testdata, testLables) 

    
    return gtmS, gtmA, rm, rA, rS



def evalGTM(gtm, symbolsCount, testdata, testLables):
    valuesA = helper.getMapValues(symbolsCount)
    results = []
    predictResults = []
    biggestScores = []
    allLableScores = []

    maxScores = dict()
    
    for lable in gtm.keys():
        maxScores[lable] =  np.sum(np.max(list(gtm[lable].values()), axis=0))
    #print(maxScores)

    answers = []
    for ti in range(len(testdata)):
        answers.append(classifyGTM(testdata[ti], testLables[ti], gtm, maxScores))
        
    asynLabels = []
    for ans in answers:
        results.append(ans[1])
        predictResults.append(ans[2])
        biggestScores.append(ans[3])
        allLableScores.append(ans[0])
        asynLabels.append(ans[4])

    acc = metrics.accuracy_score(predictResults, asynLabels)
    predicsion = metrics.precision_score(predictResults, asynLabels, average='macro')
    recall = metrics.recall_score(predictResults, asynLabels, average='macro')
    f1= metrics.f1_score(predictResults, asynLabels, average='macro')

    confidenceAcc = helper.confidenceGCR(biggestScores, results)

    return [acc, predicsion, recall, f1], [allLableScores, biggestScores], results, predictResults, confidenceAcc


def classifyGTM(trial, ylabel, rMG, maxScores):
    lableScores = dict()

    for lable in rMG.keys():
        lableScores[lable] = 0
    
    for toVi in range(len(trial)):
        toV = trial[toVi]

        for lable in rMG.keys():
            lableScores[lable] += rMG[lable][float(toV)][toVi] 

    #get final score
    for lable in rMG.keys():
        lableScores[lable] = lableScores[lable]/maxScores[lable]

    #classification
    biggestLable = list(lableScores.keys())[np.argmax(list(lableScores.values()))]
    biggestValue = lableScores[biggestLable]
    boolResult = biggestLable == ylabel

    return lableScores, boolResult, biggestLable, biggestValue, ylabel
        
    
def getSubFCAM(gMA, valuesA, indexStart, indexEnd):
    rMA = GCRPlus.nested_dict_static()
    for lable in gMA.keys():
        for fromL in valuesA:
            for toL in valuesA:
                rMA[lable][fromL][toL] = np.zeros( indexEnd-indexStart, indexEnd-indexStart)
                for i in range(indexStart,indexEnd):
                    for j in range(indexStart,indexEnd):
                        rMA[lable][fromL][toL][i-indexStart][j-indexStart] = gMA[lable][fromL][toL][i][j]
    return rMA                        

def getSubGTM(gMA, valuesA, indexStart, indexEnd):
    stm = GCRPlus.nested_dict_static()
    for lable in gMA.keys():
        for redStr in GCRPlus.gtmReductionStrings():
            for fromL in valuesA:
                stm[lable][redStr][fromL] = np.zeros(indexEnd-indexStart)
                for i in range(indexStart,indexEnd):
                    stm[lable][redStr][fromL][i-indexStart] = gMA[lable][redStr][fromL][i]
    return stm


def getLasaThresholds(saliencyMap, data, thresholdFactor, do3DData=False, do3rdStep=False, globalT=True, axis1= 2, axis2=0, axis3=1, op1='sum',op2='sum',op3='sum'):
    if do3DData:
        saliencyMap = helper.doCombiStep(op1, saliencyMap, axis1)
        saliencyMap = helper.doCombiStep(op2, saliencyMap, axis2) 
        saliencyMap = helper.doCombiStep(op3, saliencyMap, axis3) 
    elif do3rdStep:
        saliencyMap = helper.doCombiStep(op3, saliencyMap, axis3) 

    X_sax = np.array(data).squeeze()
    heats = saliencyMap.squeeze()


    if globalT:
        lowestScore = np.min(saliencyMap)
        highestScore = np.max(saliencyMap)
    else:
        lowestScore = np.min(saliencyMap, axis=(1))
        highestScore = np.max(saliencyMap, axis=(1))

    return (highestScore-lowestScore) * thresholdFactor + lowestScore

def doSimpleLasaReduction(saliencyMap, data, thresholdFactor, trainCombis, saliencyCombis, doFidelity=False, do3DData=False, do3rdStep=False, globalT=True, maskValue=-2, axis1= 2, axis2=0, axis3=1, op1='sum',op2='sum',op3='sum'):
    print('new ROAR start')
    newX = []
    reduction = []

    if do3DData:
        saliencyMap = helper.doCombiStep(op1, saliencyMap, axis1)
        saliencyMap = helper.doCombiStep(op2, saliencyMap, axis2) 
        saliencyMap = helper.doCombiStep(op3, saliencyMap, axis3) 
    elif do3rdStep:
        saliencyMap = helper.doCombiStep(op3, saliencyMap, axis3) 

    X_sax = np.array(data).copy()
    heats = saliencyMap.squeeze()


    if globalT:
        lowestScore = np.min(saliencyMap)
        highestScore = np.max(saliencyMap)
    else:
        lowestScore = np.min(saliencyMap, axis=(1))
        highestScore = np.max(saliencyMap, axis=(1))

    threshold = (highestScore-lowestScore) * thresholdFactor + lowestScore
    if not globalT:
        threshold = threshold[:,None]

    print(saliencyCombis.shape)
    print(trainCombis.shape)
    print(np.array(threshold).shape)
    if doFidelity:

        X_sax[saliencyMap > threshold] = maskValue
        newTrainCombis = trainCombis
        newTrainCombis[saliencyCombis > threshold][:, 1:] = maskValue
    else:

        X_sax[saliencyMap < threshold] = maskValue
        newTrainCombis = trainCombis.copy()       
        
        if globalT:
            newTrainCombis[:, 1:][saliencyCombis[:,1:] < threshold] = maskValue
        else:
            nShape =  saliencyCombis[:,1:].shape
            tholds = (saliencyCombis[:,1:].reshape((len(threshold), -1)) < threshold).reshape(nShape)
            newTrainCombis[:, 1:][tholds] = maskValue

    reduction =  np.sum(X_sax == maskValue) / len(X_sax.flatten())

    return X_sax, reduction, newTrainCombis

def doSimpleLasaROAR(saliencyMap, data, threshold, doBaselineT=False, doFidelity=False, do3DData=False, do3rdStep=False, axis1= 2, axis2=0, axis3=1, op1='max',op2='sum',op3='max'):
    print('new ROAR start')
    newX = []
    reduction = []

    if do3DData:
        saliencyMap = helper.doCombiStep(op1, saliencyMap, axis1)
        saliencyMap = helper.doCombiStep(op2, saliencyMap, axis2) 
        saliencyMap = helper.doCombiStep(op3, saliencyMap, axis3) 
    elif do3rdStep:
        saliencyMap = helper.doCombiStep(op3, saliencyMap, axis3) 

    X_sax = np.array(data).squeeze()
    heats = saliencyMap.squeeze()
    heats = preprocessing.minmax_scale(heats, axis=1)


    if doBaselineT:
        cutOff = threshold
        threshold = np.max(np.array(heats)[:,-1 * cutOff:], axis=1)

    for index in range(len(saliencyMap)):
                
            X_ori = X_sax[index]
            heat = heats[index] 


        
            if doBaselineT:
                borderHeat = threshold[index]
            else:
                maxHeat = np.average(heat)
                borderHeat = maxHeat*threshold
        
            fitleredSet = []
            skips = 0 
            for h in range(len(heat)):
                if validataHeat(heat[h], borderHeat, doFidelity):
                    fitleredSet.append(X_ori[h])
                else:
                    fitleredSet.append(-2)
                    skips += 1

            reduction.append(skips/len(heat))
            newX.append([fitleredSet])

    newX = np.array(newX, dtype=np.float32)
    newX = np.moveaxis(newX, 1,2)

    return newX, reduction


def validataHeat(value, heat, doFidelity):
    if doFidelity:
        return value <= heat
    else:
        return value >= heat