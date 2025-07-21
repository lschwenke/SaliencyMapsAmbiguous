from modules import helper
from modules import GCRPlus
from joblib import Parallel, delayed
from datetime import datetime

def fillOutDicWithGCRSmall(outData, subContext, oriPred, saxPred):
    inputDict = dict()
    inputDict[subContext + ' Accuracy'] = outData[0][0]
    inputDict[subContext + ' Precision'] = outData[0][1]
    inputDict[subContext + ' Recall'] = outData[0][2]
    inputDict[subContext + ' F1'] = outData[0][3]
    inputDict[subContext + ' All Lable Scores'] = outData[1][0]
    inputDict[subContext + ' Biggest Scores'] = outData[1][1]
    inputDict[subContext + ' Correct Results'] = outData[2]
    inputDict[subContext + ' Predictions'] = outData[3]
    inputDict[subContext + ' confidence'] = outData[4]

    if len(oriPred) > 0:
        oriFidelity = helper.modelFidelity(oriPred, outData[3])
    else:
        oriFidelity = -1
    saxFidelity = helper.modelFidelity(saxPred, outData[3])
    inputDict[subContext + ' Model Fidelity (Ori)'] = oriFidelity
    inputDict[subContext + ' Model Fidelity (Sax)'] = saxFidelity

    return inputDict  

def fillOutDicWithNNOutSmall(outData, outOri, outSax):
    inputDict = dict()
    inputDict['Val Accuracy'] = outData[0][0]
    inputDict['Val Precision'] = outData[0][1]
    inputDict['Val Recall'] = outData[0][2]
    inputDict['Val F1'] = outData[0][3]
    inputDict['Test Accuracy'] = outData[1][0]
    inputDict['Test Precision'] = outData[1][1]
    inputDict['Test Recall'] = outData[1][2]
    inputDict['Test F1'] = outData[1][3]
    inputDict['Train Predictions'] = outData[2][0]
    inputDict['Val Predictions'] = outData[2][1]
    inputDict['Test Predictions'] = outData[3]

    inputDict['Val Complexity'] = outData[5][0]
    inputDict['Test Complexity'] = outData[5][1]
    inputDict['Val Shifts'] = outData[12]
    inputDict['Test Shifts'] = outData[13]
    inputDict['Val Reduction'] = outData[18][0]
    inputDict['Test Reduction'] = outData[18][1]
    inputDict['Train Skip Counter'] = outData[19][0]
    inputDict['Val Skip Counter'] = outData[19][1]
    inputDict['Test Skip Counter'] = outData[19][2]


    if len(outOri) > 0:
        oriFidelityTrain = helper.modelFidelity(outOri[2][0], outData[2][0])
        oriFidelityVal = helper.modelFidelity(outOri[2][1], outData[2][1])
        oriFidelityTest = helper.modelFidelity(outOri[3], outData[3])
    else:
        oriFidelityTrain = -1
        oriFidelityVal = -1
        oriFidelityTest = -1
    if len(outSax) > 0:
        saxFidelityTrain = helper.modelFidelity(outSax[2][0], outData[2][0])
        saxFidelityVal = helper.modelFidelity(outSax[2][1], outData[2][1])
        saxFidelityTest = helper.modelFidelity(outSax[3], outData[3])
    else:
        saxFidelityTrain = -1
        saxFidelityVal = -1
        saxFidelityTest = -1

    inputDict['Train Model Fidelity (Ori)'] = oriFidelityTrain
    inputDict['Train Model Fidelity (Sax)'] = saxFidelityTrain
    inputDict['Val Model Fidelity (Ori)'] = oriFidelityVal
    inputDict['Val Model Fidelity (Sax)'] = saxFidelityVal
    inputDict['Test Model Fidelity (Ori)'] = oriFidelityTest
    inputDict['Test Model Fidelity (Sax)'] = saxFidelityTest
    return inputDict

def getGCROptions():
    gcrList = [' Sum FCAM', ' r.Avg FCAM', ' Sum CRCAM', ' r.Avg CRCAM', 'gtm']
    return gcrList

#abstractionString, outOri, outSax, rMS, rMA, x_train1, predictions, x_test, y_testy, self.valuesA, self.gtmAbstractions
def classifyGCR(abstractionString, oriPredTest, saxPredTest, oriPredTrain, saxPredTrain, rMS, rMA, x_train1, predictions, x_test, y_testy, valuesA, gtmAbstractions, classifyString, rM, doTest = True, doPenalty=False):
    resultDict = dict()
    #FCAM
    abstractionString2 = abstractionString + ' Sum FCAM'
    if classifyString == ' Sum FCAM':
        print(abstractionString2)
        maxScores = GCRPlus.calcFCAMMaxScore(rMS, valuesA)
        gcrSFcamOut = GCRPlus.classFullAtt(rMS, x_test, y_testy, maxScores, rM, doPenalty=doPenalty)
        inputDict1 = fillOutDicWithGCRSmall(gcrSFcamOut, 'test', oriPredTest, saxPredTest)
        if doTest:
            gcrSFcamOutT = GCRPlus.classFullAtt(rMS, x_train1, predictions, maxScores, rM, doPenalty=doPenalty)
            inputDict2 = fillOutDicWithGCRSmall(gcrSFcamOutT, 'train', oriPredTrain, saxPredTrain)
            resultDict[abstractionString2] = {**inputDict1, **inputDict2} 
        else:
            resultDict[abstractionString2] = inputDict1

    abstractionString2 = abstractionString + ' r.Avg FCAM'
    if classifyString == ' r.Avg FCAM':
        print(abstractionString2)
        maxScores = GCRPlus.calcFCAMMaxScore(rMA, valuesA)
        gcrAFcamOut = GCRPlus.classFullAtt(rMA, x_test, y_testy, maxScores, rM, doPenalty=doPenalty)
        inputDict1 = fillOutDicWithGCRSmall(gcrAFcamOut, 'test',oriPredTest, saxPredTest)
        if doTest:
            gcrAFcamOutT = GCRPlus.classFullAtt(rMA, x_train1, predictions, maxScores, rM, doPenalty=doPenalty)
            inputDict2 = fillOutDicWithGCRSmall(gcrAFcamOutT, 'train',oriPredTrain, saxPredTrain)
            resultDict[abstractionString2] = {**inputDict1, **inputDict2} 
        else:
            resultDict[abstractionString2] = inputDict1

    #CRCAM
    abstractionString2 = abstractionString + ' Sum CRCAM'
    if classifyString == ' Sum CRCAM':
        print(abstractionString2)
        maxScores = GCRPlus.calcCRCAMMaxScore(rMA, 'x')
        gcrSCrcamOut = GCRPlus.xAttentionMatch(rMA, x_test, y_testy, 'average', maxScores, rM)
        inputDict1 = fillOutDicWithGCRSmall(gcrSCrcamOut, 'test', oriPredTest, saxPredTest)
        if doTest:
            gcrSCrcamOutT = GCRPlus.xAttentionMatch(rMA, x_train1, predictions, 'average', maxScores, rM)
            inputDict2 = fillOutDicWithGCRSmall(gcrSCrcamOutT, 'train', oriPredTrain, saxPredTrain)
            resultDict[abstractionString2] = {**inputDict1, **inputDict2} 
        else:
            resultDict[abstractionString2] = inputDict1

    abstractionString2 = abstractionString + ' r.Avg CRCAM'
    if classifyString == ' r.Avg CRCAM':
        print(abstractionString2)
        maxScores = GCRPlus.calcCRCAMMaxScore(rMA, 'xAvg')
        gcrACrcamOut = GCRPlus.xAttentionMatch(rMA, x_test, y_testy, 'average+', maxScores, rM)
        inputDict1 = fillOutDicWithGCRSmall(gcrACrcamOut, 'test', oriPredTest, saxPredTest)
        if doTest:
            gcrACrcamOutT = GCRPlus.xAttentionMatch(rMA, x_train1, predictions, 'average+', maxScores, rM)
            inputDict2 = fillOutDicWithGCRSmall(gcrACrcamOutT, 'train', oriPredTrain, saxPredTrain)
            resultDict[abstractionString2] = {**inputDict1, **inputDict2} 
        else:
            resultDict[abstractionString2] = inputDict1

    print("starting gtm")

    bestGTMIndex = 0
    bestGTMAcc = 0
    if classifyString == "gtm":
        #GTM

        
        for reductionInt in range(len(gtmAbstractions)):
            abstractionString2 = abstractionString + ' ' + gtmAbstractions[reductionInt]
            
            print(abstractionString2)
            
            gtmOut = GCRPlus.calcFullAbstractAttention(rMA,reductionInt, x_test, y_testy)
            if gtmOut[0][0] > bestGTMAcc:
                bestGTMAcc = gtmOut[0][0]
                bestGTMIndex = reductionInt
            inputDict1 = fillOutDicWithGCRSmall(gtmOut, 'test', oriPredTest, saxPredTest)
            if doTest:
                gtmOutT = GCRPlus.calcFullAbstractAttention(rMA,reductionInt, x_train1, predictions)
                inputDict2 = fillOutDicWithGCRSmall(gtmOutT, 'train', oriPredTrain, saxPredTrain)
                resultDict[abstractionString2] = {**inputDict1, **inputDict2} 
            else:
                resultDict[abstractionString2] = inputDict1

    return bestGTMIndex, resultDict, classifyString


def doGCRClassify(abstractionString, rMA, rMS, rM, oriPredTest, saxPredTest, oriPredTrain, saxPredTrain, x_train1, predictions, x_test, y_testy, gtmAbstractions, valuesA, doTest = True, doPenalty=False):
    answers = []
    methodStart = datetime.now()
    answers = Parallel(n_jobs=5, prefer="threads")(delayed(classifyGCR)(abstractionString, oriPredTest, saxPredTest, oriPredTrain, saxPredTrain, rMS, rMA, x_train1, predictions, x_test, y_testy, valuesA, gtmAbstractions, option, rM, doTest = doTest, doPenalty=doPenalty) for option in getGCROptions())
    resultDict = dict()
    print('GCR classify full done' + str(datetime.now() - methodStart))
    bestGTMIndex =0
    for ans in answers:
        if ans[2] == 'gtm':
            bestGTMIndex = ans[0]
        
        resultDict = {**resultDict, **ans[1]} 
        
    return bestGTMIndex, resultDict

def doTThresholdGCRFull(tSet, combination, fold, attentionQ, x_train1, y_train1, order, step1, step2, num_of_classes, valuesA,  oriPredTest, saxPredTest, oriPredTrain, saxPredTrain,  predictions, x_test, y_testy, gtmAbstractions, doTest = True):
    abstractionString = "Threshold GCR " 
    configString = "Threshold: " + str(tSet) + "; Combination: " + combination #+ ' ' + str(fold)
    rMA, rMS, rM = GCRPlus.makeAttention(attentionQ, x_train1, y_train1, order, step1, step2, num_of_classes, valuesA, doThreshold=True, doMax=False, doPenalty=False, threshold=tSet, reducePredictions=False)
    bestGTMIndex, resultDict = doGCRClassify(abstractionString, rMA, rMS, rM, oriPredTest, saxPredTest, oriPredTrain, saxPredTrain, x_train1, predictions, x_test, y_testy, gtmAbstractions, valuesA, doTest = doTest)
    
    return resultDict, configString

def doTThresholdGCRFidelityFull(tSet, combination, fold, attentionQ, x_train1, y_train1, order, step1, step2, num_of_classes, valuesA, oriPredTest, saxPredTest, oriPredTrain, saxPredTrain, predictions, x_test, y_testy, gtmAbstractions, doTest = True):
    abstractionString = "Fidelity Threshold "
    configString = "Threshold: " + str(tSet) + "; Combination: " + combination #+ ' ' + str(fold)
    rMA, rMS, rM = GCRPlus.makeAttention(attentionQ, x_train1, y_train1, order, step1, step2, num_of_classes, valuesA, doThreshold=True, doFidelity=True, doMax=False, doPenalty=False, threshold=tSet)
    bestGTMIndex, resultDict = doGCRClassify(abstractionString, rMA, rMS, rM, oriPredTest, saxPredTest, oriPredTrain, saxPredTrain, x_train1, predictions, x_test, y_testy, gtmAbstractions, valuesA, doTest = doTest)
    
    return resultDict, configString
    
def doPenaltyGCRFull(pMode, combination, fold, attentionQ, x_train1, y_train1, order, step1, step2, num_of_classes, valuesA, oriPredTest, saxPredTest, oriPredTrain, saxPredTrain,  predictions, x_test, y_testy, gtmAbstractions, doTest = True):
    abstractionString = "Penalty GCR "# + combination #+ ' ' + str(fold)
    configString = "PenaltyMode: " + str(pMode) + "; Combination: " + combination #+ ' ' + str(fold)
    rMA, rMS, rM = GCRPlus.makeAttention(attentionQ, x_train1, y_train1, order, step1, step2, num_of_classes, valuesA, doThreshold=False, doMax=False, doPenalty=True, penaltyMode=pMode, reducePredictions=False)
    bestGTMIndex, resultDict = doGCRClassify(abstractionString, rMA, rMS, rM, oriPredTest, saxPredTest, oriPredTrain, saxPredTrain, x_train1, predictions, x_test, y_testy, gtmAbstractions, valuesA, doTest = doTest, doPenalty=True)

    return resultDict, configString
