from sacred import Experiment
import seml
import warnings

import torch

from sklearn.metrics import accuracy_score

import os
import random
import numpy as np
from scipy import stats

from sklearn.model_selection import StratifiedKFold


#from modules import transformer
from modules import GCRPlus
from modules import dataset_selecter as ds
from modules import pytorchTrain as pt
from modules import saliencyHelper as sh
from modules import helper

from sklearn.ensemble import RandomForestClassifier

import ViT_LRP
import cnn_LRP

from datetime import datetime

import xgboost
from sklearn.model_selection import StratifiedKFold

ex = Experiment()
seml.setup_logger(ex)


@ex.post_run_hook
def collect_stats(_run):
    seml.collect_exp_stats(_run)


@ex.config
def config():
    overwrite = None
    db_collection = None
    if db_collection is not None:
        ex.observers.append(seml.create_mongodb_observer(db_collection, overwrite=overwrite))

def arrayToString(indexes):
    out = ""
    for i in indexes:
        out = out + ',' + str(i)
    return out


class ExperimentWrapper:
    """
    A simple wrapper around a sacred experiment, making use of sacred's captured functions with prefixes.
    This allows a modular design of the configuration, where certain sub-dictionaries (e.g., "data") are parsed by
    specific method. This avoids having one large "main" function which takes all parameters as input.
    """

    def __init__(self, init_all=True):
        if init_all:
            self.init_all()

    #init before the experiment!
    @ex.capture(prefix="init")
    def baseInit(self, nrFolds: int, patience: int, seed_value: int):
        self.seed_value = seed_value
        os.environ['PYTHONHASHSEED']=str(seed_value)# 2. Set `python` built-in pseudo-random generator at a fixed value
        random.seed(seed_value)# 3. Set `numpy` pseudo-random generator at a fixed value
        np.random.RandomState(seed_value)
        np.random.seed(seed_value)
        torch.manual_seed(0)

        #save some variables for later
        self.kf = StratifiedKFold(nrFolds, shuffle=True, random_state=seed_value)
        self.fold = 0
        self.nrFolds = nrFolds
        self.seed_value = seed_value       
        self.patience = patience

        #init gpu
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using {self.device} device")


    # Load the dataset
    @ex.capture(prefix="data")
    def init_dataset(self, dataset: str, toplevel: str, symbols: int, nrEmpty: int, andStack: int, orStack: int, xorStack: int, nrAnds: int, nrOrs: int, nrxor: int, trueIndexes, test_size: float, orOffSet: int, xorOffSet: int, redaundantIndexes):
        """
        Perform dataset loading, preprocessing etc.
        Since we set prefix="data", this method only gets passed the respective sub-dictionary, enabling a modular
        experiment design.
        """

        self.X_train, self.X_test, self.y_train, self.y_test, self.y_trainy, self.y_testy, self.seqSize, self.dataName, self.num_of_classes, self.symbolCount = ds.datasetSelector(dataset, self.seed_value, topLevel=toplevel, symbols = symbols, nrEmpty = nrEmpty, andStack = andStack, orStack = orStack, xorStack = xorStack, nrAnds = nrAnds, nrOrs = nrOrs, nrxor = nrxor, trueIndexes=trueIndexes, test_size=test_size, orOffSet=orOffSet, xorOffSet=xorOffSet, redaundantIndexes=redaundantIndexes)

        self.test_size = test_size
        self.inDim = self.X_train.shape[1]
        self.dataset = dataset
        self.toplevel = toplevel
        self.symbols = symbols
        self.nrEmpty = nrEmpty
        self.andStack = andStack
        self.orStack = orStack
        self.xorStack = xorStack
        self.nrAnds = nrAnds
        self.nrOrs = nrOrs
        self.nrxor = nrxor
        self.trueIndexes = trueIndexes
        self.orOffSet = orOffSet
        self.xorOffSet = xorOffSet
        self.redaundantIndexes = redaundantIndexes


        self.dsName = str(self.dataName) +  '-l' + str(toplevel) +'-s' +  str(symbols)+'-e' +  str(nrEmpty)+'-a' +  str(andStack)+'-o' +  str(orStack)+'-x' +  str(xorStack)+'-na' +  str(nrAnds)+'-no' +  str(nrOrs)+'-nx' +  str(nrxor)+'-i' +  arrayToString(trueIndexes)+'-t' +  str(test_size)+'-oo' +  str(orOffSet)+'-xo' +  str(xorOffSet) +'-r' + arrayToString(redaundantIndexes)
        print(self.dsName)


    #all inits
    def init_all(self):
        """
        Sequentially run the sub-initializers of the experiment.
        """
        self.baseInit()
        self.init_dataset()

    def printTime(self):
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Current Time =", current_time)


    # def trainExperiment(self, useSaves: bool, modelType: str, batch_size: int, epochs: int, numOfLayers: int, header:int, dmodel: int, dfff: int, dropout: float, att_dropout: float, doSkip: bool, doBn: bool, doClsTocken: bool, stride: int, kernal_size: int, nc: int, thresholdSet, methods): #, foldModel: int):
    # one experiment run with a certain config set. MOST OF THE IMPORTANT STUFF IS DONE HERE!!!!!!!
    @ex.capture(prefix="model")
    def trainExperiment(self, useSaves: bool, hypers, batch_size: int, stride: int, kernal_size: int, nc: int, thresholdSet, methods): #, foldModel: int):

        print('Dataname:')
        print(self.dsName)
        self.printTime()
        warnings.filterwarnings('ignore')   

        fullResults = dict()
        
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

        fullResultDir = 'presults'
        filteredResults= 'filteredResults'

        #don't recalculate already finished experiments
        wname = pt.getWeightName(self.dsName, self.dataName, batch_size, epochs, numOfLayers, header, dmodel, dfff, dropout, att_dropout, doSkip, doBn, doClsTocken, learning = False, results = True, resultsPath=filteredResults)
        if os.path.isfile(wname + '.pkl'):
            fullResults["Error"] = "dataset " + self.dsName + " already done: " + str(self.seqSize) + "; name: " + wname
            print('Already Done ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            print("dataset " + self.dataName + " already done: " + str(self.seqSize) + "; name: " + wname)
        
            return "dataset " + self.dataName + "already done: " + str(self.seqSize)  + "; name: " + wname 

        wname = pt.getWeightName(self.dsName, self.dataName, batch_size, epochs, numOfLayers, header, dmodel, dfff, dropout, att_dropout, doSkip, doBn, doClsTocken, learning = False, results = True, resultsPath=fullResultDir)
        if os.path.isfile(wname + '.pkl'):
            print('Already Done training ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            print("trained dataset " + self.dataName + " already done: " + str(self.seqSize) + "; name: " + wname)
            
            results = helper.load_obj(str(wname))

            for index, v in np.ndenumerate(results):
                fullResults = v
        else:
            fullResults['testData'] = self.X_test
            fullResults['testTarget'] = self.y_test

            #save all params!
            fullResults['params'] = dict()
            fullResults['params']['patience'] = self.patience
            fullResults['params']['fileName'] = self.dsName
            fullResults['params']['epochs'] = epochs
            fullResults['params']['batchSize'] = batch_size
            fullResults['params']['useSaves'] = useSaves
            fullResults['params']['numOfLayers'] = numOfLayers
            fullResults['params']['header'] = header
            fullResults['params']['dmodel'] = dmodel
            fullResults['params']['dfff'] = dfff
            fullResults['params']['dropout'] = dropout
            fullResults['params']['att_dropout'] = att_dropout
            fullResults['params']['doSkip'] = doSkip
            fullResults['params']['doBn'] = doBn
            fullResults['params']['modelType'] = modelType
            fullResults['params']['doClsTocken'] = doClsTocken
            fullResults['params']['thresholdSet'] = thresholdSet
            fullResults['params']['dataset'] = self.dataset
            fullResults['params']['toplevel'] = self.toplevel
            fullResults['params']['symbols'] = self.symbols
            fullResults['params']['nrEmpty'] = self.nrEmpty
            fullResults['params']['andStack'] = self.andStack
            fullResults['params']['orStack'] = self.orStack
            fullResults['params']['xorStack'] = self.xorStack
            fullResults['params']['nrAnds'] = self.nrAnds
            fullResults['params']['nrOrs'] = self.nrOrs
            fullResults['params']['nrxor'] = self.nrxor
            fullResults['params']['trueIndexes'] = self.trueIndexes
            fullResults['params']['test_size'] = self.test_size
            fullResults['params']['orOffSet'] = self.orOffSet
            fullResults['params']['xorOffSet'] = self.xorOffSet
            fullResults['params']['redaundantIndexes'] = self.redaundantIndexes
            print(fullResults['params'])

            fullResults['results'] = dict()
            resultDict = fullResults['results']
            resultDict['trainPred'] = []
            resultDict['trainAcc'] = []
            resultDict['trainLoss'] = []

            resultDict['valPred'] = []
            resultDict['valAcc'] = []
            resultDict['valLoss'] = []

            resultDict['testPred'] = []
            resultDict['testAcc'] = []
            resultDict['testLoss'] = []

            resultDict['trainData'] = []
            resultDict['trainTarget'] = []
            resultDict['valData'] = []
            resultDict['valTarget'] = []

            resultDict['treeScores'] = []
            resultDict['treeImportances'] = []


            print('Base data shapes:')
            print(self.X_train.shape)
            print(self.X_test.shape)


            self.fold = 0
            for train, test in self.kf.split(self.X_train, self.y_trainy):

                self.fold+=1
                print(f"Fold #{self.fold}")
                
                
                if self.test_size > 0 and  self.test_size < 1:
                    x_train1 = self.X_train[train]
                    x_val = self.X_train[test]
                    y_train1 = self.y_train[train]
                    y_trainy1 = self.y_trainy[train]
                    y_val = self.y_train[test]
                    
                else:
                    x_train1 = self.X_train.copy()
                    x_val = self.X_train.copy()
                    y_train1 = self.y_train.copy()
                    y_trainy1 = self.y_trainy.copy()
                    y_val = self.y_train.copy()

                x_test = self.X_test.copy()

                if modelType == 'CNN':
                    model = cnn_LRP.ResNetLikeClassifier(outputs=self.num_of_classes, inDim=self.inDim, num_hidden_layers=numOfLayers, nc=nc, nf=dmodel, dropout=dropout, maskValue = -2, stride=stride, kernel_size=kernal_size, doSkip=doSkip, doBn=doBn)
                    
                    if True:
                        x_train1 = np.expand_dims(x_train1,1)
                        x_val = np.expand_dims(x_val,1)
                        x_test = np.expand_dims(x_test,1)
                elif modelType == 'Transformer':
                    model = ViT_LRP.TSModel(num_hidden_layers=numOfLayers, inDim=self.inDim, dmodel=dmodel, dfff=dfff, num_heads=header, num_classes=self.num_of_classes, dropout=dropout, att_dropout=att_dropout, doClsTocken=doClsTocken)
                elif modelType == 'Tree':
                    
                    model = xgboost.XGBRegressor(n_estimators=100, learning_rate=0.1)

                    x_train1 = np.squeeze(x_train1)
                    x_test = np.squeeze(x_test)
                    x_val = np.squeeze(x_val)
                    y_train1 = np.argmax(y_train1, axis=1)+1
                    y_val = np.argmax(y_val, axis=1)+1
                    y_test = np.argmax(self.y_test, axis=1)+1
                else:  
                    raise ValueError('Not a valid model type: ' + modelType)

                print('Train data shapes:')
                print(x_train1.shape)
                print(x_test.shape)

                if modelType == 'Tree':
                    model, trainPred, trainAcc, trainLoss, valPred, valAcc, valLoss, testPred, testAcc, testLoss = pt.trainTree(model, x_train1, y_train1, x_val, y_val, x_test, y_test)
                else:
                    model.double()
                    model.to(self.device)
                    model, trainPred, trainAcc, trainLoss, valPred, valAcc, valLoss, testPred, testAcc, testLoss = pt.trainBig(self.device, model, x_train1, y_train1, x_val, y_val, x_test, self.patience, useSaves, self.y_test, batch_size, epochs, fileAdd=self.dsName)
                    model.eval()


                train_predictions = np.argmax(y_train1, axis=1)+1
                test_predictions = np.argmax(self.y_test, axis=1)+1

                newT = np.squeeze(x_train1)
                newG = np.squeeze(x_test)
                clf = RandomForestClassifier()
                clf.fit(newT, train_predictions)
                scores = clf.score(newG, test_predictions)
                resultDict['treeScores'].append(scores)
                resultDict['treeImportances'].append(clf.feature_importances_)
                
                resultDict['trainPred'].append(trainPred) 
                resultDict['trainAcc'].append(trainAcc)
                resultDict['trainLoss'].append(trainLoss)

                resultDict['valPred'].append(valPred)
                resultDict['valAcc'].append(valAcc)
                resultDict['valLoss'].append(valLoss)

                resultDict['testPred'].append(testPred)
                resultDict['testAcc'].append(testAcc)
                resultDict['testLoss'].append(testLoss)

                resultDict['trainData'].append(x_train1)
                resultDict['trainTarget'].append(y_train1)
                resultDict['valData'].append(x_val)
                resultDict['valTarget'].append(y_val)

                if 'saliency' not in fullResults.keys():
                    fullResults['saliency'] = dict()
                saliencies = fullResults['saliency']

                for method in methods[modelType].keys():
                    for submethod in methods[modelType][method]:
                        if method+'-'+submethod not in saliencies.keys():
                            saliencies[method+'-'+submethod] = dict()
                            outMap = saliencies[method+'-'+submethod]
                            outMap['Fidelity'] = dict()
                            outMap['Infidelity'] = dict()
                            outMap['outTrain'] = []
                            outMap['outVal'] = []
                            outMap['outTest'] = []
                            outMap['modelTrain'] = []
                            outMap['modelVal'] = []
                            outMap['modelTest'] = []
                            outMap['means'] = dict()
                            outMap['means']['outTrain'] = []
                            outMap['means']['outVal'] = []
                            outMap['means']['outTest'] = []
                            outMap['means']['modelTrain'] = []
                            outMap['means']['modelVal'] = []
                            outMap['means']['modelTest'] = []
                            outMap['classes'] = dict()
                            for c in range(self.num_of_classes):
                                outMap['classes'][str(c)] = dict()
                                outMap['classes'][str(c)]['outTrain']= []
                                outMap['classes'][str(c)]['outVal'] = []
                                outMap['classes'][str(c)]['outTest'] = []
                            outMap['TargetClasses'] = dict()
                            outMap['ModelClasses'] = dict()
                        outMap = saliencies[method+'-'+submethod]
                        if submethod.startswith('smooth'):
                            smooth = True
                        else:
                            smooth = False
                        

                        outTrain, outVal, outTest, data3D, data2D = sh.getSaliencyMap(outMap, "out", self.device, self.num_of_classes, modelType, method, submethod, model, x_train1, x_val, x_test, trainPred, valPred, testPred, smooth, doClassBased=True)
                        sh.mapSaliency(outMap['ModelClasses'], self.num_of_classes, outTrain, trainPred, outVal, valPred, outTest, testPred, do3DData=data3D)

                        for doFidelity in [False]:#[True, False]:
                            if doFidelity:
                                rfDict = outMap['Fidelity']
                            else:
                                rfDict = outMap['Infidelity']
                            for threshold in thresholdSet:
                                print('Starting threshold:')
                                print(threshold)
                                if threshold == 'baseline':
                                    newTrain, trainReduction = sh.doSimpleLasaROAR(outTrain, x_train1, self.nrEmpty,  doBaselineT=True, doFidelity=doFidelity, do3DData=data3D, do3rdStep=data2D)
                                    newVal, valReduction = sh.doSimpleLasaROAR(outVal, x_val, self.nrEmpty, doBaselineT=True, doFidelity=doFidelity, do3DData=data3D, do3rdStep=data2D)
                                    newTest, testReduction = sh.doSimpleLasaROAR(outTest, x_test, self.nrEmpty,  doBaselineT=True, doFidelity=doFidelity, do3DData=data3D, do3rdStep=data2D)
                                else:
                                    newTrain, trainReduction = sh.doSimpleLasaROAR(outTrain, x_train1, threshold, doFidelity=doFidelity, do3DData=data3D, do3rdStep=data2D)
                                    newVal, valReduction = sh.doSimpleLasaROAR(outVal, x_val, threshold, doFidelity=doFidelity, do3DData=data3D, do3rdStep=data2D)
                                    newTest, testReduction = sh.doSimpleLasaROAR(outTest, x_test, threshold, doFidelity=doFidelity, do3DData=data3D, do3rdStep=data2D)


                                if modelType == 'CNN':
                                    model2 = cnn_LRP.ResNetLikeClassifier(outputs=self.num_of_classes, inDim=self.inDim, num_hidden_layers=numOfLayers, nc=1, nf=dmodel, dropout=dropout, maskValue = -2, stride=1, kernel_size=3, doSkip=doSkip, doBn=doBn)
                                    

                                    newTrain = np.expand_dims(newTrain,1)
                                    newVal = np.expand_dims(newVal,1)
                                    newTest = np.expand_dims(newTest,1)
                                elif modelType == 'Transformer':
                                    model2 = ViT_LRP.TSModel(num_hidden_layers=numOfLayers, inDim=self.inDim, dmodel=dmodel, dfff=dfff, num_heads=header, num_classes=self.num_of_classes, dropout=dropout, att_dropout=att_dropout, doClsTocken=doClsTocken)
                                elif modelType == 'Tree':
                                    model2 = xgboost.XGBRegressor(n_estimators=100, learning_rate=0.1)

                                    newTrain = np.squeeze(newTrain)
                                    newTest = np.squeeze(newTest)
                                    newVal = np.squeeze(newVal)
                                else: 
                                    raise ValueError('Not a valid model type: ' + modelType)
                                
                                if modelType == 'Tree':
                                    model2, trainPred2, trainAcc2, trainLoss2, valPred2, valAcc2, valLoss2, testPred2, testAcc2, testLoss2 = pt.trainTree(model2, newTrain, y_train1, newVal, y_val, newTest, y_test)
                                else:
                                    model2.double()
                                    model2.to(self.device)
                                    model2, trainPred2, trainAcc2, trainLoss2, valPred2, valAcc2, valLoss2, testPred2, testAcc2, testLoss2 = pt.trainBig(self.device, model2, newTrain, y_train1, newVal, y_val, newTest, self.patience, False, self.y_test, batch_size, epochs, fileAdd=self.dsName+"-lasa")
                                    model2.eval()

                                if str(threshold) not in rfDict.keys():
                                    rfDict[str(threshold)] = dict()
                                    rftDict = rfDict[str(threshold)]
                                    rftDict['trainPred'] = []
                                    rftDict['trainAcc'] = []
                                    rftDict['trainLoss'] = []

                                    rftDict['valPred'] = []
                                    rftDict['valAcc'] = []
                                    rftDict['valLoss'] = []

                                    rftDict['testPred'] = []
                                    rftDict['testAcc'] = []
                                    rftDict['testLoss'] = []

                                    rftDict['treeScores'] = []
                                    rftDict['treeImportances'] = []

                                    
                                    rftDict['trainReduction'] = []
                                    rftDict['valReduction'] = []
                                    rftDict['testReduction'] = []

                                    rftDict['approx data'] = dict()
                                    subrftDict = rftDict['approx data']

                                    subrftDict['train minAcc'] = []
                                    subrftDict['train bestAcc'] = []
                                    subrftDict['train predBest'] = []
                                    subrftDict['train baselineAcc'] = []

                                    subrftDict['val minAcc'] = []
                                    subrftDict['val bestAcc'] = []
                                    subrftDict['val predBest'] = []
                                    subrftDict['val baselineAcc'] = []

                                    subrftDict['test minAcc'] = []
                                    subrftDict['test bestAcc'] = []
                                    subrftDict['test predBest'] = []
                                    subrftDict['test baselineAcc'] = []

                                    subrftDict['train valueMap'] = []
                                    subrftDict['val valueMap'] = []
                                    subrftDict['test valueMap'] = []
                                    
                                    subrftDict['train conValueMap'] = []
                                    subrftDict['val conValueMap'] = []
                                    subrftDict['test conValueMap'] = []

                                    subrftDict['train valueMap mp'] = []
                                    subrftDict['val valueMap mp'] = []
                                    subrftDict['test valueMap mp'] = []
                                    
                                    subrftDict['train conValueMap mp'] = []
                                    subrftDict['val conValueMap mp'] = []
                                    subrftDict['test conValueMap mp'] = []

                                rftDict = rfDict[str(threshold)]
                                subrftDict = rftDict['approx data']

                                rftDict['trainPred'].append(trainPred2)
                                rftDict['trainAcc'].append(trainAcc2)
                                rftDict['trainLoss'].append(trainLoss2)

                                rftDict['valPred'].append(valPred2)
                                rftDict['valAcc'].append(valAcc2)
                                rftDict['valLoss'].append(valLoss2)

                                rftDict['testPred'].append(testPred2)
                                rftDict['testAcc'].append(testAcc2)
                                rftDict['testLoss'].append(testLoss2)

                                rftDict['trainReduction'].append(trainReduction)
                                rftDict['valReduction'].append(valReduction)
                                rftDict['testReduction'].append(testReduction)


                                train_predictions = np.argmax(y_train1, axis=1)+1
                                test_predictions = np.argmax(self.y_test, axis=1)+1

                                newT = np.squeeze(newTrain)
                                newG = np.squeeze(newTest)

                                clf = RandomForestClassifier()
                                clf.fit(newT, train_predictions)
                                scores = clf.score(newG, test_predictions)
                                rftDict['treeScores'].append(scores)
                                rftDict['treeImportances'].append(clf.feature_importances_)

        saveName = pt.getWeightName(self.dsName, self.dataName, batch_size, epochs, numOfLayers, header, dmodel, dfff, dropout, att_dropout, doSkip, doBn, doClsTocken, learning = False, results = True, resultsPath=fullResultDir)
        print(saveName)
        helper.save_obj(fullResults, str(saveName))
        saveName = self.evaluateAndSaveResults(fullResults, useSaves, hypers, batch_size, stride, kernal_size, nc, thresholdSet, methods, filteredResults)

        

        self.printTime()

        return saveName


    def evaluateAndSaveResults(self, res, useSaves: bool, hypers, batch_size: int, stride: int, kernal_size: int, nc: int, thresholdSet, methods, filteredResults):
        self.printTime()
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

        fullResults = res
        
        trueIndexes = res['params']['trueIndexes']
        nrSymbols = res['params']['symbols']
        symbolA = helper.getMapValues(nrSymbols)
        symbolA = np.array(symbolA)
        trueSymbols = symbolA[trueIndexes]
        falseSymbols = np.delete(symbolA, trueIndexes)
        res['params']

        andStack = res['params']['andStack']
        orStack = res['params']['orStack']
        xorStack = res['params']['xorStack']
        nrAnds = res['params']['nrAnds']
        nrOrs = res['params']['nrOrs']
        nrxor = res['params']['nrxor']
        orOffSet = res['params']['orOffSet']
        xorOffSet = res['params']['xorOffSet']
        topLevel = res['params']['toplevel']

        tl = res['testTarget'] 
        gt = res['testData']
        num_of_classes = len(set(list(tl.flatten())))
        
        finalResults = dict()
        finalResults['performance'] = []
        finalResults['tree performance'] = []
        finalResults['treeImportances'] = []
        finalResults['baseline'] = []

        finalResults['gcr'] = dict()
        for s in res['saliency'].keys():
            finalResults[s] = dict()

            finalResults[s]['saliency'] = []
            finalResults[s]['gtm rAvg acc'] = []
            finalResults[s]['gtm sum acc'] = [] 

            finalResults[s]['avgCorrelation'] = []
            finalResults[s]['percentPValueBroken'] = []
            finalResults[s]['andCorrelation'] = []
            finalResults[s]['orCorrelation'] = []
            finalResults[s]['xorCorrelation'] = []
            finalResults[s]['irrelevantCorrelation'] = []
            finalResults[s]['neighbourhoodCorrelation'] = []
            
            
            
            
            finalResults['gcr'][s] = dict()
            finalResults['gcr'][s]['rMA'] = dict()
            finalResults['gcr'][s]['rMS'] = dict()
            finalResults['gcr'][s]['rMA']['gcr'] = []
            finalResults['gcr'][s]['rMS']['gcr'] = []
            for gtmAbst in GCRPlus.gtmReductionStrings():
                finalResults['gcr'][s][gtmAbst] = dict()

            for sKeys in finalResults['gcr'][s].keys():
                finalResults['gcr'][s][sKeys]['acc'] =[]
                finalResults['gcr'][s][sKeys]['predicsion'] =[]
                finalResults['gcr'][s][sKeys]['recall'] =[]
                finalResults['gcr'][s][sKeys]['f1'] =[]
                finalResults['gcr'][s][sKeys]['predictResults'] =[]

            thresholds = res['params']['thresholdSet']
            
            for t in thresholds:
                t = 't'+str(t)
                finalResults[s][t] = dict()
                finalResults[s][t]['treeScores'] = []
                finalResults[s][t]['lasa acc'] = []
                finalResults[s][t]['lasa red'] = []
                finalResults[s][t]['PositiveValueRemoval'] = []
                finalResults[s][t]['PositiveValueRemoval 1s'] = []
                finalResults[s][t]['NegativeValueRemoval'] = []
                finalResults[s][t]['NegativeValueRemoval 1s'] = []
                finalResults[s][t]['Tendeny mean'] = []
                finalResults[s][t]['Tendeny median'] = []
                finalResults[s][t]['Tendeny 1s'] = []
                finalResults[s][t]['RemovalChancePP'] = []
                finalResults[s][t]['RemovalChancePosLable'] = []
                finalResults[s][t]['RemovalChanceNegLable'] = []
                finalResults[s][t]['RemovalChancePInputValue'] = dict()
                finalResults[s][t]['DoubleAssigmentTruthTable'] = dict()
                finalResults[s][t]['DoubleAssigmentTruthTableClasses'] = dict()
                finalResults[s][t]['DoubleAssigmentTableClasses'] = dict()
                finalResults[s][t]['DoubleAssigmentFull'] = dict()
                finalResults[s][t]['DoubleAssigmentTruthTableMin'] = dict()
                finalResults[s][t]['DoubleAssigmentFullPercent'] = dict()
                finalResults[s][t]['DoubleAssigmentTruthTableMinPercent'] = dict()
                finalResults[s][t]['LogicalAcc'] = []
                finalResults[s][t]['LogicalAccStatistics'] = []                
                finalResults[s][t]['MaxMappingAcc'] = [] #first Layer optimization

                for c in range(2):#NOTE fix for multi classes
                    finalResults[s][t][c] = dict()
                    finalResults[s][t][c]['Tendeny mean'] = []
                    finalResults[s][t][c]['Tendeny median'] = []
                    finalResults[s][t][c]['Tendeny 1s'] = []
                    finalResults[s][t][c]['RemovalChancePP'] = []
                    finalResults[s][t][c]['PositiveValueRemoval'] = []
                    finalResults[s][t][c]['NegativeValueRemoval'] = []

                #addMaskedValue
                finalResults['gcr'][s][t] = dict()
                for tv in [True, False]:
                    finalResults['gcr'][s][t][tv] = dict()
                    finalResults['gcr'][s][t][tv]['rMA'] = dict()
                    finalResults['gcr'][s][t][tv]['rMS'] = dict()
                    finalResults['gcr'][s][t][tv]['rMA']['gcr'] = []
                    finalResults['gcr'][s][t][tv]['rMS']['gcr'] = []
                    for gtmAbst in GCRPlus.gtmReductionStrings():
                        finalResults['gcr'][s][t][tv][gtmAbst] = dict()

                    for sKeys in finalResults['gcr'][s][t][tv].keys():
                        finalResults['gcr'][s][t][tv][sKeys]['acc'] =[]
                        finalResults['gcr'][s][t][tv][sKeys]['predicsion'] =[]
                        finalResults['gcr'][s][t][tv][sKeys]['recall'] =[]
                        finalResults['gcr'][s][t][tv][sKeys]['f1'] =[]
                        finalResults['gcr'][s][t][tv][sKeys]['predictResults'] =[]

            for c in range(2):
                finalResults[s][c] = dict()
                finalResults[s][c]['saliency'] = []
                finalResults[s][c]['saliencyPerInputValue'] = dict() 
                finalResults[s][c]['wrongImportanceMeanPercent'] = []
                finalResults[s][c]['wrongImportancePercent'] = [] 
                finalResults[s][c]['generalInformationBelowBaseline'] = [] 
                finalResults[s][c]['neededInformationBelowBaseline'] = []
                finalResults[s][c]['neededInformationStackBroken'] = []

            finalResults[s]['saliencyPerStack'] = dict() 
            finalResults[s]['saliencyPerInputValue'] = dict() 


            finalResults[s]['wrongImportanceMeanPercent'] = [] #Avg of all folds!
            finalResults[s]['wrongImportancePercent'] = [] #For each fold seperatly!



        for s in res['results']['treeScores']:
            finalResults['tree performance'].append(s)

        for s in res['results']['treeImportances']:
            finalResults['treeImportances'].append(s)

        for s in res['results']['testAcc']:
            finalResults['performance'].append(s)

        tl = res['testTarget']
        baselineAcc = accuracy_score(np.zeros(len(tl)), tl.argmax(axis=1))
        finalResults['baseline'].append(baselineAcc)

        for k in res['saliency'].keys():
            stackSaliencyStacks = dict()
            saliencyStacksValues = dict()
            valueSaliency = dict()
            valueClassSaliency = dict()

            for v in np.mean(np.array(res['saliency'][k]['outTest']), axis=(1)):
                finalResults[k]['saliency'].append(v)

            for c in res['saliency'][k]['classes'].keys():
                for v in np.mean(np.array(res['saliency'][k]['outTest'])[:,np.argmax(tl, axis=1)== int(c)], axis=(1)):
                    finalResults[k][int(c)]['saliency'].append(v)

            for f in range(len(res['saliency'][k]['outTest'])):
                sd = res['saliency'][k]['outTest'][f]
                do3DData = False
                do2DData = False
                if len(sd.shape) > 3:
                    do3DData = True
                elif len(sd.shape) > 2:
                    do2DData = True
                _, saliencyStacks, _, _ = sh.getPredictionMaps('', '', gt, sd, tl, num_of_classes, topLevel, nrSymbols, andStack, orStack, xorStack, nrAnds, nrOrs, nrxor, orOffSet, xorOffSet, trueIndexes)
                for k2 in saliencyStacks.keys():
                    if k2 not in finalResults[k]['saliencyPerStack'].keys():
                        finalResults[k]['saliencyPerStack'][k2] = dict()
                        finalResults[k]['saliencyPerStack'][k2][0] = []
                        finalResults[k]['saliencyPerStack'][k2][1] = []
                    finalResults[k]['saliencyPerStack'][k2][0].append(np.mean(saliencyStacks[k2][0], axis=0))
                    finalResults[k]['saliencyPerStack'][k2][1].append(np.mean(saliencyStacks[k2][1], axis=0))

                saliencyStacksValues = sh.splitSaliencyPerStack(saliencyStacksValues, sd, gt, gt, tl, num_of_classes, nrSymbols, andStack, orStack, xorStack, nrAnds, nrOrs, nrxor, orOffSet, xorOffSet, trueIndexes, do3DData=do3DData, do3rdStep=do2DData)

                valueSaliency = sh.splitPerValue(valueSaliency, sd, gt, nrSymbols)

                valueClassSaliency = sh.splitPerValueAndClass(valueClassSaliency, sd, gt, nrSymbols, num_of_classes, tl)

            for l in range(len(gt[0])):
                for k2 in valueSaliency.keys():
                    valueSaliency[k2][l] = np.mean(valueSaliency[k2][l])
            for k2 in valueSaliency.keys():
                if k2 not in finalResults[k]['saliencyPerInputValue'].keys():
                    finalResults[k]['saliencyPerInputValue'][k2] = []
                finalResults[k]['saliencyPerInputValue'][k2].append(valueSaliency[k2])


            for c in res['saliency'][k]['classes'].keys():
                c = int(c)
                for l in range(len(gt[0])):
                    for k2 in valueClassSaliency[c].keys():
                        valueClassSaliency[c][k2][l] = np.nan_to_num(np.mean(valueClassSaliency[c][k2][l]))
                for k2 in valueClassSaliency[c].keys():
                    if k2 not in finalResults[k][c]['saliencyPerInputValue'].keys():
                        finalResults[k][c]['saliencyPerInputValue'][k2] = []
                    finalResults[k][c]['saliencyPerInputValue'][k2].append(valueClassSaliency[c][k2])                      

            
            saliency1Ds = []
            for f in np.array(res['saliency'][k]['outTest']):
                saliency1Ds.append(sh.reduceMap(f, do3DData=do3DData, do3rdStep=do2DData))
            saliency1Ds = np.array(saliency1Ds)

            wrongImportanceMean = np.zeros(len(res['saliency'][k]['means']['outTest'][0]))
            for r in np.mean(saliency1Ds, axis=1):
                baseline = np.max(r[-1* res['params']['nrEmpty']:])

                for l in range(len(res['saliency'][k]['means']['outTest'][0])-res['params']['nrEmpty']):
                    if r[l] < baseline:
                        wrongImportanceMean[l] += 1
                    for p in range(res['params']['nrEmpty']):
                        if r[l] < r[-1*(p+1)]:
                            wrongImportanceMean[-1*(p+1)] += 1

            finalResults[k]['wrongImportanceMeanPercent'] = wrongImportanceMean / len(saliency1Ds)
            

            
            for c in res['saliency'][k]['classes'].keys():
                wrongImportanceMeanC = np.zeros(len(res['saliency'][k]['means']['outTest'][0]))
                for r in np.mean(saliency1Ds[:,np.argmax(tl, axis=1)== int(c)], axis=(1)):
                    baseline = np.max(r[-1* res['params']['nrEmpty']:])
                    for l in range(len(res['saliency'][k]['means']['outTest'][0])-res['params']['nrEmpty']):
                        if r[l] < baseline:
                            wrongImportanceMeanC[l] += 1
                        for p in range(res['params']['nrEmpty']):
                            if r[l] < r[-1*(p+1)]:
                                wrongImportanceMeanC[-1*(p+1)] += 1
                finalResults[k][int(c)]['wrongImportanceMeanPercent'] = wrongImportanceMeanC / len(saliency1Ds)
            
            for f in saliency1Ds:
                wrongImportance = np.zeros(len(res['saliency'][k]['outTest'][0][0]))
                for r in f:
                    baseline = np.max(r[-1* (res['params']['nrEmpty']):])
                    for l in range(len(res['saliency'][k]['outTest'][0][0])-res['params']['nrEmpty']):
                        if r[l] < baseline:
                            wrongImportance[l] += 1
                        for p in range(res['params']['nrEmpty']):
                            if r[l] < r[-1*(p+1)]:
                                wrongImportance[-1*(p+1)] += 1
                finalResults[k]['wrongImportancePercent'].append(wrongImportance/ len(f)) 

            for c in res['saliency'][k]['classes'].keys():
                for f in saliency1Ds[:,np.argmax(tl, axis=1)== int(c)]:
                    wrongImportanceC = np.zeros(len(res['saliency'][k]['outTest'][0][0]))
                    for r in f:
                        baseline = np.max(r[-1* (res['params']['nrEmpty']):])
                        for l in range(len(res['saliency'][k]['outTest'][0][0])-res['params']['nrEmpty']):
                            if r[l] < baseline:
                                wrongImportanceC[l] += 1
                            for p in range(res['params']['nrEmpty']):
                                if r[l] < r[-1*(p+1)]:
                                    wrongImportanceC[-1*(p+1)] += 1
                    finalResults[k][int(c)]['wrongImportancePercent'].append(wrongImportanceC/ len(f)) 
                    
            

            for f in saliency1Ds:
                wongMeaningImportanceTemp, countImportanceMeaningTemp = sh.getPredictionSaliency(res['params']['nrEmpty'], f, gt, res['saliency'][k]['classes'].keys(), tl, nrSymbols, andStack, orStack, xorStack, nrAnds, nrOrs, nrxor, orOffSet, xorOffSet, trueIndexes)
                for c in wongMeaningImportanceTemp.keys():
                    finalResults[k][c]['generalInformationBelowBaseline'].append(wongMeaningImportanceTemp[c]/countImportanceMeaningTemp[c])

                brokenImportanceTemp, brokenStackImportanceTemp = sh.getStackImportanceBreak(res['params']['nrEmpty'], f, gt, res['saliency'][k]['classes'].keys(), tl, nrSymbols, andStack, orStack, xorStack, nrAnds, nrOrs, nrxor, orOffSet, xorOffSet, trueIndexes, topLevel)
                for c in brokenImportanceTemp.keys():
                    finalResults[k][c]['neededInformationBelowBaseline'].append(brokenImportanceTemp[c])
                    finalResults[k][c]['neededInformationStackBroken'].append(brokenStackImportanceTemp[c])         

            korrelationIndication = dict()
            for f in range(len(res['saliency'][k]['outTest'])):
                sd = saliency1Ds[f]

                for l in range(len(sd[0])):
                    x = sd[:, l]
                    for l2 in range(len(sd[0])):

                        y = sd[:, l2]
                        pear = stats.pearsonr(x, y)

                        if l not in korrelationIndication.keys():
                            korrelationIndication[l] = dict()
                        if l2 not in korrelationIndication[l].keys():
                            korrelationIndication[l][l2] = []

                        korrelationIndication[l][l2].append(np.abs(np.nan_to_num(np.array(pear))))

            
            fullMean = []
            perNeighborhood = [[],[]]
            notSignificant = 0
            nrInputs = 0

            andStart = 0
            andEnd = (nrAnds * andStack)
            andRange = range(andStart,andEnd)
            orStart = andEnd
            orEnd = andEnd + (nrOrs * orStack - orOffSet * orStack)
            orRange = range(orStart,orEnd)
            xorStart = orEnd
            xorEnd = orEnd+ (nrxor * xorStack - xorOffSet * xorStack)
            xorRange = range(xorStart,xorEnd)
            irrelevantStart = xorEnd
            irrelevantEnd = len(res['saliency'][k]['outTest'][0][0]) 
            irreRange = range(irrelevantStart, irrelevantEnd)

            andCorrelations = []
            for i in range(andEnd-andStart):
                andCorrelations.append([])
            orCorrelations = []
            for i in range(orEnd-orStart):
                orCorrelations.append([])
            xorCorrelations = []
            for i in range(xorEnd-xorStart):
                xorCorrelations.append([])
            irrelevantCorrelations = []
            for i in range(irrelevantEnd-irrelevantStart):
                irrelevantCorrelations.append([])

            for key in korrelationIndication:
                if key -1 in korrelationIndication[key].keys():
                    perNeighborhood[0].append(np.array(korrelationIndication[key][key -1])[:,0])
                if key +1 in korrelationIndication[key].keys():
                    perNeighborhood[1].append(np.array(korrelationIndication[key][key +1])[:,0])
                for key2 in korrelationIndication[key]:
                    if key != key2:
                        if key in andRange and key2 in andRange: 
                            andCorrelations[key2-andStart].append(np.array(korrelationIndication[key][key2])[:,0])
                        if key in orRange and key2 in orRange: 
                            orCorrelations[key2-orStart].append(np.array(korrelationIndication[key][key2])[:,0])
                        if key in xorRange and key2 in xorRange: 
                            xorCorrelations[key2-xorStart].append(np.array(korrelationIndication[key][key2])[:,0])
                        if key in irreRange and key2 in irreRange: 
                            irrelevantCorrelations[key2-irrelevantStart].append(np.array(korrelationIndication[key][key2])[:,0])

                        fullMean.append(np.array(korrelationIndication[key][key2])[:,0])
                        notSignificant += np.sum(np.array(korrelationIndication[key][key2])[:,1] >= 0.05)
                        nrInputs += len(np.array(korrelationIndication[key][key2])[:,1])

            finalResults[k]['avgCorrelation'].append(np.mean(fullMean))
            finalResults[k]['percentPValueBroken'].append(notSignificant/nrInputs)

            if len(andCorrelations) != 0:
                if(len(np.array(andCorrelations).shape) == 2):
                    finalResults[k]['andCorrelation'].append(np.mean(andCorrelations, axis=(1)) - np.mean(fullMean))
                else:
                    finalResults[k]['andCorrelation'].append(np.mean(andCorrelations, axis=(1,2)) - np.mean(fullMean))
            if len(orCorrelations) != 0:
                if(len(np.array(orCorrelations).shape) == 2):
                    finalResults[k]['orCorrelation'].append(np.mean(orCorrelations, axis=(1)) - np.mean(fullMean))
                else:
                    finalResults[k]['orCorrelation'].append(np.mean(orCorrelations, axis=(1,2)) - np.mean(fullMean))
            if len(xorCorrelations) != 0:
                if(len(np.array(xorCorrelations).shape) == 2):
                    finalResults[k]['xorCorrelation'].append(np.mean(xorCorrelations, axis=(1)) - np.mean(fullMean))
                else:
                    finalResults[k]['xorCorrelation'].append(np.mean(xorCorrelations, axis=(1,2)) - np.mean(fullMean))

            if len(irrelevantCorrelations) != 0:
                if(len(np.array(xorCorrelations).shape) == 2):
                    finalResults[k]['irrelevantCorrelation'].append(np.mean(irrelevantCorrelations, axis=(1)) - np.mean(fullMean))
                else:
                    finalResults[k]['irrelevantCorrelation'].append(np.mean(irrelevantCorrelations, axis=(1,2)) - np.mean(fullMean))
            if len(perNeighborhood) != 0:
                if(len(np.array(perNeighborhood).shape) == 2):
                    finalResults[k]['neighbourhoodCorrelation'].append(np.mean(perNeighborhood, axis=(1)) - np.mean(fullMean))
                else:
                    finalResults[k]['neighbourhoodCorrelation'].append(np.mean(perNeighborhood, axis=(1,2)) - np.mean(fullMean))
        
        print('starting GCR')
        self.printTime()
        doBaseGCR = False
        if doBaseGCR:
            for normName in GCRPlus.getAllReductionNames():
                for ti, trainData in enumerate(fullResults['results']['trainData']):

                    for k in res['saliency'].keys():
                        sd = res['saliency'][k]['outTest'][ti]
                        tsd = res['saliency'][k]['outTrain'][ti]

                        do3DData = False
                        do2DData = False
                        if len(tsd.shape) > 3:
                            do3DData = True
                        elif len(tsd.shape) > 2:
                            do2DData = True


                        #TODO ranges einbauen!!!!
                        sGCRResults = sh.do3DGCR(tsd, trainData, res['results']['trainPred'][ti], gt, res['results']['testPred'][ti], num_of_classes, nrSymbols, doMetrics=False, addMaskedValue=False, reductionName=normName, do3DData=do3DData, do2DData=do2DData)
                    
                        print(finalResults['gcr'].keys())
                        finalResults['gcr'][k]['rMS']['acc'].append(sGCRResults[0][0][0])
                        finalResults['gcr'][k]['rMS']['gcr'].append(sGCRResults[-1])
                        finalResults['gcr'][k]['rMS']['predicsion'].append(sGCRResults[0][0][1])
                        finalResults['gcr'][k]['rMS']['recall'].append(sGCRResults[0][0][2])
                        finalResults['gcr'][k]['rMS']['f1'].append(sGCRResults[0][0][3])


                        finalResults['gcr'][k]['rMA']['gcr'].append(sGCRResults[-2])
                        finalResults['gcr'][k]['rMA']['acc'].append(sGCRResults[1][0][0])
                        finalResults['gcr'][k]['rMA']['predicsion'].append(sGCRResults[1][0][1])
                        finalResults['gcr'][k]['rMA']['recall'].append( sGCRResults[1][0][2])
                        finalResults['gcr'][k]['rMA']['f1'].append(sGCRResults[1][0][3])


                        for gtmAbstact in GCRPlus.gtmReductionStrings():
                            finalResults['gcr'][k][gtmAbstact]['acc'].append(sGCRResults[2][gtmAbstact]['performance'][0][0])
                            finalResults['gcr'][k][gtmAbstact]['predicsion'].append(sGCRResults[2][gtmAbstact]['performance'][0][1])
                            finalResults['gcr'][k][gtmAbstact]['recall'].append(sGCRResults[2][gtmAbstact]['performance'][0][2])
                            finalResults['gcr'][k][gtmAbstact]['f1'].append(sGCRResults[2][gtmAbstact]['performance'][0][3])

        print('finished basic GCR')
        self.printTime()
        for t in thresholds:
            self.thresholdsProcess(res, t, finalResults, doGCR=True)
                

        saveName = pt.getWeightName(self.dsName, self.dataName, batch_size, epochs, numOfLayers, header, dmodel, dfff, dropout, att_dropout, doSkip, doBn, doClsTocken,learning = False, results = True, resultsPath=filteredResults)
        helper.save_obj(finalResults, str(saveName))

        return saveName


  
    def thresholdsProcess(self, res, thold, finalResults, doGCR=False):
        
        gt = res['testData']#.squeeze()
        tl = res['testTarget']
        
        num_of_classes = len(set(list(tl.flatten())))
        trueIndexes = res['params']['trueIndexes']
        nrSymbols = res['params']['symbols']
        symbolA = helper.getMapValues(nrSymbols)
        symbolA = np.array(symbolA)
        trueSymbols = symbolA[trueIndexes]
        falseSymbols = np.delete(symbolA, trueIndexes)

        andStack = res['params']['andStack']
        orStack = res['params']['orStack']
        xorStack = res['params']['xorStack']
        nrAnds = res['params']['nrAnds']
        nrOrs = res['params']['nrOrs']
        nrxor = res['params']['nrxor']
        orOffSet = res['params']['orOffSet']
        xorOffSet = res['params']['xorOffSet']
        topLevel = res['params']['toplevel']


        for k in res['saliency'].keys():
            negVRem = [] # percent of neg removals'
            posVRem = [] # percent of pos removals'
            tendency = [] # tendency of pos or neg removal

            rcpp = [] #removal chance per position
            rcplp = [] #removal chance per position lable pos
            rcpln = [] #removal chance per position lable neg
            nrem = [] #how often is neg value (-1) removed
            prem = [] #how often is pos value (+1) removed
            sds = [] #saliencyMaps
            stackValioCount = dict() # double truth table assignment

            for f in range(len(res['saliency'][k]['outTest'])):
                negVRem.append([])
                posVRem.append([])
                tendency.append([])
                sd = res['saliency'][k]['outTest'][f]
                tsd = res['saliency'][k]['outTrain'][f]
                traingt = res['results']['trainData'][f]#.squeeze()
                traintl = res['results']['trainTarget'][f]
                sds.append(sd)
                do3DData = False
                do2DData = False
                if len(sd.shape) > 3:
                    do3DData = True
                elif len(sd.shape) > 2:
                    do2DData = True


                for t in [thold]:
                    t2 = 't'+str(t)
                    finalResults[k][t2]['lasa acc'].append(res['saliency'][k]['Infidelity'][str(t)]['testAcc'][f])
                    finalResults[k][t2]['lasa red'].append(res['saliency'][k]['Infidelity'][str(t)]['testReduction'][f])
                    finalResults[k][t2]['treeScores'].append(res['saliency'][k]['Infidelity'][str(t)]['treeScores'][f])

                    if t == 'baseline':
                        newTrain, trainReduction = sh.doSimpleLasaROAR(tsd, traingt, res['params']['nrEmpty'], doBaselineT=True, doFidelity=False, do3DData=do3DData, do3rdStep=do2DData)
                        newTest, testReduction = sh.doSimpleLasaROAR(sd, gt, res['params']['nrEmpty'], doBaselineT=True, doFidelity=False, do3DData=do3DData, do3rdStep=do2DData)
                    else:
                        newTrain, trainReduction = sh.doSimpleLasaROAR(tsd, traingt, t, doFidelity=False, do3DData=do3DData, do3rdStep=do2DData)
                        newTest, testReduction = sh.doSimpleLasaROAR(sd, gt, t, doFidelity=False, do3DData=do3DData, do3rdStep=do2DData)

                    logicMax, logicPred = sh.logicAcc(newTest, tl, nrSymbols, topLevel, trueIndexes=trueIndexes, andStack = andStack, orStack = orStack, xorStack = xorStack, nrAnds = nrAnds, nrOrs = nrOrs, nrxor = nrxor, orOffSet=orOffSet, xorOffSet=xorOffSet)
                    logicMaxStatistics, logicPredStatistics = sh.logicAccGuess(newTest, tl, nrSymbols, topLevel, trueIndexes=trueIndexes, andStack = andStack, orStack = orStack, xorStack = xorStack, nrAnds = nrAnds, nrOrs = nrOrs, nrxor = nrxor, orOffSet=orOffSet, xorOffSet=xorOffSet)
                    

                    finalResults[k][t2]['LogicalAcc'].append(logicMax)
                    finalResults[k][t2]['LogicalAccStatistics'].append(logicMaxStatistics)


                    print('finished logic acc for ' + str(f) + str(k) + str(t2))
                    self.printTime()


                    for normName in GCRPlus.getAllReductionNames():
                        for addMaskedValue in [True, False]:

                            sGCRResults = sh.do3DGCR(tsd, newTrain, res['results']['trainPred'][f], newTest, res['results']['testPred'][f], num_of_classes, nrSymbols, addMaskedValue=addMaskedValue, doMetrics=False, reductionName=normName, do3DData=do3DData, do2DData=do2DData)
                        

                            finalResults['gcr'][k][t2][addMaskedValue]['rMS']['gcr'].append(sGCRResults[-1])
                            finalResults['gcr'][k][t2][addMaskedValue]['rMS']['acc'].append(sGCRResults[0][0][0])
                            finalResults['gcr'][k][t2][addMaskedValue]['rMS']['predicsion'].append(sGCRResults[0][0][1])
                            finalResults['gcr'][k][t2][addMaskedValue]['rMS']['recall'].append(sGCRResults[0][0][2])
                            finalResults['gcr'][k][t2][addMaskedValue]['rMS']['f1'].append(sGCRResults[0][0][3])

                            finalResults['gcr'][k][t2][addMaskedValue]['rMA']['gcr'].append(sGCRResults[-2])
                            finalResults['gcr'][k][t2][addMaskedValue]['rMA']['acc'].append( sGCRResults[1][0][0])
                            finalResults['gcr'][k][t2][addMaskedValue]['rMA']['predicsion'].append(sGCRResults[1][0][1])
                            finalResults['gcr'][k][t2][addMaskedValue]['rMA']['recall'].append( sGCRResults[1][0][2])
                            finalResults['gcr'][k][t2][addMaskedValue]['rMA']['f1'].append(sGCRResults[1][0][3])

                            for gtmAbstact in GCRPlus.gtmReductionStrings():
                                finalResults['gcr'][k][t2][addMaskedValue][gtmAbstact]['acc'].append(sGCRResults[2][gtmAbstact]['performance'][0][0])
                                finalResults['gcr'][k][t2][addMaskedValue][gtmAbstact]['predicsion'].append(sGCRResults[2][gtmAbstact]['performance'][0][1])
                                finalResults['gcr'][k][t2][addMaskedValue][gtmAbstact]['recall'].append(sGCRResults[2][gtmAbstact]['performance'][0][2])
                                finalResults['gcr'][k][t2][addMaskedValue][gtmAbstact]['f1'].append(sGCRResults[2][gtmAbstact]['performance'][0][3])


                    print('finished t GCR for ' + str(f) + str(k) + str(t2))
                    self.printTime()

                    lableMatches = np.argmax(res['results']['testPred'][f], axis=1) == np.argmax(res['saliency'][k]['Infidelity'][str(t)]['testPred'][f], axis=1)

                    tlM = res['saliency'][k]['Infidelity'][str(t)]['testPred'][f][lableMatches]
                    newTestM = newTest[lableMatches]
                    gtM = gt[lableMatches]
                    sdM = sd[lableMatches]
                    
                    
                    fullTestValueMap, _, _, _ = sh.getPredictionMaps('', '', gtM, newTestM, tlM, num_of_classes, topLevel, nrSymbols, 1, 0, 0, len(newTest[0].flatten())-res['params']['nrEmpty'],0 , 0, 0, 0, trueIndexes)
                    print('finished getPredictionMaps for ' + str(f) + str(k) + str(t2))
                    self.printTime()

                    k2 = 'and0'
                    A = np.unique(np.array(fullTestValueMap[k2][0]), axis=0)
                    if len(A.shape) >= 3:
                        A = A.squeeze(axis=2)
                    B = np.unique(np.array(fullTestValueMap[k2][1]), axis=0)
                    if len(B.shape) >= 3:
                        B = B.squeeze(axis=2)


                    
                    if(len(A) == 0 or len(B) == 0):
                        m = []
                    else:
                        m = (A[:, None] == B).all(-1).any(1)
                    if k2 not in finalResults[k][t2]['DoubleAssigmentFull'].keys():
                        finalResults[k][t2]['DoubleAssigmentFull'][k2] = []
                        finalResults[k][t2]['DoubleAssigmentFullPercent'][k2] = []
                    if len(m) == 0:
                        finalResults[k][t2]['DoubleAssigmentFull'][k2].append(-1)     
                        finalResults[k][t2]['DoubleAssigmentFullPercent'][k2].append(-1)     
                    else:
                        finalResults[k][t2]['DoubleAssigmentFull'][k2].append(len(A[m]))
                        if len(A) < len(B):
                            finalResults[k][t2]['DoubleAssigmentFullPercent'][k2].append(len(A[m]) / len(A))
                        else:
                            finalResults[k][t2]['DoubleAssigmentFullPercent'][k2].append(len(A[m]) / len(B))

                    testValueMap, testConditionMap, classConValueMap, minConValueMap = sh.getPredictionMaps('', '', gtM, newTestM, tlM, num_of_classes, topLevel, nrSymbols, andStack, orStack, xorStack, nrAnds, nrOrs, nrxor, orOffSet, xorOffSet, trueIndexes)


                    for k2 in testValueMap.keys():

                        A = np.unique(np.array(testValueMap[k2][0]), axis=0)
                        if len(A.shape) >= 3:
                            A = A.squeeze(axis=2)
                        B = np.unique(np.array(testValueMap[k2][1]), axis=0)
                        if len(B.shape) >= 3:
                            B = B.squeeze(axis=2)


                        if(len(A) == 0 or len(B) == 0):
                            m = []
                        else:
                            m = (A[:, None] == B).all(-1).any(1)

                        if k2 not in finalResults[k][t2]['DoubleAssigmentTableClasses'].keys():
                            finalResults[k][t2]['DoubleAssigmentTableClasses'][k2] = []
                        if len(m) == 0:
                            finalResults[k][t2]['DoubleAssigmentTableClasses'][k2].append(-1)     
                        else:
                            finalResults[k][t2]['DoubleAssigmentTableClasses'][k2].append(len(A[m]))
                            
                    for k2 in testConditionMap.keys():
                        A = np.unique(np.array(testConditionMap[k2][0]), axis=0)
                        if len(A.shape) >= 3:
                            A = A.squeeze(axis=2)
                        B = np.unique(np.array(testConditionMap[k2][1]), axis=0)
                        if len(B.shape) >= 3:
                            B = B.squeeze(axis=2)


                        if(len(A) == 0 or len(B) == 0):
                            m = []
                        else:
                            m = (A[:, None] == B).all(-1).any(1)

                        if k2 not in finalResults[k][t2]['DoubleAssigmentTruthTable'].keys():
                            finalResults[k][t2]['DoubleAssigmentTruthTable'][k2] = []
                        if len(m) == 0:
                            finalResults[k][t2]['DoubleAssigmentTruthTable'][k2].append(-1)     
                        else:
                            finalResults[k][t2]['DoubleAssigmentTruthTable'][k2].append(len(A[m]))
                    
                    for k2 in classConValueMap.keys():
                        A = np.unique(np.array(classConValueMap[k2][0][0]), axis=0)
                        if len(A.shape) >= 3:
                            A = A.squeeze(axis=2)
                        B = np.unique(np.array(classConValueMap[k2][1][1]), axis=0)
                        if len(B.shape) >= 3:
                            B = B.squeeze(axis=2)


                        if(len(A) == 0 or len(B) == 0):
                            m = []
                        else:
                            m = (A[:, None] == B).all(-1).any(1)

                        if k2 not in finalResults[k][t2]['DoubleAssigmentTruthTableClasses'].keys():
                            finalResults[k][t2]['DoubleAssigmentTruthTableClasses'][k2] = []
                        if len(m) == 0:
                            finalResults[k][t2]['DoubleAssigmentTruthTableClasses'][k2].append(-1)     
                        else:
                            finalResults[k][t2]['DoubleAssigmentTruthTableClasses'][k2].append(len(A[m]))

                    for k2 in minConValueMap.keys():
                        A = np.unique(np.array(minConValueMap[k2][0]), axis=0)
                        if len(A.shape) >= 3:
                            A = A.squeeze(axis=2)
                        B = np.unique(np.array(minConValueMap[k2][1]), axis=0)
                        if len(B.shape) >= 3:
                            B = B.squeeze(axis=2)


                        if(len(A) == 0 or len(B) == 0):
                            m = []
                        else:
                            m = (A[:, None] == B).all(-1).any(1)
                        if k2 not in finalResults[k][t2]['DoubleAssigmentTruthTableMin'].keys():
                            finalResults[k][t2]['DoubleAssigmentTruthTableMin'][k2] = []
                            finalResults[k][t2]['DoubleAssigmentTruthTableMinPercent'][k2] = []
                        if len(m) == 0:
                            finalResults[k][t2]['DoubleAssigmentTruthTableMin'][k2].append(-1)    
                            finalResults[k][t2]['DoubleAssigmentTruthTableMinPercent'][k2].append(-1)
                        else:
                            finalResults[k][t2]['DoubleAssigmentTruthTableMin'][k2].append(len(A[m]))
                            if len(A) < len(B):
                                finalResults[k][t2]['DoubleAssigmentTruthTableMinPercent'][k2].append(len(A[m]) / len(A))
                            else:
                                finalResults[k][t2]['DoubleAssigmentTruthTableMinPercent'][k2].append(len(A[m]) / len(B))


                    print('finished DCA for ' + str(f) + str(k) + str(t2))
                    self.printTime()
            
                    
                    rcpp.append(np.sum([newTest == -2], axis=1)/len(newTest))

                    finalResults[k][t2]['RemovalChancePP'].append(rcpp[f])

                    ntc = newTest[np.argmax(tl, axis=1) == 1]
                    rcplp.append(np.sum([ntc == -2], axis=1)/len(ntc))

                    ntc = newTest[np.argmax(tl, axis=1) == 0]
                    rcpln.append(np.sum([ntc == -2], axis=1)/len(ntc))
                    gtf = np.where(newTest == -2, gt, -2)

                    finalResults[k][t2][1]['RemovalChancePP'].append(rcplp[f])
                    finalResults[k][t2][0]['RemovalChancePP'].append(rcpln[f])

                    nrem.append(np.sum(np.isin(gtf,falseSymbols), axis=(0))/np.sum(np.isin(gt,falseSymbols), axis=(0))) 
                    prem.append(np.sum(np.isin(gtf,trueSymbols), axis=(0))/np.sum(np.isin(gt,trueSymbols), axis=(0)))

                    finalResults[k][t2]['RemovalChancePosLable'].append(prem[f])
                    finalResults[k][t2]['RemovalChanceNegLable'].append(nrem[f])
                    for s in symbolA:
                        if s not in  finalResults[k][t2]['RemovalChancePInputValue'].keys():
                            finalResults[k][t2]['RemovalChancePInputValue'][s] = []
                        finalResults[k][t2]['RemovalChancePInputValue'][s].append(np.sum(gtf == s, axis=0)/(np.sum(gt == s, axis=0)))
                    

                    for i,a in enumerate(np.sum(np.isin(gtf,falseSymbols), axis=(1))/len(newTest[0])):
                        negVRem[f].append(a/testReduction[i])
                        if (negVRem[f][-1] > 0.5):
                            tendency[f].append(negVRem[f][-1])
                        else:
                            tendency[f].append(1 - negVRem[f][-1])
                    for i,a in enumerate(np.sum(np.isin(gtf,trueSymbols), axis=(1))/len(newTest[0])):
                        posVRem[f].append(a/testReduction[i])

                    posVRem[f] = np.array(posVRem[f])
                    finalResults[k][t2]['PositiveValueRemoval'].append(np.mean(posVRem[f], axis=(0)))
                    finalResults[k][t2]['PositiveValueRemoval 1s'].append(np.sum(np.where(posVRem[f] == 1., posVRem[f], 0))/len(posVRem[f]))

                    negVRem[f] = np.array(negVRem[f])
                    finalResults[k][t2]['NegativeValueRemoval'].append(np.mean(negVRem[f], axis=(0)))
                    finalResults[k][t2]['NegativeValueRemoval 1s'].append(np.sum(np.where(negVRem[f] == 1., negVRem[f], 0))/len(negVRem[f]))


                    tendency[f] = np.array(tendency[f])
                    finalResults[k][t2]['Tendeny mean'].append(np.mean(tendency[f], axis=(0)))
                    finalResults[k][t2]['Tendeny median'].append(np.median(tendency[f], axis=(0)))
                    finalResults[k][t2]['Tendeny 1s'].append(np.sum(np.where(tendency[f] == 1., tendency[f], 0))/len(tendency[f]))

                    for c in res['saliency'][k]['classes'].keys():
                        lables = np.argmax(tl, axis=1) == int(c)
                        finalResults[k][t2][int(c)]['NegativeValueRemoval'].append(np.mean(negVRem[f][lables], axis=(0)))
                        finalResults[k][t2][int(c)]['PositiveValueRemoval'].append(np.mean(posVRem[f][lables], axis=(0)))

                        finalResults[k][t2][int(c)]['Tendeny mean'].append(np.mean(tendency[f][lables], axis=(0)))
                        finalResults[k][t2][int(c)]['Tendeny median'].append(np.median(tendency[f][lables], axis=(0)))
                        finalResults[k][t2][int(c)]['Tendeny 1s'].append(np.sum(np.where(tendency[f][lables] == 1., tendency[f][lables], 0))/len(tendency[f][lables]))
                    
                    print('finished Rest for ' + str(f) + str(k) + str(t2))
                    self.printTime()

# We can call this command, e.g., from a Jupyter notebook with init_all=False to get an "empty" experiment wrapper,
# where we can then for instance load a pretrained model to inspect the performance.
@ex.command(unobserved=True)
def get_experiment(init_all=False):
    print('get_experiment')
    experiment = ExperimentWrapper(init_all=init_all)
    return experiment


# This function will be called by default. Note that we could in principle manually pass an experiment instance,
# e.g., obtained by loading a model from the database or by calling this from a Jupyter notebook.
@ex.automain
def train(experiment=None):
    if experiment is None:
        experiment = ExperimentWrapper()
    return experiment.trainExperiment()
