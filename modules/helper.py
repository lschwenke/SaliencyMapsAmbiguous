import numpy as np
import dill as pickle
import math
import os
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from pyts.approximation import SymbolicAggregateApproximation

def doCombiStep(step, field, axis) -> np.ndarray:
    if(step == 'max'):
        return np.max(field, axis=axis)
    elif (step == 'sum'):
        return np.sum(field, axis=axis)

#flatten an 3D np array
def flatten(X, pos = -1):
    flattened_X = np.empty((X.shape[0], X.shape[2]))  # sample x features array.
    if pos == -1:
        pos = X.shape[1]-1
    for i in range(X.shape[0]):
        flattened_X[i] = X[i, pos, :]
    return(flattened_X)

# Scale 3D array. X = 3D array, scalar = scale object from sklearn. Output = scaled 3D array.
def scale(X, scaler):
    for i in range(X.shape[0]):
        X[i, :, :] = scaler.transform(X[i, :, :])
    return X

# Symbolize a 3D array. X = 3D array, scalar = SAX symbolizer object. Output = symbolic 3D string array.
def symbolize(X, scaler):
    X_s = scaler.transform(X)
    return X_s

# translate the a string [a,e] between 
def trans(val, vocab) -> float:
    for i in range(len(vocab)):
        if val == vocab[i]:
            halfSize = (len(vocab)-1)/2
            return (i - halfSize) / halfSize
    return -2

def getMapValues(size):
    vMap = []
    for i in range(size):
        halfSize = (size-1)/2
        vMap.append(round((i - halfSize) / halfSize, 4))
    return vMap

def symbolizeTrans(X, scaler, bins = 5):
    vocab = scaler._check_params(bins)
    X_s = scaler.transform(X)
    for i in range(X.shape[0]):
        X = X.astype(float)
                
        for j in range(X.shape[1]):
            X[i][j] = trans(X_s[i][j], vocab)
    return X

def symbolizeTrans2(X, scaler, bins = 5):
    vocab = scaler._check_params(bins)
    for i in range(X.shape[0]):
        #X = X.astype('U13')
        X_s = X.astype(str) 
        z1 = scaler.transform(np.array([X[i, :, :][:,0]]))
        X_s[i, :, :][:,0] = z1
        for j in range(X.shape[1]):
            X[i][j][0] = trans(X_s[i][j][0], vocab)
    return X

def split_dataframe(df, chunk_size = 10000): 
    chunks = list()
    num_chunks = len(df) // chunk_size + 1
    for i in range(num_chunks):
        chunks.append(df[i*chunk_size:(i+1)*chunk_size])
    return chunks

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        np.save(f, obj)

def save_obj2(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(f, obj)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return np.load(f, allow_pickle=True)
		
def load_obj2(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)
    
def truncate(n):
    return int(n * 1000) / 1000

def ceFull(data):
    complexDists = []
    for d in data:
        complexDists.append(ce(d))
        
    return complexDists, np.mean(complexDists) 

def ce(data):
    summer = 0
    for i in range(len(data)-1):
        summer += math.pow(data[i] - data[i+1], 2)
    return math.sqrt(summer)


def modelFidelity(modelPrediction, interpretationPrediction):
    summer = 0
    for i in range(len(modelPrediction)):
        if modelPrediction[i] == interpretationPrediction[i]:
            summer += 1
    return summer / len(modelPrediction)

def collectLAAMs(earlyPredictor, x_test, order, step1, step2):
    limit = 500
    attentionQ0 = []
    attentionQ1 = []
    attentionQ2 = []

    for bor in range(int(math.ceil(len(x_test)/limit))):
        attOut = earlyPredictor.predict([x_test[bor*limit:(bor+1)*limit]])
        attentionQ0.extend(attOut[0]) 
        attentionQ1.extend(attOut[1])

        if len(attentionQ2) == 0:
            attentionQ2 = attOut[2]
        else:
            for k in range(len(attentionQ2)):
                
                attentionQ2[k] = np.append(attentionQ2[k], attOut[2][k], 0)
    
    attentionFQ = [np.array(attentionQ0), np.array(attentionQ1), np.array(attentionQ2)]
    
    if(order == 'lh'):
        axis1 = 0
        axis2 = 1
    elif(order == 'hl'):
        axis1 = 2
        axis2 = 0
    else:
        axis1 = 0
        axis2 = 1   

    attentionFQ[1] = doCombiStep(step1, attentionFQ[2], axis1)
    attentionFQ[1] = doCombiStep(step2, attentionFQ[1], axis2) 

    return attentionFQ[1]


def laamConsistency(laams, fromIndex, consistancyLabels):
    results = dict()
    innerFolddistance = dict()
    outerdistance = dict()
    innerClassDistance = dict()
    for combi in laams.keys():
        innerFolddistance[combi] = []
        outerdistance[combi] = []
        innerClassDistance[combi] = []
        for j in range(len(laams[combi][fromIndex])):
            outerdistance[combi].append([])
            for fold in range(len(laams[combi])):
                if fold != fromIndex:
                    outerdistance[combi][j].append(matrixEucDistance(laams[combi][fromIndex][j],laams[combi][fold][j]))

        for fold in range(len(laams[combi])): 
            innerFolddistance[combi].append([])
            for j in range(len(laams[combi][fold])):
                if j != fromIndex and consistancyLabels[j] != consistancyLabels[fromIndex]:
                    innerFolddistance[combi][fold].append(matrixEucDistance(laams[combi][fold][fromIndex],laams[combi][fold][j]))
        
        for fold in range(len(laams[combi])): 
            innerClassDistance[combi].append([])
            for j in range(len(laams[combi][fold])):
                if j != fromIndex and consistancyLabels[j] == consistancyLabels[fromIndex]:
                    innerClassDistance[combi][fold].append(matrixEucDistance(laams[combi][fold][fromIndex],laams[combi][fold][j]))
    
    #from one sample to another of a different class
    results["innerFold"] = innerFolddistance
    #from one sample to another of the same class
    results["innerClass"] = innerClassDistance
    #same sample between folds
    results["outer"] = outerdistance
    return results

def matrixEucDistance(matrix1, matrix2):
    summer = 0
    for i in range(len(matrix1)):
        for j in range(len(matrix1[i])):
            summer += math.pow(matrix1[i][j] - matrix2[i][j], 2)
    return math.sqrt(summer)

def confidenceGCR(bestScores, correctness):
    top80Len = int(len(bestScores) * 0.80)
    top50Len = int(len(bestScores) * 0.50)
    top20Len = int(len(bestScores) * 0.20)
    top10Len = int(len(bestScores) * 0.10)
    
    top80Ind = sorted(range(len(bestScores)), key=lambda x: bestScores[x])[-top80Len:]
    top50Ind = sorted(range(len(bestScores)), key=lambda x: bestScores[x])[-top50Len:]
    top20Ind = sorted(range(len(bestScores)), key=lambda x: bestScores[x])[-top20Len:]
    top10Ind = sorted(range(len(bestScores)), key=lambda x: bestScores[x])[-top10Len:]

    nCorrectness = np.array(correctness)
    top80Acc = (sum(nCorrectness[top80Ind])/len(nCorrectness[top80Ind]))
    top50Acc = (sum(nCorrectness[top50Ind])/len(nCorrectness[top50Ind]))
    top20Acc = (sum(nCorrectness[top20Ind])/len(nCorrectness[top20Ind]))
    top10Acc = (sum(nCorrectness[top10Ind])/len(nCorrectness[top10Ind]))
    
    return top80Acc, top50Acc, top20Acc, top10Acc

def confidenceGCR2(bestScores, correctness, steps):
    results = []
    nCorrectness = np.array(correctness)
    step = 1
    while step > 0:
        topLen = int(len(bestScores) * step)
        topInd = sorted(range(len(bestScores)), key=lambda x: bestScores[x])[-topLen:]
        
        results.append(sum(nCorrectness[topInd])/len(nCorrectness[topInd]))
        step = step - (1/steps)
    return results

def fidelityConfidenceGCR(bestScores, correctness, saxResults):
    top80Len = int(len(bestScores) * 0.80)
    top50Len = int(len(bestScores) * 0.50)
    top20Len = int(len(bestScores) * 0.20)
    top10Len = int(len(bestScores) * 0.10)
    
    top80Ind = sorted(range(len(bestScores)), key=lambda x: bestScores[x])[-top80Len:]
    top50Ind = sorted(range(len(bestScores)), key=lambda x: bestScores[x])[-top50Len:]
    top20Ind = sorted(range(len(bestScores)), key=lambda x: bestScores[x])[-top20Len:]
    top10Ind = sorted(range(len(bestScores)), key=lambda x: bestScores[x])[-top10Len:]

    nCorrectness = np.array(correctness)
    nSaxResults = np.array(saxResults)
    top80Fidelity = modelFidelity(nCorrectness[top80Ind], nSaxResults[top80Ind])
    top50Fidelity = modelFidelity(nCorrectness[top50Ind], nSaxResults[top50Ind])
    top20Fidelity = modelFidelity(nCorrectness[top20Ind], nSaxResults[top20Ind])
    top10Fidelity = modelFidelity(nCorrectness[top10Ind], nSaxResults[top10Ind])


    return top80Fidelity, top50Fidelity, top20Fidelity, top10Fidelity



def preprocessData(x_train1, x_val, X_test, y_train1, y_val, y_test, y_trainy, y_testy, binNr, symbolsCount, dataName, useEmbed = False, useSaves = False, doSymbolify = True, multiVariant=False):    
    
    x_test = X_test.copy()
    
    if(useEmbed):
        processedDataName = "./saves/"+str(dataName)+ '-bin' + str(binNr) + '-symbols' + str(symbolsCount) + '+embedding'
    else:
        processedDataName = "./saves/"+str(dataName)+ '-bin' + str(binNr) + '-symbols' + str(symbolsCount)
    fileExists = os.path.isfile(processedDataName +'.pkl')

    if(fileExists and useSaves):
        print('found file! Start loading file!')
        res = load_obj(processedDataName)


        for index, v in np.ndenumerate(res):
            print(index)
            res = v
        res.keys()

        x_train1 = res['X_train']
        #x_train1 = res['X_val']
        x_test = res['X_test']
        x_val = res['X_val']
        X_train_ori = res['X_train_ori']
        X_test_ori = res['X_test_ori']
        y_trainy = res['y_trainy']
        y_train1 = res['y_train']
        y_test = res['y_test']
        y_testy = res['y_testy']
        y_val = res['y_val']
        X_val_ori = res['X_val_ori']
        print(x_test.shape)
        print(x_train1.shape)
        print(y_test.shape)
        print(y_train1.shape)
        print('SHAPES loaded')
        
    else:
        print('SHAPES:')
        print(x_test.shape)
        print(x_train1.shape)
        print(x_val.shape)
        print(y_test.shape)
        print(y_train1.shape)

        x_train1 = x_train1.squeeze()
        x_val = x_val.squeeze()
        x_test = x_test.squeeze()
        
        trainShape = x_train1.shape
        valShape = x_val.shape
        testShape = x_test.shape
        
        if multiVariant:
            X_test_ori = x_test.copy()
            X_val_ori = x_val.copy()
            X_train_ori = x_train1.copy()
            for i in range(trainShape[-1]):
                x_train2 = x_train1[:,:,i]
                x_val2 = x_val[:,:,i]
                x_test2 = x_test[:,:,i]
                print('####')
                print(x_train2.shape)

                trainShape2 = x_train2.shape
                valShape2 = x_val2.shape
                testShape2 = x_test2.shape
        
                scaler = StandardScaler()    
                scaler = scaler.fit(x_train1.reshape((-1,1)))
                x_train2 = scaler.transform(x_train2.reshape(-1, 1)).reshape(trainShape2)##
                x_val2 = scaler.transform(x_val2.reshape(-1, 1)).reshape(valShape2)
                x_test2 = scaler.transform(x_test2.reshape(-1, 1)).reshape(testShape2)

                sax = SymbolicAggregateApproximation(n_bins=symbolsCount, strategy='uniform')
                sax.fit(x_train2)

                if(useEmbed):
                    x_train2 = symbolize(x_train2, sax)
                    x_val2 = symbolize(x_val2, sax)
                    x_test2 = symbolize(x_test2, sax)
                else:
                    x_train2 = symbolizeTrans(x_train2, sax, bins = symbolsCount)
                    x_val2 = symbolizeTrans(x_val2, sax, bins = symbolsCount)
                    x_test2 = symbolizeTrans(x_test2, sax, bins = symbolsCount)
                print(x_train2.shape)
                #x_train1 = np.expand_dims(x_train1, axis=2)

                x_train1[:,:,i] = x_train2      
                x_val[:,:,i] = x_val2
                x_test[:,:,i] = x_test2
                
            #x_train1 = x_train1.reshape(trainShape[0],-1,1)
            #x_val = x_val.reshape(valShape[0],-1,1)
            #x_test = x_test.reshape(testShape[0],-1,1)
            print(x_train1.shape)
            

        else:    
            if(doSymbolify):
                scaler = StandardScaler()    
                scaler = scaler.fit(x_train1.reshape((-1,1)))
                x_train1 = scaler.transform(x_train1.reshape(-1, 1)).reshape(trainShape)
                x_val = scaler.transform(x_val.reshape(-1, 1)).reshape(valShape)
                x_test = scaler.transform(x_test.reshape(-1, 1)).reshape(testShape)

                X_test_ori = x_test.copy()
                X_val_ori = x_val.copy()
                X_train_ori = x_train1.copy()


                sax = SymbolicAggregateApproximation(n_bins=symbolsCount, strategy='uniform')
                sax.fit(x_train1)

                if(useEmbed):
                    x_train1 = symbolize(x_train1, sax)
                    x_val = symbolize(x_val, sax)
                    x_test = symbolize(x_test, sax)
                else:
                    x_train1 = symbolizeTrans(x_train1, sax, bins = symbolsCount)
                    x_val = symbolizeTrans(x_val, sax, bins = symbolsCount)
                    x_test = symbolizeTrans(x_test, sax, bins = symbolsCount)
            else:
                X_test_ori = x_test.copy()
                X_val_ori = x_val.copy()
                X_train_ori = x_train1.copy()


            x_train1 = np.expand_dims(x_train1, axis=2)
            x_val = np.expand_dims(x_val, axis=2)
            x_test = np.expand_dims(x_test, axis=2)   
            X_test_ori = np.expand_dims(X_test_ori, axis=2)   
            X_train_ori = np.expand_dims(X_train_ori, axis=2) 
            X_val_ori = np.expand_dims(X_val_ori, axis=2) 
            
            

        print('saves shapes:')
        print(x_test.shape)
        print(x_train1.shape)

        #save sax results to only calculate them once
        resultsSave = {
            'X_train':x_train1,
            'X_train_ori':X_train_ori,
            'X_test':x_test,
            'X_test_ori':X_test_ori,
            'X_val': x_val,
            'X_val_ori':X_val_ori,
            'y_trainy':y_trainy,
            'y_train':y_train1,
            'y_val': y_val,
            'y_test':y_test,
            'y_testy':y_testy
        }
        save_obj(resultsSave, processedDataName)
    return x_train1, x_val, x_test, y_train1, y_val, y_test, X_train_ori, X_val_ori, X_test_ori, y_trainy, y_testy