from skmultiflow.ADCN.ADCN_process.utilsADCN import dataLoader, plotPerformance
from skmultiflow.ADCN.ADCN_process.ADCNbasic import ADCN
from skmultiflow.ADCN.ADCN_process.ADCNmainloop import ADCNmain
from skmultiflow.ADCN.ADCN_process.model import simpleMPL, singleMPL
import numpy as np
import pdb
import torch
import random
from torchvision import datasets, transforms

class ADCNevaluator(object):
    """ The ADCN evaluation method.

    An alternative design to holdout ADCN evaluation.

    This evaluator is made special for ADCN only. Since ADCN evolve according to
    data, the evaluator must pass the new data for structure forming. The evaluator
    also created to measure the classifier performance on the spesific data stream.

    This method consists of using each sample to test the model, which means
    to make a predictions, and then the same sample is used to train the model. 
    This way the model is always tested on samples that it hasn't seen yet.

    Parameters
    ----------
    max_samples: int (Default: 20000)
        | The maximum number of samples to process during the evaluation.

    batch_size: int (Default: 1000)
        | The number of samples to pass at a time to the model(s).

    pretrain_size: int (Default: 500)
        | The number of samples to use to train the model before starting the evaluation. Used to enforce a 'warm' start.

    metrics: list, optional (Default: ['accuracy', 'ari','nmi','f1','precision','recall','kappa','gmean'])
        | The list of metrics to track during the evaluation. Also defines the metrics that will be displayed in plots
          and/or logged into the output file. Valid options are
        | **Classification**
        | 'accuracy'
        | 'ari'
        | 'nmi'
        | 'f1'
        | 'precision'
        | 'recall'
        | 'kappa'
        | 'gmean'
        
    stream_type : int (Default: 2)
        | The type of stream to use to train the model
        - 1 - Data stream from matlab file
        - 2 - Data stream from data generator

    Examples
    --------
    >>> # The first example demonstrates how to evaluate one model
    >>> from skmultiflow.data import SEAGenerator
    >>> from skmultiflow.ADCN import ADCNclassifier
    >>> from skmultiflow.evaluation import ADCNevaluator
    >>>
    >>> # Set the stream
    >>> stream = SEAGenerator(random_state=1)
    >>>
    >>> # Set the model
    >>> model = ADCNclassifier(stream, HidNodeMul = 10, ExtraFeatureMul =4, FeatureClusMul = 2)
    >>>
    >>> # Set the evaluator
    >>>
    >>> evaluator = evaluator = ADCNevaluator(max_samples = 20000, 
                                                pretrain_size = 300
                                                ,batch_size = 1000
                                                ,stream = stream,
                                                stream_type=2)
    >>> 
    >>>
    >>> # Run evaluation
    >>> evaluator.evaluate(model)
    """
    def __init__(self,stream,
                 max_samples= 20000,
                 stream_type = 1,
                 batch_size = 1000,
                 pretrain_size = 500,
                 show_plot = True,
                 metrics= None):
        self.max_samples = max_samples
        self.batch_size = batch_size
        self.show_plot = show_plot
        self.pretrain_size = int(pretrain_size/(stream.n_features -1))
        self.device = torch.device('cuda')
        if metrics is None:
            self.metrics = [
                'accuracy',
                'ari',
                'nmi',
                'f1',
                'precision',
                'recall',
                'kappa',
                'gmean'
                ]
        else:
            if isinstance(metrics,list):
                self.metrics = metrics
            else:
                raise ValueError("Attribute 'metrics' must be 'None' or 'list', passed {}".format(type(metrics)))
        
        if stream_type == 1:
            self.dataStream = dataLoader(stream,
                                         batchSize= self.batch_size,
                                         nEachClassSamples=self.pretrain_size)
        else:
            self.dataStream = dataLoaderGenerator(stream,
                                                  batchSize= self.batch_size,
                                                  max_samples=self.max_samples,
                                                  nEachClassSamples=self.pretrain_size)
        
    def evaluate(self,model):
        allMetrics = []
        ADCNnet = ADCN(model.nOutput,
                       nInput = model.nExtractedFeature,
                       nHiddenNode = model.nFeaturClustering)
        ADCNnet.ADCNcnn = simpleMPL(model.nInput,
                                    nNodes = model.nHidNodeExtractor,
                                    nOutput = model.nExtractedFeature)
        ADCNnet.desiredLabels = model.classes
        ADCNnet, performanceHistory, allPerformance = ADCNmain(ADCNnet,
                                                               self.dataStream,
                                                               metrics = self.metrics,
                                                               device = self.device,
                                                               batch_size = self.batch_size,
                                                               show_plot = self.show_plot)
        performanceHistory1 = performanceHistory
        allMetrics.append(allPerformance)
        print('\n')
        print(self.metrics)
        for i in self.metrics:
            print(str(i) + " : "+str(round(allMetrics[0][get_metrics(i)],2)))
def get_metrics(reference):
    switch = {
        "accuracy": 0,
        "ari": 1,
        "nmi": 2,
        "f1": 3,
        "precision": 4,
        "recall": 5,
        "kappa" : 6,
        "gmean" :7
    }
        
    return switch.get(reference)
class dataLoaderGenerator(object):
    def __init__(self, stream,
                 batchSize = 1000,
                 max_samples = 20000,
                 nEachClassSamples = 500):
        self.stream_name = stream.get_data_info()
        self.stream  = stream
        self.batchSize = batchSize
        self.max_samples = max_samples
        self.loadDataFromGenerator(nEachClassSamples,max_samples)
    def loadDataFromGenerator(self,
                              nEachClassSamples,
                              max_samples):
        data1 = self.stream.next_sample(max_samples)
        data = torch.from_numpy(data1[0])
        data = data.float()
        self.labeledData  = data[:,0:-1]
        label = torch.from_numpy(data1[1])
        self.labeledLabel = label.long()
        self.nData = data.shape[0]
        self.nBatch = int(self.nData/self.batchSize)
        self.nInput = self.labeledData.shape[1]
        self.nOutput = torch.unique(self.labeledLabel).shape[0]
        self.classes = torch.unique(self.labeledLabel).tolist()
        self.nLabeledData = self.labeledData.shape[0]

        if nEachClassSamples is not None:
            selectedLabeledData = torch.Tensor(self.nOutput*nEachClassSamples,self.nInput)
            selectedLabeledLabel = torch.LongTensor(self.nOutput*nEachClassSamples)

            idx = 0
            selectedDataIdx = []
            for iClass in self.classes:
                # print(iClass)
                dataCount = 0

                for iData in range(0,self.nLabeledData):
                    # print(iData)
                    if self.labeledLabel[iData] == iClass:
                        selectedLabeledData[idx] = self.labeledData[iData]
                        selectedLabeledLabel[idx] = self.labeledLabel[iData]
                        # print(labeledLabel[iData])
                        idx += 1
                        dataCount += 1

                        selectedDataIdx.append(iData)
                        iData += 1

                    if dataCount == nEachClassSamples:
                        break
            remainderData = deleteRowTensor(self.labeledData,selectedDataIdx,2)
            remainderLabel = deleteRowTensor(self.labeledLabel,selectedDataIdx,2)
            
            self.nLabeledData = selectedLabeledData.shape[0]

            # shuffle, so that the label is not in order
            indices = torch.randperm(self.nLabeledData)
            self.labeledData = selectedLabeledData[indices]
            self.labeledLabel = selectedLabeledLabel[indices]
            self.nLabeledData = self.labeledData.shape[0]
            
            # unlabeled data
            self.unlabeledData = remainderData
            self.unlabeledLabel = remainderLabel
            
            self.nUnlabeledData = self.unlabeledData.shape[0]
            self.nBatch = int(self.nUnlabeledData/self.batchSize)
            self.taskIndicator = (torch.zeros(self.nBatch).long()).tolist()
def deleteRowTensor(x, index,
                    mode = 1):
    if mode == 1:
        x = x[torch.arange(x.size(0)) != index] 
    elif mode == 2:
        # delete more than 1 row
        # index is a list of deleted row
        allRow = torch.arange(x.size(0)).tolist()
        
        for ele in sorted(index, reverse = True):  
            del allRow[ele]

        remainderRow = torch.tensor(allRow).long()

        x = x[remainderRow]
    
    return x