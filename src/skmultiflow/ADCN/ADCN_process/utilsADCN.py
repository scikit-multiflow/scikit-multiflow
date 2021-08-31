import numpy as np
import pandas as pd
import time 
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import linalg as LA
import scipy
from scipy import io
import sklearn
from sklearn import preprocessing
import pdb
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from skmultiflow.ADCN.ADCN_process.rotateImage import Rotate

class meanStdCalculator(object):
	# developed and modified from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    # license BSD 3-Clause "New" or "Revised" License
    def __init__(self):
        self.mean = 0.0
        self.mean_old = 0.0
        self.std = 0.001
        self.count = 0.0
        self.minMean = 100.0
        self.minStd = 100.0
        self.M_old = 0.0
        self.M = 0.0
        self.S = 0.0
        self.S_old = 0.0
        
    def updateMeanStd(self, data, cnt = 1):
        self.data = data
        self.mean_old = self.mean  # copy.deepcopy(self.mean)
        self.M_old = self.count*self.mean_old
        self.M = self.M_old + data
        self.S_old = self.S     # copy.deepcopy(self.S)
        if self.count > 0:
            self.S = self.S_old + ((self.count*data - self.M_old)**2)/(self.count*(self.count + cnt) + 0.0001)
        
        self.count += cnt
        self.mean = self.mean_old + (data-self.mean_old)/((self.count + 0.0001))  # np.divide((data-self.mean_old),self.count + 0.0001)
        self.std = np.sqrt(self.S/(self.count + 0.0001))
        
        # if (self.std != self.std).any():
        #     print('There is NaN in meanStd')
        #     pdb.set_trace()
    
    def resetMinMeanStd(self):
        self.minMean = self.mean  # copy.deepcopy(self.mean)
        self.minStd = self.std   # copy.deepcopy(self.std)
        
    def updateMeanStdMin(self):
        if self.mean < self.minMean:
            self.minMean = self.mean  # copy.deepcopy(self.mean)
        if self.std < self.minStd:
            self.minStd = self.std   # copy.deepcopy(self.std)

    def reset(self):
        self.__init__()

class mnistLoader(object):
    def __init__(self, 
                 labeldSamples,
                 unlabeldSamples, 
                 testingBatchSize = 1000, 
                 nEachClassSamples = None):
        # np.random.seed(0)
        # torch.manual_seed(0)
        # random.seed(0)
        self.batchSize = testingBatchSize
        self.loadData(labeldSamples, unlabeldSamples, nEachClassSamples)
        
    def loadData(self, 
                 labeldSamples, 
                 unlabeldSamples, 
                 nEachClassSamples):
        # labeled data
        labeledData = torch.Tensor(10000,1,28,28)
        labeledLabel = torch.LongTensor(10000)

        for idx, example in enumerate(labeldSamples):
            labeledData[idx] = example[0]
            labeledLabel[idx] = example[1]
            
        self.labeledData = labeledData
        self.labeledLabel = labeledLabel

        self.classes = torch.unique(self.labeledLabel).tolist()
        self.nOutput = torch.unique(self.labeledLabel).shape[0]
        self.nLabeledData = self.labeledData.shape[0]
        
        if nEachClassSamples is not None:
            selectedLabeledData  = torch.Tensor(self.nOutput*nEachClassSamples,1,28,28)
            selectedLabeledLabel = torch.LongTensor(self.nOutput*nEachClassSamples)

            idx = 0
            selectedDataIdx = []
            for iClass in self.classes:
                # print(iClass)
                dataCount = 0

                for iData in range(0,self.nLabeledData):
                    # print(iData)
                    if labeledLabel[iData] == iClass:
                        selectedLabeledData[idx]  = self.labeledData[iData]
                        selectedLabeledLabel[idx] = self.labeledLabel[iData]
                        # print(labeledLabel[iData])
                        idx += 1
                        dataCount += 1

                        selectedDataIdx.append(iData)
                        iData += 1

                    if dataCount == nEachClassSamples:
                        break

            remainderData = deleteRowTensor(self.labeledData, selectedDataIdx, 2)
            remainderLabel = deleteRowTensor(self.labeledLabel, selectedDataIdx, 2)
            
            self.nLabeledData = selectedLabeledData.shape[0]

            # shuffle
            indices = torch.randperm(self.nLabeledData)
            self.labeledData = selectedLabeledData[indices]
            self.labeledLabel = selectedLabeledLabel[indices]
        
        # unlabeled data
        unlabeledData = torch.Tensor(60000,1,28,28)
        unlabeledLabel = torch.LongTensor(60000)

        for idx, example in enumerate(unlabeldSamples):
            unlabeledData[idx]  = example[0]
            unlabeledLabel[idx] = example[1]
        
        if nEachClassSamples is not None:
            unlabeledData = torch.cat((unlabeledData,remainderData),0)
            unlabeledLabel  = torch.cat((unlabeledLabel,remainderLabel),0)

        self.unlabeledData = unlabeledData
        self.unlabeledLabel = unlabeledLabel
        
        self.nUnlabeledData = self.unlabeledData.shape[0]
        self.nBatch = int(self.nUnlabeledData/self.batchSize)
        self.taskIndicator  = (torch.zeros(self.nBatch).long()).tolist()
        
        print('Number of output: ', self.nOutput)
        print('Number of labeled data: ', self.nLabeledData)
        print('Number of unlabeled data: ', self.nUnlabeledData)
        print('Number of unlabeled data batch: ', self.nBatch)

    def createTask(self, nTask = 2, taskList = [], taskType = 1):
        # random seed control
        # random.seed(0)
        if (taskType == 2 or taskType == 3) and (len(taskList) != nTask):
            raise NameError('list of rotaion angle should be the same with the number of task')

        self.taskIndicator = []

        # clone labeled data
        transformedLabeledData = self.labeledData.clone()
        transformedLabeledLabel = self.labeledLabel.clone()
        finalLabeledData = {}
        finalLabeledLabel = {}

        # clone unlabeled data
        transformedUnlabeledData = self.unlabeledData.clone()
        transformedUnlabeledLabel = self.unlabeledLabel.clone()
        finalUnlabeledData = {}
        finalUnlabeledLabel = {}

        # testing data
        unlabeledDataTest = {}
        unlabeledLabelTest = {}
        
        # number of data and batch for each task
        self.nTask = nTask
        self.nLabeledDataPerTask = int(self.nLabeledData/nTask)
        self.nBatchPerTask = int(self.nBatch/nTask)
        self.nBatch = self.nBatchPerTask*nTask
        self.nUnlabeledDataPerTask = self.nBatchPerTask*self.batchSize

        nUnlabeledDataTest = 0

        for iTask in range(0,nTask):
            # load data
            # iTask = iTask + 1
            self.taskIndicator = self.taskIndicator + (iTask*torch.ones(self.nBatchPerTask-1).long()).tolist()

            # load labeled data
            taskLabeledData = transformedLabeledData[(iTask)*self.nLabeledDataPerTask:(iTask+1)*self.nLabeledDataPerTask]
            taskLabeledLabel = transformedLabeledLabel[(iTask)*self.nLabeledDataPerTask:(iTask+1)*self.nLabeledDataPerTask]

            # load unlabeled data
            taskUnlabeledData = transformedUnlabeledData[(iTask)*self.nUnlabeledDataPerTask:(iTask+1)*self.nUnlabeledDataPerTask]
            taskUnlabeledLabel = transformedUnlabeledLabel[(iTask)*self.nUnlabeledDataPerTask:(iTask+1)*self.nUnlabeledDataPerTask]

            if self.nUnlabeledDataPerTask != taskUnlabeledData.shape[0] or self.nLabeledDataPerTask != taskLabeledData.shape[0]:
                raise NameError('Mismatch data size')

            # permute
            if taskType == 1:
                # reshape image
                taskLabeledData = taskLabeledData.view(taskLabeledData.size(0),-1)
                taskUnlabeledData = taskUnlabeledData.view(taskUnlabeledData.size(0),-1)

                # create permutation
                col_idxs = list(range(taskLabeledData.shape[1]))
                random.shuffle(col_idxs)

                # permute labeled data
                taskLabeledData = taskLabeledData[:, torch.tensor(col_idxs)]

                # permute unlabeled data
                taskUnlabeledData = taskUnlabeledData[:, torch.tensor(col_idxs)]

                # reshape back image
                taskLabeledData = taskLabeledData.reshape(taskLabeledData.size(0),1,28,28)
                taskUnlabeledData = taskUnlabeledData.reshape(taskUnlabeledData.size(0),1,28,28)

            elif taskType == 2:
                # rotate labeled data
                for idx, _ in enumerate(taskLabeledData):
                    taskLabeledData[idx] = Rotate(random.randint(taskList[iTask][0],
                                                  taskList[iTask][1]))(taskLabeledData[idx])

                # rotate unlabeled data
                for idx, _ in enumerate(taskUnlabeledData):
                    taskUnlabeledData[idx] = Rotate(random.randint(taskList[iTask][0],
                                                    taskList[iTask][1]))(taskUnlabeledData[idx])                    
            
            elif taskType == 3:
                # split MNIST, into 5 tasks
                self.nOutputPerTask = 2

                taskLabeledDataSplit = torch.Tensor().float()
                taskLabeledLabelSplit = torch.Tensor().long()

                taskUnlabeledDataSplit = torch.Tensor().float()
                taskUnlabeledLabelSplit = torch.Tensor().long()

                for iClass in taskList[iTask]:
                    # split labeled data
                    taskLabeledDataSplit = torch.cat((taskLabeledDataSplit,
                                                      transformedLabeledData[transformedLabeledLabel==iClass]),
                                                    0)
                    taskLabeledLabelSplit = torch.cat((taskLabeledLabelSplit,
                                                       transformedLabeledLabel[transformedLabeledLabel==iClass]),
                                                    0)

                    # split unlabeled data
                    taskUnlabeledDataSplit = torch.cat((taskUnlabeledDataSplit,
                                                        transformedUnlabeledData[transformedUnlabeledLabel==iClass]),
                                                      0)
                    taskUnlabeledLabelSplit = torch.cat((taskUnlabeledLabelSplit,
                                                         transformedUnlabeledLabel[transformedUnlabeledLabel==iClass]),
                                                        0)

                # shuffle labeled data
                taskLabeledData = taskLabeledDataSplit
                taskLabeledLabel = taskLabeledLabelSplit
                
                row_idxs = list(range(taskLabeledData.shape[0]))
                random.shuffle(row_idxs)
                taskLabeledData = taskLabeledData[torch.tensor(row_idxs), :]
                taskLabeledLabel = taskLabeledLabel[torch.tensor(row_idxs)]

                # shuffle unlabeled data
                taskUnlabeledData = taskUnlabeledDataSplit
                taskUnlabeledLabel = taskUnlabeledLabelSplit

                row_idxs = list(range(taskUnlabeledData.shape[0]))
                random.shuffle(row_idxs)
                taskUnlabeledData = taskUnlabeledData[torch.tensor(row_idxs), :]
                taskUnlabeledLabel = taskUnlabeledLabel[torch.tensor(row_idxs)]

            # store labeled data and labels
            finalLabeledData[iTask] = taskLabeledData
            finalLabeledLabel[iTask] = taskLabeledLabel

            # store unlabeled data and labels
            finalUnlabeledData[iTask] = taskUnlabeledData[self.batchSize:]
            finalUnlabeledLabel[iTask] = taskUnlabeledLabel[self.batchSize:]

            # store unlabeled data for testing
            unlabeledDataTest[iTask] = taskUnlabeledData[:self.batchSize]
            unlabeledLabelTest[iTask] = taskUnlabeledLabel[:self.batchSize]  
            nUnlabeledDataTest += unlabeledDataTest[iTask].shape[0]        

        # labeled data
        self.labeledData = finalLabeledData
        self.labeledLabel = finalLabeledLabel

        # unlabeled data
        self.unlabeledData = finalUnlabeledData
        self.unlabeledLabel = finalUnlabeledLabel

        # testing data
        self.unlabeledDataTest = unlabeledDataTest
        self.unlabeledLabelTest = unlabeledLabelTest

        # update size
        self.nBatchPerTask = self.nBatchPerTask - 1
        self.nBatch  = self.nBatchPerTask*nTask
        self.nUnlabeledDataPerTask  = self.nBatchPerTask*self.batchSize
        self.nUnlabeledDataTest  = nUnlabeledDataTest

        print('Number of task: ', nTask)
        print('Number of labeled data per task: ', self.nLabeledDataPerTask)
        print('Number of unlabeled data per task: ', self.nUnlabeledDataPerTask)
        print('Number of unlabeled data batch per task: ', self.nBatchPerTask)
        print('Number of unlabeled data test: ', self.nUnlabeledDataTest)

    def createDrift(self, 
                    nDrift = 2, 
                    taskList = [], 
                    taskType = 1):
        # random seed control
        # random.seed(0)
        if (taskType == 2) and (len(taskList) != nDrift):
            raise NameError('list of rotaion angle should be the same with the number of task')

        # clone unlabeled data
        transformedUnlabeledData = self.unlabeledData.clone()
        transformedUnlabeledLabel = self.unlabeledLabel.clone()
        finalUnlabeledData = torch.Tensor().float()
        finalUnlabeledLabel = torch.Tensor().long()
        
        # number of data and batch for each task
        self.nDrift = nDrift
        self.nBatchPerTask = int(self.nBatch/nDrift)
        self.nBatch = self.nBatchPerTask*nDrift
        self.nUnlabeledDataPerTask  = self.nBatchPerTask*self.batchSize

        for iTask in range(0,nDrift):
            # load data
            # iTask = iTask + 1
            self.taskIndicator = self.taskIndicator + (iTask*torch.ones(self.nBatchPerTask).long()).tolist()

            # load unlabeled data
            taskUnlabeledData = transformedUnlabeledData[(iTask)*self.nUnlabeledDataPerTask:(iTask+1)*self.nUnlabeledDataPerTask]
            taskUnlabeledLabel = transformedUnlabeledLabel[(iTask)*self.nUnlabeledDataPerTask:(iTask+1)*self.nUnlabeledDataPerTask]

            if self.nUnlabeledDataPerTask != taskUnlabeledData.shape[0]:
                raise NameError('Mismatch data size')

            # permute
            if taskType == 1:
                # reshape image
                taskUnlabeledData = taskUnlabeledData.view(taskUnlabeledData.size(0),-1)

                # create permutation
                col_idxs = list(range(taskLabeledData.shape[1]))
                random.shuffle(col_idxs)

                # permute unlabeled data
                taskUnlabeledData = taskUnlabeledData[:, torch.tensor(col_idxs)]

                # reshape back image
                taskUnlabeledData = taskUnlabeledData.reshape(taskUnlabeledData.size(0),1,28,28)

            elif taskType == 2:
                # rotate unlabeled data
                for idx, _ in enumerate(taskUnlabeledData):
                    taskUnlabeledData[idx] = Rotate(random.randint(taskList[iTask][0],
                                                                   taskList[iTask][1]))(taskUnlabeledData[idx])

            # store unlabeled data and labels
            finalUnlabeledData  = torch.cat((finalUnlabeledData,taskUnlabeledData),0)
            finalUnlabeledLabel = torch.cat((finalUnlabeledLabel,taskUnlabeledLabel),0)

        # unlabeled data
        self.unlabeledData  = finalUnlabeledData
        self.unlabeledLabel = finalUnlabeledLabel

class omniglotLoader(object):
    def __init__(self, 
                dataTrain, 
                labelTrain, 
                propLabeled = 0.2, 
                propUnlabeled = 0.5, 
                propTest = 0.3, 
                nDataPerClass = 20, 
                nTask = 10, 
                nClassperTask = 100):
        # np.random.seed(0)
        # torch.manual_seed(0)
        # random.seed(0)
        self.loadData(dataTrain, 
                      labelTrain,
                      propLabeled, 
                      propUnlabeled, 
                      propTest, 
                      nDataPerClass, 
                      nTask, 
                      nClassperTask)
        self.nOutputPerTask = nClassperTask
        self.nTask = nTask
        
    def loadData(self, 
                dataTrain, 
                labelTrain, 
                propLabeled, 
                propUnlabeled, 
                propTest, 
                nDataPerClass, 
                nTask, 
                nClassperTask):
        finalLabeledData = {}
        finalLabeledLabel = {}

        finalUnlabeledData = {}
        finalUnlabeledLabel = {}

        unlabeledDataTest = {}
        unlabeledLabelTest = {}

        labeledData = torch.Tensor().float()
        labeledLabel = torch.Tensor().long()

        unlabeledData = torch.Tensor().float()
        unlabeledLabel = torch.Tensor().long()

        testData = torch.Tensor().float()
        testLabel = torch.Tensor().long()
        
        nUnlabeled = int(propUnlabeled*nDataPerClass)
        nLabeled = int(propLabeled*nDataPerClass)
        nTest = int(propTest*nDataPerClass)

        for i in range(0,torch.unique(labelTrain).shape[0]):
            dataClass = dataTrain [labelTrain==i]
            labelClass = labelTrain[labelTrain==i]

            unlabeledData = torch.cat((unlabeledData,dataClass[0:nUnlabeled]),0)
            unlabeledLabel = torch.cat((unlabeledLabel,labelClass[0:nUnlabeled]),0)

            labeledData = torch.cat((labeledData,dataClass[nUnlabeled:(nUnlabeled+nLabeled)]),0)
            labeledLabel = torch.cat((labeledLabel,labelClass[nUnlabeled:(nUnlabeled+nLabeled)]),0)

            testData = torch.cat((testData,dataClass[(nUnlabeled+nLabeled):nDataPerClass]),0)
            testLabel = torch.cat((testLabel,labelClass[(nUnlabeled+nLabeled):nDataPerClass]),0)
        
        self.nOutput = labeledLabel.max().item() + 1
        
        for iTask in range(0,nTask):
            if iTask < nTask-1:
                minClass = iTask*nClassperTask
                maxClass = nClassperTask + (iTask)*nClassperTask
            else:
                minClass = iTask*nClassperTask
                maxClass = labeledLabel.max().item() + 1

            labeledDataTask = torch.Tensor().float()
            labeledLabelTask = torch.Tensor().long()

            unlabeledDataTask = torch.Tensor().float()
            unlabeledLabelTask  = torch.Tensor().long()

            testDataTask = torch.Tensor().float()
            testLabelTask  = torch.Tensor().long()

            for iClass in range(minClass,maxClass):
                labeledDataiTask = labeledData [labeledLabel==iClass]
                labeledLabeliTask = labeledLabel[labeledLabel==iClass]

                unlabeledDataiTask  = unlabeledData [unlabeledLabel==iClass]
                unlabeledLabeliTask = unlabeledLabel[unlabeledLabel==iClass]

                testDataiTask = testData [testLabel==iClass]
                testLabeilTask = testLabel[testLabel==iClass]

                # augment
                labeledDataTask = torch.cat((labeledDataTask,labeledDataiTask.unsqueeze(1)),0)
                labeledLabelTask = torch.cat((labeledLabelTask,labeledLabeliTask),0)

                unlabeledDataTask = torch.cat((unlabeledDataTask,unlabeledDataiTask.unsqueeze(1)),0)
                unlabeledLabelTask  = torch.cat((unlabeledLabelTask,unlabeledLabeliTask),0)

                testDataTask = torch.cat((testDataTask,testDataiTask.unsqueeze(1)),0)
                testLabelTask  = torch.cat((testLabelTask,testLabeilTask),0)

            finalLabeledData[iTask] = labeledDataTask
            finalLabeledLabel[iTask] = labeledLabelTask

            finalUnlabeledData[iTask] = unlabeledDataTask
            finalUnlabeledLabel[iTask] = unlabeledLabelTask

            unlabeledDataTest[iTask] = testDataTask
            unlabeledLabelTest[iTask] = testLabelTask

            if (testLabelTask.max().item() != unlabeledLabelTask.max().item() or 
                testLabelTask.max().item() != labeledLabelTask.max().item() or
                labeledLabelTask.max().item() != unlabeledLabelTask.max().item() or 
                testLabelTask.min().item() != unlabeledLabelTask.min().item() or 
                testLabelTask.min().item() != labeledLabelTask.min().item() or
                labeledLabelTask.min().item() != unlabeledLabelTask.min().item()):
                print('ERROR')
                break

        # labeled data
        self.labeledData = finalLabeledData
        self.labeledLabel = finalLabeledLabel

        # unlabeled data
        self.unlabeledData = finalUnlabeledData
        self.unlabeledLabel = finalUnlabeledLabel

        # testing data
        self.unlabeledDataTest = unlabeledDataTest
        self.unlabeledLabelTest = unlabeledLabelTest

        print('Unlabeled data: ', unlabeledData.shape[0])
        print('Labeled data: ', labeledData.shape[0])
        print('Test data: ', testData.shape[0])

class cifarLoader(object):
    def __init__(self,
                labeldSamples,
                unlabeldSamples,
                testingBatchSize = 1000,
                nEachClassSamples = None):
        # np.random.seed(0)
        # torch.manual_seed(0)
        # random.seed(0)
        self.batchSize = testingBatchSize
        self.loadData(labeldSamples, 
                      unlabeldSamples,
                      nEachClassSamples)
        
    def loadData(self,
                 labeldSamples,
                 unlabeldSamples,
                 nEachClassSamples):
        # labeled data
        labeledData = torch.Tensor(10000,3,32,32)
        labeledLabel = torch.LongTensor(10000)

        for idx, example in enumerate(labeldSamples):
            labeledData[idx] = example[0]
            labeledLabel[idx] = example[1]
            
        self.labeledData = labeledData
        self.labeledLabel = labeledLabel

        self.classes = torch.unique(self.labeledLabel).tolist()
        self.nOutput = torch.unique(self.labeledLabel).shape[0]
        self.nLabeledData = self.labeledData.shape[0]
        
        if nEachClassSamples is not None:
            selectedLabeledData  = torch.Tensor(self.nOutput*nEachClassSamples,3,32,32)
            selectedLabeledLabel = torch.LongTensor(self.nOutput*nEachClassSamples)

            idx = 0
            selectedDataIdx = []
            for iClass in self.classes:
                # print(iClass)
                dataCount = 0

                for iData in range(0,self.nLabeledData):
                    # print(iData)
                    if labeledLabel[iData] == iClass:
                        selectedLabeledData[idx] = self.labeledData[iData]
                        selectedLabeledLabel[idx] = self.labeledLabel[iData]
                        # print(labeledLabel[iData])
                        idx += 1
                        dataCount += 1

                        selectedDataIdx.append(iData)
                        iData += 1

                    if dataCount == nEachClassSamples:
                        break

            remainderData = deleteRowTensor(self.labeledData, selectedDataIdx, 2)
            remainderLabel = deleteRowTensor(self.labeledLabel, selectedDataIdx, 2)
            
            self.nLabeledData = selectedLabeledData.shape[0]

            # shuffle
            indices = torch.randperm(self.nLabeledData)
            self.labeledData  = selectedLabeledData[indices]
            self.labeledLabel = selectedLabeledLabel[indices]
        
        # unlabeled data
        unlabeledData = torch.Tensor(60000,3,32,32)
        unlabeledLabel = torch.LongTensor(60000)

        for idx, example in enumerate(unlabeldSamples):
            unlabeledData[idx] = example[0]
            unlabeledLabel[idx] = example[1]
        
        if nEachClassSamples is not None:
            unlabeledData = torch.cat((unlabeledData,remainderData),0)
            unlabeledLabel  = torch.cat((unlabeledLabel,remainderLabel),0)

        self.unlabeledData = unlabeledData
        self.unlabeledLabel = unlabeledLabel
        
        self.nUnlabeledData = self.unlabeledData.shape[0]
        self.nBatch = int(self.nUnlabeledData/self.batchSize)
        self.taskIndicator = (torch.zeros(self.nBatch).long()).tolist()
        
        print('Number of output: ', self.nOutput)
        print('Number of labeled data: ', self.nLabeledData)
        print('Number of unlabeled data: ', self.nUnlabeledData)
        print('Number of unlabeled data batch: ', self.nBatch)

    def createTask(self, 
                   nTask = 2, 
                   taskList = [], 
                   taskType = 1):
        # random seed control
        # random.seed(0)
        if (taskType == 2 or taskType == 3) and (len(taskList) != nTask):
            raise NameError('list of rotaion angle should be the same with the number of task')

        self.taskIndicator = []

        # clone labeled data
        transformedLabeledData = self.labeledData.clone()
        transformedLabeledLabel = self.labeledLabel.clone()
        finalLabeledData = {}
        finalLabeledLabel = {}

        # clone unlabeled data
        transformedUnlabeledData = self.unlabeledData.clone()
        transformedUnlabeledLabel = self.unlabeledLabel.clone()
        finalUnlabeledData = {}
        finalUnlabeledLabel = {}

        # testing data
        unlabeledDataTest = {}
        unlabeledLabelTest = {}
        
        # number of data and batch for each task
        self.nTask = nTask
        self.nLabeledDataPerTask = int(self.nLabeledData/nTask)
        self.nBatchPerTask = int(self.nBatch/nTask)
        self.nBatch = self.nBatchPerTask*nTask
        self.nUnlabeledDataPerTask  = self.nBatchPerTask*self.batchSize

        nUnlabeledDataTest = 0

        for iTask in range(0,nTask):
            # load data
            # iTask = iTask + 1
            

            # load labeled data
            taskLabeledData = transformedLabeledData[(iTask)*self.nLabeledDataPerTask:(iTask+1)*self.nLabeledDataPerTask]
            taskLabeledLabel = transformedLabeledLabel[(iTask)*self.nLabeledDataPerTask:(iTask+1)*self.nLabeledDataPerTask]

            # load unlabeled data
            taskUnlabeledData = transformedUnlabeledData[(iTask)*self.nUnlabeledDataPerTask:(iTask+1)*self.nUnlabeledDataPerTask]
            taskUnlabeledLabel = transformedUnlabeledLabel[(iTask)*self.nUnlabeledDataPerTask:(iTask+1)*self.nUnlabeledDataPerTask]

            if self.nUnlabeledDataPerTask != taskUnlabeledData.shape[0] or self.nLabeledDataPerTask != taskLabeledData.shape[0]:
                raise NameError('Mismatch data size')

            # permute
            if taskType == 1:
                # reshape image
                taskLabeledData = taskLabeledData.view(taskLabeledData.size(0),-1)
                taskUnlabeledData = taskUnlabeledData.view(taskUnlabeledData.size(0),-1)

                # create permutation
                col_idxs = list(range(taskLabeledData.shape[1]))
                random.shuffle(col_idxs)

                # permute labeled data
                taskLabeledData = taskLabeledData[:, torch.tensor(col_idxs)]

                # permute unlabeled data
                taskUnlabeledData = taskUnlabeledData[:, torch.tensor(col_idxs)]

                # reshape back image
                taskLabeledData = taskLabeledData.reshape(taskLabeledData.size(0),3,32,32)
                taskUnlabeledData = taskUnlabeledData.reshape(taskUnlabeledData.size(0),3,32,32)

            elif taskType == 2:
                # rotate labeled data
                for idx, _ in enumerate(taskLabeledData):
                    taskLabeledData[idx] = Rotate(random.randint(taskList[iTask][0],
                                                                 taskList[iTask][1]))(taskLabeledData[idx])

                # rotate unlabeled data
                for idx, _ in enumerate(taskUnlabeledData):
                    taskUnlabeledData[idx] = Rotate(random.randint(taskList[iTask][0], 
                                                                   taskList[iTask][1]))(taskUnlabeledData[idx])                    
            
            elif taskType == 3:
                # split MNIST, into 5 tasks
                self.nOutputPerTask = 2

                taskLabeledDataSplit = torch.Tensor().float()
                taskLabeledLabelSplit = torch.Tensor().long()

                taskUnlabeledDataSplit = torch.Tensor().float()
                taskUnlabeledLabelSplit = torch.Tensor().long()

                for iClass in taskList[iTask]:
                    # split labeled data
                    taskLabeledDataSplit = torch.cat((taskLabeledDataSplit,
                                                      transformedLabeledData[transformedLabeledLabel==iClass]),0)
                    taskLabeledLabelSplit = torch.cat((taskLabeledLabelSplit,
                                                       transformedLabeledLabel[transformedLabeledLabel==iClass]),0)

                    # split unlabeled data
                    taskUnlabeledDataSplit = torch.cat((taskUnlabeledDataSplit,
                                                        transformedUnlabeledData[transformedUnlabeledLabel==iClass]),0)
                    taskUnlabeledLabelSplit = torch.cat((taskUnlabeledLabelSplit,
                                                        transformedUnlabeledLabel[transformedUnlabeledLabel==iClass]),0)

                # shuffle labeled data
                taskLabeledData = taskLabeledDataSplit
                taskLabeledLabel = taskLabeledLabelSplit
                
                row_idxs = list(range(taskLabeledData.shape[0]))
                random.shuffle(row_idxs)
                taskLabeledData = taskLabeledData[torch.tensor(row_idxs), :]
                taskLabeledLabel = taskLabeledLabel[torch.tensor(row_idxs)]

                # shuffle unlabeled data
                taskUnlabeledData = taskUnlabeledDataSplit
                taskUnlabeledLabel = taskUnlabeledLabelSplit

                row_idxs = list(range(taskUnlabeledData.shape[0]))
                random.shuffle(row_idxs)
                taskUnlabeledData = taskUnlabeledData[torch.tensor(row_idxs), :]
                taskUnlabeledLabel = taskUnlabeledLabel[torch.tensor(row_idxs)]

            # store labeled data and labels
            finalLabeledData[iTask]  = taskLabeledData
            finalLabeledLabel[iTask] = taskLabeledLabel

            # store unlabeled data and labels
            finalUnlabeledData[iTask]  = taskUnlabeledData[self.batchSize:]
            finalUnlabeledLabel[iTask] = taskUnlabeledLabel[self.batchSize:]
            self.taskIndicator = self.taskIndicator + (iTask*torch.ones(int(finalUnlabeledData[iTask].shape[0]/self.batchSize)).long()).tolist()

            # store unlabeled data for testing
            unlabeledDataTest[iTask] = taskUnlabeledData[:self.batchSize]
            unlabeledLabelTest[iTask] = taskUnlabeledLabel[:self.batchSize]  
            nUnlabeledDataTest += unlabeledDataTest[iTask].shape[0]        


        # labeled data
        self.labeledData = finalLabeledData
        self.labeledLabel = finalLabeledLabel

        # unlabeled data
        self.unlabeledData = finalUnlabeledData
        self.unlabeledLabel = finalUnlabeledLabel

        # testing data
        self.unlabeledDataTest = unlabeledDataTest
        self.unlabeledLabelTest = unlabeledLabelTest

        # update size
        self.nBatchPerTask = int(finalUnlabeledData[iTask].shape[0]/self.batchSize)
        self.nBatch = self.nBatchPerTask*nTask
        self.nUnlabeledDataPerTask = int(finalUnlabeledData[iTask].shape[0]/self.batchSize)*self.batchSize
        self.nUnlabeledDataTest = nUnlabeledDataTest

        print('Number of task: ', nTask)
        print('Number of labeled data per task: ', self.nLabeledDataPerTask)
        print('Number of unlabeled data per task: ', self.nUnlabeledDataPerTask)
        print('Number of unlabeled data batch per task: ', self.nBatchPerTask)
        print('Number of unlabeled data test: ', self.nUnlabeledDataTest)

    def createDrift(self, 
                    nDrift = 2, 
                    taskList = [], 
                    taskType = 1):
        # random seed control
        # random.seed(0)
        if (taskType == 2) and (len(taskList) != nDrift):
            raise NameError('list of rotaion angle should be the same with the number of task')

        # clone unlabeled data
        transformedUnlabeledData  = self.unlabeledData.clone()
        transformedUnlabeledLabel = self.unlabeledLabel.clone()
        finalUnlabeledData = torch.Tensor().float()
        finalUnlabeledLabel = torch.Tensor().long()
        
        # number of data and batch for each task
        self.nDrift = nDrift
        self.nBatchPerTask = int(self.nBatch/nDrift)
        self.nBatch = self.nBatchPerTask*nDrift
        self.nUnlabeledDataPerTask = self.nBatchPerTask*self.batchSize

        for iTask in range(0,nDrift):
            # load data
            # iTask = iTask + 1
            self.taskIndicator = self.taskIndicator + (iTask*torch.ones(self.nBatchPerTask).long()).tolist()

            # load unlabeled data
            taskUnlabeledData = transformedUnlabeledData[(iTask)*self.nUnlabeledDataPerTask:(iTask+1)*self.nUnlabeledDataPerTask]
            taskUnlabeledLabel = transformedUnlabeledLabel[(iTask)*self.nUnlabeledDataPerTask:(iTask+1)*self.nUnlabeledDataPerTask]

            if self.nUnlabeledDataPerTask != taskUnlabeledData.shape[0]:
                raise NameError('Mismatch data size')

            # permute
            if taskType == 1:
                # reshape image
                taskUnlabeledData = taskUnlabeledData.view(taskUnlabeledData.size(0),-1)

                # create permutation
                col_idxs = list(range(taskLabeledData.shape[1]))
                random.shuffle(col_idxs)

                # permute unlabeled data
                taskUnlabeledData = taskUnlabeledData[:, torch.tensor(col_idxs)]

                # reshape back image
                taskUnlabeledData = taskUnlabeledData.reshape(taskUnlabeledData.size(0),3,32,32)

            elif taskType == 2:
                # rotate unlabeled data
                for idx, _ in enumerate(taskUnlabeledData):
                    taskUnlabeledData[idx] = Rotate(random.randint(taskList[iTask][0], taskList[iTask][1]))(taskUnlabeledData[idx])

            # store unlabeled data and labels
            finalUnlabeledData  = torch.cat((finalUnlabeledData,taskUnlabeledData),0)
            finalUnlabeledLabel = torch.cat((finalUnlabeledLabel,taskUnlabeledLabel),0)

        # unlabeled data
        self.unlabeledData  = finalUnlabeledData
        self.unlabeledLabel = finalUnlabeledLabel

class svhnLoader(object):
    def __init__(self,
                labeldSamples, 
                unlabeldSamples, 
                testingBatchSize = 1000, 
                nEachClassSamples = None):
        # np.random.seed(0)
        # torch.manual_seed(0)
        # random.seed(0)
        self.batchSize = testingBatchSize
        self.loadData(labeldSamples, unlabeldSamples, nEachClassSamples)
        
    def loadData(self, 
                labeldSamples, 
                unlabeldSamples,
                nEachClassSamples):
        # labeled data
        labeledData = torch.Tensor(26032,3,32,32)
        labeledLabel = torch.LongTensor(26032)

        for idx, example in enumerate(labeldSamples):
            labeledData[idx] = example[0]
            labeledLabel[idx] = example[1]
            
        self.labeledData = labeledData
        self.labeledLabel = labeledLabel

        self.classes = torch.unique(self.labeledLabel).tolist()
        self.nOutput = torch.unique(self.labeledLabel).shape[0]
        self.nLabeledData = self.labeledData.shape[0]
        
        if nEachClassSamples is not None:
            selectedLabeledData = torch.Tensor(self.nOutput*nEachClassSamples,3,32,32)
            selectedLabeledLabel = torch.LongTensor(self.nOutput*nEachClassSamples)

            idx = 0
            selectedDataIdx = []
            for iClass in self.classes:
                # print(iClass)
                dataCount = 0

                for iData in range(0,self.nLabeledData):
                    # print(iData)
                    if labeledLabel[iData] == iClass:
                        selectedLabeledData[idx] = self.labeledData[iData]
                        selectedLabeledLabel[idx] = self.labeledLabel[iData]
                        # print(labeledLabel[iData])
                        idx += 1
                        dataCount += 1

                        selectedDataIdx.append(iData)
                        iData += 1

                    if dataCount == nEachClassSamples:
                        break

            remainderData = deleteRowTensor(self.labeledData, selectedDataIdx, 2)
            remainderLabel = deleteRowTensor(self.labeledLabel, selectedDataIdx, 2)
            
            self.nLabeledData = selectedLabeledData.shape[0]

            # shuffle
            indices = torch.randperm(self.nLabeledData)
            self.labeledData = selectedLabeledData[indices]
            self.labeledLabel = selectedLabeledLabel[indices]
        
        # unlabeled data
        unlabeledData = torch.Tensor(73257,3,32,32)
        unlabeledLabel = torch.LongTensor(73257)

        for idx, example in enumerate(unlabeldSamples):
            unlabeledData[idx] = example[0]
            unlabeledLabel[idx] = example[1]
        
        if nEachClassSamples is not None:
            unlabeledData = torch.cat((unlabeledData,remainderData),0)
            unlabeledLabel = torch.cat((unlabeledLabel,remainderLabel),0)

        self.unlabeledData = unlabeledData
        self.unlabeledLabel = unlabeledLabel
        
        self.nUnlabeledData = self.unlabeledData.shape[0]
        self.nBatch = int(self.nUnlabeledData/self.batchSize)
        self.taskIndicator  = (torch.zeros(self.nBatch).long()).tolist()
        
        print('Number of output: ', self.nOutput)
        print('Number of labeled data: ', self.nLabeledData)
        print('Number of unlabeled data: ', self.nUnlabeledData)
        print('Number of unlabeled data batch: ', self.nBatch)

    def createTask(self, 
                   nTask = 2, 
                   taskList = [], 
                   taskType = 1):
        # random seed control
        # random.seed(0)
        if (taskType == 2 or taskType == 3) and (len(taskList) != nTask):
            raise NameError('list of rotaion angle should be the same with the number of task')

        self.taskIndicator = []

        # clone labeled data
        transformedLabeledData = self.labeledData.clone()
        transformedLabeledLabel = self.labeledLabel.clone()
        finalLabeledData = {}
        finalLabeledLabel = {}

        # clone unlabeled data
        transformedUnlabeledData = self.unlabeledData.clone()
        transformedUnlabeledLabel = self.unlabeledLabel.clone()
        finalUnlabeledData = {}
        finalUnlabeledLabel = {}

        # testing data
        unlabeledDataTest = {}
        unlabeledLabelTest = {}
        
        # number of data and batch for each task
        self.nTask = nTask
        self.nLabeledDataPerTask = int(self.nLabeledData/nTask)
        self.nBatchPerTask = int(self.nBatch/nTask)
        self.nBatch = self.nBatchPerTask*nTask
        self.nUnlabeledDataPerTask  = self.nBatchPerTask*self.batchSize

        nUnlabeledDataTest = 0

        for iTask in range(0,nTask):
            # load data
            # iTask = iTask + 1
            

            # load labeled data
            taskLabeledData = transformedLabeledData[(iTask)*self.nLabeledDataPerTask:(iTask+1)*self.nLabeledDataPerTask]
            taskLabeledLabel = transformedLabeledLabel[(iTask)*self.nLabeledDataPerTask:(iTask+1)*self.nLabeledDataPerTask]

            # load unlabeled data
            taskUnlabeledData = transformedUnlabeledData[(iTask)*self.nUnlabeledDataPerTask:(iTask+1)*self.nUnlabeledDataPerTask]
            taskUnlabeledLabel = transformedUnlabeledLabel[(iTask)*self.nUnlabeledDataPerTask:(iTask+1)*self.nUnlabeledDataPerTask]

            if self.nUnlabeledDataPerTask != taskUnlabeledData.shape[0] or self.nLabeledDataPerTask != taskLabeledData.shape[0]:
                raise NameError('Mismatch data size')

            # permute
            if taskType == 1:
                # reshape image
                taskLabeledData = taskLabeledData.view(taskLabeledData.size(0),-1)
                taskUnlabeledData = taskUnlabeledData.view(taskUnlabeledData.size(0),-1)

                # create permutation
                col_idxs = list(range(taskLabeledData.shape[1]))
                random.shuffle(col_idxs)

                # permute labeled data
                taskLabeledData = taskLabeledData[:, torch.tensor(col_idxs)]

                # permute unlabeled data
                taskUnlabeledData = taskUnlabeledData[:, torch.tensor(col_idxs)]

                # reshape back image
                taskLabeledData = taskLabeledData.reshape(taskLabeledData.size(0),3,32,32)
                taskUnlabeledData = taskUnlabeledData.reshape(taskUnlabeledData.size(0),3,32,32)

            elif taskType == 2:
                # rotate labeled data
                for idx, _ in enumerate(taskLabeledData):
                    taskLabeledData[idx] = Rotate(random.randint(taskList[iTask][0], taskList[iTask][1]))(taskLabeledData[idx])

                # rotate unlabeled data
                for idx, _ in enumerate(taskUnlabeledData):
                    taskUnlabeledData[idx] = Rotate(random.randint(taskList[iTask][0], taskList[iTask][1]))(taskUnlabeledData[idx])                    
            
            elif taskType == 3:
                # split MNIST, into 5 tasks
                self.nOutputPerTask = 2

                taskLabeledDataSplit = torch.Tensor().float()
                taskLabeledLabelSplit = torch.Tensor().long()

                taskUnlabeledDataSplit = torch.Tensor().float()
                taskUnlabeledLabelSplit = torch.Tensor().long()

                for iClass in taskList[iTask]:
                    # split labeled data
                    taskLabeledDataSplit = torch.cat((taskLabeledDataSplit,transformedLabeledData[transformedLabeledLabel==iClass]),0)
                    taskLabeledLabelSplit = torch.cat((taskLabeledLabelSplit,transformedLabeledLabel[transformedLabeledLabel==iClass]),0)

                    # split unlabeled data
                    taskUnlabeledDataSplit = torch.cat((taskUnlabeledDataSplit,transformedUnlabeledData[transformedUnlabeledLabel==iClass]),0)
                    taskUnlabeledLabelSplit = torch.cat((taskUnlabeledLabelSplit,transformedUnlabeledLabel[transformedUnlabeledLabel==iClass]),0)

                # shuffle labeled data
                taskLabeledData = taskLabeledDataSplit
                taskLabeledLabel = taskLabeledLabelSplit
                
                row_idxs = list(range(taskLabeledData.shape[0]))
                random.shuffle(row_idxs)
                taskLabeledData = taskLabeledData[torch.tensor(row_idxs), :]
                taskLabeledLabel = taskLabeledLabel[torch.tensor(row_idxs)]

                # shuffle unlabeled data
                taskUnlabeledData = taskUnlabeledDataSplit
                taskUnlabeledLabel = taskUnlabeledLabelSplit

                row_idxs = list(range(taskUnlabeledData.shape[0]))
                random.shuffle(row_idxs)
                taskUnlabeledData = taskUnlabeledData[torch.tensor(row_idxs), :]
                taskUnlabeledLabel = taskUnlabeledLabel[torch.tensor(row_idxs)]

            # store labeled data and labels
            finalLabeledData[iTask] = taskLabeledData
            finalLabeledLabel[iTask] = taskLabeledLabel

            # store unlabeled data and labels
            finalUnlabeledData[iTask] = taskUnlabeledData[self.batchSize:]
            finalUnlabeledLabel[iTask] = taskUnlabeledLabel[self.batchSize:]
            self.taskIndicator = self.taskIndicator + (iTask*torch.ones(int(finalUnlabeledData[iTask].shape[0]/self.batchSize)).long()).tolist()

            # store unlabeled data for testing
            unlabeledDataTest[iTask] = taskUnlabeledData[:self.batchSize]
            unlabeledLabelTest[iTask] = taskUnlabeledLabel[:self.batchSize]  
            nUnlabeledDataTest += unlabeledDataTest[iTask].shape[0]        


        # labeled data
        self.labeledData = finalLabeledData
        self.labeledLabel = finalLabeledLabel

        # unlabeled data
        self.unlabeledData = finalUnlabeledData
        self.unlabeledLabel = finalUnlabeledLabel

        # testing data
        self.unlabeledDataTest = unlabeledDataTest
        self.unlabeledLabelTest = unlabeledLabelTest

        # update size
        self.nBatchPerTask = int(finalUnlabeledData[iTask].shape[0]/self.batchSize)
        self.nBatch = self.nBatchPerTask*nTask
        self.nUnlabeledDataPerTask = int(finalUnlabeledData[iTask].shape[0]/self.batchSize)*self.batchSize
        self.nUnlabeledDataTest = nUnlabeledDataTest

        print('Number of task: ', nTask)
        print('Number of labeled data per task: ', self.nLabeledDataPerTask)
        print('Number of unlabeled data per task: ', self.nUnlabeledDataPerTask)
        print('Number of unlabeled data batch per task: ', self.nBatchPerTask)
        print('Number of unlabeled data test: ', self.nUnlabeledDataTest)

    def createDrift(self, 
                    nDrift = 2, 
                    taskList = [], 
                    taskType = 1):
        # random seed control
        # random.seed(0)
        if (taskType == 2) and (len(taskList) != nDrift):
            raise NameError('list of rotaion angle should be the same with the number of task')

        # clone unlabeled data
        transformedUnlabeledData = self.unlabeledData.clone()
        transformedUnlabeledLabel = self.unlabeledLabel.clone()
        finalUnlabeledData = torch.Tensor().float()
        finalUnlabeledLabel = torch.Tensor().long()
        
        # number of data and batch for each task
        self.nDrift = nDrift
        self.nBatchPerTask = int(self.nBatch/nDrift)
        self.nBatch = self.nBatchPerTask*nDrift
        self.nUnlabeledDataPerTask = self.nBatchPerTask*self.batchSize

        for iTask in range(0,nDrift):
            # load data
            # iTask = iTask + 1
            self.taskIndicator = self.taskIndicator + (iTask*torch.ones(self.nBatchPerTask).long()).tolist()

            # load unlabeled data
            taskUnlabeledData = transformedUnlabeledData[(iTask)*self.nUnlabeledDataPerTask:(iTask+1)*self.nUnlabeledDataPerTask]
            taskUnlabeledLabel = transformedUnlabeledLabel[(iTask)*self.nUnlabeledDataPerTask:(iTask+1)*self.nUnlabeledDataPerTask]

            if self.nUnlabeledDataPerTask != taskUnlabeledData.shape[0]:
                raise NameError('Mismatch data size')

            # permute
            if taskType == 1:
                # reshape image
                taskUnlabeledData = taskUnlabeledData.view(taskUnlabeledData.size(0),-1)

                # create permutation
                col_idxs = list(range(taskLabeledData.shape[1]))
                random.shuffle(col_idxs)

                # permute unlabeled data
                taskUnlabeledData = taskUnlabeledData[:, torch.tensor(col_idxs)]

                # reshape back image
                taskUnlabeledData = taskUnlabeledData.reshape(taskUnlabeledData.size(0),3,32,32)

            elif taskType == 2:
                # rotate unlabeled data
                for idx, _ in enumerate(taskUnlabeledData):
                    taskUnlabeledData[idx] = Rotate(random.randint(taskList[iTask][0], taskList[iTask][1]))(taskUnlabeledData[idx])

            # store unlabeled data and labels
            finalUnlabeledData = torch.cat((finalUnlabeledData,taskUnlabeledData),0)
            finalUnlabeledLabel = torch.cat((finalUnlabeledLabel,taskUnlabeledLabel),0)

        # unlabeled data
        self.unlabeledData = finalUnlabeledData
        self.unlabeledLabel = finalUnlabeledLabel

class dataLoader(object):
    def __init__(self, 
                fileName, 
                batchSize = 1000, 
                nEachClassSamples = 500, 
                maxMinNorm = False, 
                zScoreNorm = False):
        self.fileName = fileName
        self.batchSize = batchSize
        self.maxMinNorm = maxMinNorm
        self.zScoreNorm = zScoreNorm
        self.loadDataFromMatFile(nEachClassSamples)
        
    def loadDataFromMatFile(self, nEachClassSamples):
        data1 = scipy.io.loadmat(self.fileName)  # change your folder
        data = data1.get('data')
        data = torch.from_numpy(data)
        data = data.float()
        self.labeledData = data[:,0:-1]
        label = data[:,-1]
        self.labeledLabel = label.long()
        self.nData = data.shape[0]
        self.nBatch = int(self.nData/self.batchSize)
        self.nInput = self.labeledData.shape[1]
        self.nOutput = torch.unique(self.labeledLabel).shape[0]
        # print('Number of input: ', self.nInput)
        # print('Number of output: ', self.nOutput)
        # print('Number of batch: ', self.nBatch)
        self.classes = torch.unique(self.labeledLabel).tolist()
        self.nLabeledData = self.labeledData.shape[0]

        if self.zScoreNorm:
            self.zScoreNormalization()

        if self.maxMinNorm:
            self.maxMinNormalization()

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

            remainderData = deleteRowTensor(self.labeledData, selectedDataIdx, 2)
            remainderLabel = deleteRowTensor(self.labeledLabel, selectedDataIdx, 2)
            
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
            print('Number of output: ', self.nOutput)
            print('Number of labeled data: ', self.nLabeledData)
            print('Number of unlabeled data: ', self.nUnlabeledData)
            print('Number of unlabeled data batch: ', self.nBatch)

    def splitTrainTestData(self, testProportion = 0.3):
        self.data, self.testingData, self.label, self.testingLabel = train_test_split(self.data, self.label,
                                                                                      test_size = testProportion,
                                                                                      random_state = 0,
                                                                                      stratify = self.label)
        self.nData = self.testingData.shape[0]
        self.nBatch = int(self.nData/self.batchSize)
    
    def maxMinNormalization(self):
        self.labeledData = torch.from_numpy(preprocessing.minmax_scale(self.labeledData, feature_range=(0.001, 1))).float()
        
    def zScoreNormalization(self):
        self.labeledData = torch.from_numpy(scipy.stats.zscore(self.labeledData, axis=0)).float()

def probitFunc(meanIn,stdIn):
    stdIn += 0.0001  # for safety
    out = meanIn/(torch.ones(1) + (np.pi/8)*stdIn**2)**0.5
    
    return out

def reduceLabeledData(dataTrain, labelTrain, nLabeled):
    labeledData = torch.Tensor().float()
    labeledLabel  = torch.Tensor().long()

    nData = dataTrain [labelTrain==torch.unique(labelTrain)[0].item()].shape[0]
    nLabeled = int(nLabeled*nData)

    min_i = torch.unique(labelTrain)[0].item()
    max_i = torch.unique(labelTrain)[-1].item()

    for i in range(min_i, max_i + 1):
        dataClass = dataTrain [labelTrain==i]
        labelClass = labelTrain[labelTrain==i]

        labeledData = torch.cat((labeledData,dataClass[0:nLabeled]),0)
        labeledLabel = torch.cat((labeledLabel,labelClass[0:nLabeled]),0)

    # shuffle
    try:
        row_idxs = list(range(labeledData.shape[0]))
        random.shuffle(row_idxs)
        labeledData  = labeledData[torch.tensor(row_idxs), :]
        labeledLabel = labeledLabel[torch.tensor(row_idxs)]
    except:
        pdb.set_trace()

    return labeledData, labeledLabel

def stableSoftmax(data):
    # data is in the form of numpy array n x m, where n is the number of data point and m is the number of classes
    # output = exp(output - max(output,[],2));
    # output = output./sum(output, 2);
    data = data/np.max(data,1)[:,None]
    data = np.exp(data)
    data = data/np.sum(data,1)[:,None]

    return data

def deleteRowTensor(x, index, mode = 1):
    if mode == 1:
        x = x[torch.arange(x.size(0))!=index] 
    elif mode == 2:
        # delete more than 1 row
        # index is a list of deleted row
        allRow = torch.arange(x.size(0)).tolist()
        
        for ele in sorted(index, reverse = True):  
            del allRow[ele]

        remainderRow = torch.tensor(allRow).long()

        x = x[remainderRow]
    
    return x

def deleteColTensor(x,index):
    x = x.transpose(1,0)
    x = x[torch.arange(x.size(0))!=index]
    x = x.transpose(1,0)
    
    return x

def clusteringLoss(latentFeatures, oneHotClust, centroids):
    # criterion = nn.MSELoss()
    # lossClust = criterion(latentFeatures, torch.matmul(oneHotClust,centroids))      
    # ((latentFeatures-torch.matmul(oneHotClust,centroids))**2).mean()
    # pdb.set_trace()
    # torch.dist(y,x,2)
    lossClust = torch.mean(torch.norm(latentFeatures - torch.matmul(oneHotClust,centroids),dim=1))
    
    return lossClust

def maskingNoise(x, noiseIntensity = 0.1):
    # noiseStr: the ammount of masking noise 0~1*100%
    
    nData, nInput = x.shape
    nMask = np.max([int(noiseIntensity*nInput),1])
    for i,_ in enumerate(x):
        maskIdx = np.random.randint(nInput,size = nMask)
        x[i][maskIdx] = 0
    
    return x

def show_image(x):
    plt.imshow(x.numpy())

def imageNoise(x, noiseIntensity = 0.3, device = torch.device('cpu')):
    noiseIntensity = 0.3

    noise = torch.from_numpy(noiseIntensity * np.random.normal(loc= 0.5, scale= 0.5, size= x.shape)).float().to(device)
    X_train_noisy = x + noise
    X_train_noisy = torch.clamp(X_train_noisy, 0., 1.)

    return x

def reinitNet(cnn, netlist):
    for netLen in range(len(netlist)):
        netlist[netLen].network.apply(weight_reset)

    cnn.apply(weight_reset)
        
    return netlist, cnn

def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

def showImage(img):
    plt.imshow(img[0].numpy())
    plt.show()

def plotPerformance(Iter,accuracy,hiddenNode,hiddenLayer,nCluster):
    SMALL_SIZE = 10
    MEDIUM_SIZE = 12
    BIGGER_SIZE = 14

    plt.rc('font', size=8)                   # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    fig, axes = plt.subplots(4,1,figsize=(8, 12))
#     fig.tight_layout()

    ax1 = axes[0]
    ax2 = axes[1]
    ax3 = axes[2]
    ax4 = axes[3]
    
    ax1.plot(Iter,accuracy,'k-')
#     ax1.set_title('Testing accuracy')
    ax1.set_ylabel('Áccuracy (%)')
#     ax1.set_xlabel('Number of bathces')
    ax1.yaxis.tick_right()
    ax1.autoscale_view('tight')
    ax1.set_ylim(ymin=0,ymax=100)
    ax1.set_xlim(xmin=0,xmax=len(Iter))

    ax2.plot(Iter,hiddenNode,'k-')
#     ax2.set_title('Testing loss')
    ax2.set_ylabel('Hidden node')
#     ax2.set_xlabel('Number of bathces')
    ax2.yaxis.tick_right()
    ax2.autoscale_view('tight')
    ax2.set_ylim(ymin=0)
    ax2.set_xlim(xmin=0,xmax=len(Iter))

    ax3.plot(Iter,hiddenLayer,'k-')
#     ax3.set_title('Hidden node evolution')
    ax3.set_ylabel('Hidden layer')
#     ax3.set_xlabel('Number of bathces')
    ax3.yaxis.tick_right()
    ax3.autoscale_view('tight')
    ax3.set_ylim(ymin=0)
    ax3.set_xlim(xmin=0,xmax=len(Iter))

    ax4.plot(Iter,nCluster,'k-')
#     ax4.set_title('Hidden layer evolution')
    ax4.set_ylabel('Cluster')
    ax4.set_xlabel('Number of bathces')
    ax4.yaxis.tick_right()
    ax4.autoscale_view('tight')
    ax4.set_ylim(ymin=0)
    ax4.set_xlim(xmin=0,xmax=len(Iter))