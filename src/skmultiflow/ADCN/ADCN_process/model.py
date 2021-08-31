import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import linalg as LA
import pdb
import random
from skmultiflow.ADCN.ADCN_process.utilsADCN import deleteRowTensor, deleteColTensor

# random seed control
# np.random.seed(0)
# torch.manual_seed(0)
# random.seed(0)

# ============================= Cluster model =============================
class cluster(object):
    def __init__(self, nInput,
                 initData = None,
                 nInitCluster = 2,
                 clusterGrowing = True,
                 desiredLabels = [0,1]):
        # create centroids
        self.nInput = nInput
        self.nClass = len(desiredLabels)
        self.nCluster = nInitCluster

        if initData is None or nInitCluster > 2:
            self.centroids = torch.rand(self.nCluster, nInput).numpy()
        else:
            self.centroids = initData.numpy()
            self.nCluster = self.centroids.shape[0]

        self.support = np.ones(self.nCluster, dtype=int)
        
        # initial distance
        self.meanDistance = np.ones(self.nCluster)
        self.stdDistance = np.zeros(self.nCluster)
        
        self.growClustInd = False
        self.clusterGrowing = clusterGrowing
        self.insignificantCluster = []
    
    # ============================= evaluation =============================
    def distanceMeasure(self, data):
        self.data  = data
        l2NormDistance = LA.norm(self.centroids - data,axis=1)
        self.nearestCentroid = np.argmin(l2NormDistance,0)

    def detectNovelPattern(self, newData, winCentroid, winClsIdx):
        winDistance = LA.norm(winCentroid - newData,axis=1)

        # estimate distance
        miuOld = self.meanDistance[winClsIdx]
        self.meanDistance[winClsIdx] = (self.meanDistance[winClsIdx] 
                                        - ((self.meanDistance[winClsIdx] - winDistance)
                                        / (self.support[winClsIdx])))
        variance                     = ((self.stdDistance[winClsIdx])**2 + (miuOld)**2 
                                        - (self.meanDistance[winClsIdx])**2 + (((winDistance)**2 
                                        - (self.stdDistance[winClsIdx])**2 - (miuOld)**2 )
                                        / (self.support[winClsIdx])))
        
        self.stdDistance[winClsIdx] = np.sqrt(variance)
        
        # grow cluster condition
        dynamicSigma = 2*np.exp(-winDistance) + 2  # 2
        growClusterCondition = (self.meanDistance[winClsIdx] + dynamicSigma*self.stdDistance[winClsIdx])
        
        if winDistance > growClusterCondition:
            self.growClustInd = True
        else:
            self.growClustInd = False

    def identifyInsignificantCluster(self):
        self.insignificantCluster = []
        if self.nCluster > self.nClass:
            self.insignificantCluster = np.where(np.max(self.allegianceClusterClass,1) < 0.55)[0].tolist()
    
    # ============================= evolving =============================    
    def growCluster(self, newCentroid):
        self.nCluster += 1
        self.centroids = np.concatenate((self.centroids,newCentroid),0)
        self.meanDistance = np.concatenate((self.meanDistance,np.array([self.distance.min()])),0)
        self.stdDistance = np.concatenate((self.stdDistance,np.zeros(1)),0)
        self.support = np.concatenate((self.support,np.ones(1)),0)
        
    def removeCluster(self, index):
        self.nCluster -= len(index)
        self.centroids = np.delete(self.centroids, index, 0)
        self.allegianceClusterClass = np.delete(self.allegianceClusterClass, index, 0)
        self.meanDistance = np.delete(self.meanDistance, index, 0)
        self.stdDistance = np.delete(self.stdDistance, index, 0)
        self.support = np.delete(self.support, index, 0)
        self.insignificantCluster = []
        
    def growInput(self, constant = None):
        self.nInput += 1
        if constant is None:
            self.centroids  = np.concatenate((self.centroids,np.zeros([self.nCluster,1], dtype = float)), 1)
        elif constant is not None:
            self.centroids  = np.concatenate((self.centroids,constant*np.ones([self.nCluster,1], dtype = float)), 1)
        
    def deleteInput(self, index):
        self.nInput -= 1
        self.centroids = np.delete(self.centroids, index, 1)
    
    # ============================= update =============================
    def augmentLabeledSamples(self, data, labels):
        # data is the encoded data of the respective layer
        # data and labels are numpy arrays
        self.labeledData = np.concatenate((self.labeledData, data),0)
        self.labels = np.concatenate((self.labels, labels),0)

        # update labels count
        uniqueLabels = np.unique(labels)
        for iLabel in uniqueLabels:
            iLabelsCount = np.array([(data[labels == iLabel]).shape[0]])
            self.labelsCount[iLabel] = self.labelsCount[iLabel] + iLabelsCount

    def updateCentroid(self, newData,
                       winClsIdx,
                       addSupport = True):
        if addSupport:
            self.support[winClsIdx] = self.support[winClsIdx] + 1
            
        self.centroids[winClsIdx] = (self.centroids[winClsIdx] 
                                    - ((self.centroids[winClsIdx] - newData)
                                    / (self.support[winClsIdx])))
        self.distance             = LA.norm(self.centroids - newData, axis=1)

    def updateAllegiance(self, data, labels, 
                         randomTesting = False):
        # data and labels are in numpy array format
        
        nData = data.shape[0]
        # allegiance = np.empty([self.nCluster,nData],dtype=float)*0.0
        allegiance = np.zeros((self.nCluster,nData),dtype=float).astype(np.float32)
        for i,iCentroid in enumerate(self.centroids):
            allegiance[i] = (np.exp(-LA.norm(iCentroid - data,axis=1)))
        self.allegiance = allegiance/np.max(allegiance,0)               # equation 6 from paper STEM
        
        uniqueLabels = np.unique(labels)

        if randomTesting:
            # np.empty([self.nCluster,self.nClass],dtype=float)
            allegianceClusterClass = np.random.rand(self.nCluster,self.nClass).astype(np.float32)  
        else:
            allegianceClusterClass = np.zeros((self.nCluster,self.nClass),dtype=float).astype(np.float32)

        for iCluster in range(0,self.nCluster):
            for iClass in uniqueLabels:
                allegianceClusterClass[iCluster][iClass] = (np.sum(self.allegiance[iCluster][labels==iClass])
                                                            / len(labels[labels==iClass]))                            # equation 7 from paper STEM

        # pdb.set_trace()
        self.allegianceClusterClass = allegianceClusterClass

        if np.isnan(np.sum(self.allegianceClusterClass)) or np.isnan(np.sum(self.allegiance)):
            pdb.set_trace()

    def fit(self, trainData, epoch = 1):
        nTrainData = trainData.shape[0]
        
        growCount = 0

        clusterHistory = []
        
        for iEpoch in range(0,epoch):
            shuffled_indices = torch.randperm(nTrainData)
            
            for iData in range(0,nTrainData):
                # load data
                indices = shuffled_indices[iData:iData+1]
                minibatch_xTrain = trainData[indices]
                minibatch_xTrain = minibatch_xTrain.numpy()

                self.distanceMeasure(minibatch_xTrain)

                # update clusters
                if iEpoch == 0:
                    self.updateCentroid(minibatch_xTrain, 
                                        self.nearestCentroid)
                else:
                    self.updateCentroid(minibatch_xTrain, 
                                        self.nearestCentroid, 
                                        addSupport = False)

                self.detectNovelPattern(minibatch_xTrain, 
                                        self.centroids[self.nearestCentroid],
                                        self.nearestCentroid)

                if self.growClustInd and self.clusterGrowing:
                    # grow clusters
                    self.growCluster(minibatch_xTrain)
                    growCount += 1

            clusterHistory.append(self.nCluster)

        # reassigning empty clusters
        self.clusterReassigning()
        
        self.clusterHistory = clusterHistory
        # update allegiance
        # self.updateAllegiance()

    def clusterReassigning(self):
        if self.support.min() == 1:
            singleToneCluster = np.where(self.support <= 2)[0].tolist()
            randCandidateList = np.where(self.support > 50)[0]

            if len(singleToneCluster) > 0 and len(randCandidateList.tolist()) > 0:
                for iClusterIdx in singleToneCluster:
                    randCandidateIdx  = random.choice(randCandidateList)
                    self.centroids[iClusterIdx] = self.centroids[randCandidateIdx] + 0.007  # np.random.rand(1)

    # ============================= prediction =============================    
    def predict(self, testData):
        nTestData  = testData.shape[0]
        testData   = testData.numpy()
        prediction = np.zeros([nTestData,self.nClass])
        for i,iAllegiance in enumerate(self.allegianceClusterClass):
            # equation 8 from paper STEM
            centroidAllegiance = (iAllegiance*np.expand_dims(np.exp(
                                    -LA.norm(self.centroids[i] - testData,axis=1)),0).T) 
            # equation 8 from paper STEM       
            prediction += centroidAllegiance                                
        # pdb.set_trace()
        self.score           = prediction
        # equation 8 from paper STEM
        self.predictedLabels = np.argmax(prediction,1)                      
        
    def getCluster(self,testData):
        nTestData  = testData.shape[0]
        testData   = testData.numpy()
        score      = np.zeros((self.nCluster,nTestData),dtype=float).astype(np.float32)
        # score      = np.empty([self.nCluster,nTestData],dtype=float)*0.0
        
        for i,iCentroid in enumerate(self.centroids):
            distance = LA.norm(iCentroid - testData,axis=1)
            score[i] = distance
        
        self.predictedClusters = np.argmax(score,0)

# ============================= Main network =============================
class smallAE():
    def __init__(self, no_input, no_hidden):
        self.network = basicAE(no_input, no_hidden)
        self.netUpdateProperties()
        
    def getNetProperties(self):
        print(self.network)
        print('No. of AE inputs :',self.nNetInput)
        print('No. of AE nodes :',self.nNodes)
        print('No. of AE parameters :',self.nParameters)
    
    def getNetParameters(self):
        print('Input weight: \n', self.network.linear.weight)
        print('Input bias: \n', self.network.linear.bias)
        print('Bias decoder: \n', self.network.biasDecoder)
    
    def netUpdateProperties(self):
        self.nNetInput = self.network.linear.in_features
        self.nNodes = self.network.linear.out_features
        self.nParameters = (self.network.linear.in_features*self.network.linear.out_features 
                            + len(self.network.linear.bias.data) 
                            + len(self.network.biasDecoder))
    
    # ============================= evolving =============================    
    def nodeGrowing(self, nNewNode = 1, 
                    device = torch.device('cpu')):
        nNewNodeCurr = self.nNodes + nNewNode
        
        # grow node
        newWeight  = nn.init.xavier_uniform_(torch.empty(nNewNode, self.nNetInput)).to(device)
        self.network.linear.weight.data  = torch.cat((self.network.linear.weight.data,
                                                      newWeight),0)  # grow input weights
        self.network.linear.bias.data    = torch.cat((self.network.linear.bias.data,
                                                      torch.zeros(nNewNode).to(device)),0) 
        self.network.linear.out_features = nNewNodeCurr
        del self.network.linear.weight.grad
        del self.network.linear.bias.grad
        
        self.netUpdateProperties()
    
    def nodePruning(self, pruneIdx,
                    nPrunedNode = 1):
        nNewNodeCurr = self.nNodes - nPrunedNode  # prune a node
        
        # prune node for current layer, output
        self.network.linear.weight.data = deleteRowTensor(self.network.linear.weight.data,
                                                          pruneIdx)  # prune input weights
        self.network.linear.bias.data = deleteRowTensor(self.network.linear.bias.data,
                                                        pruneIdx)  # prune input bias
        self.network.linear.out_features = nNewNodeCurr
        del self.network.linear.weight.grad
        del self.network.linear.bias.grad
        
        self.netUpdateProperties()
        
    def inputGrowing(self, nNewInput = 1,
                     device = torch.device('cpu')):
        nNewInputCurr = self.nNetInput + nNewInput

        # grow input weight
        newWeightNext = nn.init.xavier_uniform_(torch.empty(self.nNodes, nNewInput)).to(device)
        self.network.linear.weight.data = torch.cat((self.network.linear.weight.data,newWeightNext),1)
        self.network.biasDecoder.data = torch.cat((self.network.biasDecoder.data,
                                                   torch.zeros(nNewInput).to(device))
                                                  ,0)
        
        del self.network.linear.weight.grad
        del self.network.biasDecoder.grad

        self.network.linear.in_features = nNewInputCurr
        self.netUpdateProperties()
        
    def inputPruning(self, pruneIdx, nPrunedNode = 1):
        nNewInputCurr = self.nNetInput - nPrunedNode

        # prune input weight of next layer
        self.network.linear.weight.data = deleteColTensor(self.network.linear.weight.data,pruneIdx)
        self.network.biasDecoder.data = deleteRowTensor(self.network.biasDecoder.data,pruneIdx)
        
        del self.network.linear.weight.grad
        del self.network.biasDecoder.grad

        # update input features
        self.network.linear.in_features = nNewInputCurr
        self.netUpdateProperties()

class linearlizationlayer(nn.Module):
    def __init__(self):
        super(linearlizationlayer, self).__init__()
        
    def forward(self, x):
        
        return x

# ============================= Encoder Decoder =============================
class basicAE(nn.Module):
    def __init__(self, no_input, no_hidden):
        super(basicAE, self).__init__()
        # hidden layer
        self.linear = nn.Linear(no_input, no_hidden,  bias=True)
        self.activation = nn.ReLU(inplace=True)
        nn.init.xavier_uniform_(self.linear.weight)
        self.linear.bias.data.zero_()

        # decoder
        self.biasDecoder = nn.Parameter(torch.zeros(no_input))
        
    def forward(self, x, mode = 1):
        # mode 1. encode, 2. decode
        if mode == 1:
            # encoder
            x = self.linear(x)
            x = self.activation(x)                          # encoded output

        if mode == 2:
            # decoder
            x = F.linear(x, self.linear.weight.t()) + self.biasDecoder
            x = self.activation(x)                          # reconstructed input for end-to-end cnn
        
        return x

class mlpAePMNIST(nn.Module):
    def __init__(self, nNodes = 384, nOutput = 196):
        super(mlpAePMNIST, self).__init__()
        # hidden layer 1
        self.linear1 = nn.Linear(784, nNodes, bias=True)
        nn.init.xavier_uniform_(self.linear1.weight)
        self.linear1.bias.data.zero_()

        # hidden layer 2
        self.linear2 = nn.Linear(nNodes, nOutput,  bias=True)
        nn.init.xavier_uniform_(self.linear2.weight)
        self.linear2.bias.data.zero_()

        # activation function
        self.activation1 = nn.ReLU(inplace=True)
        self.activation2 = nn.Sigmoid()

        # decoder
        self.biasDecoder1 = nn.Parameter(torch.zeros(784))
        self.biasDecoder2 = nn.Parameter(torch.zeros(nNodes))
        
    def forward(self, x, mode = 1):
        # mode 1. encode, 2. decode
        if mode == 1:
            # encoder
            x = x.view(x.size(0),-1)
            x = self.linear1(x)
            x = self.activation1(x)                          
            x = self.linear2(x)
            x = self.activation1(x)                     # encoded output

        if mode == 2:
            # decoder
            x = F.linear(x, self.linear2.weight.t()) + self.biasDecoder2
            x = self.activation1(x)
            x = F.linear(x, self.linear1.weight.t()) + self.biasDecoder1
            x = self.activation2(x)                          # reconstructed input for end-to-end cnn
            x = x.reshape(x.size(0),1,28,28)
        
        return x

class simpleMPL(nn.Module):
    def __init__(self, nInput, 
                 nNodes = 10,
                 nOutput = 5):
        super(simpleMPL, self).__init__()
        # hidden layer 1
        self.linear1 = nn.Linear(nInput, nNodes, bias=True)
        nn.init.xavier_uniform_(self.linear1.weight)
        self.linear1.bias.data.zero_()

        # hidden layer 2
        self.linear2 = nn.Linear(nNodes, nOutput,  bias=True)
        nn.init.xavier_uniform_(self.linear2.weight)
        self.linear2.bias.data.zero_()

        # activation function
        self.activation1 = nn.ReLU(inplace=True)
        self.activation2 = nn.Sigmoid()

        # decoder
        self.biasDecoder1 = nn.Parameter(torch.zeros(nInput))
        self.biasDecoder2 = nn.Parameter(torch.zeros(nNodes))
        
    def forward(self, x, mode = 1):
        # mode 1. encode, 2. decode
        if mode == 1:
            # encoder
            x = self.linear1(x)
            x = self.activation1(x)                          
            x = self.linear2(x)
            x = self.activation1(x)                     # encoded output

        if mode == 2:
            # decoder
            x = F.linear(x, self.linear2.weight.t()) + self.biasDecoder2
            x = self.activation1(x)
            x = F.linear(x, self.linear1.weight.t()) + self.biasDecoder1
            x = self.activation2(x)                          # reconstructed input for end-to-end
        
        return x

class singleMPL(nn.Module):
    def __init__(self, nInput, nNodes = 10):
        super(singleMPL, self).__init__()
        # hidden layer 1
        self.linear1 = nn.Linear(nInput, nNodes, bias=True)
        nn.init.xavier_uniform_(self.linear1.weight)
        self.linear1.bias.data.zero_()

        # activation function
        self.activation1 = nn.ReLU(inplace=True)
        self.activation2 = nn.Sigmoid()

        # decoder
        self.biasDecoder1 = nn.Parameter(torch.zeros(nInput))
        
    def forward(self, x, mode = 1):
        # mode 1. encode, 2. decode
        if mode == 1:
            # encoder
            x = self.linear1(x)
            x = self.activation1(x)                       # encoded output

        if mode == 2:
            # decoder
            x = F.linear(x, self.linear1.weight.t()) + self.biasDecoder1
            x = self.activation2(x)                          # reconstructed input for end-to-end
        
        return x

class ConvAeMNIST(nn.Module):
    def __init__(self):
        super(ConvAeMNIST, self).__init__()
        ## encoder layers ##
        # conv layer (depth from 3 --> 16), 3x3 kernels
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)  
        
        # conv layer (depth from 16 --> 4), 3x3 kernels
        self.conv2 = nn.Conv2d(16, 4, 3, padding=1)
        
        # pooling layer to reduce x-y dims by two; kernel and stride of 2
        self.pool = nn.MaxPool2d(2, 2)

        ## decoder layers ##
        ## a kernel of 2 and a stride of 2 will increase the spatial dims by 2
        self.t_conv1 = nn.ConvTranspose2d(4, 16, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(16, 1, 2, stride=2)
        self.activation = nn.Sigmoid()

    def forward(self, x, mode = 1):
        # mode 1. encode, 2. decode
        if mode == 1:
            ## encode ##
            # add hidden layers with relu activation function
            # and maxpooling after
            x = F.relu(self.conv1(x))
            x = self.pool(x)
            
            # add second hidden layer
            x = F.relu(self.conv2(x))
            x = self.pool(x)  # compressed representation
            x = x.view(x.size(0),-1)

        if mode == 2:
            ## decode ##
            # add transpose conv layers, with relu activation function
            x = x.reshape(x.size(0),4,7,7)
            x = F.relu(self.t_conv1(x))
            
            # output layer (with sigmoid for scaling from 0 to 1)
            x = self.activation(self.t_conv2(x))
                
        return x

# class ConvAeCIFAR(nn.Module):
#     def __init__(self):
#         super(ConvAeCIFAR, self).__init__()
#         ## encoder layers ##
#         # conv layer (depth from 3 --> 16), 3x3 kernels
#         self.conv1 = nn.Conv2d(3, 32, 3, padding=1)  
        
#         # conv layer (depth from 16 --> 4), 3x3 kernels
#         self.conv2 = nn.Conv2d(32, 4, 3, padding=1)
        
#         # pooling layer to reduce x-y dims by two; kernel and stride of 2
#         self.pool  = nn.MaxPool2d(2, 2)

#         ## decoder layers ##
#         ## a kernel of 2 and a stride of 2 will increase the spatial dims by 2
#         self.t_conv1    = nn.ConvTranspose2d(4, 32, 2, stride=2)
#         self.t_conv2    = nn.ConvTranspose2d(32, 3, 2, stride=2)
#         self.activation = nn.Sigmoid()

#     def forward(self, x, mode = 1):
#         # mode 1. encode, 2. decode
#         if mode == 1:
#             ## encode ##
#             # add hidden layers with relu activation function
#             # and maxpooling after
#             x = F.relu(self.conv1(x))
#             x = self.pool(x)
            
#             # add second hidden layer
#             x = F.relu(self.conv2(x))
#             x = self.pool(x)  # compressed representation
#             x = x.view(x.size(0),-1)

#         if mode == 2:
#             ## decode ##
#             # add transpose conv layers, with relu activation function
#             x = x.reshape(x.size(0),4,8,8)
#             x = F.relu(self.t_conv1(x))
            
#             # output layer (with sigmoid for scaling from 0 to 1)
#             x = self.activation(self.t_conv2(x))
                
#         return x

class ConvAeCIFAR(nn.Module):
    def __init__(self):
        super(ConvAeCIFAR, self).__init__()
        ## encoder layers ##
        self.conv1 = nn.Conv2d(3 , 12, 4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(12, 24, 4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(24, 48, 4, stride=2, padding=1)

        ## decoder layers ##
        self.t_conv1 = nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1)
        self.t_conv2 = nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1)
        self.t_conv3 = nn.ConvTranspose2d(12, 3 , 4, stride=2, padding=1)
        self.activation = nn.Sigmoid()

    def forward(self, x, mode = 1):
        # mode 1. encode, 2. decode
        if mode == 1:
            ## encode ##
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            x = x.view(x.size(0),-1)

        if mode == 2:
            ## decode ##
            # add transpose conv layers, with relu activation function
            x = x.reshape(x.size(0),48,4,4)
            x = F.relu(self.t_conv1(x))
            x = F.relu(self.t_conv2(x))
            
            # output layer (with sigmoid for scaling from 0 to 1)
            x = self.activation(self.t_conv3(x))
                
        return x

# ============================= For LWF =============================
class ADCNoldtask():
    def __init__(self, taskId):
        # random seed control
        # np.random.seed(0)
        # torch.manual_seed(0)
        # random.seed(0)

        # initial network
        self.ADCNcnn = []
        self.ADCNae = []
        self.taskId = taskId
        
        # properties
        self.nInput = 0
        self.nOutput = 0
        self.nHiddenLayer = 1
        self.nHiddenNode = 0

    # ============================= forward pass =============================
    def forward(self, x, device = torch.device('cpu')):
        # encode decode in end-to-end manner
        # prepare model
        self.ADCNcnn = self.ADCNcnn.to(device)

        # prepare data
        x = x.to(device)

        # forward encoder CNN
        x = self.ADCNcnn(x)

        # feedforward from input layer to latent space, encode
        for iLayer,_ in enumerate(self.ADCNae):
            currnet = self.ADCNae[iLayer].network
            obj = currnet.train()
            obj = obj.to(device)
            x = obj(x)

        # feedforward from latent space to output layer, decode
        for iLayer in range(len(self.ADCNae)-1,0-1,-1):
            currnet = self.ADCNae[iLayer].network
            obj     = currnet.train()
            obj     = obj.to(device)
            x       = obj(x, 2)

        # forward decoder CNN
        x = self.ADCNcnn(x, 2)
            
        return x

# ============================= For Comparison =============================
class simpleNet(nn.Module):
    def __init__(self):
        super(simpleNet, self).__init__()
        # hidden layer 1
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)  
        
        # hidden layer 2
        self.conv2 = nn.Conv2d(16, 4, 3, padding=1)

        # pooling layer to reduce x-y dims by two; kernel and stride of 2
        self.pool = nn.MaxPool2d(2, 2)
        
        # output layer
        self.linear = nn.Linear(196, 10,  bias=True)
        nn.init.xavier_uniform_(self.linear.weight)
        self.linear.bias.data.zero_()

        # activation function
        self.activation = nn.ReLU(inplace=True)
        
    def forward(self, x):
        # forward
        x = x.view(x.size(0),-1)
        x = self.conv1(x)
        x = self.activation(x)                          
        x = self.conv2(x)
        x = self.activation(x)
        x = self.pool(x)            # compressed representation
        x = x.view(x.size(0),-1)
        x = self.linear(x)
        
        return x

class simpleNetPMNIST(nn.Module):
    def __init__(self):
        super(simpleNetPMNIST, self).__init__()
        # hidden layer
        self.linear1 = nn.Linear(784, 384, bias=True)
        nn.init.xavier_uniform_(self.linear1.weight)
        self.linear1.bias.data.zero_()

        self.linear2 = nn.Linear(384, 196,  bias=True)
        nn.init.xavier_uniform_(self.linear2.weight)
        self.linear2.bias.data.zero_()

        self.linear3 = nn.Linear(196, 10,  bias=True)
        nn.init.xavier_uniform_(self.linear3.weight)
        self.linear3.bias.data.zero_()

        # activation function
        self.activation = nn.ReLU(inplace=True)
        
    def forward(self, x):
        # forward
        x = x.view(x.size(0),-1)
        x = self.linear1(x)
        x = self.activation(x)                          
        x = self.linear2(x)
        x = self.activation(x)
        x = self.linear3(x)
        
        return x