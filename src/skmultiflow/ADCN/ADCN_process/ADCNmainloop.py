import numpy as np
import torch
import random
import time
import pdb
import sys
import skmultiflow.utils.constants as constants
from skmultiflow.ADCN.ADCN_classifier.utilsADCN import meanStdCalculator, plotPerformance, reduceLabeledData
from skmultiflow.ADCN.ADCN_classifier.model import cluster
from sklearn.metrics import precision_score, normalized_mutual_info_score, \
    adjusted_rand_score, recall_score, f1_score, cohen_kappa_score,fowlkes_mallows_score
import progressbar

#import other folder
from skmultiflow.ADCN.ADCN_classifier.evaluation_data_buffer import EvaluationDataBuffer
from skmultiflow.ADCN.ADCN_classifier.visualization.evaluation_visualizer import EvaluationVisualizer

def update_progress_bar(curr, total, steps):
        progress = curr / total
        progress_bar = round(progress * steps)
        print('\r', '#' * progress_bar + '-' * (steps - progress_bar),
              '[{:.0%}] '.format(progress), end='')
        print("\n")
        sys.stdout.flush()

def ADCNmain(ADCNnet,dataStreams,
            metrics,batch_size,
            show_plot = True,
            nLabeled = 1,
            layerGrowing = True,
            nodeEvolution = True,
            clusterGrowing = True,
            lwfLoss = True,
            clusteringLoss = True,
            trainingBatchSize = 16,
            noOfEpoch = 1,
            device = torch.device('cpu')):
    # random seed control
    # np.random.seed(0)
    # torch.manual_seed(0)
    # random.seed(0)
    
    # performance metrics
    # performanceMetrics = meanStd()   # [accuracy,testingLoss,testingTime,trainingTime]
    visualizer = None
    Accuracy  = []
    ARI = []    
    NMI = []
    F1 = []
    Precision = []
    Recall = []
    Kappa = []
    Gmean = []
    testingTime = []
    trainingTime = []
    # testingLoss  = []

    prevBatchData = []
        
    Y_pred = []
    Y_true = []
    Iter = []

    # for figure
    AccuracyHistory = []
    nHiddenLayerHistory = []
    nHiddenNodeHistory = []
    nClusterHistory = []
    
    # network evolution
    # netEvolution = meanStdCalculator()   # [nHiddenNode,nHiddenLayer]
    nHiddenNode = []
    nHiddenLayer = []
    nCluster = []
    layerCount = 0
    


    # Metrics + Plot
    data_dict = {}
    for metric in metrics:
        data_ids = ["mean", "current"]
        if metric == "true_vs_predict":
            data_ids = ["y_true", "y_pred"]
        data_dict[metric] = data_ids

        data_buffer = EvaluationDataBuffer(data_dict=data_dict)
    if show_plot:
        visualizer = EvaluationVisualizer(dataset_name=dataStreams.stream_name,
                                          task_type="classification",
                                          n_wait = 0,
                                          n_models = 1,
                                          metrics=metrics,
                                          model_names="ADCN",
                                          data_dict=data_dict)
    

    # initialization phase
    start_initialization_train = time.time()
    ADCNnet.initialization(dataStreams.labeledData, layerCount, 
                            batchSize = trainingBatchSize, device = device)

    if nLabeled == 1:
        allegianceData = dataStreams.labeledData.clone()
        allegianceLabel = dataStreams.labeledLabel.clone()
    elif nLabeled < 1:
        # reduced labeled data
        allegianceData, allegianceLabel = reduceLabeledData(dataStreams.labeledData.clone(),
                                                            dataStreams.labeledLabel.clone(),
                                                            nLabeled)
        print('Number of allegiance data: ', allegianceData.shape[0])

    # update allegiance
    # ADCNnet.updateAllegiance(dataStreams.labeledData, dataStreams.labeledLabel)

    end_initialization_train = time.time()
    initialization_time = end_initialization_train - start_initialization_train

    for i in range(len(ADCNnet.hiddenNodeHist)):
        Iter.append(i)
        nHiddenLayerHistory.append(ADCNnet.nHiddenLayer)
        AccuracyHistory.append(0)

    nHiddenNodeHistory = ADCNnet.hiddenNodeHist
    nClusterHistory    = ADCNnet.clusterHistory

    ## batch loop, handling unlabeled samples
    # training is conducted with single epoch. The initialization of a new layer uses epoch.
    # bar = progressbar.ProgressBar(dataStreams.nBatch)
    # bar = progressbar.ProgressBar()
    
    sample_id = 0
    for iBatch in range(0,dataStreams.nBatch):
        # if iBatch%10 == 0 or iBatch == 0:
        # print(iBatch,'-th batch')

        # load data
        batchIdx = iBatch + 1
        batchData = dataStreams.unlabeledData[(batchIdx-1)*dataStreams.batchSize:batchIdx*dataStreams.batchSize]
        batchLabel = dataStreams.unlabeledLabel[(batchIdx-1)*dataStreams.batchSize:batchIdx*dataStreams.batchSize]

        # update
        start_train = time.time()

        if iBatch > 0 and layerGrowing:
            # drift detection
            ADCNnet.driftDetection(batchData, previousBatchData)

            if ADCNnet.driftStatus == 2:
                # grow layer if drift is confirmed driftStatus == 2
                ADCNnet.layerGrowing()
                layerCount += 1

                # initialization phase
                ADCNnet.initialization(dataStreams.labeledData,
                                       layerCount, 
                                       batchSize = trainingBatchSize,
                                       device = device)

        # training data preparation
        previousBatchData = batchData.clone()
        batchData, batchLabel = ADCNnet.trainingDataPreparation(batchData, batchLabel)

        # training
        if ADCNnet.driftStatus == 0 or ADCNnet.driftStatus == 2:  # only train if it is stable or drift
            ADCNnet.fit(batchData, epoch = noOfEpoch)
            ADCNnet.updateNetProperties()

            # update allegiance
            ADCNnet.updateAllegiance(allegianceData, allegianceLabel)

        end_train = time.time()
        training_time = end_train - start_train

        # testing
        ADCNnet.testing(batchData, batchLabel)

        sample_id = sample_id + batch_size
        current_y_pred = ADCNnet.predictedLabel.tolist()
        current_y_true = ADCNnet.trueClassLabel.tolist()
        # update_metrics
        for metric in metrics:
            values = [[],[]]
            if metric == "true_vs_predict":
                values[0].append(ADCNnet.trueClassLabel)
                values[1].append(ADCNnet.predictedLabel)
                data_buffer.update_data(sample_id = sample_id,
                                        metric_id=metric,
                                        data_id='y_true',
                                        value=values[0]) 
                data_buffer.update_data(sample_id = sample_id,
                                        metric_id=metric,
                                        data_id='y_pred',
                                        value=values[1])  
            elif metric == "accuracy":
                values[0].append(np.mean(Accuracy)/100)
                values[1].append(ADCNnet.accuracy/100)
                data_buffer.update_data(sample_id = sample_id,
                                        metric_id=metric,
                                        data_id='mean',
                                        value=values[0]) 
                data_buffer.update_data(sample_id = sample_id,
                                        metric_id=metric,
                                        data_id='current',
                                        value=values[1])  
            elif metric == "ari":
                values[0].append(np.mean(ARI))
                values[1].append(adjusted_rand_score(current_y_true, current_y_pred))
                data_buffer.update_data(sample_id = sample_id,
                                        metric_id=metric,
                                        data_id='mean',
                                        value=values[0]) 
                data_buffer.update_data(sample_id = sample_id,
                                        metric_id=metric,
                                        data_id='current',
                                        value=values[1])
            elif metric == "nmi":
                values[0].append(np.mean(NMI))
                values[1].append(normalized_mutual_info_score(current_y_true, current_y_pred))
                data_buffer.update_data(sample_id = sample_id,
                                        metric_id=metric,
                                        data_id='mean',
                                        value=values[0]) 
                data_buffer.update_data(metric_id=metric,
                                        data_id='current',
                                        value=values[1]) 
            elif metric == "f1":
                values[0].append(np.mean(F1))
                values[1].append(f1_score(current_y_true, current_y_pred,
                                          average='weighted'))
                data_buffer.update_data(sample_id = sample_id,
                                        metric_id=metric,
                                        data_id='mean',
                                        value=values[0]) 
                data_buffer.update_data(sample_id = sample_id,
                                        metric_id=metric,
                                        data_id='current',
                                        value=values[1]) 
            elif metric == "precision":
                values[0].append(np.mean(Precision))
                values[1].append(precision_score(current_y_true, current_y_pred,
                                                 average='weighted'))
                data_buffer.update_data(sample_id = sample_id,
                                        metric_id=metric,
                                        data_id='mean',
                                        value=values[0]) 
                data_buffer.update_data(metric_id=metric,
                                        data_id='current',
                                        value=values[1]) 
            elif metric == "recall":
                values[0].append(np.mean(Recall))
                values[1].append(recall_score(current_y_true, current_y_pred,
                                              average='weighted'))
                data_buffer.update_data(sample_id = sample_id,
                                        metric_id=metric,
                                        data_id='mean',
                                        value=values[0]) 
                data_buffer.update_data(metric_id=metric,
                                        data_id='current',
                                        value=values[1]) 
            elif metric == "kappa":
                values[0].append(np.mean(Kappa))
                values[1].append(cohen_kappa_score(current_y_true, current_y_pred))
                data_buffer.update_data(sample_id = sample_id,
                                        metric_id=metric,
                                        data_id='mean',
                                        value=values[0]) 
                data_buffer.update_data(metric_id=metric,
                                        data_id='current',
                                        value=values[1]) 
            elif metric == "gmean":
                values[0].append(np.mean(Gmean))
                values[1].append(fowlkes_mallows_score(current_y_true,current_y_pred))
                data_buffer.update_data(sample_id = sample_id,
                                        metric_id=metric,
                                        data_id='mean',
                                        value=values[0]) 
                data_buffer.update_data(sample_id = sample_id,
                                        metric_id=metric,
                                        data_id='current',
                                        value=values[1])    
        
        
        if show_plot and visualizer is not None:
            visualizer.on_new_train_step(sample_id,data_buffer)


        # if iBatch > 0:
        Y_pred = Y_pred + ADCNnet.predictedLabel.tolist()
        Y_true = Y_true + ADCNnet.trueClassLabel.tolist()
        # if iBatch > 0:
        # calculate performance
        Accuracy.append(ADCNnet.accuracy)
        ARI.append(adjusted_rand_score(current_y_true, current_y_pred))
        NMI.append(normalized_mutual_info_score(current_y_true, current_y_pred))
        F1.append(f1_score(current_y_true, current_y_pred,
                           average='weighted'))
        Precision.append(precision_score(current_y_true, current_y_pred,
                                         average='weighted'))
        Recall.append(recall_score(current_y_true, current_y_pred,
                                   average='weighted'))
        Kappa.append(cohen_kappa_score(current_y_true, current_y_pred))
        Gmean.append(fowlkes_mallows_score(current_y_true,current_y_pred))

        AccuracyHistory.append(ADCNnet.accuracy)
        testingTime.append(ADCNnet.testingTime)
        trainingTime.append(training_time)
        # print("Accuracy : "),
        # print(ADCNnet.accuracy)
        # testingLoss.append(ADCNnet.testingLoss)
        
        # calculate network evolution
        nHiddenLayer.append(ADCNnet.nHiddenLayer)
        nHiddenNode.append(ADCNnet.nHiddenNode)
        nCluster.append(ADCNnet.nCluster)

        nHiddenLayerHistory.append(ADCNnet.nHiddenLayer)
        nHiddenNodeHistory.append(ADCNnet.nHiddenNode)
        nClusterHistory.append(ADCNnet.nCluster)

        Iter.append(iBatch + i + 1)

        #if iBatch%10 == 0 or iBatch == 0:
        #    print('Accuracy: ',np.mean(Accuracy))

        update_progress_bar(iBatch,dataStreams.nBatch,20)
    
    #print('\n')
    #print('=== Performance result ===')
    #print('Accuracy: ',np.mean(Accuracy),'(+/-)',np.std(Accuracy))
    #print('ARI: ',adjusted_rand_score(Y_true, Y_pred))
    #print('NMI: ',normalized_mutual_info_score(Y_true, Y_pred))
    #print('F1 score: ',f1_score(Y_true, Y_pred, average='weighted'))
    #print('Precision: ',precision_score(Y_true, Y_pred, average='weighted'))
    #print('Recall: ',recall_score(Y_true, Y_pred, average='weighted'))
    #print('Testing Time: ',np.mean(testingTime),'(+/-)',np.std(testingTime))
    #print('Training Time: ',np.mean(trainingTime) + initialization_time,'(+/-)',np.std(trainingTime))
    #print('Total Time: ',np.sum(trainingTime) + initialization_time)
    # print('Testing Loss: ',np.mean(testingLoss),'(+/-)',np.std(testingLoss))
    
    # print('\n')
    # print('=== Average network evolution ===')
    # print('Number of layer: ',np.mean(nHiddenLayer),'(+/-)',np.std(nHiddenLayer))
    # print('Total hidden node: ',np.mean(nHiddenNode),'(+/-)',np.std(nHiddenNode))
    # print('Number of cluster: ',np.mean(nCluster),'(+/-)',np.std(nCluster))

    # print('\n')
    # print('=== Final network structure ===')
    # ADCNnet.getNetProperties()
    
    # print('\n')f1_score
    # print('=== Precision Recall ===')
    # print(classification_report(Y_true, Y_pred))
    
    # 0: accuracy
    # 1: ARI
    # 2: NMI
    # 3: f1_score
    # 4: precision_score
    # 5: recall_score
    # 6: training_time
    # 7: testingTime
    # 8: nHiddenLayer
    # 9: nHiddenNode
    # 10: nCluster

    allPerformance = [
        np.mean(Accuracy),
        adjusted_rand_score(Y_true, Y_pred),
        normalized_mutual_info_score(Y_true, Y_pred),
        f1_score(Y_true, Y_pred, average='weighted'),
        precision_score(Y_true, Y_pred, average='weighted'),
        recall_score(Y_true, Y_pred, average='weighted'),
        (np.mean(trainingTime) + initialization_time),np.mean(testingTime),
        cohen_kappa_score(Y_true, Y_pred),
        fowlkes_mallows_score(Y_true,Y_pred),
        ADCNnet.nHiddenLayer,
        ADCNnet.nHiddenNode,
        ADCNnet.nCluster
    ]

    performanceHistory = [
        Iter,
        AccuracyHistory,
        nHiddenLayerHistory,
        nHiddenNodeHistory,
        nClusterHistory
    ]

    
    visualizer.hold()
    return ADCNnet, performanceHistory, allPerformance

# ============================= Multi Task Learning =============================
def ADCNmainMT(ADCNnet, dataStreams,
               nLabeled = 1, layerGrowing = True,
               nodeEvolution = True, clusterGrowing = True,
               lwfLoss = True, clusteringLoss = True,
               trainingBatchSize = 16, noOfEpoch = 1,
               device = torch.device('cpu')):
    # for multi task learning
    # random seed control
    # np.random.seed(0)
    # torch.manual_seed(0)
    # random.seed(0)
    
    # performance metrics
    # performanceMetrics = meanStd()   # [accuracy,testingLoss,testingTime,trainingTime]
    Accuracy = []
    testingTime = []
    trainingTime = []
    # testingLoss  = []

    prevBatchData = []
    
    # multi task
    currTask = 0
    prevTask = 0
    postTaskAcc = []
    preTaskAcc = []
    
    Y_pred = []
    Y_true = []
    Iter = []

    # for figure
    AccuracyHistory = []
    nHiddenLayerHistory = []
    nHiddenNodeHistory = []
    nClusterHistory = []
    
    # network evolution
    # netEvolution = meanStdCalculator()   # [nHiddenNode,nHiddenLayer]
    nHiddenNode = []
    nHiddenLayer = []
    nCluster = []
    layerCount = 0
    
    # initiate network to handle the new task, trained on the initial data in the current task
    start_initialization_train = time.time()
    ADCNnet.initialization(dataStreams.labeledData[currTask], layerCount, 
                            batchSize = trainingBatchSize, device = device)

    end_initialization_train = time.time()
    initialization_time = end_initialization_train - start_initialization_train

    # collection of labeled data
    if nLabeled == 1:
        labeledData = dataStreams.labeledData[currTask]
        labeledLabel = dataStreams.labeledLabel[currTask]
    elif nLabeled < 1:
        # reduced labeled data
        labeledData, labeledLabel = reduceLabeledData(dataStreams.labeledData[currTask].clone(), 
                                                      dataStreams.labeledLabel[currTask].clone(), 
                                                      nLabeled)
        print('Number of initial allegiance data: ', labeledData.shape[0])
    

    for i in range(len(ADCNnet.hiddenNodeHist)):
        Iter.append(i)
        nHiddenLayerHistory.append(ADCNnet.nHiddenLayer)
        AccuracyHistory.append(0)

    nHiddenNodeHistory = ADCNnet.hiddenNodeHist
    nClusterHistory = ADCNnet.clusterHistory

    ## batch loop, handling unlabeled samples
    # training is conducted with single epoch. The initialization of a new layer uses epoch.
    # bar = progressbar.ProgressBar(max_value=dataStreams.nBatch)
    # bar = progressbar.ProgressBar()
    
    batchIdx = 0
    for iBatch in range(0, dataStreams.nBatch):
        currTask = dataStreams.taskIndicator[iBatch]

        # update
        start_train = time.time()

        if currTask != prevTask and currTask > prevTask:
            batchIdx = 0

            # store previous task model
            ADCNnet.storeOldModel(prevTask)

            # test on the prev task before entering curr task. For calculating BWT. 
            prevBatchData = dataStreams.unlabeledDataTest[prevTask]
            prevBatchLabel = dataStreams.unlabeledLabelTest[prevTask]

            ADCNnet.testing(prevBatchData, prevBatchLabel)

            postTaskAcc.append(ADCNnet.accuracy)

            # test on the current task after finishing prev task. For calculating FWT.
            currBatchData = dataStreams.unlabeledDataTest[currTask]
            currBatchLabel = dataStreams.unlabeledLabelTest[currTask]

            # update allegiance
            ADCNnet.updateAllegiance(labeledData, labeledLabel, randomTesting = True)

            ADCNnet.testing(currBatchData, currBatchLabel)

            preTaskAcc.append(ADCNnet.accuracy)

            # initiate network to handle the new task, trained on the initial data in the current task
            ADCNnet.fitCL(dataStreams.labeledData[currTask],
                          reconsLoss = True, epoch = 50,
                          unlabeled = False)

            # augment the collection of unlabeled samples ***************
            if nLabeled == 1:
                labeledData  = torch.cat((labeledData,dataStreams.labeledData[currTask]),0)
                labeledLabel = torch.cat((labeledLabel,dataStreams.labeledLabel[currTask]),0)
            elif nLabeled < 1:
                reducedData, reducedLabel = reduceLabeledData(dataStreams.labeledData[currTask].clone(), 
                                                              dataStreams.labeledLabel[currTask].clone(),
                                                              nLabeled)
                labeledData  = torch.cat((labeledData,reducedData),0)
                labeledLabel = torch.cat((labeledLabel,reducedLabel),0)
                print('Number of newly added allegiance data: ', reducedData.shape[0])

        # load data
        batchIdx   = batchIdx + 1
        batchData  = dataStreams.unlabeledData[currTask][(batchIdx-1)*dataStreams.batchSize:batchIdx*dataStreams.batchSize]
        batchLabel = dataStreams.unlabeledLabel[currTask][(batchIdx-1)*dataStreams.batchSize:batchIdx*dataStreams.batchSize]

        print(batchIdx,'-th batch',currTask,'-th task')

        if iBatch > 0 and layerGrowing:
            # if batchData.shape[0] == 0:
            #     continue

            # drift detection
            ADCNnet.driftDetection(batchData, previousBatchData)

            if ADCNnet.driftStatus == 2:
                # grow layer if drift is confirmed driftStatus == 2
                ADCNnet.layerGrowing()
                layerCount += 1

                # initialization phase
                # need to augment data from previous task also?
                ADCNnet.initialization(dataStreams.labeledData[currTask],
                                       layerCount, 
                                       batchSize = trainingBatchSize,
                                       device = device)

                # ADCNnet.fitEndtoEnd(batchData)

        # training data preparation
        previousBatchData = batchData.clone()
        batchData, batchLabel = ADCNnet.trainingDataPreparation(batchData, batchLabel)

        # training
        if ADCNnet.driftStatus == 0 or ADCNnet.driftStatus == 2:  # only train if it is stable or drift
            ADCNnet.fit(batchData, epoch = noOfEpoch)
            ADCNnet.updateNetProperties()

            # multi task training
            if len(ADCNnet.ADCNold) > 0 and ADCNnet.regStrLWF != 0.0:
                ADCNnet.fitCL(batchData)

            # update allegiance
            ADCNnet.updateAllegiance(labeledData, labeledLabel)

        end_train = time.time()
        training_time = end_train - start_train

        # testing
        ADCNnet.testing(batchData, batchLabel)
        # if iBatch > 0:
        Y_pred = Y_pred + ADCNnet.predictedLabel.tolist()
        Y_true = Y_true + ADCNnet.trueClassLabel.tolist()

        prevTask = dataStreams.taskIndicator[iBatch]

        Accuracy.append(ADCNnet.accuracy)
        AccuracyHistory.append(ADCNnet.accuracy)
        testingTime.append(ADCNnet.testingTime)
        trainingTime.append(training_time)

        # calculate performance
        if iBatch%10 == 0 or iBatch == 0:
            print('Accuracy: ',np.mean(Accuracy))
        # testingLoss.append(ADCNnet.testingLoss)
        
        # calculate network evolution
        nHiddenLayer.append(ADCNnet.nHiddenLayer)
        nHiddenNode.append(ADCNnet.nHiddenNode)
        nCluster.append(ADCNnet.nCluster)

        nHiddenLayerHistory.append(ADCNnet.nHiddenLayer)
        nHiddenNodeHistory.append(ADCNnet.nHiddenNode)
        nClusterHistory.append(ADCNnet.nCluster)

        Iter.append(iBatch + i + 1)

        # bar.update(iBatch+1)
    
    # final test, all tasks, except the last task. For calculating BWT
    allTaskAccuracies = []
    Y_predTasks       = []
    Y_trueTasks       = []
    for iTask in range(len(dataStreams.unlabeledData)-1):
        ADCNnet.testing(dataStreams.unlabeledDataTest[iTask],
                        dataStreams.unlabeledLabelTest[iTask])
        allTaskAccuracies.append(ADCNnet.accuracy)

        Y_predTasks = Y_predTasks + ADCNnet.predictedLabel.tolist()
        Y_trueTasks = Y_trueTasks + ADCNnet.trueClassLabel.tolist()

    BWT = 1/(dataStreams.nTask-1)*(np.sum(allTaskAccuracies)-np.sum(postTaskAcc))

    # test on the last task
    ADCNnet.testing(dataStreams.unlabeledDataTest[len(dataStreams.unlabeledData)-1], 
                    dataStreams.unlabeledLabelTest[len(dataStreams.unlabeledData)-1])
    allTaskAccuracies.append(ADCNnet.accuracy)

    # test with random initialization. For calculating FWT.
    b_matrix = []
    
    for iTask in range(1, len(dataStreams.unlabeledData)):
        ADCNnet.randomTesting(dataStreams.unlabeledDataTest[iTask],
                              dataStreams.unlabeledLabelTest[iTask])
        b_matrix.append(ADCNnet.accuracy)

    FWT = 1/(dataStreams.nTask-1)*(np.sum(preTaskAcc)-np.sum(b_matrix))

    print('\n')
    print('=== Performance result ===')
    print('Prequential Accuracy: ',np.mean(Accuracy),'(+/-)',np.std(Accuracy))
    print('Prequential F1 score: ',f1_score(Y_true, Y_pred, average='weighted'))
    print('Prequential ARI: ',adjusted_rand_score(Y_true, Y_pred))
    print('Prequential NMI: ',normalized_mutual_info_score(Y_true, Y_pred))
    print('Mean Task Accuracy: ',np.mean(allTaskAccuracies),'(+/-)',np.std(allTaskAccuracies))
    print('All Task Accuracy: ',allTaskAccuracies)
    print('Post Task Accuracy: ',postTaskAcc)       # test results on the prev task before entering curr task.
    print('Pre Task Accuracy: ',preTaskAcc)         # test results on the current task after finishing prev task.
    print('B Matrix: ',b_matrix)         # test results on the current task after finishing prev task.
    # print('F1 score: ',f1_score(Y_true, Y_pred, average='weighted'))
    # print('Precision: ',precision_score(Y_true, Y_pred, average='weighted'))
    # print('Recall: ',recall_score(Y_true, Y_pred, average='weighted'))
    print('BWT: ',BWT)
    print('FWT: ',FWT)
    print('Testing Time: ',np.mean(testingTime),'(+/-)',np.std(testingTime))
    print('Training Time: ',np.mean(trainingTime) + initialization_time,'(+/-)',np.std(trainingTime))
    print('Total Time: ',np.sum(trainingTime) + initialization_time)
    # print('Testing Loss: ',np.mean(testingLoss),'(+/-)',np.std(testingLoss))
    
    print('\n')
    print('=== Average network evolution ===')
    print('Number of layer: ',np.mean(nHiddenLayer),'(+/-)',np.std(nHiddenLayer))
    print('Total hidden node: ',np.mean(nHiddenNode),'(+/-)',np.std(nHiddenNode))
    print('Number of cluster: ',np.mean(nCluster),'(+/-)',np.std(nCluster))

    print('\n')
    print('=== Final network structure ===')
    ADCNnet.getNetProperties()

    # 0: accuracy
    # 1: all tasks accuracy
    # 2: BWT
    # 3: FWT
    # 4: ARI
    # 5: NMI
    # 6: f1_score
    # 7: precision_score
    # 8: recall_score
    # 9: training_time
    # 10: testingTime
    # 11: nHiddenLayer
    # 12: nHiddenNode
    # 13: nCluster

    allPerformance = [
        np.mean(Accuracy),
        np.mean(allTaskAccuracies),
        BWT,
        FWT,
        adjusted_rand_score(Y_true, Y_pred),
        normalized_mutual_info_score(Y_true, Y_pred),
        f1_score(Y_true, Y_pred, average='weighted'),
        precision_score(Y_true, Y_pred, average='weighted'),
        recall_score(Y_true, Y_pred, average='weighted'),
        (np.mean(trainingTime) + initialization_time),
        np.mean(testingTime),
        ADCNnet.nHiddenLayer,
        ADCNnet.nHiddenNode,
        ADCNnet.nCluster
    ]
    
    # print('\n')f1_score
    # print('=== Precision Recall ===')
    # print(classification_report(Y_true, Y_pred))

    performanceHistory = [
        Iter,
        AccuracyHistory,
        nHiddenLayerHistory,
        nHiddenNodeHistory,
        nClusterHistory
    ]

    return ADCNnet, performanceHistory, allPerformance

    