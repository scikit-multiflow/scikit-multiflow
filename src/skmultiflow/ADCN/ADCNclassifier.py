from skmultiflow.ADCN.ADCN_process.ADCNbasic import ADCN
from skmultiflow.ADCN.ADCN_process.model import simpleMPL, singleMPL
import numpy as np
import pdb
import torch
import random
from torchvision import datasets, transforms
import numpy as np

device = torch.device('cuda')
class ADCNclassifier(object):
    def __init__(self,stream,
                 HidNodeMul = 4,
                 ExtraFeatureMul = 4,
                 FeatureClusMul = 2):
        self.nInput = stream.n_features -1
        self.nHidNodeExtractor = self.nInput*HidNodeMul
        self.nExtractedFeature = self.nInput*ExtraFeatureMul
        self.nFeaturClustering = self.nInput*FeatureClusMul
        self.nOutput = len(stream.target_values)
        self.classes = stream.target_values

    """ Autonomous Deep Clustering Network classifier.

    A classifier with unsupervised learning approach. The classifier that maximize 
    feature extractor and deep network approach. The deep network is crafted with 
    a self-evolving property. Both network width and depth are self-generated.
    To classify, the self-clustering measure is performed in the deep embedding 
    space where every layer of the network produces their own set of clusters. 
    The unsupervised learning approach neglect the labeling cost and effort in 
    implementation.

    Parameters
    ----------
    stream: <class skmultiflow.data> 
        | Data used to initialize the network
    HidNodeMul : int (default=4)
        | Multiplier of input feature to setup hidden node of the feature extractor
    ExtraFeatureMul : int (default=4)
        | Multiplier of input feature to setup feature in the hidden node of the feature extractor
    FeatureClusMul : int (default=2)
        | Initial clustering multiplier 

    Examples
    --------
    >>> # Imports
    >>> from skmultiflow.data import HyperplaneGenerator
    >>> from skmultiflow.ADCN import ADCNclassifier
    >>> import numpy as np
    >>>
    >>> # Setup a data stream
    >>> stream = HyperplaneGenerator(random_state=1)
    >>> # Prepare stream for use
    >>>
    >>> # Setup the ADCN classifier
    >>> model = ADCNclassifier(stream)
    >>>
    >>> # Set the evaluator
    >>> 
    >>> evaluator = evaluator = ADCNevaluator(max_samples = 20000, 
                                                pretrain_size = 300
                                                ,batch_size = 1000
                                                ,stream = stream,
                                                stream_type=2)
    >>> # Run evaluation
    >>> evaluator.evaluate(model)
    """
    
    def Initiate(self):
        ADCNnet = ADCN(self.nOutput,
                       nInput = self.nInput,
                       nHiddenNode = self.nFeaturClustering)
        ADCNnet.ADCNcnn = simpleMPL(self.nInput,
                                    nNodes = self.nHidNodeExtractor,
                                    nOutput = self.nExtractedFeature)
        ADCNnet.desiredLabels = self.classes
        return ADCNnet