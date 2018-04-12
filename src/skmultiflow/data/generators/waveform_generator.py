import numpy as np
from skmultiflow.data.base_stream import Stream
from skmultiflow.core.utils.validation import check_random_state


class WaveformGenerator(Stream):
    """ WaveformGenerator

    Generates instances with 21 numeric attributes and 3 targets, based 
    on a random differentiation of some base waveforms. Supports noise 
    addition, but in this case the generator will have 40 attribute 
    instances
     
    Parameters
    ----------
    seed: int
        Seed for random generation of instances (Default: 23)
    add_noise: bool
        Add noise (Default: False)
        
    Examples
    --------
    >>> # Imports
    >>> from skmultiflow.data.generators.waveform_generator import WaveformGenerator
    >>> # Setting up the stream
    >>> stream = WaveformGenerator(seed=774, add_noise=True)
    >>> stream.prepare_for_use()
    >>> # Retrieving one sample
    >>> stream.next_sample()
    (array([[ -3.87277692e-03,   5.35892351e-01,  -6.07354638e-02,
          1.70731601e+00,   5.34361689e-01,  -1.77051944e-01,
          1.14121806e+00,   1.35608518e-01,   1.41239266e+00,
          3.54064724e+00,   3.07032776e+00,   4.51698567e+00,
          4.68043133e+00,   3.56075018e+00,   3.83788037e+00,
          2.71987164e+00,   4.77706723e-01,   2.12187988e+00,
          1.59313816e+00,  -5.11689592e-01,   5.99317674e-01,
          2.14508816e-01,  -1.05534090e+00,  -1.34679419e-01,
          5.32610078e-01,  -1.39251668e+00,   1.13220325e+00,
          3.04748552e-01,   1.41454012e+00,   6.73651106e-01,
          1.85981832e-01,  -1.76774471e+00,   3.31777766e-02,
          8.17011922e-02,   1.70686324e+00,   1.10471095e+00,
         -5.08874759e-01,   4.16279862e-01,  -4.26805543e-01,
          9.94596567e-01]]), array([ 2.]))
    >>> # Retrieving 2 samples
    >>> stream.next_sample(2)
    (array([[ -6.72385828e-01,   1.51039782e+00,   5.64599422e-01,
          2.77481410e+00,   2.27653785e+00,   4.40016488e+00,
          3.87856303e+00,   4.90321750e+00,   4.40651078e+00,
          5.07337409e+00,   3.23727692e+00,   2.99724461e+00,
          1.46384329e+00,   1.30042173e+00,   3.67083253e-02,
          3.80546239e-01,  -2.05337011e+00,   6.06889589e-01,
         -1.10649679e+00,   3.38098465e-01,  -8.33683681e-01,
         -3.35283052e-02,  -6.65394037e-01,  -1.09290608e+00,
          4.15778821e-01,   3.64210364e-01,   1.18695445e+00,
          2.77980322e-01,   8.69224059e-01,  -4.93428014e-01,
         -1.08745643e+00,  -9.80906438e-01,   4.12116697e-01,
          2.39579703e-01,   1.53145126e+00,   6.26022691e-01,
          9.82996997e-02,   8.33911055e-01,   8.55830752e-02,
          1.54462877e+00],
       [  3.34017183e-01,  -5.00919347e-01,   2.67311051e+00,
          3.23473039e+00,   2.04091185e+00,   5.62868585e+00,
          4.79611194e+00,   4.14500688e+00,   5.76366418e+00,
          4.18105491e+00,   4.73064582e+00,   3.03461230e+00,
          1.79417942e+00,  -9.84100765e-01,   1.34212863e+00,
          1.29337991e-01,   6.08571939e-01,  -8.56504577e-01,
          2.95358587e-01,   9.12880505e-01,   2.88118824e-01,
         -4.49398914e-01,   5.44025828e-03,  -1.78535212e+00,
          1.41541455e-01,  -6.91216596e-01,  -8.66808201e-02,
         -1.27541907e-01,  -5.38038710e-01,  -1.19807563e+00,
          1.03113317e+00,   2.39999025e-01,   5.24084853e-02,
          1.04314518e+00,   3.20412032e+00,   1.26117112e+00,
         -7.10479419e-01,   4.60132194e-01,  -5.63210805e-02,
         -1.56883271e-01]]), array([ 1.,  1.]))
    >>> # Generators will have infinite remaining instances, so it returns -1
    >>> stream.n_remaining_samples()
    -1
    >>> stream.has_more_samples()
    True
        
    """
    
    NUM_CLASSES = 3
    NUM_BASE_ATTRIBUTES = 21
    TOTAL_ATTRIBUTES_INCLUDING_NOISE = 40
    H_FUNCTION = np.array([[0, 1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1, 0],
                           [0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1, 0, 0, 0, 0, 0]])

    def __init__(self, seed=None, add_noise=False):
        super().__init__()
        self.seed = seed
        self.random_state = None
        self.add_noise = add_noise
        self.n_num_features = self.NUM_BASE_ATTRIBUTES
        self.n_classes = self.NUM_CLASSES
        self.n_targets = 1
        self.n_samples = 0
        self.n_features = self.n_num_features

        self.__configure()

    def __configure(self):
        self.n_samples = 100000
        self.n_features = self.TOTAL_ATTRIBUTES_INCLUDING_NOISE if self.add_noise else self.NUM_BASE_ATTRIBUTES
        if self.add_noise:
            self.n_num_features = self.TOTAL_ATTRIBUTES_INCLUDING_NOISE
        self.n_classes = self.NUM_CLASSES
        self.features_labels = ["att_num_" + str(i) for i in range(self.n_features)]
        self.outputs_labels = ["class"]

    def prepare_for_use(self):
        self.restart()

    def n_remaining_samples(self):
        return -1

    def has_more_samples(self):
        return True

    def next_sample(self, batch_size=1):
        """ next_sample
        
        An instance is generated based on the parameters passed. If noise 
        is included the total number of attributes will be 40, if it's not 
        included there will be 21 attributes. In both cases there is one 
        classification task, which chooses one between three labels.
        
        After the number of attributes is chosen, the algorithm will randomly
        choose one of the hard coded waveforms, as well as random multipliers. 
        For each attribute, the actual value generated will be a a combination 
        of the hard coded functions, with the multipliers and a random value.
        
        Furthermore, if noise is added the attributes from 21 to 40 will be 
        replaced with a random normal value.
        
        Parameters
        ----------
        batch_size: int
            The number of samples to return.
            
        Returns
        -------
        tuple or tuple list
            Return a tuple with the features matrix and the labels matrix 
            for the batch_size samples that were requested.
        
        """
        if self.has_noise():
            data = np.zeros([batch_size, self.TOTAL_ATTRIBUTES_INCLUDING_NOISE + 1])
        else:
            data = np.zeros([batch_size, self.NUM_BASE_ATTRIBUTES + 1])

        for j in range(batch_size):
            self.sample_idx += 1
            waveform = self.random_state.randint(0, self.NUM_CLASSES)
            choice_a = 1 if (waveform == 2) else 0
            choice_b = 1 if (waveform == 0) else 2
            multiplier_a = self.random_state.rand()
            multiplier_b = 1.0 - multiplier_a

            for i in range(self.NUM_BASE_ATTRIBUTES):
                data[j, i] = multiplier_a*self.H_FUNCTION[choice_a][i] \
                            + multiplier_b*self.H_FUNCTION[choice_b][i] \
                            + self.random_state.normal()

            if self.has_noise():
                for i in range(self.NUM_BASE_ATTRIBUTES, self.TOTAL_ATTRIBUTES_INCLUDING_NOISE):
                    data[j, i] = self.random_state.normal()

            data[j, data[j].size-1] = waveform
        self.current_sample_x = data[:, :self.n_features]
        self.current_sample_y = np.ravel(data[:, self.n_features:])

        return self.current_sample_x, self.current_sample_y

    def is_restartable(self):
        return True

    def restart(self):
        self.random_state = None
        self.random_state = check_random_state(self.seed)
        self.sample_idx = 0

    def has_noise(self):
        return self.add_noise

    def get_n_cat_features(self):
        return self.n_cat_features

    def get_n_num_features(self):
        return self.n_num_features

    def get_n_features(self):
        return self.n_num_features

    def get_n_targets(self):
        return self.n_targets

    def get_feature_names(self):
        return self.features_labels

    def get_target_names(self):
        return self.outputs_labels

    def last_sample(self):
        return self.current_sample_x, self.current_sample_y

    def get_name(self):
        return "Waveform Generator - {} target, {} classes".format(self.n_targets, self.n_classes)

    def get_targets(self):
        return [i for i in range(self.n_classes)]

    def get_info(self):
        add_noise = 'True' if self.add_noise else 'False'
        return 'Waveform Generator: n_classes: ' + str(self.n_classes) + \
               '  -  n_num_features: ' + str(self.n_num_features) + \
               '  -  n_cat_features: ' + str(self.n_cat_features) + \
               '  -  add_noise: ' + add_noise + \
               '  -  sample_seed: ' + str(self.seed)
