__author__ = 'Guilherme Matsumoto'

from skmultiflow.data.base_instance_stream import BaseInstanceStream
from skmultiflow.core.utils import pseudo_random_processes as prp
from skmultiflow.core.base_object import BaseObject
import numpy as np


class RandomRBFGenerator(BaseInstanceStream, BaseObject):
    """ RandomRBFGenerator
    
    This generator produces a radial basis function stream.
    
    A number of centroids, having a random central position, a standard 
    deviation, a class label and weight, are generated. A new sample is 
    created by choosing one of the centroids at random, taking into 
    account their weights, and offsetting the attributes at a random 
    direction from the centroid's center. The offset length is drawn 
    from a Gaussian distribution.
      
    This process will create a normally distributed hypersphere of samples 
    on the surrounds of each centroid.
    
    Parameters
    ---------
    model_seed: int (Default: 21)
        The seed to be used by the model random generator.
        
    instance_seed: int (Default: 5)
        The seed to be used by the instance random generator.
        
    num_classes: int (Default: 2)
        The number of class labels to generate.
        
    num_att: int (Default: 10)
        The total number of attributes to generate.
        
    num_centroids: int (Default: 50)
        The total number of centroids to generate.
    
    """

    def __init__(self, model_seed = 21, instance_seed = 5, num_classes = 2, num_att = 10, num_centroids = 50):
        super().__init__()

        #default values
        self.num_numerical_attributes = 10
        self.num_nominal_attributes = 0
        self.num_values_per_nominal_att = 0
        self.current_instance_x = None
        self.current_instance_y = None
        self.model_seed = 21
        self.instance_seed = 5
        self.num_classes = 2
        self.num_centroids = 50
        self.centroids = None
        self.centroid_weights = None
        self.instance_random = None
        self.attributes_header = None
        self.class_header = None

        self.__configure(model_seed, instance_seed, num_classes, num_att, num_centroids)

    def __configure(self, model_seed, instance_seed, num_classes, num_att, num_centroids):
        self.model_seed = model_seed
        self.instance_seed = instance_seed
        self.num_classes = num_classes
        self.num_numerical_attributes = num_att
        self.num_centroids = num_centroids
        self.instance_random = np.random
        self.instance_random.seed(self.instance_seed)

        self.class_header = ["class"]
        self.attributes_header = []
        for i in range(self.num_numerical_attributes):
            self.attributes_header.append("NumAtt" + str(i))

    def estimated_remaining_instances(self):
        return -1

    def has_more_instances(self):
        return True

    def next_instance(self, batch_size = 1):
        """ next_instance
        
        Return batch_size samples generated by choosing a centroid at 
        random and randomly offsetting its attributes so that it is 
        placed inside the hypersphere of that centroid.
        
        Parameters
        ----------
        batch_size: int
            The number of samples to return.
        
        Returns
        -------
        Return a tuple with the features matrix and the labels matrix for 
        the batch_size samples that were requested. 
        
        """
        data = np.zeros([batch_size, self.num_numerical_attributes + 1])
        num_atts = self.num_numerical_attributes
        for j in range(batch_size):
            centroid_aux = self.centroids[prp.random_index_based_on_weights(self.centroid_weights, self.instance_random)]
            att_vals = []
            magnitude = 0.0
            for i in range(num_atts):
                att_vals.append((self.instance_random.rand() * 2.0) - 1.0)
                magnitude += att_vals[i]*att_vals[i]
            magnitude = np.sqrt(magnitude)
            desired_mag = self.instance_random.normal() * centroid_aux.std_dev
            scale = desired_mag/magnitude
            for i in range(num_atts):
                data[j, i] = centroid_aux.centre[i] + att_vals[i]*scale
            data[j, num_atts] = centroid_aux.class_label
        return (data[:, :num_atts], data[:, num_atts:])

    def prepare_for_use(self):
        self.restart()

    def is_restartable(self):
        return True

    def restart(self):
        self.generate_centroids()
        self.instance_random.seed(self.instance_seed)

    def has_more_mini_batch(self):
        return True

    def get_num_nominal_attributes(self):
        return self.num_nominal_attributes

    def get_num_numerical_attributes(self):
        return self.num_numerical_attributes

    def get_num_values_per_nominal_attribute(self):
        return self.num_values_per_nominal_att

    def get_num_attributes(self):
        return self.num_numerical_attributes + (self.num_nominal_attributes * self.num_values_per_nominal_att)

    def get_num_targets(self):
        return self.num_classes

    def get_attributes_header(self):
        return self.attributes_header

    def get_classes_header(self):
        return self.class_header

    def get_last_instance(self):
        return (self.current_instance_x, self.current_instance_y)

    def get_plot_name(self):
        return "Random RBF Generator - " + str(self.num_classes) + " class labels"

    def get_classes(self):
        c = []
        for i in range(self.num_classes):
            c.append(i)
        return c

    def generate_centroids(self):
        """ generate_centroids
        
        Sequentially creates all the centroids, choosing at random a center, 
        a label, a standard deviation and a weight. 
        
        """
        model_random = np.random
        model_random.seed(self.model_seed)
        self.centroids = []
        self.centroid_weights = []
        for i in range (self.num_centroids):
            self.centroids.append(Centroid())
            rand_centre = []
            for j in range(self.num_numerical_attributes):
                rand_centre.append(model_random.rand())
            self.centroids[i].centre = rand_centre
            self.centroids[i].class_label = model_random.randint(self.num_classes)
            self.centroids[i].std_dev = model_random.rand()
            self.centroid_weights.append(model_random.rand())

    def get_info(self):
        return 'RandomRBFGenerator: model_seed: ' + str(self.model_seed) + ' - instance_seed: ' + \
               str(self.instance_seed) + ' - num_classes: ' + str(self.num_classes) + ' - num_att: ' \
               + str(self.num_numerical_attributes) + ' - num_centroids: ' + str(self.num_centroids)

    def get_num_targeting_tasks(self):
        return 1

class Centroid:
    """ Centroid
    
    Class that stores a centroid's attributes. No further methods.
    
    """
    def __init__(self):
        self.centre = None
        self.class_label = None
        self.std_dev = None


if __name__ == "__main__":
    rrbfg = RandomRBFGenerator()
    rrbfg.prepare_for_use()
    for i in range(4):
        X, y = rrbfg.next_instance(4)
        print(X)
        print(y)

