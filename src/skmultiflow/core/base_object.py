from abc import ABCMeta, abstractmethod


class BaseObject(metaclass=ABCMeta):
    """ BaseObject
    
    The most basic object, from which classes in scikit-multiflow
    derive from. It guarantees that all classes have at least the
    two basic functions described in this base class.
    
    """

    @abstractmethod
    def get_class_type(self):
        """ get_class_type
        
        The class type is a string that identifies the type of object 
        generated by that module.
        
        Returns
        -------
        The class type
        
        """
        raise NotImplementedError

    @abstractmethod
    def get_info(self):
        """ get_info
        
        A sum up of all important characteristics of a class. 
        
        The default format of the return string is as follows: 
        ClassName: attribute_one: value_one - attribute_two: value_two \ 
        - info_one: info_one_value
        
        Returns
        -------
        string
        A string with the class' relevant information.
        
        """
        raise NotImplementedError
