from .ifn_Classifier import IfnClassifier
from .IOLIN.Regenerative import OnlineNetworkRegenerative
from .IOLIN.Basic_Incremental import BasicIncremental
from .IOLIN.pure_multiple_model import PureMultiple

from ._version import __version__

__all__ = ['IfnClassifier', 'OnlineNetworkRegenerative', 'BasicIncremental',
           'PureMultiple', '__version__']
