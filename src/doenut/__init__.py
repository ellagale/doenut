"""
Doenut

Design of Experiments Numerical Utility Toolkit
"""

from importlib.metadata import version
__version__ = version("doenut")

from . import doenut
from . import designer
from . import plot
from .doenut import *
