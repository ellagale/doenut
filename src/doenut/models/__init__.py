"""
DoENUT: models
A model is an approximation of a system, built by fitting various equations to
a dataset.

Currently, this module provides four classes - two model classes and two model
grouping classes.

Model is a basic fitted via linear regression model. It's a fairly thin layer
over sklearn's model and the base for the rest of the code.

AveragedModel is the more useful one. As well as generating a normal model, it
will also generate a set of fitting models via a leave-one-out methodology.
This allows it to calculate the Q2 cross validation correlation coefficient.

ModelSet can be used when you have multiple columns/parameters in your response
data. It will generate one Model for each response column. Similarly,
AveragedModelSet will generate one AveragedModel for each response column.

"""
from .model import Model
from .averaged_model import AveragedModel
from .model_set import ModelSet
from .averaged_model_set import AveragedModelSet
