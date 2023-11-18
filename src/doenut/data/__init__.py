"""
DoENUT: data
These classes handle the grouping and storage of DataSets for DoE.

A DataSet consists of a set of inputs and a set of responses.
This module provides two classes - a basic DataSet and ModifiableDataSet.

Mostly, you want to use ModifiableDataSet as that allows for the operations
you will need for DoE.
"""
from .modifiable_data_set import ModifiableDataSet
from .data_set import DataSet
