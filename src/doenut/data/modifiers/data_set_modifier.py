from abc import abstractmethod, ABC
import pandas as pd

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from doenut.data.data_set import DataSet


class DataSetModifier(ABC):
    """
    Parent class for all types of modifier.
    They take a dataset in, perform some form of operation on it and then
    pass it along
    """

    def __init__(self, dataset: "DataSet", **kwargs):
        """
        Does nothing, but defines the constructor for all other DataSets
        @param dataset: the dataset the modifier is being applied to.
        Use this to do things like check the size and ranges of the dataset.
        @param kwargs: any other arguments the modifier needs
        """
        pass

    @abstractmethod
    def apply_to_inputs(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the modifier to the inputs of the dataset.
        If the data needs to be changed, a deep copy should be made.
        @param data: The input data
        @return: The data post modification. If any values are being directly
        modified, a copy should be made.
        """
        pass

    @abstractmethod
    def apply_to_responses(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the modifier to the responses of the dataset.
        If the data needs to be changed, a deep copy should be made.
        @param data: The response data
        @return: The data post modification. If any values are being changed,
        a copy should be made.
        """
        pass
