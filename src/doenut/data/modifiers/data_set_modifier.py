from abc import abstractmethod, ABC
import pandas as pd

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from doenut.data.modifiable_data_set import ModifiableDataSet


class DataSetModifier(ABC):
    """
    Parent class for all types of modifier.
    They take a dataset in, perform some form of operation on it and then
    pass it along
    """

    def __init__(
        self, inputs: pd.DataFrame, responses: pd.DataFrame, **kwargs
    ):
        """
        Does nothing, but defines the constructor for all other DataSets
        @param inputs: the processed inputs up till this point
        @param responses: the processed responses up till this point
        Use this to do things like check the size and ranges of the dataset.
        @param kwargs: any other arguments the modifier needs
        """
        pass

    @abstractmethod
    def apply_to_inputs(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the modifier to the inputs of the dataset.
        @param data: The input data
        @return: The data post modification.
        """
        pass

    @abstractmethod
    def apply_to_responses(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the modifier to the responses of the dataset.
        @param data: The response data
        @return: The data post modification.
        """
        pass
