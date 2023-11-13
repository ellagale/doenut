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
        pass

    @abstractmethod
    def apply_to_inputs(self, data: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractmethod
    def apply_to_responses(self, data: pd.DataFrame) -> pd.DataFrame:
        pass
