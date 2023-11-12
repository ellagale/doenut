from abc import abstractmethod, ABC
import pandas as pd


class DataSetModifier(ABC):
    """
    Parent class for all types of modifier.
    They take a dataset in, perform some form of operation on it and then
    pass it along
    """
    def __init__(self, data: "DataSetModifier" = None):
        self.data = data

    @abstractmethod
    def _parse_inputs(self, data: pd.DataFrame) -> pd.DataFrame:
        pass

    def get_inputs(self) -> pd.DataFrame:
        return self._parse_inputs(self.data.get_inputs())

    @abstractmethod
    def _parse_responses(self, data: pd.DataFrame) -> pd.DataFrame:
        pass

    def get_responses(self) -> pd.DataFrame:
        return self._parse_responses(self.data.get_responses())

    def get_raw_inputs(self) -> pd.DataFrame:
        return self.data.get_raw_inputs()

    def get_raw_responses(self) -> pd.DataFrame:
        return self.get_responses()