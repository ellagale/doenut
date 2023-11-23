from abc import abstractmethod, ABC
import pandas as pd


class DataSetModifier(ABC):
    """Parent class for all types of modifier.
    They take a dataset in, perform some form of operation on it and then
    pass it along

    Parameters
    ----------
     inputs : pd.DataFrame
        The dataset's inputs
    responses : pd.DataFrame
        The dataset's responses
    \*\*kwargs : dict, optional
        Any extra arguments needed by individual modifiers.

    Note
    ----
    This is an abstract class and should not be used directly.
    """

    def __init__(
        self, inputs: pd.DataFrame, responses: pd.DataFrame, **kwargs
    ):
        pass

    @abstractmethod
    def apply_to_inputs(self, data: pd.DataFrame) -> pd.DataFrame:
        """Applies the modifier to the inputs of the dataset.

        Parameters
        ----------
        data :  pd.DataFrame
            The input data

        Returns
        -------
        pd.DataFrame:
            The modified input data
        """
        pass

    @abstractmethod
    def apply_to_responses(self, data: pd.DataFrame) -> pd.DataFrame:
        """Applies the modifier to the responses of the dataset.

        Parameters
        ----------
        data :  pd.DataFrame
            The response data

        Returns
        -------
        pd.DataFrame:
            The modified response data
        """
        pass
