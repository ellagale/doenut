from typing import List, Tuple
import pandas as pd
from doenut.data.modifiers.data_set_modifier import DataSetModifier

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from doenut.data.data_set import DataSet


class ColumnSelector(DataSetModifier):
    """
    DataSet Modifier to remove columns from the dataset
    """

    @classmethod
    def _parse_selector(
        cls, data: pd.DataFrame, selector: List[str | int]
    ) -> Tuple[List[str], List[int]]:
        """
        Internal helper function to take either a list of column names or
        column indices and convert it to the other.
        @param data: The data set the list applies to
        @param selector: The known selector list
        @return: Tuple of the column names and indices as lists
        """
        if isinstance(selector[0], str):  # columns provided
            # First validate it
            for col in selector:
                if col not in data.columns:
                    raise ValueError(f"Data lacks column {col}")
            return selector, [
                i for i, j in enumerate(data.columns) if j in selector
            ]
        elif isinstance(selector[0], int):  # column indices provided
            for idx in selector:
                if idx >= len(data):
                    raise IndexError(f"Index {idx} out of range for data")
            return [data.columns[i] for i in selector], selector

        raise ValueError("Type of selector needs to be string or int")

    def __init__(
        self,
        data: "DataSet",
        input_selector: List[str | int],
        response_selector: List[str | int] = None,
    ):
        super().__init__(data)

        # Parse / Validate the input selector
        if input_selector is None:
            raise ValueError("Input selector must select at least one column!")
        self.input_selector, self.input_indices = self._parse_selector(
            data.get_inputs(), input_selector
        )

        if response_selector is not None:
            (
                self.response_selector,
                self.response_indices,
            ) = self._parse_selector(data.get_responses(), response_selector)
        else:
            self.response_selector = None
            self.response_indices = None

    def apply_to_inputs(self, data: pd.DataFrame) -> pd.DataFrame:
        if not self.input_indices:
            # Should never happen, but hey
            return data
        return data.iloc[:, self.input_indices]

    def apply_to_responses(self, data: pd.DataFrame) -> pd.DataFrame:
        if not self.response_indices:
            return data
        return data.iloc[:, self.response_indices]
