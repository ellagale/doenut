from typing import List, Tuple
import pandas as pd
from doenut.data.modifiers.data_set_modifier import DataSetModifier


class ColumnSelector(DataSetModifier):
    """
    DataSet Modifier to remove columns from the dataset
    """

    @classmethod
    def _parse_selector(
        cls, data: pd.DataFrame, selector: List["str | int"]
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
        inputs: pd.DataFrame,
        responses: pd.DataFrame,
        input_selector: List["str | int"] = None,
        response_selector: List["str | int"] = None,
    ):
        """
        Used to filter specific columns from a dataset. Note that at least one
        of input_selector and response_selector must be specified.
        The selector should be a list of either column names or column indices
        @param inputs: The dataset's inputs
        @param responses: The dataset's responses
        @param input_selector: A list to filter the inputs by
        @param response_selector: A list to filter the responses by

        """
        super().__init__(inputs, responses)
        # Validate inputs
        if input_selector is None and response_selector is None:
            raise ValueError(
                "At least one of input_selector and response_selector is required."
            )

        # Parse / Validate the input selector
        if input_selector is not None:
            (self.input_selector, self.input_indices) = self._parse_selector(
                inputs, input_selector
            )
        else:
            self.input_selector = None
            self.input_indices = None

        if response_selector is not None:
            (
                self.response_selector,
                self.response_indices,
            ) = self._parse_selector(responses, response_selector)
        else:
            self.response_selector = None
            self.response_indices = None

    def apply_to_inputs(self, data: pd.DataFrame) -> pd.DataFrame:
        if not self.input_indices:
            return data
        return data.iloc[:, self.input_indices]

    def apply_to_responses(self, data: pd.DataFrame) -> pd.DataFrame:
        if not self.response_indices:
            return data
        return data.iloc[:, self.response_indices]
