from typing import Type, List

import pandas as pd

from doenut.data.data_set_filter import DataSetFilter
from doenut.data.data_set_scaler import DataSetScaler
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from doenut.data.data_set_modifier import DataSetModifier


class DataSet:
    def __init__(self, inputs: pd.DataFrame, responses: pd.DataFrame) -> None:
        super().__init__()
        if inputs is None or len(inputs) == 0:
            raise ValueError("Inputs must not be empty")
        if responses is None or len(responses) == 0:
            raise ValueError("Responses must not be empty")
        if len(inputs) != len(responses):
            raise ValueError("Inputs and Responses must have the same length")

        self.inputs = inputs
        self.responses = responses
        self.modifiers = []

    def get_inputs(self) -> pd.DataFrame:
        results = self.inputs
        for modifier in self.modifiers:
            results = modifier.apply_to_inputs(results)
        return results

    def get_responses(self) -> pd.DataFrame:
        results = self.responses
        for modifier in self.modifiers:
            results = modifier.apply_to_responses(results)
        return results

    def get_raw_inputs(self) -> pd.DataFrame:
        return self.inputs

    def get_raw_responses(self) -> pd.DataFrame:
        return self.responses

    def add_modifier(
        self, modifier: Type["DataSetModifier"], **kwargs
    ) -> None:
        self.modifiers.append(modifier(self, **kwargs))

    def filter(
        self,
        input_selector: List[str | int],
        response_selector: List[str | int] = None,
    ) -> "DataSet":
        self.add_modifier(
            DataSetFilter,
            input_selector=input_selector,
            response_selector=response_selector,
        )
        return self

    def scale(self) -> "DataSet":
        self.add_modifier(DataSetScaler)
        return self
