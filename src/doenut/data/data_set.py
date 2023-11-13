from typing import Type, List

import pandas as pd

from doenut.data.modifiers.column_selector import ColumnSelector
from doenut.data.modifiers.duplicate_averager import DuplicateAverager
from doenut.data.modifiers.duplicate_remover import DuplicateRemover
from doenut.data.modifiers.ortho_scaler import OrthoScaler
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from doenut.data.modifiers.data_set_modifier import DataSetModifier


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
            ColumnSelector,
            input_selector=input_selector,
            response_selector=response_selector,
        )
        return self

    def scale(self) -> "DataSet":
        self.add_modifier(OrthoScaler)
        return self

    def drop_duplicates(self) -> "DataSet":
        self.add_modifier(DuplicateRemover)
        return self

    def average_duplicates(self) -> "DataSet":
        self.add_modifier(DuplicateAverager)
        return self
