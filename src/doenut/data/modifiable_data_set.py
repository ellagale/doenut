import copy
from typing import Type, List

import pandas as pd

from doenut.data.data_set import DataSet
from doenut.data.modifiers.column_selector import ColumnSelector
from doenut.data.modifiers.duplicate_averager import DuplicateAverager
from doenut.data.modifiers.duplicate_remover import DuplicateRemover
from doenut.data.modifiers.ortho_scaler import OrthoScaler
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from doenut.data.modifiers.data_set_modifier import DataSetModifier


class ModifiableDataSet:
    def __init__(self, inputs: pd.DataFrame, responses: pd.DataFrame) -> None:
        if inputs is None or len(inputs) == 0:
            raise ValueError("Inputs must not be empty")
        if responses is None or len(responses) == 0:
            raise ValueError("Responses must not be empty")
        if len(inputs) != len(responses):
            raise ValueError("Inputs and Responses must have the same length")

        # the input values
        self.inputs = inputs
        self.responses = responses
        # the input values post (current) processing
        self._proc_inputs = copy.deepcopy(inputs)
        self._proc_responses = copy.deepcopy(responses)
        self.modifiers = []

    def get(self) -> DataSet:
        return DataSet(self._proc_inputs, self._proc_responses)

    def add_modifier(
        self, modifier: Type["DataSetModifier"], **kwargs
    ) -> "ModifiableDataSet":
        """
        Adds a new modifier to the stack.
        @param modifier: The new modifier to add
        @param kwargs: Any additional arguments the modifier is expecting.
        """
        modifier = modifier(self._proc_inputs, self._proc_responses, **kwargs)
        self.modifiers.append(modifier)
        self._proc_inputs = modifier.apply_to_inputs(self._proc_inputs)
        self._proc_responses = modifier.apply_to_responses(
            self._proc_responses
        )
        return self

    def filter(
        self,
        input_selector: List["str | int"],
        response_selector: List["str | int"] = None,
    ) -> "ModifiableDataSet":
        return self.add_modifier(
            ColumnSelector,
            input_selector=input_selector,
            response_selector=response_selector,
        )

    def scale(self, scale_responses: bool = True) -> "ModifiableDataSet":
        return self.add_modifier(OrthoScaler, scale_responses=scale_responses)

    def drop_duplicates(self) -> "ModifiableDataSet":
        return self.add_modifier(DuplicateRemover)

    def average_duplicates(self) -> "ModifiableDataSet":
        return self.add_modifier(DuplicateAverager)
