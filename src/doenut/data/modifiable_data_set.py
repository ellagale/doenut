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
    """
    Typically when doing DoE you will want to apply various modifiers such as
    scaling or filtering of columns to your dataset. ModifiableDataSet is
    DoENUT's mechanism to provide this.

    A base dataset consists of two pandas dataframes, one for the inputs and
    one for the responses. These should be of the same length.

    Once you have built the dataset you can add modifiers to it using
    add_modifier or (more likely) via the helper functions such as filter and
    scale. Finally, get() will then give you a DataSet object containing the
    result of applying all the modifiers.

    Be aware that modifiers are applied in the order they are added, and that
    modifiers cannot be removed once added. ModifiableDataSet makes deep
    copies of the dataframes, so the original data will not get changed.

    All the modifier functions return a link to self, so they can be used as
    per the builder pattern - i.e. so you can write code like:
      dataset = ModifiableDataset(inputs,responses).filter(list).scale()
    """

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
        input_selector: List["str | int"] = None,
        response_selector: List["str | int"] = None,
    ) -> "ModifiableDataSet":
        """
        Select a subset of the columns in this dataset.
        You must specify at least one selector.
        Each select selector can be either a list of column names or indices
        that you wish to keep.
        @param input_selector: Filter for the input data
        @param response_selector: Filter for the response data
        @return: this dataset
        """
        return self.add_modifier(
            ColumnSelector,
            input_selector=input_selector,
            response_selector=response_selector,
        )

    def scale(self, scale_responses: bool = False) -> "ModifiableDataSet":
        """
        Apply an orthographic scaling to the dataset
        i.e. apply a linear scaling so each column is in the range -1...1
        @param scale_responses: Whether to scale the reponse data as well
        @return: this dataset
        """
        return self.add_modifier(OrthoScaler, scale_responses=scale_responses)

    def drop_duplicates(self) -> "ModifiableDataSet":
        """
        Removes all duplicate rows from the dataset. The first instance of
        each duplicate will be kept.
        NOTE: while only the inputs are considered for whether a row is a
        duplicate or now, duplicates will be removed from both inputs and
        responses.
        @return: self
        """
        return self.add_modifier(DuplicateRemover)

    def average_duplicates(self) -> "ModifiableDataSet":
        """
        Removes all duplicate rows from the dataset. The first instance of
        each duplicate will be kept, and it's responses set to the average of
        all the rows that matched it.
        NOTE: only inputs values are considered for whether a row is a
        duplicate or not
        @return: self
        """
        return self.add_modifier(DuplicateAverager)
