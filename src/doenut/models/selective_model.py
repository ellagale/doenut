from typing import List

import numpy as np
import pandas as pd

from doenut.models.averaged_model import AveragedModel


class SelectiveModel(AveragedModel):
    def __init__(
        self,
        inputs: pd.DataFrame,
        responses: pd.DataFrame,
        scale_data: bool = True,
        scale_run_data: bool = True,
        fit_intercept: bool = True,
        drop_duplicates: bool = True,
        input_selector: List = None,
        response_selector: List = None,
    ) -> None:
        # first set_filter out the input and/or response columns we want
        input_column_list = list(inputs.columns)
        response_column_list = list(responses.columns)

        input_column_indices = []
        if input_selector is None:
            input_column_indices = list(range(len(inputs.columns)))
        else:
            input_sorter = np.argsort(input_column_list)
            input_column_indices = input_sorter[
                np.searchsorted(
                    input_column_list, input_selector, sorter=input_sorter
                )
            ]

        if len(input_column_indices) == 0:
            raise ValueError("No input columns specified")

        if response_selector is None:
            response_selector = responses.columns

        if len(response_selector) == 0:
            raise ValueError("No response columns specified")

        filtered_inputs = inputs.iloc[:, input_column_indices]

        # TODO:: this is effectively how current code behaves. Fix this.
        assert len(response_selector) == 1

        response_column = response_selector[0]
        filtered_responses = responses[[response_column]]

        super().__init__(
            filtered_inputs,
            filtered_responses,
            scale_data,
            scale_run_data,
            fit_intercept,
            response_column,
            drop_duplicates,
            input_selector,
        )
