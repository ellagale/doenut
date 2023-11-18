import numpy as np
import pandas as pd
from typing import Tuple
from doenut.data.modifiers.data_set_modifier import DataSetModifier


class OrthoScaler(DataSetModifier):
    """
    Takes a dataset and scales it per column using an ortho scaling to
    the range -1 ... 1
    By default only inputs are scaled, but this can be overridden.
    """

    @classmethod
    def _compute_scaling(cls, data: pd.DataFrame) -> Tuple[float, float]:
        data_max = np.max(data, axis=0)
        data_min = np.min(data, axis=0)
        mj = (data_min + data_max) / 2
        rj = (data_max - data_min) / 2
        return mj, rj

    def __init__(
        self,
        inputs: pd.DataFrame,
        responses: pd.DataFrame,
        scale_responses: bool = False,
    ) -> None:
        """
        Use this modifier to add a standard ortho scaling to the dataset, so
        it has a range of -1..1
        @param inputs: The inputs of the dataset
        @param responses: The responses of the dataset
        @param scale_responses: Whether to also scale the responses
        """
        super().__init__(inputs, responses)
        self.inputs_mj, self.inputs_rj = self._compute_scaling(inputs)
        self.responses_mj, self.responses_rj = self._compute_scaling(responses)
        self.scale_responses = scale_responses

    def apply_to_inputs(self, data: pd.DataFrame) -> pd.DataFrame:
        return (data - self.inputs_mj) / self.inputs_rj

    def apply_to_responses(self, data: pd.DataFrame) -> pd.DataFrame:
        if not self.scale_responses:
            return data
        return (data - self.responses_mj) / self.responses_rj
