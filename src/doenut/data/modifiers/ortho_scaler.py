import numpy as np
import pandas as pd

from typing import TYPE_CHECKING, Tuple
from doenut.data.modifiers.data_set_modifier import DataSetModifier

if TYPE_CHECKING:
    from doenut.data.modifiable_data_set import ModifiableDataSet


class OrthoScaler(DataSetModifier):
    """
    Takes a dataset and scales it per column using an ortho scaling to
    the range -1 ... 1
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
    ) -> None:
        super().__init__(inputs, responses)
        self.inputs_mj, self.inputs_rj = self._compute_scaling(inputs)
        self.responses_mj, self.responses_rj = self._compute_scaling(responses)

    def apply_to_inputs(self, data: pd.DataFrame) -> pd.DataFrame:
        return (data - self.inputs_mj) / self.inputs_rj

    def apply_to_responses(self, data: pd.DataFrame) -> pd.DataFrame:
        return (data - self.responses_mj) / self.responses_rj
