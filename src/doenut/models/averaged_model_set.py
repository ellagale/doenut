import pandas as pd

from doenut.data import ModifiableDataSet
from doenut.models import ModelSet
from doenut.models import AveragedModel


class AveragedModelSet(ModelSet):
    @classmethod
    def multiple_response_columns(
        cls,
        inputs: pd.DataFrame = None,
        responses: pd.DataFrame = None,
        scale_data: bool = True,
        scale_run_data: bool = True,
        fit_intercept: bool = True,
        drop_duplicates: str = "yes",
        input_selector: list = [],
    ) -> "AveragedModelSet":
        result = AveragedModelSet(
            inputs,
            responses,
            scale_data,
            scale_run_data,
            fit_intercept,
            [],
            drop_duplicates,
            input_selector,
        )
        for column in responses.columns:
            result.add_model(response_key=column)
        return result

    def __init__(
        self,
        default_inputs: pd.DataFrame = None,
        default_responses: pd.DataFrame = None,
        default_scale_data: bool = True,
        default_scale_run_data: bool = True,
        default_fit_intercept: bool = True,
        default_response_key: list = [0],
        default_drop_duplicates: str = "yes",
        default_input_selector: list = [],
    ):
        super().__init__(
            default_inputs,
            default_responses,
            default_scale_data,
            default_fit_intercept,
        )
        self.default_scale_run_data = default_scale_run_data
        self.default_response_key = default_response_key
        self.default_drop_duplicates = default_drop_duplicates
        self.default_input_selector = default_input_selector

    def add_model(
        self,
        inputs=None,
        responses=None,
        scale_data=None,
        scale_run_data=None,
        fit_intercept=None,
        response_key=None,
        drop_duplicates=None,
        input_selector=None,
    ):
        inputs = self._validate_value("inputs", inputs)
        responses = self._validate_value("responses", responses)
        scale_data = self._validate_value("scale_data", scale_data)
        scale_run_data = self._validate_value("scale_run_data", scale_run_data)
        fit_intercept = self._validate_value("fit_intercept", fit_intercept)
        response_key = self._validate_value("response_key", response_key)
        drop_duplicates = self._validate_value(
            "drop_duplicates", drop_duplicates
        )
        input_selector = self._validate_value("input_selector", input_selector)

        data = ModifiableDataSet(inputs, responses)
        # if scale_data:
        #     data.scale()
        if input_selector:
            data.filter(input_selector)
        model = AveragedModel(
            data,
            scale_data,
            scale_run_data,
            fit_intercept,
            response_key,
            drop_duplicates,
        )
        self.models.append(model)
        return model
