import logging

import pandas as pd

import doenut.utils
from doenut.data import ModifiableDataSet
from doenut.models.model_set import ModelSet
from doenut.models.averaged_model import AveragedModel

logger = doenut.utils.initialise_log(__name__, logging.DEBUG)


class AveragedModelSet(ModelSet):
    """Class to train and hold a group of related (averaged) models.
    When constructing the AveragedModelSet, you can define default values.
    Then when adding a new model to the set you only have to specify the
    parameters which differ from the default.


    Parameters
    ----------
    default_inputs: pd.DataFrame, optional
        The default inputs to the model
    default_responses: pd.DataFrame, optional
        The default responses for the model
    default_scale_data: bool, optional
        Whether to scale the data before adding to the model by default
    default_scale_run_data: bool, optional
        Whether to scale the data for each train/test set by default
    default_fit_intercept: bool, optional
        Whether to fit the model's intercept to the axis by default
    default_response_key: str, optional
        The default column to pick from the responses
    default_drop_duplicates: {'no', 'yes', 'averages'}, optional
        What to do with duplicates in the inputs, by default
    default_input_selector: List, optional
        What columns from the input data to select by default

    """

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
        logger.info("Generating AveragedModelSet")
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
            logger.debug(f"Adding model for response key {column}")
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
        """Add a new AveragedModel to the set

        Parameters
        ----------
        inputs: pd.DataFrame, optional
            The inputs to the model
        responses: pd.DataFrame, optional
            The responses for the model
        scale_data: bool, optional
            Whether to scale the data before adding to the model
        scale_run_data: bool, optional
            Whether to scale the data for each train/test set
        fit_intercept: bool, optional
            Whether to fit the model's intercept to the axis
        response_key: str, optional
            The column to pick from the responses
        drop_duplicates: {'no', 'yes', 'averages'}, optional
            What to do with duplicates in the inputs
        input_selector: List, optional
            What columns from the input data to select

        Returns
        -------
        doenut.models.AveragedModel
            The generated model

        """
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
