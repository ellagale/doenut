from typing import List, Any

import pandas as pd

from doenut.data import ModifiableDataSet
from doenut.models.model import Model


class ModelSet:
    """Class to train and hold a group of related models.
    When constructing the ModelSet, you can define default values.
    Then when adding a new model to the set you only have to specify the
    parameters which differ from the default.

    Note
    ----
    This class mostly exists as a base - you probably want :py:class:`~doenut.models.AveragedModelSet`

    Parameters
    ----------
    default_inputs: pd.DataFrame, optional
        The default inputs to the model
    default_responses: pd.DataFrame, optional
        The default responses for the model
    default_scale_data: bool, optional
        Whether to scale the data before adding to the model by default
    default_fit_intercept: bool, optional
        Whether to fit the model's intercept to the axis by default

    """

    def __init__(
        self,
        default_inputs=None,
        default_responses=None,
        default_scale_data=True,
        default_fit_intercept=True,
    ):
        self.default_inputs = default_inputs
        self.default_responses = default_responses
        self.default_scale_data = default_scale_data
        self.default_fit_intercept = default_fit_intercept
        self.models = []

    def _validate_value(self, name: str, value: Any = None) -> Any:
        if value is not None:
            return value
        default_name = f"default_{name}"
        if hasattr(self, default_name):
            value = getattr(self, default_name)
            if value is not None:
                return value
        raise ValueError(f"model set lacks default value for {name}")

    def add_model(
        self,
        inputs: pd.DataFrame = None,
        responses: pd.DataFrame = None,
        scale_data: bool = None,
        fit_intercept: bool = None,
    ):
        """Builds and adds a model to the set
        For each parameter not specified, the defaults will be used instead.

        Parameters
        ----------
        inputs: pd.DataFrame, optional
            The inputs to the model
        responses: pd.DataFrame, optional
            The responses for the model
        scale_data: bool, optional
            Whether to scale the data before adding to the model
        fit_intercept: bool, optional
            Whether to fit the model's intercept to the axis

        Returns
        -------
        doenut.models.Model
            The generated model
        """
        inputs = self._validate_value("inputs", inputs)
        responses = self._validate_value("responses", responses)
        scale_data = self._validate_value("scale_data", scale_data)
        fit_intercept = self._validate_value("fit_intercept", fit_intercept)
        dataset = ModifiableDataSet(inputs, responses).get()
        model = Model(dataset, fit_intercept)
        self.models.append(model)
        return model

    def get_r2s(self):
        """
        Get the Pearson R2 values for the models in the set

        Returns
        -------
        List[float]
            The R2 value for each model in the set.
        """

        return self.get_attributes("r2")

    def get_attributes(self, attribute: str) -> List[Any]:
        """Get a specified attribute from each model.
        Frustratingly, some are in the model, others in the sklearn model.

        Parameters
        ----------
        attribute: str
            The attribute you want from the model

        Returns
        -------
        List[Any]
            A list of the value of that attribute for each model in the set.

        Raises
        ------
        ValueError
            If the attribute is not present in either the model or the inner
            sklearn model.

        note
        ----
        If the attribute exists in both the model and the sklearn model, the
        model attribute will be the one returned.
        """
        if hasattr(self.models[0], attribute):
            return [getattr(x, attribute) for x in self.models]
        if hasattr(self.models[0].model, attribute):
            return [getattr(x.model, attribute) for x in self.models]
        raise ValueError(f"Attribute {attribute} is not in the models")
