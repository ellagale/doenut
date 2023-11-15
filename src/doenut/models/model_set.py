from typing import List, Any

from doenut.data import ModifiableDataSet
from doenut.models import Model


class ModelSet:
    """
    Class to train and hold a group of related models.
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
        self, inputs=None, responses=None, scale_data=None, fit_intercept=None
    ):
        inputs = self._validate_value("inputs", inputs)
        responses = self._validate_value("responses", responses)
        scale_data = self._validate_value("scale_data", scale_data)
        fit_intercept = self._validate_value("fit_intercept", fit_intercept)
        dataset = ModifiableDataSet(inputs, responses).get()
        model = Model(dataset, fit_intercept)
        self.models.append(model)
        return model

    def get_r2s(self):
        # return [x.r2 for x in self.models]
        return self.get_attributes("r2")

    def get_attributes(self, attribute: str) -> List:
        """
        Get a specified attribute from each model.
        Frustratingly, some are in the model, others in the model object.
        @param attribute:
        @return:
        """
        if hasattr(self.models[0], attribute):
            return [getattr(x, attribute) for x in self.models]
        if hasattr(self.models[0].model, attribute):
            return [getattr(x.model, attribute) for x in self.models]
        raise ValueError(f"Attribute {attribute} is not in the models")
