from typing import List

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

    def add_model(
        self, inputs=None, responses=None, scale_data=None, fit_intercept=None
    ):
        if inputs is None:
            inputs = self.default_inputs
        if inputs is None:
            raise ValueError(
                "inputs not specified and no default inputs given"
            )

        if responses is None:
            responses = self.default_responses
        if responses is None:
            raise ValueError(
                "responses not specified and no default responses given"
            )

        if scale_data is None:
            scale_data = self.default_scale_data
        if scale_data is None:
            raise ValueError(
                "scale_data not specified and no default scale_data given"
            )

        if fit_intercept is None:
            fit_intercept = self.default_fit_intercept
        if fit_intercept is None:
            raise ValueError(
                "fit_intercept not specified and no default fit_intercept given"
            )

        model = Model(inputs, responses, scale_data, fit_intercept)
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
