import numpy as np
import pandas as pd
import doenut
from doenut.models.model import Model


class AveragedModel(Model):
    """
    Model generated as the average of multiple models generated from a single
    set of inputs via a leave-one-out approach.
    """
    # TODO: response_key should probably be moved to selective model
    def __init__(
        self,
        inputs,
        responses,
        scale_data=True,
        scale_run_data=True,
        fit_intercept=True,
        response_key="ortho",
        drop_duplicates=True,
    ):
        """
        Constructor
        @param inputs: The inputs to create a model from as a numpy array-like
        @param responses: The ground truths for the inputs
        @param scale_data: Whether to normalise the input data
        @param scale_run_data: Whether to normalise the data for each run
        @param fit_intercept: Whether to fit the intercept to zero
        @param response_key: for multi-column responses, which one to test on
        @param drop_duplicates: whether to drop duplicate values or not.
        May also be 'average' which will cause them to be dropped, but the one
        left will have its response value(s) set to the average of all the
        duplicates.
        """
        # Call super to setup basic model
        super().__init__(inputs, responses, scale_data, fit_intercept)

        # handle checking the duplicates
        self.duplicates = None
        if isinstance(drop_duplicates, str):
            if str.lower(drop_duplicates) == "yes":
                self.duplicates = [
                    x for x in inputs[inputs.duplicated()].index
                ]
            elif str.lower(drop_duplicates) == "average":
                self.inputs, self.responses = doenut.average_replicates(
                    self.inputs, self.responses
                )
                self.duplicates = []
            elif str.lower(drop_duplicates) == "no":
                self.duplicates = []
        if isinstance(drop_duplicates, bool) and drop_duplicates:
            self.duplicates = [x for x in inputs[inputs.duplicated()].index]

        if self.duplicates is None:
            raise ValueError(
                f"Invalid drop_duplicates value {drop_duplicates}"
                " - should be boolean or  one of 'yes', 'no', 'average'"
            )

        # If there are any duplicates, remove them.
        if len(self.duplicates) > 0:
            self.inputs = self.inputs.drop(self.duplicates)
            self.responses = self.responses.drop(self.duplicates)

        # Use leave-one-out on the input data rows to generate a set of models
        self.models = []
        model_predictions = []
        r2s = []
        intercepts = []
        errors = []
        coeffs = []
        model_responses = []
        for i, row_idx in enumerate(self.inputs.index):
            test_input = self.inputs.iloc[i].to_numpy().reshape(1, -1)
            test_response = self.responses.iloc[i]
            train_input = self.inputs.drop(row_idx).to_numpy()
            train_responses = responses.drop(row_idx)
            # We need to re-scale each column, using the training data *only*,
            # but then applying the same scaling to the test data.
            if scale_run_data:
                train_input, mj, rj = doenut.orthogonal_scaling(train_input, 0)
                test_input = doenut.scale_by(test_input, mj, rj)
            model = Model(train_input, train_responses, False, fit_intercept)
            r2s.append(model.r2)
            coeffs.append(model.model.coef_)
            predictions = model.get_predictions_for(test_input)[0]
            model_predictions.append(predictions)
            model_responses.append(test_response)
            errors.append(test_response - predictions)
            intercepts.append(model.model.intercept_)
            self.models.append(model)
        self.coeffs = coeffs
        self.averaged_coeffs = np.mean(np.array(coeffs), axis=0)
        self.averaged_intercepts = np.mean(np.array(intercepts), axis=0)
        self.r2s = r2s

        # replace our initial model with the averaged one.
        self.model.coef_ = self.averaged_coeffs
        self.model.intercept_ = self.averaged_intercepts
        self.r2 = self.get_r2_for(self.inputs, self.responses)
        self.predictions = self.get_predictions_for(self.inputs)

        # Now calculate q2
        self.q2_predictions = pd.DataFrame.from_records(
            model_predictions, columns=self.responses.columns
        )
        self.q2_groundtruths = pd.DataFrame.from_records(
            model_responses, columns=self.responses.columns
        )
        self.q2 = doenut.Calculate_Q2(
            self.q2_groundtruths,
            self.q2_predictions,
            self.responses,
            response_key,
        )
