from typing import List
import numpy as np
import pandas as pd
import doenut
from doenut.models.model_set import ModelSet
from doenut.models.model import Model


class AveragedModel(Model):
    """
    Model generated as the average of multiple models generated from a single
    set of inputs via a leave-one-out approach.
    """

    # TODO: response_key should probably be moved to selective model
    def __init__(
        self,
        inputs: pd.DataFrame,
        responses: pd.DataFrame,
        scale_data: bool = True,
        scale_run_data: bool = True,
        fit_intercept: bool = True,
        response_key: str = None,
        drop_duplicates: str = True,
        input_selector: List = [],
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
        @param input_selector: Optional list of columns to filter input by
        May also be 'average' which will cause them to be dropped, but the one
        left will have its response value(s) set to the average of all the
        duplicates.
        """
        # Handle input selector
        if input_selector is not None and len(input_selector) > 0:
            inputs = inputs[input_selector]

        # Call super to set up basic model
        super().__init__(inputs, responses, scale_data, fit_intercept)

        # check the columns
        if response_key is None:
            if len(responses.columns) > 1:
                raise ValueError(
                    "No response key specified and multiple response columns"
                )
            response_key = responses.columns[0]

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
        self.models = ModelSet(None, None, scale_data, fit_intercept)
        model_predictions = []
        errors = []
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
            model = self.models.add_model(
                train_input, train_responses, False, fit_intercept
            )
            predictions = model.get_predictions_for(test_input)[0]
            model_predictions.append(predictions)
            model_responses.append(test_response)
            errors.append(test_response - predictions)
        self.coeffs = self.models.get_attributes("coef_")
        self.intercepts = self.models.get_attributes("intercept_")
        self.averaged_coeffs = np.mean(np.array(self.coeffs), axis=0)
        self.averaged_intercepts = np.mean(np.array(self.intercepts), axis=0)
        self.r2s = self.models.get_r2s()

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
