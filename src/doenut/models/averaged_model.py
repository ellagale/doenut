from typing import Tuple

import numpy as np
import pandas as pd
import doenut
import copy

from doenut.data.modifiable_data_set import ModifiableDataSet
from doenut.models.model_set import ModelSet
from doenut.models.model import Model


class AveragedModel(Model):
    """
    Model scored as the average of multiple models generated from a single
    set of inputs via a leave-one-out approach.
    """

    @classmethod
    def tune_model(
        cls,
        data: ModifiableDataSet,
        fit_intercept: bool = True,
        response_key: str = None,
        drop_duplicates: str = "yes",
    ) -> Tuple["AveragedModel", "AveragedModel"]:
        """
        Generate a pair of models from the same set of data. One using scaled
        data the other unscaled.

        The scaled model can then be used for determining which columns to drop
        for later models, and the unscaled model for checking the models
        performance against validation data (or just for using once done).

        @param data: The dataset to test against. This should be unscaled.
        @param fit_intercept: Whether to fit the intercept or not (usually yes)
        @param response_key: If there are more than one response columns,
        which to use.
        @param drop_duplicates: Whether to apply any duplicate reduction to
        the data as it is trained. Should be one of 'yes', 'no' or 'average'
        @return: A tuple of AveragedModels: (scaled, unscaled)
        """
        scaled_model = AveragedModel(
            data,
            scale_data=True,
            scale_run_data=True,
            fit_intercept=fit_intercept,
            response_key=response_key,
            drop_duplicates=drop_duplicates,
        )
        unscaled_model = AveragedModel(
            data,
            scale_data=False,
            scale_run_data=False,
            fit_intercept=fit_intercept,
            response_key=response_key,
            drop_duplicates=drop_duplicates,
        )
        return scaled_model, unscaled_model

    def __init__(
        self,
        data: ModifiableDataSet,
        scale_data: bool = True,
        scale_run_data: bool = True,
        fit_intercept: bool = True,
        response_key: str = None,
        drop_duplicates: str = "yes",
    ):
        """
        Constructor
        @param data: the data to run / test against.
        @param scale_data: Whether to scale the overall data before running it.
        @param scale_run_data: Whether to normalise the data for each run
        @param fit_intercept: Whether to fit the intercept to zero
        @param response_key: for multi-column responses, which one to test on
        @param drop_duplicates: whether to drop duplicate values or not.
        May also be 'average' which will cause them to be dropped, but the one
        left will have its response value(s) set to the average of all the
        duplicates.
        """
        proc_data = copy.deepcopy(data)
        if scale_data:
            proc_data.scale(False)
        # Call super to set up basic model
        super().__init__(proc_data.get(), fit_intercept)

        # check the columns
        responses = self.data.get_responses()
        if response_key is None:
            if len(responses.columns) > 1:
                raise ValueError(
                    "No response key specified and multiple response columns"
                )
            response_key = responses.columns[0]

        # Get the processed inputs + responses (after filtering + dedupe
        proc_inputs, proc_responses = None, None
        if isinstance(drop_duplicates, str):
            if str.lower(drop_duplicates) == "yes":
                proc_data.drop_duplicates()
            elif str.lower(drop_duplicates) == "average":
                proc_data.average_duplicates()
            elif str.lower(drop_duplicates) == "no":
                pass
            else:
                raise ValueError(
                    f"Invalid drop_duplicates value {drop_duplicates}"
                    " - should one of 'yes', 'no', 'average'"
                )

        final_data = proc_data.get()
        proc_inputs = final_data.get_inputs()
        proc_responses = final_data.get_responses()

        # Use leave-one-out on the input data rows to generate a set of models
        self.models = ModelSet(None, None, fit_intercept)
        model_predictions = []
        errors = []
        model_responses = []
        for i, row_idx in enumerate(proc_inputs.index):
            test_input = proc_inputs.iloc[i].to_numpy().reshape(1, -1)
            test_response = proc_responses.iloc[i]
            train_input = proc_inputs.drop(row_idx).to_numpy()
            train_responses = proc_responses.drop(row_idx)
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

        # replace our initial model with the averaged one to determine R2/Q2.
        self.model.coef_ = self.averaged_coeffs
        self.model.intercept_ = self.averaged_intercepts

        self.r2 = self.get_r2_for(final_data)

        # Now calculate q2
        self.q2_predictions = pd.DataFrame.from_records(
            model_predictions, columns=proc_responses.columns
        )
        self.q2_ground_truths = pd.DataFrame.from_records(
            model_responses, columns=proc_responses.columns
        )
        self.q2 = doenut.Calculate_Q2(
            self.q2_ground_truths,
            self.q2_predictions,
            proc_responses,
            response_key,
        )
        # finally make a fitted model.
        start_data = data.get()
        self.model.fit(start_data.get_inputs(), start_data.get_responses())
        self.predictions = self.get_predictions_for(proc_inputs)
