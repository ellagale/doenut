import numpy as np
import pandas as pd
import doenut
from doenut.models.model import Model


class AveragedModel(Model):
    def __init__(
        self,
        inputs,
        responses,
        scale_data=True,
        fit_intercept=True,
        response_key="ortho",
        drop_duplicates=True,
    ):
        super().__init__(inputs, responses, scale_data, fit_intercept)

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
            if self.data_is_scaled:
                train_input, mj, rj = doenut.orthogonal_scaling(train_input, 0)
                test_input = doenut.scale_by(test_input, mj, rj)
            model = Model(train_input, train_responses, False, fit_intercept)
            r2s.append(model.get_r2())
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

        # replace our initial model with the averaged one.
        self.model.coef_ = self.averaged_coeffs
        self.model.intercept_ = self.averaged_intercepts
        self.r2s = r2s

        # Now calculate q2
        self.df_pred = pd.DataFrame.from_records(
            model_predictions, columns=self.responses.columns
        )
        self.df_gt = pd.DataFrame.from_records(
            model_responses, columns=self.responses.columns
        )
        self.q2 = doenut.Calculate_Q2(
            self.df_gt, self.df_pred, self.responses, response_key
        )

        # return model, df_pred, df_GT, coeffs, R2s, R2, Q2
