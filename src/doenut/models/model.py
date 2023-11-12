import pandas as pd
from sklearn.linear_model import LinearRegression
import doenut
from doenut.data import DataSet


class Model:
    def __init__(self, data: DataSet, scale_data: bool, fit_intercept: bool) -> None:
        """
        Generate a simple model from the given values
        @param data:  The inputs and responses to the model
        @param scale_data: Whether to normalise the input data
        @param fit_intercept: Whether to fit the intercept to zero
        """
        self.data_is_scaled = scale_data
        self.data = data
        if scale_data:
            (
                self.data.data,
                self.scaling_Mj,
                self.scaling_Rj,
            ) = doenut.orthogonal_scaling(self.data.data, axis=0)
        else:
            # No scaling. Set the co-efficients to identity values.
            self.scaling_Rj = 1
            self.scaling_Mj = 0
        inputs = self.data.get()
        responses = self.data.get_filtered_responses()
        self.model = LinearRegression(fit_intercept=fit_intercept)
        self.model.fit(inputs, responses)
        self.predictions = self.get_predictions_for(inputs)
        self.r2 = self.get_r2_for(self.data)

    def get_predictions_for(self, inputs: pd.DataFrame) -> pd.DataFrame:
        """
        Generates the predictions of the model for a set of inputs
        @param inputs: The inputs to test against
        @return: the predictions from the model
        """
        return self.model.predict(inputs)

    def get_r2_for(self, data: DataSet):
        """
        Calculate the R2 Pearson coefficient for a given pairing of
        inputs and responses.
        @param data: The data to test.
        @return: the calculated R2 value as a float
        """
        return self.model.score(data.get(), data.get_filtered_responses())
