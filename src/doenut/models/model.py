import pandas as pd
from sklearn.linear_model import LinearRegression
from doenut.data.data_set import DataSet


class Model:
    """
    A simple linear regression model.

    This mostly exists as a base - you probably want AveragedModel
    """

    def __init__(self, data: DataSet, fit_intercept: bool) -> None:
        """
        Generate a simple model from the given values
        @param data: The inputs and responses to the model
        @param fit_intercept: Whether to fit the intercept to zero
        """
        self.data = data
        inputs = self.data.get_inputs()
        responses = self.data.get_responses()
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
        return self.model.score(data.get_inputs(), data.get_responses())
