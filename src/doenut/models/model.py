from sklearn.linear_model import LinearRegression
import doenut


class Model:
    def __init__(self, inputs, responses, scale_data, fit_intercept):
        """
        Generate a simple model from the given values
        @param inputs:  The inputs to create a model from as a numpy array-like
        @param responses: The ground truths for the inputs
        @param scale_data: Whether to normalise the input data
        @param fit_intercept: Whether to fit the intercept to zero
        """
        self.data_is_scaled = scale_data
        if scale_data:
            (
                self.inputs,
                self.scaling_Mj,
                self.scaling_Rj,
            ) = doenut.orthogonal_scaling(inputs, axis=0)
        else:
            # No scaling. Set the co-efficients to identity values.
            self.inputs = inputs
            self.scaling_Rj = 1
            self.scaling_Mj = 0
        self.responses = responses

        self.model = LinearRegression(fit_intercept=fit_intercept)
        self.model.fit(inputs, responses)
        self.predictions = self.get_predictions_for(self.inputs)
        self.r2 = self.get_r2_for(self.inputs, self.responses)

    def get_predictions_for(self, inputs):
        """
        Generates the predictions of the model for a set of inputs
        @param inputs: The inputs to test against as an array-like
        @return: the predictions as an array-like
        """
        return self.model.predict(inputs)

    def get_r2_for(self, inputs, responses):
        """
        Calculate the R2 Pearson coefficient for a given pairing of
        inputs and responses.
        @param inputs: Inputs to test as an array-like
        @param responses: The ground truths to test against as an array-like
        @return: the calculated R2 value as a float
        """
        return self.model.score(inputs, responses)
