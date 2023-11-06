from sklearn.linear_model import LinearRegression
import doenut


class Model:
    def __init__(self, inputs, responses, scale_data, fit_intercept):
        """ """
        self.data_is_scaled = scale_data
        if scale_data:
            (
                self.inputs,
                self.scaling_Mj,
                self.scaling_Rj,
            ) = doenut.orthogonal_scaling(inputs, axis=0)
        else:
            self.inputs = inputs
            self.scaling_Rj = 1
            self.scaling_Mj = 0
        self.responses = responses

        self.model = LinearRegression(fit_intercept=fit_intercept)
        self.model.fit(inputs, responses)
        self._predictions = None
        self._r2 = None

    def get_predictions(self):
        """Lazy evaluator for predictions"""
        if self._predictions is None:
            self._predictions = self.get_predictions_for(self.inputs)
        return self._predictions

    def get_predictions_for(self, inputs):
        return self.model.predict(inputs)

    def get_r2(self):
        if self._r2 is None:
            self._r2 = self.get_r2_for(self.inputs, self.responses)
        return self._r2

    def get_r2_for(self, inputs, responses):
        """
        Calculate the R2 Pearson coefficient for a given pairing of
        inputs and responses
        """
        return self.model.score(inputs, responses)
