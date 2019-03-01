class Modeler:
    """
    Model Factory Class
    Train a model
    """
    def __init__(self):
        """
        Constructor
        """
        self.model = None

        return

    def validate_data(self, df):
        """
        Validate data integrity before training
        Basic stasistics, data assumptions
        """
        return

    def splitData(self, df):
        """
        Split dataset
        """
        return df, df, df, df

    def train(self, X_train, y_train):
        """
        Cross validate and train a model
        """

        # sample model as template placeholder
        self.model = None

        # return training results for logging
        cv_scores = {}
        return cv_scores

    def score(self, df):
        """
        Score a dataset using a trained model
        """
        return

    def explain(self, df):
        """
        Explain a trained model's predictions
        """
        return
