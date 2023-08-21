import numpy as np
from numpy import ndarray
from sklearn.ensemble import RandomForestRegressor


class RandomForestSurrogateRegressor(RandomForestRegressor):
    """
    This class is a wrapper for the sklearn RandomForestRegressor class.
    It adds the functionality to return the uncertainty of the predictions.
    It is used as a surrogate model for the Bayesian optimization.
    """

    def __init__(
        self,
        n_estimators=100,
        *,
        criterion="squared_error",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=1.0,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        ccp_alpha=0.0,
        max_samples=None,
    ):
        super().__init__(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            ccp_alpha=ccp_alpha,
            max_samples=max_samples,
        )

    def predict(self, X, return_std=False) -> ndarray:
        """Predict class or regression value for X.

        Args:
            X (_type_):  The input samples.
            return_std (bool, optional): Allows to return the variance of the diffrent trees. Defaults to False.

        Returns:
            ndarray: The predicted classes, or the predict values.
        """
        if return_std:
            return super().predict(X), self.get_uncertainty(X)
        return super().predict(X)

    def get_uncertainty(self, X) -> ndarray:
        """_summary_

        Args:
            X (_type_): The input samples.

        Returns:
            ndarray: The variance of the predictions of the diffrent trees.
        """
        ensemble_predictions = np.array(
            [submodel.predict(X) for submodel in self.estimators_]
        )
        uncertainty = np.var(ensemble_predictions, axis=0)
        return uncertainty
