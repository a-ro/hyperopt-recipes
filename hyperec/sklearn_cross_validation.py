from typing import Dict

import numpy
from hyperopt import hp, fmin, tpe, space_eval
from hyperopt.pyll.base import scope
from sklearn.base import BaseEstimator
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import BaseCrossValidator, train_test_split, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier


def sklearn_kfold_hyper_parameter_search(max_eval: int, random_state=42) -> None:
    numpy.random.seed(random_state)

    print("loading the iris dataset...")
    inputs, outputs = load_iris(return_X_y=True)
    train_inputs, test_inputs, train_outputs, test_outputs = train_test_split(
        inputs, outputs, stratify=outputs, random_state=random_state
    )
    print()

    print("creating the model and setting up the evaluator.")
    model = DecisionTreeClassifier()
    cross_validation_splitter = StratifiedKFold(n_splits=10, random_state=random_state)
    evaluator = AccuracyModelEvaluator(model, train_inputs, train_outputs, cross_validation_splitter)
    print()

    print("finding best hyper parameter values with hyperopt")
    # define hyper space for the hyper parameters of the model.
    # this is just a subset of hyper parameters available for decision tree
    decision_tree_hyper_space = {
        "criterion": hp.choice("criterion", ["entropy", "gini"]),
        "max_depth": hp.lognormal("max_depth_value", 3, 1),
        "min_samples_split": scope.int(hp.uniform("min_samples_split", 2, 10)),
    }
    best_configuration = fmin(
        evaluator.evaluate,
        space=decision_tree_hyper_space,
        algo=tpe.suggest,
        max_evals=max_eval,
        rstate=numpy.random.RandomState(random_state),
    )
    best_hyper_parameter_values = space_eval(decision_tree_hyper_space, best_configuration)
    print(f"best hyper parameters: {best_hyper_parameter_values}")
    print()

    # evaluate model with best hyper parameters
    print("evaluating model on test set")
    model.set_params(**best_hyper_parameter_values)
    model.fit(train_inputs, train_outputs)
    predicted_outputs = model.predict(test_inputs)
    score = accuracy_score(test_outputs, predicted_outputs)
    print(f"accuracy on test set: {score}")


class AccuracyModelEvaluator:
    def __init__(
        self,
        model: BaseEstimator,
        train_inputs: numpy.array,
        train_outputs: numpy.array,
        fold_splitter: BaseCrossValidator,
    ):
        self.model = model
        self.inputs = train_inputs
        self.outputs = train_outputs
        self.fold_splitter = fold_splitter

    def evaluate(self, hyper_parameters: Dict) -> float:
        """
        Evaluation method called by hyperopt for every trial
        """
        self.model.set_params(**hyper_parameters)
        fold_scores = []
        for train_indexes, test_indexes in self.fold_splitter.split(self.inputs, self.outputs):
            self.model.fit(self.inputs[train_indexes], self.outputs[train_indexes])
            predicted_outputs = self.model.predict(self.inputs[test_indexes])
            fold_score = accuracy_score(self.outputs[test_indexes], predicted_outputs)
            fold_scores.append(fold_score)
        score = numpy.mean(fold_scores)
        # we want hyperopt to maximize the accuracy, which is equivalent to minimizing -accuracy.
        loss = -score
        return loss


if __name__ == "__main__":
    sklearn_kfold_hyper_parameter_search(max_eval=50)
