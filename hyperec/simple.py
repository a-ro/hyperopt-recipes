from typing import Dict

from hyperopt import hp, fmin, tpe

SIGMA = "sigma"
LAMBDA = "lambda"


def simple_hyper_parameter_search(
    sigma_min: float, sigma_max: float, lambda_min: float, lambda_max: float, max_evals: int
) -> None:
    """
    Simple Recipe: one evaluation function depends on two hyper_parameters: sigma and lambda.
    Use hyperopt to find the best values for sigma and lambda with a specific number of trials.
    Sample sigma uniformly between sigma_min and sigma_max.
    Sample lambda uniformly between lambda_min and lambda_max.
    :param sigma_min: min value for sigma
    :param sigma_max: max value for sigma
    :param lambda_min: min value for lambda
    :param lambda_max: max value for lambda
    :param max_evals: maximum number of trials with hyperopt
    """
    hyper_space = {
        SIGMA: hp.uniform(SIGMA, low=sigma_min, high=sigma_max),
        LAMBDA: hp.uniform(LAMBDA, low=lambda_min, high=lambda_max),
    }
    best_hyper_parameters = fmin(evaluation, space=hyper_space, algo=tpe.suggest, max_evals=max_evals)
    print(f"best hyper parameter values: {best_hyper_parameters}")


def evaluation(hyper_parameters: Dict) -> float:
    loss = (hyper_parameters[SIGMA] - hyper_parameters[LAMBDA]) ** 2
    return loss


if __name__ == "__main__":
    simple_hyper_parameter_search(sigma_min=-5, sigma_max=10.5, lambda_min=-2.5, lambda_max=3, max_evals=500)
