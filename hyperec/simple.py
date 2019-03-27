from typing import Dict

from hyperopt import hp, fmin, tpe

SIGMA = "sigma"
LAMBDA = "lambda"


def simple_hyper_parameter_search(
    sigma_min: float, sigma_max: float, lambda_min: float, lambda_max: float, max_eval: int
) -> None:
    hyper_space = {
        SIGMA: hp.uniform(SIGMA, low=sigma_min, high=sigma_max),
        LAMBDA: hp.uniform(LAMBDA, low=lambda_min, high=lambda_max),
    }
    best_hyper_parameters = fmin(evaluation, space=hyper_space, algo=tpe.suggest, max_evals=max_eval)
    print(f"best hyper parameter values: {best_hyper_parameters}")


def evaluation(hyper_parameters: Dict) -> float:
    loss = (hyper_parameters[SIGMA] - hyper_parameters[LAMBDA]) ** 2
    return loss


if __name__ == "__main__":
    simple_hyper_parameter_search(sigma_min=-5, sigma_max=10.5, lambda_min=-2.5, lambda_max=3, max_eval=500)
