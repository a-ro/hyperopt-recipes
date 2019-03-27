from typing import Dict

from hyperopt import hp, fmin, tpe, Trials, STATUS_OK

SIGMA = "sigma"
LAMBDA = "lambda"


def hyper_parameter_search_with_trials_and_multiple_loss_functions(
    sigma_min: float, sigma_max: float, lambda_min: float, lambda_max: float, max_evals: int
) -> None:
    """
    Trials & Multi Losses Recipe: multiple evaluation functions depend on two hyper_parameters (sigma and lambda).
    Goal: save the results of each configuration tried for all evaluation functions.
    Use hyperopt to find the best values for sigma and lambda with a specific number of trials.
    Sample sigma uniformly between sigma_min and sigma_max.
    Sample lambda uniformly between lambda_min and lambda_max.
    :param sigma_min: min value for sigma
    :param sigma_max: max value for sigma
    :param lambda_min: min value for lambda
    :param lambda_max: max value for lambda
    :param max_evals: maximum number of trials with hyperopt
    """
    trials = Trials()
    hyper_space = {
        SIGMA: hp.uniform(SIGMA, low=sigma_min, high=sigma_max),
        LAMBDA: hp.uniform(LAMBDA, low=lambda_min, high=lambda_max),
    }
    fmin(evaluation, space=hyper_space, algo=tpe.suggest, max_evals=max_evals, trials=trials)

    print()
    print(f"best hyper parameter values: {trials.argmin}")
    print(f"all hyper parameter values tried: {trials.vals}")
    print(f"all trial results: {trials.results}")
    print()

    print("showing results for all trials:")
    for trial in trials:
        print(f"results for trial {trial['tid']}: {trial['result']}")


def evaluation(hyper_parameters: Dict) -> Dict:
    result_dict = dict()

    # this is the loss that hyperopt will minimize, make sure to use the "loss" key in the result dict
    main_loss = (hyper_parameters[SIGMA] - hyper_parameters[LAMBDA]) ** 2
    result_dict["loss"] = main_loss

    # this is another loss that we'll be able to access in the trial results
    other_loss = hyper_parameters[SIGMA] - hyper_parameters[LAMBDA]
    result_dict["other_loss"] = other_loss

    # to tell hyperopt there were no errors
    result_dict["status"] = STATUS_OK

    # saving the hyper parameter values for that trial
    result_dict["hp"] = hyper_parameters

    return result_dict


if __name__ == "__main__":
    hyper_parameter_search_with_trials_and_multiple_loss_functions(
        sigma_min=-5, sigma_max=10.5, lambda_min=-2.5, lambda_max=3, max_evals=20
    )
