"""Use exactly sklearn AP for checkpoint selection, like the neural models."""
import numpy as np
from sklearn.metrics import average_precision_score

from train.common.data import flat_inputs


def fit(training, validation, weights, config, device, threads, seed, out):
    import xgboost as xgb
    from train.common.io import write_json
    settings = config["training"]
    params = {**config["architecture"], "objective": "binary:logistic", "tree_method": "hist",
              "device": device, "seed": seed, "nthread": threads, "disable_default_eval_metric": 1}
    train_matrix = xgb.DMatrix(flat_inputs(training), label=training["y"], weight=weights, nthread=threads)
    validation_matrix = xgb.DMatrix(flat_inputs(validation), label=validation["y"], nthread=threads)

    def metric(predictions, data):
        return "average_precision", float(average_precision_score(data.get_label(), predictions))

    history = {}
    callback = xgb.callback.EarlyStopping(rounds=settings["patience"], metric_name="average_precision",
                                          data_name="validation", maximize=True, save_best=True)
    model = xgb.train(params, train_matrix, num_boost_round=settings["rounds"],
                      evals=[(validation_matrix, "validation")], custom_metric=metric,
                      evals_result=history, callbacks=[callback], verbose_eval=20)
    model.save_model(out / "model.ubj")
    write_json(out / "history.json", history)
    info = {"best_iteration": int(model.best_iteration), "best_validation_ap": float(model.best_score),
            "boosted_rounds_saved": model.num_boosted_rounds(), "resolved_parameters": params}
    return model, info


def predict(model, split, device, threads):
    import xgboost as xgb
    result = model.predict(xgb.DMatrix(flat_inputs(split), nthread=threads))
    return np.asarray(result, dtype=np.float64)


def load(path, device, threads):
    import xgboost as xgb
    model = xgb.Booster()
    model.load_model(path)
    model.set_param({"device": device, "nthread": threads})
    return model
