"""Use exactly sklearn AP for checkpoint selection, like the neural models."""
import numpy as np
from sklearn.metrics import average_precision_score

from train.common.data import flat_inputs
from train.common.temporal import validate_start


def fit(training, validation, weights, config, device, threads, seed, out):
    import xgboost as xgb
    from train.common.io import write_json
    settings = config["training"]
    start = validate_start(config.get("input_transform", {}).get("start_bucket", 0))
    params = {**config["architecture"], "objective": "binary:logistic", "tree_method": "hist",
              "device": device, "seed": seed, "nthread": threads, "disable_default_eval_metric": 1}
    train_matrix = xgb.DMatrix(flat_inputs(training, start), label=training["y"], weight=weights, nthread=threads)
    validation_matrix = xgb.DMatrix(flat_inputs(validation, start), label=validation["y"], nthread=threads)

    def metric(predictions, data):
        return "average_precision", float(average_precision_score(data.get_label(), predictions))

    history = {}
    callback = xgb.callback.EarlyStopping(rounds=settings["patience"], metric_name="average_precision",
                                          data_name="validation", maximize=True, save_best=True)
    model = xgb.train(params, train_matrix, num_boost_round=settings["rounds"],
                      evals=[(validation_matrix, "validation")], custom_metric=metric,
                      evals_result=history, callbacks=[callback], verbose_eval=20)
    model.set_attr(circular_shift_start=str(start))
    model.save_model(out / "model.ubj")
    write_json(out / "history.json", history)
    info = {"best_iteration": int(model.best_iteration), "best_validation_ap": float(model.best_score),
            "boosted_rounds_saved": model.num_boosted_rounds(), "resolved_parameters": params}
    return model, info


def predict(model, split, device, threads):
    import xgboost as xgb
    start = validate_start(int(model.attr("circular_shift_start") or 0))
    result = model.predict(xgb.DMatrix(flat_inputs(split, start), nthread=threads))
    return np.asarray(result, dtype=np.float64)


def load(path, device, threads, circular_shift_start=None):
    import xgboost as xgb
    model = xgb.Booster()
    model.load_model(path)
    saved_start = validate_start(int(model.attr("circular_shift_start") or 0))
    if circular_shift_start is not None and saved_start != validate_start(circular_shift_start):
        raise ValueError("Checkpoint/config circular-shift mismatch")
    model.set_param({"device": device, "nthread": threads})
    return model
