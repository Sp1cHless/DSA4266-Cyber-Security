"""Fixed input-order policy shared by neural models and flattened tree inputs."""


def validate_start(start):
    if type(start) is not int or not 0 <= start < 10:
        raise ValueError("circular_shift_start must be an integer from 0 to 9")
    return start


def input_transform(start):
    validate_start(start)
    if start == 0:
        return {"name": "identity"}
    return {"name": "circular_shift", "start_bucket": start,
            "bucket_order": list(range(start, 10)) + list(range(start)),
            "features_and_mask_together": True,
            "applies_to": "train_validation_and_frozen_evaluation"}
