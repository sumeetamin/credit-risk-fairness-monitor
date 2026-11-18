import numpy as np
import pandas as pd
import pytest

from main import encode_positive_target
from predict import positive_probability


@pytest.mark.parametrize("classes,positive,expected", [
    (["0", "1"], "1", [0, 1, 0]),
    (["default", "paid"], "default", [1, 0, 1]),
])
def test_positive_label_maps_to_model_class_one(classes, positive, expected):
    values = pd.Series([classes[0], classes[1], classes[0]])
    mapped = encode_positive_target(values, positive)
    assert mapped.tolist() == expected


class FakeModel:
    classes_ = np.array([0, 1])

    def predict_proba(self, frame):
        return np.array([[0.8, 0.2], [0.1, 0.9]])


def test_prediction_uses_probability_of_remapped_positive_class():
    bundle = {"pipeline": FakeModel(), "positive_model_class": 1}
    got = positive_probability(bundle, pd.DataFrame({"x": [1, 2]}))
    assert got.tolist() == [0.2, 0.9]


def test_prediction_rejects_missing_positive_model_class():
    class InvalidModel(FakeModel):
        classes_ = np.array([0, 2])
    with pytest.raises(ValueError, match="expected positive class"):
        positive_probability({"pipeline": InvalidModel()}, pd.DataFrame({"x": [1]}))

def test_holdout_labels_decode_with_configured_positive_label():
    classes = ["default", "paid"]
    positive_label = "default"
    negative_label = next(label for label in classes if label != positive_label)
    binary_actual = np.array([1, 0, 1])
    decoded = np.where(binary_actual == 1, positive_label, negative_label)
    assert decoded.tolist() == ["default", "paid", "default"]
