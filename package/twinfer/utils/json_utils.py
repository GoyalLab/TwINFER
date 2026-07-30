import os
import json
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    """
    JSON encoder that handles numpy scalar and array types, which the
    standard json module cannot serialize natively (np.int64, np.float64,
    np.ndarray all raise TypeError otherwise).
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def make_json_safe(obj):
    if hasattr(obj, "to_dict"):      # pandas DataFrame / Series
        return obj.to_dict()
    if isinstance(obj, set):
        return list(obj)
    return obj
