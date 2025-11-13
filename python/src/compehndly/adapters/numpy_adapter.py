import logging
import numpy as np
import pyarrow as pa

from compehndly.adapters.base import ArrayAdapter

logger = logging.getLogger(__name__)


class NumpyAdapter(ArrayAdapter):
    name = "numpy"

    def matches(self, obj):
        return isinstance(obj, np.ndarray) and obj.ndim == 1

    def _to_arrow(self, obj):
        return pa.array(obj)  # zero copy for many numeric dtypes

    def from_arrow(self, arrow_obj):
        return arrow_obj.to_numpy()  # zero-copy where possible
