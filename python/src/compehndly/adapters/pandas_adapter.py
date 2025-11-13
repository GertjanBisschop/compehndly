import logging
import pyarrow as pa
import pandas as pd

from compehndly.adapters.base import ArrayAdapter

logger = logging.getLogger(__name__)


class PandasAdapter(ArrayAdapter):
    name = "pandas"

    def matches(self, obj):
        return isinstance(obj, pd.Series)

    def _to_arrow(self, obj):
        return pa.array(obj)  # may be zero-copy if ArrowExtensionArray

    def from_arrow(self, arrow_obj):
        return pd.Series(arrow_obj)
