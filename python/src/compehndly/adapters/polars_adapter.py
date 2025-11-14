import logging
import polars as pl
import pyarrow as pa

from compehndly.adapters.base import ArrayAdapter

logger = logging.getLogger(__name__)


class PolarsAdapter(ArrayAdapter):
    name = "polars"

    def matches(self, obj):
        return isinstance(obj, pl.Series)

    def _to_arrow(self, obj):
        if isinstance(obj, pl.Series):
            return obj.to_arrow()
        return pa.array(obj)

    def from_arrow(self, arrow_obj):
        return pl.from_arrow(arrow_obj)
