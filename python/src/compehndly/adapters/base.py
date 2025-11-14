import logging
import numbers
import pyarrow as pa

logger = logging.getLogger(__name__)


class ArrayAdapter:
    name: str = "base"

    def _to_arrow(self, obj):
        return pa.array(obj)

    def to_arrow(self, obj) -> pa.Array:
        if isinstance(obj, (pa.Array, pa.Scalar)):
            return obj

        # Python scalar → pass through unchanged
        elif isinstance(obj, (numbers.Number, bool, str)):
            return obj

        # Python list/iterable → fallback to Arrow
        return self._to_arrow(obj)

    def from_arrow(self, obj):
        return obj
