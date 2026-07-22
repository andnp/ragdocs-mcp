from contextlib import contextmanager
from typing import Generator
import logging


logger = logging.getLogger(__name__)


@contextmanager
def safe_writer(index, operation_name: str = "write") -> Generator:
    writer = index.writer()
    try:
        yield writer
        writer.commit()
    except Exception as e:
        logger.warning(
            "Writer operation '%s' failed, cancelling: %s",
            operation_name,
            e,
            exc_info=True,
        )
        writer.cancel()
        raise
