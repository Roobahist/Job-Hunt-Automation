from __future__ import annotations

import random
import time
from collections.abc import Callable

from job_hunt.errors import WorkflowError


def retry_transient[T](
    operation: Callable[..., T],
    *args: object,
    attempts: int = 3,
    base_delay: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
    **kwargs: object,
) -> T:
    last: WorkflowError | None = None
    for attempt in range(attempts):
        try:
            return operation(*args, **kwargs)
        except WorkflowError as exc:
            last = exc
            if not exc.retryable or attempt == attempts - 1:
                raise
            delay = exc.retry_after if exc.retry_after is not None else base_delay * (2**attempt)
            sleep(delay + random.uniform(0, max(delay * 0.2, 0.001)))
    assert last is not None
    raise last
