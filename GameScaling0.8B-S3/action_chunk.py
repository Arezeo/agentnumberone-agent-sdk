"""A single bounded action chunk; inference may run during its six slots."""
import asyncio
import time


async def execute_chunk(steps, apply_step, eligible, *, started=None, duration=.2):
    """No control escapes revalidation; no concurrent chunks share held inputs.

    `started` is resolved after the first slot is queued, allowing the next
    observation/inference to overlap this chunk. It is not an application ack.
    """
    begin = time.monotonic()
    try:
        for index, step in enumerate(steps):
            await asyncio.sleep(max(0, begin + index / 30 - time.monotonic()))
            if not eligible():
                return False
            apply_step(index, step)
            if index == 0 and started is not None and not started.done():
                started.set_result(begin)
        await asyncio.sleep(max(0, begin + duration - time.monotonic()))
        return bool(eligible())
    finally:
        if started is not None and not started.done():
            started.set_result(None)
