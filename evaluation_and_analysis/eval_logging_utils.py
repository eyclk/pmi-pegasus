"""Mirror whatever the step1-step4 scripts print into a .log file.

Those scripts report their averages by printing them, so the numbers scroll past
and are gone once the terminal buffer fills. Decorating a dataset function with
@log_printed_output_to(...) keeps the printing exactly as it was and, on top of
that, writes every printed line to a log file sitting next to the JSON output the
same function produces -- the "<name>.json" + "<name>.log" pairing that steps 5-7
already use.

tqdm draws its progress bars on stderr, so the logs stay free of progress noise.
"""

import functools
import os
import sys
from contextlib import contextmanager
from datetime import datetime


class _TeeStdout:
    """A stdout stand-in that writes to the console and to a log file."""

    def __init__(self, console, log_file):
        self.console = console
        self.log_file = log_file

    def write(self, text):
        self.console.write(text)
        self.log_file.write(text)
        # These scripts print rarely but run for hours, so pay the flush on every
        # write and keep the log complete even if the run is killed part way.
        self.log_file.flush()
        return len(text)

    def flush(self):
        self.console.flush()
        self.log_file.flush()

    def isatty(self):
        return self.console.isatty()


@contextmanager
def tee_stdout_to_log(log_path):
    """Mirror everything printed inside the block into log_path."""
    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write(f"# {os.path.basename(sys.argv[0])} -- run at "
                       f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        console = sys.stdout
        sys.stdout = _TeeStdout(console, log_file)
        try:
            yield
        finally:
            # Restored even when the wrapped function raises, so a failure part
            # way through still leaves the console usable and the log readable.
            sys.stdout = console

    print(f"Printed output also saved to {log_path}")


def log_printed_output_to(log_path):
    """Decorator form of tee_stdout_to_log(), so a function body stays untouched."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with tee_stdout_to_log(log_path):
                return func(*args, **kwargs)

        return wrapper

    return decorator
