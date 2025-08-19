import os
import sys

import logging

logger = logging.getLogger(__name__)


def run_silently(func, *args):
    """
        Executes a function while suppressing all output (stdout, stderr) and logging messages.

        :param func: The function to execute.
        :param args: The arguments to pass to the function ``func``.
        """
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        # Set logging level to CRITICAL to suppress logs
        logging.disable(logging.CRITICAL)
        try:
            func(*args)
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            logging.disable(logging.NOTSET)
