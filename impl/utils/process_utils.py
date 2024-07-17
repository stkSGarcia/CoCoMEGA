def run_silently(func, *args):
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