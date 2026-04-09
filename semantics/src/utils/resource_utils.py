
import threading
import functools
import psutil
import os


class MemoryUsageExceededException(Exception):
    """Custom exception to indicate that memory usage has exceeded a threshold."""
    pass


class TimeoutException(Exception):
    """Custom exception to indicate a timeout has occurred."""
    pass


def check_memory_usage_gb():
    """Check the current memory usage of the system."""
    rss = psutil.Process(os.getpid()).memory_info().rss
    return rss / (1024 ** 3)  # Convert bytes to gigabytes


def timeout(seconds):
    """Decorator to enforce a timeout on a function call."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = [TimeoutException(f"Function '{func.__name__}' timed out after {seconds} seconds")]

            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    result[0] = e

            thread = threading.Thread(target=target)
            thread.start()
            thread.join(seconds)
            if thread.is_alive():
                raise TimeoutException(f"Function '{func.__name__}' timed out after {seconds} seconds")
            if isinstance(result[0], Exception):
                raise result[0]
            return result[0]
        return wrapper
    return decorator
