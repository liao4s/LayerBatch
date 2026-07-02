"""
Centralized logging for nanovllm.

Usage:
    from nanovllm.utils.logger import init_logger
    logger = init_logger(__name__)

    logger.info("Server started")
    logger.warning("Something looks wrong")
    logger.error("Something failed")
"""
import logging
import sys


_INITIALIZED = False


def _setup_root_logger():
    """Configure the nanovllm root logger once."""
    global _INITIALIZED
    if _INITIALIZED:
        return
    _INITIALIZED = True

    root = logging.getLogger("nanovllm")
    root.setLevel(logging.DEBUG)
    root.propagate = False

    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            fmt="%(asctime)s %(levelname)s [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        root.addHandler(handler)


def init_logger(name: str) -> logging.Logger:
    """Create a logger under the nanovllm namespace.

    Args:
        name: Typically ``__name__``, e.g. ``"nanovllm.server"``.

    Returns:
        A :class:`logging.Logger` instance.
    """
    _setup_root_logger()
    return logging.getLogger(name)
