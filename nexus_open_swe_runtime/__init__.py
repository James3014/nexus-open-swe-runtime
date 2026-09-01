"""External Open SWE execution runtime for Nexus."""

from .opencli_web_model import OpenCLIWebChatModel, OpenCLIWebModelError

__version__ = "0.1.0"

__all__ = ["OpenCLIWebChatModel", "OpenCLIWebModelError", "__version__"]
