import inspect
import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Literal, Optional


class AbstractLoggerFactory(ABC):
    def __init__(
        self,
        handler_type: Literal["File", "Stream", "GCP"] = "Stream",
        filename: str = "logger_file.log",
        verbose: bool = False,
    ):
        self.handler_type = handler_type
        self.filename = filename
        self.verbose = verbose

    @abstractmethod
    def create_module_logger(
        self,
        module_name: Optional[str] = None,
        **kwargs: Optional[dict],
    ) -> logging.Logger:
        """
        Creates a logger instance for a specific module.

        Args:
            module_name (str, optional): Name of the module to create logger for. Defaults to None.
            **kwargs: Additional keyword arguments to configure the logger.

        Returns:
            logging.Logger: Logger instance configured for the specified module.

        Raises:
            NotImplementedError: If the subclass does not implement this abstract method.
        """

        raise NotImplementedError(
            "Subclasses must implement create_module_logger method."
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(handler_type={self.handler_type!r}, filename={self.filename!r}, verbose={self.verbose!r})"


class LoggerFactory(AbstractLoggerFactory):
    """
    LoggerFactory is responsible for creating loggers for different modules with various handler types.

    Methods
    -------
    create_module_logger(module_name: Optional[str] = None, **kwargs) -> logging.Logger
        Creates and returns a logger for the specified module with the given handler type.

    Parameters
    ----------
    module_name : Optional[str], optional
        The name of the module for which the logger is being created. Defaults to the current module name.
    **kwargs
        Additional keyword arguments to be passed to the handler.

    Returns
    -------
    logging.Logger
        A configured logger instance for the specified module.
    """

    def __init__(
        self,
        handler_type: Literal["File", "Stream", "GCP"] = "Stream",
        filename: str = "logger_file.log",
        verbose: bool = True,
    ):
        super().__init__(handler_type, filename, verbose)

    @staticmethod
    def _resolve_level(level: Optional[str | int], verbose: bool) -> int:
        if isinstance(level, int):
            return level

        if isinstance(level, str):
            configured = level.strip().upper()
        else:
            configured = os.getenv("MCP_LOG_LEVEL", "").strip().upper()

        if configured:
            parsed = logging.getLevelName(configured)
            if isinstance(parsed, int):
                return parsed
            raise ValueError(f"Invalid log level: {configured}")

        return logging.DEBUG if verbose else logging.INFO

    @staticmethod
    def _remove_managed_handlers(logger: logging.Logger) -> None:
        managed_handlers = [
            handler
            for handler in logger.handlers
            if getattr(handler, "_mcp_logger_factory_managed", False)
        ]
        for handler in managed_handlers:
            logger.removeHandler(handler)
            handler.close()

    @staticmethod
    def _caller_module_name() -> str:
        frame = inspect.stack()[2]
        module = inspect.getmodule(frame[0])
        return module.__name__ if module else "unknown"

    def create_module_logger(
        self,
        module_name: Optional[str] = None,
        **kwargs: Any,
    ) -> logging.Logger:
        if module_name is None:
            # Dynamically get the name of the calling module.
            module_name = self._caller_module_name()

        level = self._resolve_level(kwargs.pop("level", None), self.verbose)
        log_format = kwargs.pop(
            "log_format",
            os.getenv(
                "MCP_LOG_FORMAT",
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            ),
        )
        datefmt = kwargs.pop("datefmt", os.getenv("MCP_LOG_DATEFMT") or None)
        propagate = kwargs.pop("propagate", False)
        force_reconfigure = kwargs.pop("force_reconfigure", True)

        logger = logging.getLogger(module_name)
        logger.setLevel(level)
        logger.propagate = propagate

        if force_reconfigure:
            self._remove_managed_handlers(logger)
        else:
            for handler in logger.handlers:
                if getattr(handler, "_mcp_logger_factory_managed", False):
                    return logger

        handler: logging.Handler
        match self.handler_type:
            case "File":
                log_dir = os.path.join(os.path.dirname(__file__), "logging")
                os.makedirs(log_dir, exist_ok=True)
                file_path = os.path.join(log_dir, self.filename)
                kwargs.setdefault("encoding", "utf-8")
                handler = logging.FileHandler(filename=file_path, **kwargs)
            case "Stream":
                handler = logging.StreamHandler(**kwargs)
            case "GCP":
                try:
                    from google.cloud.logging import Client
                    from google.cloud.logging_v2.handlers import CloudLoggingHandler

                    client = Client()
                    handler = CloudLoggingHandler(client, **kwargs)
                except Exception as e:
                    raise RuntimeError(
                        "Failed to create GCP logging handler. Ensure 'google-cloud-logging' is installed and credentials are configured."
                    ) from e
            case _:
                raise ValueError(f"Invalid handler type: {self.handler_type}")

        formatter = logging.Formatter(log_format, datefmt=datefmt)
        handler.setFormatter(formatter)
        handler._mcp_logger_factory_managed = True  # type: ignore[attr-defined]
        logger.addHandler(handler)

        return logger

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(handler_type={self.handler_type!r}, filename={self.filename!r}, verbose={self.verbose!r})"
