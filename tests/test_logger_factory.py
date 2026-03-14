import logging
from pathlib import Path

from mcp_sandbox.services.logger import LoggerFactory


def test_create_module_logger_is_idempotent_for_managed_handlers() -> None:
    factory = LoggerFactory(handler_type="Stream", verbose=False)
    logger = factory.create_module_logger(
        module_name="tests.logger.idempotent",
        force_reconfigure=False,
    )
    factory.create_module_logger(
        module_name="tests.logger.idempotent",
        force_reconfigure=False,
    )

    managed_handlers = [
        handler
        for handler in logger.handlers
        if getattr(handler, "_mcp_logger_factory_managed", False)
    ]

    assert len(managed_handlers) == 1
    assert logger.level == logging.INFO


def test_file_handler_writes_log_entry() -> None:
    log_name = "pytest_logger_quality.log"
    factory = LoggerFactory(handler_type="File", filename=log_name, verbose=True)
    logger = factory.create_module_logger(
        module_name="tests.logger.file",
        force_reconfigure=True,
    )

    message = "logger quality smoke test"
    logger.info(message)

    log_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "mcp_sandbox"
        / "services"
        / "logging"
        / log_name
    )

    assert log_path.exists()
    assert message in log_path.read_text(encoding="utf-8")
