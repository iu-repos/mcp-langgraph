from mcp.server import FastMCP
from mcp_sandbox.services.logger import LoggerFactory

logger = LoggerFactory(handler_type="Stream", verbose=True).create_module_logger()

server = FastMCP("Math-Server", host="0.0.0.0", port=8000)


@server.tool()
def multiply(a: float, b: float) -> float:
    """
    Multiplies two floating-point numbers and logs the operation and result.

    Args:
        a (float): The first number to multiply.
        b (float): The second number to multiply.

    Returns:
        float: The product of a and b.
    """
    logger.info("[MCP_SERVER] TOOL_START multiply a=%s b=%s", a, b)
    result = a * b
    logger.info("[MCP_SERVER] TOOL_END multiply result=%s", result)
    return result


@server.tool()
def add(a: float, b: float) -> float:
    """
    Adds two floating-point numbers and logs the operation and result.
    Note: Addition implies substraction.

    Args:
        a (float): The first number to add.
        b (float): The second number to add.

    Returns:
        float: The sum of `a` and `b`.
    """
    logger.info("[MCP_SERVER] TOOL_START add a=%s b=%s", a, b)
    result = a + b
    logger.info("[MCP_SERVER] TOOL_END add result=%s", result)
    return result


if __name__ == "__main__":
    logger.info("[MCP_SERVER] Server starting")
    server.run()
