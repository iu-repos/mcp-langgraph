import uvicorn
from typing import TypedDict, Annotated, Sequence, Dict
from fastapi import FastAPI
from contextlib import asynccontextmanager
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage
from langchain.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages
from mcp_sandbox.config import global_config as glob
from mcp_sandbox.services.logger import LoggerFactory
from mcp_sandbox.utils.proxy_chat_model import ProxyChatModel

logger = LoggerFactory(handler_type="Stream", verbose=True).create_module_logger()


class AgentState(TypedDict):
    messages: Annotated[Sequence[AIMessage | HumanMessage], add_messages]


class AskRequest(BaseModel):
    query: str


# Define local tools (i.e. Non-MCP server tools)
# -----------------------------------------------
@tool
def local_divide(a: float, b: float) -> float:
    """
    Divides two floating-point numbers and logs the operation and result.
    Handles division by zero by returning infinity.

    Args:
        a (float): The dividend (number to be divided).
        b (float): The divisor (number to divide by).

    Returns:
        float: The quotient of a divided by b, or infinity if b is zero.
    """
    logger.info(f"➗ DIVIDE CALLED: {a} / {b}")
    if b == 0:
        logger.warning(f"⚠️  DIVIDE WARNING: Division by zero, returning infinity")
        result = float("inf")
    else:
        result = a / b
    logger.info(f"➗ DIVIDE RESULT: {result}")
    return result


@tool
def local_power(a: float, b: float) -> float:
    """
    Raises a floating-point number to the power of another and logs the operation and result.
    Note: Power operation implies exponentiation and root operations.

    Args:
        a (float): The base number to be raised to a power.
        b (float): The exponent (power to raise the base to).

    Returns:
        float: The result of a raised to the power of b.
    """
    logger.info(f"🔀 POWER CALLED: {a} ^ {b}")
    result = float(a**b)
    logger.info(f"🔀 POWER RESULT: {result}")
    return result


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting MCP client...")
    try:
        client = MultiServerMCPClient(
            {
                "mytools": {
                    "command": "python",
                    "args": [f"{glob.CODE_DIR}/resources/mcp_server.py"],
                    "transport": "stdio",
                }
            }
        )
        mcp_tools = await client.get_tools(server_name="mytools")
        logger.info(f"✅ Loaded {len(mcp_tools)} MCP tools.")

        local_tools = [local_divide, local_power]
        all_tools = local_tools + mcp_tools

        logger.info(f"🛠️  Total tools available: {len(all_tools)}")

        llm = ProxyChatModel()

        model_with_tools = llm.bind_tools(all_tools)
        tool_node = ToolNode(all_tools)

        # Create the state graph
        # -----------------------
        logger.info("📊 Creating state graph for agent...")
        graph = StateGraph(AgentState)
        graph.add_node("call_model", lambda state: call_model(state, model_with_tools))
        graph.add_node("tools", tool_node)
        graph.add_edge(START, "call_model")
        graph.add_conditional_edges("call_model", tools_condition)
        graph.add_edge("tools", "call_model")
        graph.set_finish_point("call_model")
        app.state.graph = graph.compile()
        yield
    except Exception as e:
        logger.exception(f"❌ Error during MCP client startup: {e}")
        raise


app = FastAPI(title="MCP Client (FastAPI)", lifespan=lifespan, version="0.1.0")


@app.get("/")
def health():
    status = f"Hi there, your service is up! 😊 Version = {app.version}"
    return status


@app.post("/calculate")
async def ask(req: AskRequest) -> Dict[str, str]:
    query = req.query
    logger.info("[CLIENT] /calculate request received")
    logger.info("[CLIENT] User query: %s", query)
    state = {"messages": [HumanMessage(content=query)]}
    # Run the graph asynchronously
    final = await app.state.graph.ainvoke(state)
    answer = final["messages"][-1].content
    logger.info("[CLIENT] Final answer: %s", answer)
    logger.info("[CLIENT] /calculate request completed")
    return {"answer": answer}


def call_model(state: AgentState, model_with_tools) -> Dict[str, Sequence[AIMessage]]:
    """
    Calls a language model with the provided agent state and model interface, handling tool calls and logging.

    Args:
        state (AgentState): The current agent state containing at least a 'messages' key with the conversation history.
        model_with_tools: The model interface object that supports an 'invoke' method for message processing.

    Returns:
        dict: A dictionary containing the model's response message(s) under the 'messages' key.

    Raises:
        Exception: Propagates any exception encountered during model invocation after logging the error.
    """
    try:
        message_count = len(state.get("messages", []))
        logger.info("[GRAPH] call_model invoked with %s message(s)", message_count)

        response = model_with_tools.invoke(state["messages"])
        logger.info("[GRAPH] Model response: %s", response.content)

        if hasattr(response, "tool_calls") and response.tool_calls:
            logger.info("[GRAPH] Tool call count: %s", len(response.tool_calls))
            logger.info(
                "[GRAPH] Tool calls requested: %s",
                [call["name"] for call in response.tool_calls],
            )
            for idx, call in enumerate(response.tool_calls, start=1):
                logger.info(
                    "[GRAPH] Tool call %s -> id=%s name=%s args=%s",
                    idx,
                    call.get("id"),
                    call.get("name"),
                    call.get("args"),
                )
        else:
            logger.info("[GRAPH] No tool call requested by model")

        return {"messages": [response]}
    except Exception as e:
        logger.exception(f"❌ Error during model invocation: {e}")
        raise


_LOG_FMT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
_ACCESS_FMT = '%(asctime)s - %(name)s - %(levelname)s - %(client_addr)s - "%(request_line)s" %(status_code)s'

_UVICORN_LOG_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {"()": "logging.Formatter", "fmt": _LOG_FMT},
        "access": {"()": "uvicorn.logging.AccessFormatter", "fmt": _ACCESS_FMT},
    },
    "handlers": {
        "default": {
            "formatter": "default",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stderr",
        },
        "access": {
            "formatter": "access",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
    },
    "loggers": {
        "uvicorn": {"handlers": ["default"], "level": "INFO", "propagate": False},
        "uvicorn.error": {"handlers": ["default"], "level": "INFO", "propagate": False},
        "uvicorn.access": {"handlers": ["access"], "level": "INFO", "propagate": False},
    },
}

if __name__ == "__main__":
    uvicorn.run(
        "app_mcp_client:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_config=_UVICORN_LOG_CONFIG,
    )
