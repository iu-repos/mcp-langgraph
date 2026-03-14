# Model Context Protocol Playground

This project provides a basic code example how to use *LangGraph* as agentic orchestrator with a mix of 'local' tools and tools provided via simple (Fast-) MCP server. The example use case here is answering basic arithmetic user questions. The MCP client can be either called directly or via a Streamlit UI. 

�🚀 **What you'll find here:**
- **MCP Server** FastMCP with custom tools (multiply, add)
- **MCP Client** FastAPI with LangGraph agent orchestration  
- **Local Tools** (divide, power) alongside MCP tools
- **Streamlit Chat UI** to interact with your AI agent
- **Docker deployment** for easy containerized setup
- **Real-time tool usage** visibility and debugging

## Package structure

```
├── CHANGELOG.md
├── data
├── docker-compose.yml
├── Dockerfile.client
├── Dockerfile.server
├── Dockerfile.ui
├── Makefile
├── pyproject.toml
├── README.md
├── src
│   ├── mcp_sandbox
│   │   ├── config
│   │   ├── resources
│   │   ├── services
│   │   └── utils
│   └── notebooks
├── chat_ui.py
├── app_mcp_client.py
├── tests
```

## Install project

Optional: get [*uv*](https://github.com/astral-sh/uv) package manager
```bash
python -m pip install uv           
```

Install package dependencies for local development:
```bash
uv venv --python 3.13
source .venv/bin/activate 
uv sync --group dev
```

### Configure LLM proxy (optional)
This project can use an OpenAI-compatible proxy via `proxy_llm.py`. Put the following in a `.env` file in the repo root:

```env
PROXY_API_KEY=<your-api-key>
PROXY_BASE_URL=<your-proxy-base-url>  # no trailing slash
# Optional: override the proxy model (default: chatgpt-4o)
# PROXY_MODEL=chatgpt-4o
```

Get started locally:
```bash
make all-services                      # UI + MCP Client + MCP Server
make local-mcp-client                  # MCP Client
make local-mcp-server                  # MCP Server only
```

Run containerized stack:
```bash
make up-full                           # UI + MCP Client + MCP Server
```

## API Usage

Once the MCP client is running (locally on port 8080 or containerized on port 8081), you can interact with it directly using curl:

### Local development (port 8080):
```bash
# Simple calculation using MCP tools
curl -X POST http://localhost:8080/calculate \
  -H "Content-Type: application/json" \
  -d '{"query": "What is 15 multiplied by 7?"}'

# Complex calculation using both MCP and local tools
curl -X POST http://localhost:8080/calculate \
  -H "Content-Type: application/json" \
  -d '{"query": "Calculate 100 divided by 4, then multiply by 3"}'

# Power calculation using local tools
curl -X POST http://localhost:8080/calculate \
  -H "Content-Type: application/json" \
  -d '{"query": "What is 2 to the power of 10?"}'
```

### Containerized deployment (port 8081):
```bash
curl -X POST http://localhost:8081/calculate \
  -H "Content-Type: application/json" \
  -d '{"query": "Add 25 and 17, then multiply by 2"}'
```

The API will return a JSON response with the AI agent's answer:
```json
{
  "answer": "15 multiplied by 7 equals 105."
}
```
