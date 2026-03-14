#!/usr/bin/env python3
try:
    from mcp_sandbox.utils.proxy_chat_model import ProxyChatModel

    print("Import successful")
    model = ProxyChatModel()
    print("Instantiation successful")
    print(f"Model name: {model.model_name}")
except Exception as e:
    print(f"Error: {e}")
    import traceback

    traceback.print_exc()
