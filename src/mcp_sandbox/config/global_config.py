import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

package_root = Path(__file__).parent.parent.parent.parent

# Path-derived defaults work in both local and Docker layouts because package_root
# resolves from this installed module location.
defaults = {
    "CODE_DIR": str(package_root / "src" / "mcp_sandbox"),
    "DATA_PKG_DIR": str(package_root / "data"),
}

# -----------------------------------------------------------
# Set environment variables with defaults if not already set
for env, value in defaults.items():
    os.environ.setdefault(env, value)

CODE_DIR: str = os.environ["CODE_DIR"]
DATA_PKG_DIR: str = os.environ["DATA_PKG_DIR"]
