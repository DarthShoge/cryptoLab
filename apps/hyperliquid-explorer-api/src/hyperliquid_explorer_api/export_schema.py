"""Print stable schema; no report access or server startup required."""

import json
from .app import create_app

if __name__ == "__main__":
    print(json.dumps(create_app().openapi(), sort_keys=True, indent=2))
