# Legacy exchange arbitrage

This directory contains an archival, unmaintained exchange-arbitrage subsystem. It is excluded from the uv workspace, the default package installation, and the default test suite.

The subsystem requires separate dependency and configuration recovery before it can be used. It expects a `config.json` file, performs exchange and network discovery, and includes order and account operations. No live credentials or secrets should be added to this repository.

`main.py` invokes `run_real_arb` during module execution. Do not import or run it without first completing a code, security, and credential review.
