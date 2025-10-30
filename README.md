## Thor Shipping Dashboard

This is a Streamlit app that visualizes AYN Thor shipping progress and estimates ship dates from your three‑digit order prefix.

### Prerequisites
- Python 3.12 (the project is pinned to 3.12)
- [uv](https://github.com/astral-sh/uv) package manager installed
- macOS/Linux or Windows

### First Time Setup
```bash
# From the project root
uv venv .venv --python=3.12
source .venv/bin/activate

# Install dependencies from lockfile / pyproject
uv sync
```

### Running the app
```bash
source .venv/bin/activate
uv run streamlit run thor_shipping_dashboard/app.py
```