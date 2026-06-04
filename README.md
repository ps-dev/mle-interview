# MLE Interview

## Setup

### Option A — GitHub Codespaces (recommended)

Click **Code → Open with Codespaces** on the repository page. The environment builds automatically — no local installation needed.

### Option B — Local Dev Container (VS Code)

1. Install [Visual Studio Code](https://code.visualstudio.com/) and the [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) extension
2. Install [Docker Desktop](https://www.docker.com/products/docker-desktop/)
3. Clone this repo and open it in VS Code (`code .`)
4. VS Code will prompt you to reopen in a Dev Container — select **Yes**
   - If you miss the prompt: `⇧⌘P` → **Dev Containers: Rebuild and Reopen in Container**

> **Note (corporate networks / Zscaler):** If your machine uses Zscaler SSL inspection, ensure your Zscaler certificate bundle exists at `~/.ssl/zscaler-bundle.pem` before building the container. The build handles this automatically.

Once the container starts, dependencies are installed automatically via `make setup`. You do not need to run it manually.


## Make Commands

### Test

```
make test
```

Runs all tests in `./tests` with pytest. Use this to verify each task as you work through the interview.

### Train

```
make train
```

Trains the model using `model/main.py` and exports a SavedModel to `serving/interests-model/1/`.

### Serve Model

```
make serve-model
```

Starts TF Serving via Docker Compose on port 8501. Run this after `make train` — the model must be trained first.

### Serve API

```
make serve-api
```

Starts the Flask development server on port 5005. Requires the model server to be running for inference endpoints to work.


## API

```
GET http://localhost:5005/interests/<user_handle>
```

Optional query parameters:
- `top_k` (int) — number of interests to return (default: 10)
- `min_probability` (float) — filter results below this probability score

Example response:

```json
{
    "user_handle": "e337a675-46f5-437e-aa72-5d43643b5461",
    "name": "Kisiza",
    "type": "B2B",
    "interests": [
        {
            "id": "29e020bb-7a02-41db-b21f-527a8ef4dfdf",
            "label": "Accounting technician",
            "probability": 0.94
        },
        {
            "id": "012abcd1-3a6a-4803-a47e-42f46b402024",
            "label": "Field seismologist",
            "probability": 0.87
        },
        {
            "id": "a1b028dd-8464-4c63-85e8-ae29ea184fc7",
            "label": "Designer, industrial/product",
            "probability": 0.81
        }
    ]
}
```
