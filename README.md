# spark_sam_viewer

Interactive local 3DGS segmentation tool built as a standalone Spark frontend + FastAPI backend project.

V1 scope:

- upload one 3DGS-specific `.ply`
- navigate the splat scene in real time in the browser
- click 3D positive / negative prompts on the scene
- request an async 2D SAM3 preview for the current view
- commit the preview into a persistent 3D visible mask
- download `mask.npy` or the filtered `.ply`

## Architecture

- `frontend/`: Vite + React + `@sparkjsdev/spark`
- `backend/`: FastAPI + in-project 3DGS PLY parser / projector + SAM3 predictor

The browser is responsible for realtime viewport interaction and prompt picking. Segmentation is asynchronous: the frontend sends the current camera, the current Spark viewport screenshot, and 3D prompt positions to the backend. The backend runs SAM3 on that screenshot, projects the 3D gaussians into the same camera, applies a simple z-buffer over gaussian centers, and maps the 2D mask back to the current visible 3D set.

## Requirements

- Python 3.10+
- Node.js 18+
- CUDA-capable GPU for SAM3 inference
- a local SAM3 checkpoint file
- an importable Python package named `sam3` in the backend environment

Important:

- use Python 3.11 or 3.12 for the backend environment; do not use Python 3.14 for this stack
- `SAM3_CHECKPOINT` must point to an existing local checkpoint path; this project does not download weights automatically
- uploaded `.ply` files must be 3DGS splat PLYs with gaussian attributes, not generic point cloud PLYs
- backend PyTorch is pinned to `torch==2.7.0+cu118`
- this repo no longer imports code from sibling repositories at runtime

## Environment

Backend reads these variables:

- `SAM3_CHECKPOINT`: absolute path to the local SAM3 checkpoint
- `SPARK_SAM_VIEWER_DEVICE`: device string used for SAM3 loading
- `SPARK_SAM_VIEWER_SESSION_ROOT`: where uploaded session files and exports are staged
- `SPARK_SAM_VIEWER_CORS_ORIGIN`: CORS origin, defaults to `*`

Example:

```bash
export SAM3_CHECKPOINT=/absolute/path/to/sam3.pt
export SPARK_SAM_VIEWER_DEVICE=cuda
```

## Backend

Install backend dependencies in your Python environment:

```bash
cd /Users/lanjie/Proj/3dgs/spark_sam_viewer/backend
pip install -r requirements.txt
```

If you are provisioning a fresh server, prefer a Python 3.11 virtual environment:

```bash
cd /Users/lanjie/Proj/3dgs/spark_sam_viewer/backend
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

The requirements file pulls PyTorch from the CUDA 11.8 wheel index and pins:

```text
torch==2.7.0+cu118
```

The backend also expects the `sam3` Python package to be installed into the same environment. This project does not add sibling repository paths to `sys.path`.

Start the API:

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8765
```

Health check:

```bash
curl http://127.0.0.1:8765/api/healthz
```

## Frontend

Install frontend dependencies:

```bash
cd /Users/lanjie/Proj/3dgs/spark_sam_viewer/frontend
npm install
```

Start Vite:

```bash
npm run dev
```

The frontend dev server runs on `http://127.0.0.1:5173` by default and proxies `/api` to the backend on `http://127.0.0.1:8765`.

## Interaction Flow

1. Load one 3DGS `.ply`.
2. Orbit / zoom in Spark.
3. Click the scene to place positive or negative 3D prompts.
4. Press `Preview` to render the current view and run SAM3.
5. Inspect the 2D preview in the right panel.
6. Press `Union` to keep the previewed object, `Invert` to subtract it, or `Reset` to restore the full scene.
7. Download the current mask or filtered `.ply`.

## Notes

- The right panel preview is view-dependent by design.
- The preview mask is generated from the current Spark canvas image, then mapped back onto the uploaded 3DGS points with projective visibility.
- `Union` builds the kept selection over time.
- `Invert` removes the current preview from the visible selection.
- `Reset` restores full visibility and clears the committed selection state.

## Verification

Minimal local checks that do not require downloading anything:

```bash
python3 -m compileall /Users/lanjie/Proj/3dgs/spark_sam_viewer/backend/app
```

If frontend dependencies are already installed:

```bash
cd /Users/lanjie/Proj/3dgs/spark_sam_viewer/frontend
npm run build
```
