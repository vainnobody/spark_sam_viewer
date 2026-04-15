# spark_sam_viewer

Interactive local 3DGS segmentation tool built with a Spark frontend and a FastAPI backend.

V1 scope:

- upload one 3DGS-specific `.ply`
- navigate the splat scene in real time in the browser
- click 3D positive / negative prompts on the scene
- request an async 2D SAM3 preview for the current view
- commit the preview into a persistent 3D visible mask
- download `mask.npy` or the filtered `.ply`

## Architecture

- `frontend/`: Vite + React + `@sparkjsdev/spark`
- `backend/`: FastAPI + `semantic-gaussians` renderer + local `sam3`

The browser is responsible for realtime viewport interaction and prompt picking. Segmentation is asynchronous: the frontend sends the current camera and 3D prompt positions to the backend, the backend renders the matching 2D view, runs SAM3, projects the 2D mask back to visible gaussians, and returns a preview.

## Requirements

- Python 3.10+
- Node.js 18+
- CUDA-capable GPU
- local `semantic-gaussians/` repo under the same `/Users/lanjie/Proj/3dgs` workspace, or override with env vars below
- local `sam3/` repo under the same `/Users/lanjie/Proj/3dgs` workspace, or override with env vars below
- a local SAM3 checkpoint file

Important:

- the backend depends on `semantic-gaussians`, which uses CUDA-only rasterization extensions in this repo layout
- `SAM3_CHECKPOINT` must point to an existing local checkpoint path; this project does not download weights automatically
- uploaded `.ply` files must be 3DGS splat PLYs with gaussian attributes, not generic point cloud PLYs

## Environment

Backend reads these variables:

- `SAM3_CHECKPOINT`: absolute path to the local SAM3 checkpoint
- `SPARK_SAM_VIEWER_DEVICE`: device string used for SAM3 loading; keep this on CUDA for the default repo stack
- `SPARK_SAM_VIEWER_REPO_ROOT`: override repo root if `semantic-gaussians/` and `sam3/` are elsewhere
- `SPARK_SAM_VIEWER_SEMANTIC_ROOT`: explicit path to `semantic-gaussians`
- `SPARK_SAM_VIEWER_SAM3_ROOT`: explicit path to `sam3`
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
