import { useRef, useState, type ChangeEvent } from "react";
import { commitMask, createSession, requestPreview } from "./lib/api";
import { Inspector } from "./components/Inspector";
import { SplatWorkspace, type WorkspaceHandle } from "./components/SplatWorkspace";
import type { PromptMode, PromptPoint } from "./types";

export default function App() {
  const workspaceRef = useRef<WorkspaceHandle | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const [promptMode, setPromptMode] = useState<PromptMode>("positive");
  const [prompts, setPrompts] = useState<PromptPoint[]>([]);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [status, setStatus] = useState("Load a 3DGS PLY to start a session.");
  const [previewImage, setPreviewImage] = useState<string | null>(null);
  const [busyPreview, setBusyPreview] = useState(false);
  const [busyCommit, setBusyCommit] = useState(false);
  const [totalCount, setTotalCount] = useState<number | null>(null);
  const [visibleCount, setVisibleCount] = useState<number | null>(null);

  const handleFileSelection = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file || !workspaceRef.current) {
      return;
    }

    setStatus(`Loading ${file.name} into Spark and backend session…`);
    setPreviewImage(null);
    setPrompts([]);
    setSessionId(null);
    setTotalCount(null);
    setVisibleCount(null);

    try {
      const [session] = await Promise.all([
        createSession(file),
        workspaceRef.current.loadFile(file),
      ]);
      setSessionId(session.sessionId);
      setTotalCount(session.numSplats);
      setVisibleCount(session.numSplats);
      setStatus(`Loaded ${file.name}. Click on the scene to place prompts.`);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Failed to load file.";
      setStatus(message);
    } finally {
      event.target.value = "";
    }
  };

  const handlePromptAdd = (world: [number, number, number], label: 1 | -1) => {
    setPreviewImage(null);
    setPrompts((current) => [
      ...current,
      {
        id: crypto.randomUUID(),
        world,
        label,
      },
    ]);
  };

  const handlePreview = async () => {
    if (!sessionId || !workspaceRef.current) {
      return;
    }
    const camera = workspaceRef.current.buildCameraPayload();
    if (!camera) {
      setStatus("Viewer camera is not ready.");
      return;
    }

    setBusyPreview(true);
    setStatus("Projecting prompts and requesting segmentation preview…");
    try {
      const response = await requestPreview(sessionId, camera, prompts);
      setPreviewImage(response.previewImage);
      setVisibleCount(response.visibleCount);
      setStatus(`Preview ready. ${response.previewCount.toLocaleString()} splats matched.`);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Preview failed.";
      setStatus(message);
    } finally {
      setBusyPreview(false);
    }
  };

  const handleCommit = async (op: "union" | "invert" | "reset") => {
    if (!sessionId || !workspaceRef.current) {
      return;
    }

    setBusyCommit(true);
    setStatus(op === "reset" ? "Resetting visible splats…" : `Applying ${op} to the current mask…`);
    try {
      const response = await commitMask(sessionId, op);
      workspaceRef.current.applyVisibleMask(response.visibleMaskBitset);
      setVisibleCount(response.visibleCount);
      setPreviewImage(null);
      setPrompts([]);
      workspaceRef.current.clearPromptsVisuals();
      setStatus(
        op === "reset"
          ? "Selection reset. Full visibility restored."
          : `Applied ${op}. ${response.visibleCount.toLocaleString()} splats remain visible.`,
      );
    } catch (error) {
      const message = error instanceof Error ? error.message : "Commit failed.";
      setStatus(message);
    } finally {
      setBusyCommit(false);
    }
  };

  return (
    <div className="app-shell">
      <input
        ref={fileInputRef}
        className="visually-hidden"
        type="file"
        accept=".ply"
        onChange={handleFileSelection}
      />

      <main className="app-grid">
        <section className="workspace-panel">
          <SplatWorkspace
            ref={workspaceRef}
            promptMode={promptMode}
            prompts={prompts}
            onPromptAdd={handlePromptAdd}
            onSceneError={(message) => {
              if (message) {
                setStatus(message);
              }
            }}
            onSceneReady={(count) => {
              setTotalCount(count);
              setVisibleCount(count);
            }}
          />
        </section>

        <Inspector
          sessionReady={Boolean(sessionId)}
          sessionId={sessionId}
          promptMode={promptMode}
          prompts={prompts}
          previewImage={previewImage}
          hasPreview={previewImage !== null}
          promptCount={prompts.length}
          visibleCount={visibleCount}
          totalCount={totalCount}
          status={status}
          busyPreview={busyPreview}
          busyCommit={busyCommit}
          onChooseFile={() => fileInputRef.current?.click()}
          onPromptModeChange={setPromptMode}
          onPreview={handlePreview}
          onCommit={handleCommit}
          onClearPrompts={() => {
            setPrompts([]);
            setPreviewImage(null);
            workspaceRef.current?.clearPromptsVisuals();
            setStatus("Prompt stack cleared.");
          }}
        />
      </main>
    </div>
  );
}
