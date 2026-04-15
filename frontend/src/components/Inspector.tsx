import type { PromptMode, PromptPoint } from "../types";

type InspectorProps = {
  sessionReady: boolean;
  promptMode: PromptMode;
  prompts: PromptPoint[];
  previewImage: string | null;
  hasPreview: boolean;
  promptCount: number;
  visibleCount: number | null;
  totalCount: number | null;
  status: string;
  busyPreview: boolean;
  busyCommit: boolean;
  onChooseFile: () => void;
  onPromptModeChange: (mode: PromptMode) => void;
  onPreview: () => void;
  onCommit: (op: "union" | "invert" | "reset") => void;
  onClearPrompts: () => void;
  sessionId: string | null;
};

export function Inspector(props: InspectorProps) {
  const {
    sessionReady,
    promptMode,
    prompts,
    previewImage,
    hasPreview,
    promptCount,
    visibleCount,
    totalCount,
    status,
    busyPreview,
    busyCommit,
    onChooseFile,
    onPromptModeChange,
    onPreview,
    onCommit,
    onClearPrompts,
    sessionId,
  } = props;

  return (
    <aside className="inspector">
      <div className="inspector__hero">
        <p className="eyebrow">Spark + SAM3</p>
        <h1>3DGS Object Cut</h1>
        <p className="hero-copy">
          Load one Gaussian PLY, click positive and negative prompts in 3D, preview the
          projected mask, then keep or subtract the result without reloading the splat field.
        </p>
      </div>

      <div className="inspector__section">
        <button className="primary-button" onClick={onChooseFile}>
          Load 3DGS PLY
        </button>
        <div className="metric-row">
          <span>Prompts</span>
          <strong>{promptCount}</strong>
        </div>
        <div className="metric-row">
          <span>Visible splats</span>
          <strong>
            {visibleCount === null || totalCount === null ? "—" : `${visibleCount.toLocaleString()} / ${totalCount.toLocaleString()}`}
          </strong>
        </div>
        <p className="status-copy">{status}</p>
      </div>

      <div className="inspector__section">
        <p className="section-label">Prompt polarity</p>
        <div className="segmented">
          <button
            className={promptMode === "positive" ? "is-active" : ""}
            onClick={() => onPromptModeChange("positive")}
          >
            Positive
          </button>
          <button
            className={promptMode === "negative" ? "is-active" : ""}
            onClick={() => onPromptModeChange("negative")}
          >
            Negative
          </button>
        </div>
        <div className="action-row">
          <button disabled={!sessionReady || promptCount === 0 || busyPreview} onClick={onPreview}>
            {busyPreview ? "Previewing…" : "Preview"}
          </button>
          <button disabled={promptCount === 0} onClick={onClearPrompts}>
            Clear points
          </button>
        </div>
      </div>

      <div className="inspector__section">
        <p className="section-label">Selection flow</p>
        <div className="action-grid">
          <button disabled={!sessionReady || !hasPreview || busyCommit} onClick={() => onCommit("union")}>
            {busyCommit ? "Applying…" : "Union"}
          </button>
          <button disabled={!sessionReady || !hasPreview || busyCommit} onClick={() => onCommit("invert")}>
            Invert
          </button>
          <button disabled={!sessionReady || busyCommit} onClick={() => onCommit("reset")}>
            Reset
          </button>
        </div>
      </div>

      <div className="inspector__section preview-panel">
        <div className="preview-header">
          <p className="section-label">Preview</p>
          <span>{previewImage ? "Current view" : "Awaiting request"}</span>
        </div>
        <div className={`preview-frame ${previewImage ? "has-image" : ""}`}>
          {previewImage ? (
            <img src={previewImage} alt="Segmentation preview" />
          ) : (
            <p>Preview overlays land here after a segmentation request.</p>
          )}
        </div>
      </div>

      <div className="inspector__section">
        <p className="section-label">Prompt stack</p>
        <div className="prompt-list">
          {prompts.length === 0 ? (
            <p className="empty-copy">Click directly on the splats to add prompts.</p>
          ) : (
            prompts.map((prompt, index) => (
              <div className="prompt-row" key={prompt.id}>
                <span className={`prompt-dot ${prompt.label > 0 ? "positive" : "negative"}`} />
                <div>
                  <strong>{prompt.label > 0 ? "Keep" : "Reject"}</strong>
                  <p>
                    #{index + 1} · {prompt.world.map((value) => value.toFixed(3)).join(", ")}
                  </p>
                </div>
              </div>
            ))
          )}
        </div>
      </div>

      <div className="inspector__footer">
        <a
          className={sessionId ? "download-link" : "download-link is-disabled"}
          href={sessionId ? `/api/sessions/${sessionId}/export/mask` : undefined}
        >
          Download mask
        </a>
        <a
          className={sessionId ? "download-link" : "download-link is-disabled"}
          href={sessionId ? `/api/sessions/${sessionId}/export/ply` : undefined}
        >
          Download filtered PLY
        </a>
      </div>
    </aside>
  );
}
