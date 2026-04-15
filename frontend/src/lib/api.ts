import type {
  CameraPayload,
  CommitResponse,
  PreviewResponse,
  PromptPoint,
  SessionCreated,
} from "../types";

async function parseResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const data = (await response.json().catch(() => null)) as { detail?: string } | null;
    throw new Error(data?.detail ?? `Request failed with status ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export async function createSession(file: File): Promise<SessionCreated> {
  const body = new FormData();
  body.append("file", file);
  const response = await fetch("/api/sessions", {
    method: "POST",
    body,
  });
  return parseResponse<SessionCreated>(response);
}

export async function requestPreview(
  sessionId: string,
  camera: CameraPayload,
  imageDataUrl: string,
  points: PromptPoint[],
): Promise<PreviewResponse> {
  const response = await fetch(`/api/sessions/${sessionId}/preview`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      camera,
      imageDataUrl,
      points: points.map((point) => ({
        world: point.world,
        label: point.label,
      })),
    }),
  });
  return parseResponse<PreviewResponse>(response);
}

export async function commitMask(
  sessionId: string,
  op: "union" | "invert" | "reset",
): Promise<CommitResponse> {
  const response = await fetch(`/api/sessions/${sessionId}/commit`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ op }),
  });
  return parseResponse<CommitResponse>(response);
}

export function downloadUrl(sessionId: string, kind: "mask" | "ply"): string {
  return kind === "mask"
    ? `/api/sessions/${sessionId}/export/mask`
    : `/api/sessions/${sessionId}/export/ply`;
}
