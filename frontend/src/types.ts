export type PromptMode = "positive" | "negative";

export type PromptPoint = {
  id: string;
  world: [number, number, number];
  screen: [number, number];
  label: 1 | -1;
};

export type CameraPayload = {
  width: number;
  height: number;
  aspect: number;
  fovY: number;
  near: number;
  far: number;
  position: [number, number, number];
  quaternion: [number, number, number, number];
  viewMatrix: number[][];
  projectionMatrix: number[][];
};

export type SessionCreated = {
  sessionId: string;
  numSplats: number;
  bbox: {
    min: [number, number, number];
    max: [number, number, number];
    center: [number, number, number];
  };
  warnings: string[];
};

export type PreviewResponse = {
  previewImage: string;
  previewCount: number;
  previewMaskBitset: string;
  visibleCount: number;
  visibleMaskBitset: string;
};

export type CommitResponse = {
  visibleCount: number;
  visibleMaskBitset: string;
};
