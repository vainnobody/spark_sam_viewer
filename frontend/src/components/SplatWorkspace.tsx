import {
  forwardRef,
  useEffect,
  useImperativeHandle,
  useRef,
} from "react";
import * as THREE from "three";
import {
  SparkControls,
  SparkRenderer,
  SplatMesh,
} from "@sparkjsdev/spark";
import type { CameraPayload, PromptMode, PromptPoint } from "../types";
import { decodeMaskBitset } from "../lib/bitset";
import { matrixToRows } from "../lib/camera";

type WorkspaceProps = {
  promptMode: PromptMode;
  prompts: PromptPoint[];
  onPromptAdd: (world: [number, number, number], label: 1 | -1) => void;
  onSceneError: (message: string | null) => void;
  onSceneReady: (count: number) => void;
};

export type WorkspaceHandle = {
  loadFile: (file: File) => Promise<void>;
  buildCameraPayload: () => CameraPayload | null;
  captureViewImage: () => string | null;
  applyVisibleMask: (encoded: string) => void;
  clearPromptsVisuals: () => void;
};

type SceneRefs = {
  renderer: THREE.WebGLRenderer | null;
  scene: THREE.Scene | null;
  camera: THREE.PerspectiveCamera | null;
  controls: SparkControls | null;
  spark: SparkRenderer | null;
  mesh: SplatMesh | null;
  originalOpacity: Float32Array | null;
  numSplats: number;
  markers: THREE.Group | null;
  animationId: number | null;
};

type PackedSplatsHandle = {
  numSplats: number;
};

type PackedSplatRecord = {
  center: THREE.Vector3;
  scales: THREE.Vector3;
  quaternion: THREE.Quaternion;
  opacity: number;
  color: THREE.Color;
};

type MutablePackedSplats = {
  needsUpdate: boolean;
  getSplat: (index: number) => PackedSplatRecord;
  setSplat: (
    index: number,
    center: THREE.Vector3,
    scales: THREE.Vector3,
    quaternion: THREE.Quaternion,
    opacity: number,
    color: THREE.Color,
  ) => void;
};

type MutableSplatMesh = SplatMesh & {
  initialized: Promise<unknown>;
  packedSplats?: PackedSplatsHandle & MutablePackedSplats;
  needsUpdate: boolean;
  updateGenerator: () => void;
  updateVersion?: () => void;
  forEachSplat?: (
    callback: (
      index: number,
      center: THREE.Vector3,
      scales: THREE.Vector3,
      quaternion: THREE.Quaternion,
      opacity: number,
      color: THREE.Color,
    ) => void,
  ) => void;
  getBoundingBox: (includeAll?: boolean) => THREE.Box3;
};

export const SplatWorkspace = forwardRef<WorkspaceHandle, WorkspaceProps>(function SplatWorkspace(
  props,
  ref,
) {
  const { promptMode, prompts, onPromptAdd, onSceneError, onSceneReady } = props;
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const sceneRef = useRef<SceneRefs>({
    renderer: null,
    scene: null,
    camera: null,
    controls: null,
    spark: null,
    mesh: null,
    originalOpacity: null,
    numSplats: 0,
    markers: null,
    animationId: null,
  });
  const raycasterRef = useRef(new THREE.Raycaster());

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) {
      return;
    }

    const renderer = new THREE.WebGLRenderer({
      canvas,
      antialias: false,
      alpha: true,
      preserveDrawingBuffer: true,
    });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setClearColor(new THREE.Color("#06131c"), 1);

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(48, 1, 0.01, 1000);
    camera.position.set(1.6, 1.2, 1.8);
    camera.lookAt(0, 0, 0);

    const spark = new SparkRenderer({ renderer });
    const controls = new SparkControls({ canvas: renderer.domElement });
    const markers = new THREE.Group();
    scene.add(spark);
    scene.add(camera);
    scene.add(markers);

    sceneRef.current = {
      ...sceneRef.current,
      renderer,
      scene,
      camera,
      controls,
      spark,
      markers,
    };

    const resize = () => {
      const width = canvas.clientWidth || 1;
      const height = canvas.clientHeight || 1;
      renderer.setSize(width, height, false);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
    };

    const animate = () => {
      const now = performance.now() * 0.001;
      markers.children.forEach((child, index) => {
        const scale = 1 + Math.sin(now * 3.5 + index * 0.5) * 0.08;
        child.scale.setScalar(scale);
      });
      controls.update(camera);
      renderer.render(scene, camera);
      sceneRef.current.animationId = requestAnimationFrame(animate);
    };

    resize();
    window.addEventListener("resize", resize);
    sceneRef.current.animationId = requestAnimationFrame(animate);

    return () => {
      window.removeEventListener("resize", resize);
      if (sceneRef.current.animationId !== null) {
        cancelAnimationFrame(sceneRef.current.animationId);
      }
      disposeGroup(markers);
      sceneRef.current.mesh?.dispose();
      renderer.dispose();
    };
  }, []);

  useEffect(() => {
    const markers = sceneRef.current.markers;
    if (!markers) {
      return;
    }
    disposeGroup(markers);
    prompts.forEach((prompt) => {
      if (prompt.world.some((value) => !Number.isFinite(value))) {
        return;
      }
      const geometry = new THREE.SphereGeometry(0.026, 18, 18);
      const material = new THREE.MeshBasicMaterial({
        color: prompt.label > 0 ? "#ff6a3d" : "#2dc6ff",
      });
      const mesh = new THREE.Mesh(geometry, material);
      mesh.position.set(...prompt.world);
      markers.add(mesh);
    });
  }, [prompts]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) {
      return;
    }

    const onPointerDown = (event: PointerEvent) => {
      try {
        const { camera, scene, mesh } = sceneRef.current;
        if (!camera || !scene || !mesh) {
          return;
        }
        const rect = canvas.getBoundingClientRect();
        const pointer = new THREE.Vector2(
          ((event.clientX - rect.left) / rect.width) * 2 - 1,
          -((event.clientY - rect.top) / rect.height) * 2 + 1,
        );
        const raycaster = raycasterRef.current;
        raycaster.setFromCamera(pointer, camera);
        const hits = raycaster.intersectObject(mesh, true);
        if (hits.length === 0) {
          return;
        }
        const point = hits[0]?.point;
        if (
          !point
          || !Number.isFinite(point.x)
          || !Number.isFinite(point.y)
          || !Number.isFinite(point.z)
        ) {
          onSceneError("Spark raycast returned an invalid prompt point. Try clicking a denser region of the splat.");
          return;
        }
        onPromptAdd(
          [point.x, point.y, point.z],
          promptMode === "positive" ? 1 : -1,
        );
      } catch (error) {
        const message = error instanceof Error ? error.message : "Unknown click interaction error.";
        onSceneError(`Click interaction failed: ${message}`);
      }
    };

    canvas.addEventListener("pointerdown", onPointerDown);
    return () => canvas.removeEventListener("pointerdown", onPointerDown);
  }, [onPromptAdd, promptMode]);

  useImperativeHandle(
    ref,
    () => ({
      async loadFile(file: File) {
        const { scene, mesh: currentMesh, camera, controls, markers } = sceneRef.current;
        if (!scene || !camera || !controls || !markers) {
          throw new Error("Workspace is not initialized.");
        }

        if (currentMesh) {
          scene.remove(currentMesh);
          currentMesh.dispose();
        }
        disposeGroup(markers);
        sceneRef.current.originalOpacity = null;
        sceneRef.current.mesh = null;
        sceneRef.current.numSplats = 0;
        onSceneError(null);

        try {
          const fileBytes = await file.arrayBuffer();
          const mesh = new SplatMesh({
            fileBytes,
            fileName: file.name,
            raycastable: true,
          }) as MutableSplatMesh;
          await mesh.initialized;
          const packedSplats = mesh.packedSplats;
          if (!packedSplats) {
            throw new Error("Spark did not expose packed splats for this file.");
          }

          scene.add(mesh);
          sceneRef.current.mesh = mesh;
          sceneRef.current.originalOpacity = extractOriginalOpacity(mesh, packedSplats.numSplats);
          sceneRef.current.numSplats = packedSplats.numSplats;
          fitCameraToMesh(camera, controls, mesh);
          onSceneReady(packedSplats.numSplats);
        } catch (error) {
          const message = error instanceof Error ? error.message : "Failed to load splat file.";
          onSceneError(message);
          throw error;
        }
      },

      buildCameraPayload() {
        const { camera, renderer, controls, scene } = sceneRef.current;
        if (!camera || !renderer || !controls || !scene) {
          return null;
        }
        renderCurrentView(renderer, scene, camera, controls);

        const size = new THREE.Vector2();
        renderer.getDrawingBufferSize(size);
        camera.updateMatrixWorld();
        camera.updateProjectionMatrix();
        return {
          width: Math.max(1, Math.floor(size.x)),
          height: Math.max(1, Math.floor(size.y)),
          aspect: camera.aspect,
          fovY: THREE.MathUtils.degToRad(camera.fov),
          near: camera.near,
          far: camera.far,
          position: [camera.position.x, camera.position.y, camera.position.z],
          quaternion: [
            camera.quaternion.x,
            camera.quaternion.y,
            camera.quaternion.z,
            camera.quaternion.w,
          ],
          viewMatrix: matrixToRows(camera.matrixWorldInverse),
          projectionMatrix: matrixToRows(camera.projectionMatrix),
        };
      },

      captureViewImage() {
        const { renderer, scene, camera, controls, markers } = sceneRef.current;
        if (!renderer || !scene || !camera || !controls) {
          return null;
        }

        const markersWereVisible = markers?.visible ?? false;
        if (markers) {
          markers.visible = false;
        }

        try {
          renderCurrentView(renderer, scene, camera, controls);
          return renderer.domElement.toDataURL("image/jpeg", 0.9);
        } finally {
          if (markers) {
            markers.visible = markersWereVisible;
          }
          renderCurrentView(renderer, scene, camera, controls);
        }
      },

      applyVisibleMask(encoded: string) {
        const { originalOpacity, numSplats, mesh } = sceneRef.current;
        if (!originalOpacity || !mesh) {
          return;
        }
        const mutableMesh = mesh as MutableSplatMesh;
        const packedSplats = mutableMesh.packedSplats;
        if (!packedSplats) {
          return;
        }
        const mask = decodeMaskBitset(encoded, numSplats);
        for (let index = 0; index < numSplats; index += 1) {
          const splat = packedSplats.getSplat(index);
          packedSplats.setSplat(
            index,
            splat.center,
            splat.scales,
            splat.quaternion,
            mask[index] ? originalOpacity[index] : 0,
            splat.color,
          );
        }
        packedSplats.needsUpdate = true;
        mutableMesh.needsUpdate = true;
        mutableMesh.updateVersion?.();
      },

      clearPromptsVisuals() {
        if (sceneRef.current.markers) {
          disposeGroup(sceneRef.current.markers);
        }
      },
    }),
    [onSceneError, onSceneReady],
  );

  return (
    <div className="workspace-shell">
      <canvas ref={canvasRef} className="workspace-canvas" />
      <div className="workspace-hint">
        <span />
        Click directly on the splats to place prompts.
      </div>
    </div>
  );
});

function renderCurrentView(
  renderer: THREE.WebGLRenderer,
  scene: THREE.Scene,
  camera: THREE.PerspectiveCamera,
  controls: SparkControls,
) {
  controls.update(camera);
  camera.updateMatrixWorld();
  camera.updateProjectionMatrix();
  renderer.render(scene, camera);
}

function extractOriginalOpacity(mesh: MutableSplatMesh, count: number): Float32Array {
  const opacity = new Float32Array(count);
  if (mesh.forEachSplat) {
    mesh.forEachSplat((index, _center, _scales, _quaternion, splatOpacity) => {
      opacity[index] = splatOpacity;
    });
    return opacity;
  }

  const packedSplats = mesh.packedSplats;
  if (!packedSplats) {
    return opacity;
  }
  for (let index = 0; index < count; index += 1) {
    opacity[index] = packedSplats.getSplat(index).opacity;
  }
  return opacity;
}

function disposeGroup(group: THREE.Group) {
  while (group.children.length > 0) {
    const child = group.children[0];
    group.remove(child);
    if (child instanceof THREE.Mesh) {
      child.geometry.dispose();
      if (Array.isArray(child.material)) {
        child.material.forEach((material) => material.dispose());
      } else {
        child.material.dispose();
      }
    }
  }
}

function fitCameraToMesh(
  camera: THREE.PerspectiveCamera,
  controls: SparkControls,
  mesh: SplatMesh,
) {
  const splatMesh = mesh as MutableSplatMesh;
  const box = splatMesh.getBoundingBox(true);
  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3());
  const radius = Math.max(size.length() * 0.5, 0.2);
  const distance = radius / Math.sin(THREE.MathUtils.degToRad(camera.fov) * 0.5) * 0.9;
  const direction = new THREE.Vector3(1.1, 0.85, 1.3).normalize();
  const start = camera.position.clone();
  const target = center.clone().add(direction.multiplyScalar(distance));
  const startTime = performance.now();
  const duration = 420;
  const controlsWithTarget = controls as SparkControls & { target?: THREE.Vector3 };
  if (controlsWithTarget.target) {
    controlsWithTarget.target.copy(center);
  }

  const tick = (now: number) => {
    const t = Math.min((now - startTime) / duration, 1);
    const eased = 1 - Math.pow(1 - t, 3);
    camera.position.lerpVectors(start, target, eased);
    camera.lookAt(center);
    if (controlsWithTarget.target) {
      controlsWithTarget.target.copy(center);
    }
    controls.update(camera);
    if (t < 1) {
      requestAnimationFrame(tick);
    }
  };

  requestAnimationFrame(tick);
}
