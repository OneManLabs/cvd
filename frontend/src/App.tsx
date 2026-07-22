import { useState, useEffect, useCallback } from "react";
import { Header, type AppMode } from "./components/layout/Header";
import { Sidebar } from "./components/layout/Sidebar";
import { Viewport } from "./components/input/Viewport";
import { ResultsPanel } from "./components/output/ResultsPanel";
import { TrainingPanel } from "./components/training/TrainingPanel";
import { useWebSocket } from "./hooks/useWebSocket";
import {
  fetchDevices,
  fetchModels,
  loadModel,
  unloadModel,
  uploadModel,
  inferImage,
  type ModelInfo,
  type InferenceResult,
} from "./lib/api";
import type { SourceType } from "./lib/constants";

export default function App() {
  const [mode, setMode] = useState<AppMode>("inference");
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [activeModels, setActiveModels] = useState<string[]>([]);
  const [loadingModels, setLoadingModels] = useState<Set<string>>(new Set());
  const [device, setDevice] = useState("Detecting...");
  const [deviceId, setDeviceId] = useState<string>("cpu");
  const [hardware, setHardware] = useState<"cuda" | "metal" | "cpu">("cpu");
  const [source, setSource] = useState<SourceType>("webcam");
  const [confidence, setConfidence] = useState(0.5);
  const [iou, setIou] = useState(0.45);
  const [isRunning, setIsRunning] = useState(false);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imageResults, setImageResults] = useState<InferenceResult[]>([]);
  const [canvasSize, setCanvasSize] = useState({ width: 640, height: 480 });

  const ws = useWebSocket();

  useEffect(() => {
    let cancelled = false;

    const initializeWorkspace = async () => {
      let preferredDevice = "cpu";

      try {
        const d = await fetchDevices();
        if (cancelled) return;
        const rec = d.available.find((a) => a.id === d.recommended);
        setDevice(rec?.name || d.recommended);
        setDeviceId(d.recommended);
        preferredDevice = d.recommended;
        if (d.recommended.startsWith("cuda")) setHardware("cuda");
        else if (d.recommended === "mps") setHardware("metal");
        else setHardware("cpu");
      } catch {
        if (!cancelled) setDevice("CPU (backend offline)");
      }

      try {
        const registry = await fetchModels();
        if (cancelled) return;
        setModels(registry);

        const bundledModel = registry.find((model) => model.id === "custom-yolo11n");
        if (!bundledModel) return;

        setLoadingModels(new Set([bundledModel.id]));
        try {
          const loadedModel = await loadModel(bundledModel.id, preferredDevice);
          if (cancelled) return;
          setModels((current) => current.map((model) => model.id === loadedModel.id ? loadedModel : model));
          setActiveModels([loadedModel.id]);
        } catch (error) {
          console.error("Failed to preload the bundled ONNX model:", error);
        } finally {
          if (!cancelled) setLoadingModels(new Set());
        }
      } catch {
        // The empty registry state explains that the backend is not available.
      }
    };

    initializeWorkspace();
    return () => { cancelled = true; };
  }, []);

  const handleToggleModel = useCallback(
    async (modelId: string) => {
      if (activeModels.includes(modelId)) {
        setActiveModels((prev) => prev.filter((id) => id !== modelId));
        try { await unloadModel(modelId); } catch { /* The backend can already be offline. */ }
        setModels((prev) => prev.map((m) => (m.id === modelId ? { ...m, loaded: false } : m)));
      } else {
        setLoadingModels((prev) => new Set([...prev, modelId]));
        try {
          const updated = await loadModel(modelId, deviceId);
          setModels((prev) => prev.map((m) => (m.id === modelId ? updated : m)));
          setActiveModels((prev) => [...prev, modelId]);
        } catch (err) {
          console.error("Failed to load model:", err);
        } finally {
          setLoadingModels((prev) => { const next = new Set(prev); next.delete(modelId); return next; });
        }
      }
    },
    [activeModels, deviceId]
  );

  const handleUploadModel = useCallback(async (file: File) => {
    try {
      const info = await uploadModel(file);
      setModels((prev) => [...prev, info]);
    } catch (err) {
      console.error("Failed to upload model:", err);
    }
  }, []);

  const handleFrame = useCallback(
    (canvas: HTMLCanvasElement) => {
      if (activeModels.length === 0) return;
      setCanvasSize({ width: canvas.width, height: canvas.height });
      canvas.toBlob(
        (blob) => {
          if (!blob) return;
          const reader = new FileReader();
          reader.onload = () => {
            const base64 = (reader.result as string).split(",")[1];
            ws.sendFrame({ model_ids: activeModels, frame: base64, settings: { confidence, iou, img_size: 640 } });
          };
          reader.readAsDataURL(blob);
        },
        "image/jpeg",
        0.7
      );
    },
    [activeModels, confidence, iou, ws]
  );

  const handleToggleRun = useCallback(() => {
    if (isRunning) {
      setIsRunning(false);
      ws.disconnect();
    } else {
      if (activeModels.length === 0) return;
      ws.connect();
      setIsRunning(true);
    }
  }, [isRunning, activeModels, ws]);

  const handleImageSelected = useCallback(
    async (file: File, preview: string) => {
      setImageFile(file);
      setImagePreview(preview);
      setImageResults([]);
      if (activeModels.length === 0) return;

      const img = new window.Image();
      img.onload = async () => {
        setCanvasSize({ width: img.naturalWidth, height: img.naturalHeight });
        const results: InferenceResult[] = [];
        for (const modelId of activeModels) {
          try { results.push(await inferImage(modelId, file, { confidence, iou })); } catch { /* Continue with the remaining models. */ }
        }
        setImageResults(results);
      };
      img.src = preview;
    },
    [activeModels, confidence, iou]
  );

  useEffect(() => {
    if (source === "image" && imageFile && imagePreview && activeModels.length > 0) {
      handleImageSelected(imageFile, imagePreview);
    }
  }, [activeModels, source, imageFile, imagePreview, handleImageSelected]);

  const displayResults = source === "image" ? imageResults : ws.results;

  return (
    <div className="cvd-app flex h-screen flex-col overflow-hidden">
      <Header mode={mode} onModeChange={setMode} device={device} hardware={hardware} />

      <main className="cvd-workspace flex flex-1 overflow-hidden">
        {mode === "inference" ? (
          <>
            <Sidebar
              models={models}
              activeModels={activeModels}
              onToggleModel={handleToggleModel}
              onUploadModel={handleUploadModel}
              loadingModels={loadingModels}
              confidence={confidence}
              iou={iou}
              onConfidenceChange={setConfidence}
              onIouChange={setIou}
              hardware={hardware}
            />
            <Viewport
              source={source}
              onSourceChange={setSource}
              isRunning={isRunning}
              onToggleRun={handleToggleRun}
              onFrame={handleFrame}
              onImageSelected={handleImageSelected}
              imagePreview={imagePreview}
              onClearImage={() => { setImagePreview(null); setImageFile(null); setImageResults([]); }}
              results={displayResults}
              canvasSize={canvasSize}
              activeModelCount={activeModels.length}
            />
            <ResultsPanel results={displayResults} isRunning={isRunning} activeModels={activeModels} />
          </>
        ) : (
          <TrainingPanel deviceId={deviceId} />
        )}
      </main>
    </div>
  );
}
