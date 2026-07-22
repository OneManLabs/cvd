import { useState, useEffect, useCallback, useRef } from "react";
import { Upload, FolderOpen, Play, Pause, TrendingDown, Target, Clock, Trash2 } from "lucide-react";
import { Button } from "../ui/Button";
import { Label } from "../ui/Label";
import { Progress } from "../ui/Progress";
import { ScrollArea } from "../ui/ScrollArea";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "../ui/Select";
import { Slider } from "../ui/Slider";
import { TRAINING_PRESETS } from "../../lib/constants";
import {
  fetchDatasets,
  uploadDataset,
  deleteDataset,
  startTraining,
  fetchJobs,
  stopTraining,
  type DatasetInfo,
  type TrainingJob,
} from "../../lib/training-api";

interface TrainingPanelProps {
  deviceId: string;
}

const BASE_MODELS = [
  { id: "yolo26n.pt", name: "YOLO26 Nano" },
  { id: "yolo26s.pt", name: "YOLO26 Small" },
  { id: "yolo26m.pt", name: "YOLO26 Medium" },
  { id: "yolo26l.pt", name: "YOLO26 Large" },
  { id: "yolo26x.pt", name: "YOLO26 XLarge" },
  { id: "yolo11n.pt", name: "YOLO11 Nano" },
  { id: "yolo11s.pt", name: "YOLO11 Small" },
  { id: "yolo11m.pt", name: "YOLO11 Medium" },
];

export function TrainingPanel({ deviceId }: TrainingPanelProps) {
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [selectedDataset, setSelectedDataset] = useState<string | null>(null);
  const [jobs, setJobs] = useState<TrainingJob[]>([]);
  const [uploading, setUploading] = useState(false);

  const [preset, setPreset] = useState<keyof typeof TRAINING_PRESETS>("standard");
  const [config, setConfig] = useState({ ...TRAINING_PRESETS.standard });
  const [baseModel, setBaseModel] = useState("yolo26n.pt");

  const pollRef = useRef<ReturnType<typeof setInterval>>(undefined);

  useEffect(() => {
    fetchDatasets().then(setDatasets).catch(() => {});
    fetchJobs().then(setJobs).catch(() => {});
  }, []);

  useEffect(() => {
    const hasRunning = jobs.some((j) => j.status === "running" || j.status === "pending");
    if (hasRunning) {
      pollRef.current = setInterval(() => fetchJobs().then(setJobs).catch(() => {}), 2000);
    }
    return () => { if (pollRef.current) clearInterval(pollRef.current); };
  }, [jobs]);

  const handlePresetChange = (value: string) => {
    const key = value as keyof typeof TRAINING_PRESETS;
    setPreset(key);
    setConfig({ ...TRAINING_PRESETS[key] });
  };

  const handleUpload = useCallback(async (file: File) => {
    setUploading(true);
    try {
      const ds = await uploadDataset(file);
      setDatasets((prev) => [...prev, ds]);
      setSelectedDataset(ds.id);
    } catch (err) {
      console.error("Upload failed:", err);
    } finally {
      setUploading(false);
    }
  }, []);

  const handleDelete = useCallback(async (id: string) => {
    await deleteDataset(id);
    setDatasets((prev) => prev.filter((d) => d.id !== id));
    if (selectedDataset === id) setSelectedDataset(null);
  }, [selectedDataset]);

  const handleStart = useCallback(async () => {
    if (!selectedDataset) return;
    try {
      const job = await startTraining({
        dataset_id: selectedDataset,
        base_model: baseModel,
        epochs: config.epochs,
        batch_size: config.batchSize,
        img_size: config.imgSize,
        lr0: config.lr,
        device: deviceId,
      });
      setJobs((prev) => [...prev, job]);
    } catch (err) {
      console.error("Training start failed:", err);
    }
  }, [selectedDataset, baseModel, config, deviceId]);

  const handleStop = useCallback(async (jobId: string) => {
    await stopTraining(jobId);
    setJobs((prev) => prev.map((j) => j.id === jobId ? { ...j, status: "stopped" } : j));
  }, []);

  const isTraining = jobs.some((j) => j.status === "running");
  const activeJob = [...jobs].reverse().find((j) => j.status === "running" || j.status === "completed");
  const selectedDs = datasets.find((d) => d.id === selectedDataset);

  return (
    <div className="flex h-full flex-1">
      {/* Left: Configuration */}
      <aside className="flex w-72 flex-col border-r border-border bg-sidebar">
        <div className="border-b border-sidebar-border p-3">
          <h2 className="text-sm font-semibold">Training Configuration</h2>
        </div>

        <ScrollArea className="flex-1">
          <div className="flex flex-col gap-4 p-3">
            {/* Dataset */}
            <div className="rounded-lg bg-sidebar-accent p-3">
              <div className="mb-3 flex items-center gap-2">
                <FolderOpen className="h-4 w-4 text-primary" />
                <span className="text-sm font-medium">Dataset</span>
              </div>

              {selectedDs ? (
                <div className="flex flex-col gap-2">
                  <div className="rounded bg-muted/50 p-2">
                    <p className="text-sm font-medium">{selectedDs.name}</p>
                    <div className="mt-1 flex gap-3 text-xs text-muted-foreground">
                      <span>{selectedDs.train_images} images</span>
                      <span>{selectedDs.num_classes} classes</span>
                    </div>
                    <div className="mt-2 flex gap-2 text-xs">
                      <span className="rounded bg-primary/20 px-1.5 py-0.5 text-primary">
                        Train: {selectedDs.train_images}
                      </span>
                      <span className="rounded bg-secondary px-1.5 py-0.5">
                        Val: {selectedDs.val_images}
                      </span>
                    </div>
                  </div>
                  <Button variant="outline" size="sm" className="w-full text-xs" onClick={() => setSelectedDataset(null)}>
                    Change Dataset
                  </Button>
                </div>
              ) : (
                <div className="flex flex-col gap-2">
                  {datasets.map((ds) => (
                    <div
                      key={ds.id}
                      className="flex items-center justify-between rounded bg-muted/50 p-2 cursor-pointer hover:bg-muted transition-colors"
                      onClick={() => setSelectedDataset(ds.id)}
                    >
                      <div>
                        <p className="text-xs font-medium">{ds.name}</p>
                        <p className="text-[10px] text-muted-foreground">{ds.train_images} images, {ds.num_classes} classes</p>
                      </div>
                      <button onClick={(e) => { e.stopPropagation(); handleDelete(ds.id); }} className="text-muted-foreground hover:text-destructive">
                        <Trash2 className="h-3 w-3" />
                      </button>
                    </div>
                  ))}
                  <label className={`flex cursor-pointer flex-col items-center gap-2 rounded-lg border border-dashed border-border p-4 transition-colors hover:border-primary/50 hover:bg-primary/5 ${uploading ? "opacity-50 pointer-events-none" : ""}`}>
                    <Upload className="h-6 w-6 text-muted-foreground" />
                    <div className="text-center">
                      <p className="text-xs font-medium">{uploading ? "Uploading..." : "Upload YOLO Dataset"}</p>
                      <p className="text-xs text-muted-foreground">.zip file</p>
                    </div>
                    <input type="file" accept=".zip" className="hidden" onChange={(e) => {
                      const file = e.target.files?.[0];
                      if (file) handleUpload(file);
                      e.target.value = "";
                    }} />
                  </label>
                </div>
              )}
            </div>

            {/* Base Model */}
            <div className="flex flex-col gap-2">
              <Label className="text-xs">Base Model</Label>
              <Select value={baseModel} onValueChange={setBaseModel}>
                <SelectTrigger className="h-8 text-xs">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {BASE_MODELS.map((m) => (
                    <SelectItem key={m.id} value={m.id} className="text-xs">{m.name}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {/* Preset */}
            <div className="flex flex-col gap-2">
              <Label className="text-xs">Training Preset</Label>
              <Select value={preset} onValueChange={handlePresetChange}>
                <SelectTrigger className="h-8 text-xs">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="quick" className="text-xs">Quick (10 epochs)</SelectItem>
                  <SelectItem value="standard" className="text-xs">Standard (100 epochs)</SelectItem>
                  <SelectItem value="thorough" className="text-xs">Thorough (300 epochs)</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {/* Hyperparameters */}
            <div className="flex flex-col gap-3 rounded-lg bg-sidebar-accent p-3">
              <span className="text-xs font-medium">Hyperparameters</span>

              <div className="flex flex-col gap-1.5">
                <div className="flex items-center justify-between">
                  <Label className="text-xs text-muted-foreground">Epochs</Label>
                  <span className="text-xs font-medium">{config.epochs}</span>
                </div>
                <Slider value={[config.epochs]} onValueChange={([v]) => setConfig({ ...config, epochs: v })} min={1} max={500} step={1} />
              </div>

              <div className="flex flex-col gap-1.5">
                <div className="flex items-center justify-between">
                  <Label className="text-xs text-muted-foreground">Batch Size</Label>
                  <span className="text-xs font-medium">{config.batchSize}</span>
                </div>
                <Slider value={[config.batchSize]} onValueChange={([v]) => setConfig({ ...config, batchSize: v })} min={1} max={64} step={1} />
              </div>

              <div className="flex flex-col gap-1.5">
                <div className="flex items-center justify-between">
                  <Label className="text-xs text-muted-foreground">Learning Rate</Label>
                  <span className="text-xs font-medium font-mono">{config.lr}</span>
                </div>
                <Slider value={[config.lr * 1000]} onValueChange={([v]) => setConfig({ ...config, lr: v / 1000 })} min={0.1} max={100} step={0.1} />
              </div>

              <div className="flex flex-col gap-1.5">
                <div className="flex items-center justify-between">
                  <Label className="text-xs text-muted-foreground">Image Size</Label>
                  <span className="text-xs font-medium">{config.imgSize}px</span>
                </div>
                <Slider value={[config.imgSize]} onValueChange={([v]) => setConfig({ ...config, imgSize: v })} min={320} max={1280} step={32} />
              </div>
            </div>
          </div>
        </ScrollArea>

        <div className="border-t border-sidebar-border p-3">
          <Button
            className="w-full gap-2"
            variant={isTraining ? "destructive" : "default"}
            onClick={isTraining ? () => { const j = jobs.find((j) => j.status === "running"); if (j) handleStop(j.id); } : handleStart}
            disabled={!selectedDataset && !isTraining}
          >
            {isTraining ? (<><Pause className="h-4 w-4" /> Stop Training</>) : (<><Play className="h-4 w-4" /> Start Training</>)}
          </Button>
        </div>
      </aside>

      {/* Right: Monitor */}
      <div className="flex flex-1 flex-col bg-background">
        <div className="border-b border-border p-3">
          <h2 className="text-sm font-semibold">Training Monitor</h2>
        </div>

        <div className="flex-1 p-4">
          {activeJob && activeJob.metrics.epoch > 0 ? (
            <div className="flex flex-col gap-6">
              {/* Progress */}
              <div className="rounded-xl bg-card p-4">
                <div className="mb-4 flex items-center justify-between">
                  <div>
                    <p className="text-sm text-muted-foreground">Training Progress</p>
                    <p className="text-2xl font-semibold">
                      Epoch {activeJob.metrics.epoch} / {activeJob.metrics.total_epochs}
                    </p>
                  </div>
                  <div className="flex items-center gap-2 text-sm text-muted-foreground">
                    <Clock className="h-4 w-4" />
                    <span>{Math.round(activeJob.progress * 100)}% complete</span>
                  </div>
                </div>
                <Progress value={activeJob.progress * 100} className="h-2" />
              </div>

              {/* Metrics */}
              <div className="grid grid-cols-3 gap-4">
                <div className="rounded-xl bg-card p-4">
                  <div className="flex items-center gap-2 text-muted-foreground">
                    <TrendingDown className="h-4 w-4" />
                    <span className="text-xs">Loss</span>
                  </div>
                  <p className="mt-2 text-3xl font-semibold tabular-nums text-chart-1">
                    {activeJob.metrics.box_loss.toFixed(4)}
                  </p>
                </div>
                <div className="rounded-xl bg-card p-4">
                  <div className="flex items-center gap-2 text-muted-foreground">
                    <Target className="h-4 w-4" />
                    <span className="text-xs">mAP@50</span>
                  </div>
                  <p className="mt-2 text-3xl font-semibold tabular-nums text-chart-2">
                    {(activeJob.metrics.mAP50 * 100).toFixed(1)}%
                  </p>
                </div>
                <div className="rounded-xl bg-card p-4">
                  <div className="flex items-center gap-2 text-muted-foreground">
                    <Target className="h-4 w-4" />
                    <span className="text-xs">mAP@50-95</span>
                  </div>
                  <p className="mt-2 text-3xl font-semibold tabular-nums text-chart-3">
                    {(activeJob.metrics.mAP50_95 * 100).toFixed(1)}%
                  </p>
                </div>
              </div>

              {/* Log */}
              <div className="rounded-xl bg-card p-4">
                <span className="text-sm font-medium">Training Log</span>
                <div className="mt-3 rounded bg-muted/50 p-3 font-mono text-xs space-y-1">
                  <p className="text-muted-foreground">
                    [Epoch {activeJob.metrics.epoch}/{activeJob.metrics.total_epochs}] box: {activeJob.metrics.box_loss.toFixed(4)}, cls: {activeJob.metrics.cls_loss.toFixed(4)}, dfl: {activeJob.metrics.dfl_loss.toFixed(4)}
                  </p>
                  <p className="text-muted-foreground">
                    mAP50: {activeJob.metrics.mAP50.toFixed(3)}, mAP50-95: {activeJob.metrics.mAP50_95.toFixed(3)}, lr: {activeJob.metrics.lr.toExponential(2)}
                  </p>
                  {activeJob.status === "completed" && (
                    <p className="text-primary">Training complete! Model saved.</p>
                  )}
                  {activeJob.error && (
                    <p className="text-destructive">{activeJob.error}</p>
                  )}
                </div>
              </div>
            </div>
          ) : (
            <div className="flex h-full items-center justify-center">
              <div className="text-center">
                <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-full bg-muted">
                  <Play className="h-8 w-8 text-muted-foreground" />
                </div>
                <p className="font-medium">Ready to Train</p>
                <p className="mt-1 text-sm text-muted-foreground">
                  Configure your settings and click Start Training
                </p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
