import { useState } from "react";
import { Search, ChevronRight, ChevronDown, Upload, Check, Loader2 } from "lucide-react";
import { ScrollArea } from "../ui/ScrollArea";
import { Slider } from "../ui/Slider";
import { Button } from "../ui/Button";
import { Input } from "../ui/Input";
import { TASK_TYPES, type TaskType } from "../../lib/constants";
import { cn } from "../../lib/utils";
import type { ModelInfo } from "../../lib/api";

interface ModelSidebarProps {
  models: ModelInfo[];
  activeModels: string[];
  onToggleModel: (modelId: string) => void;
  onUploadModel: (file: File) => void;
  loadingModels: Set<string>;
  confidence: number;
  iou: number;
  onConfidenceChange: (value: number) => void;
  onIouChange: (value: number) => void;
  hardware: "cuda" | "metal" | "cpu";
}

type Recommendation = "recommended" | "capable" | "slow";

function getRecommendation(hardware: "cuda" | "metal" | "cpu", sizeKey: string, task: string): Recommendation {
  const smallSizes = ["n", "s"];
  const medSizes = ["n", "s", "m"];
  const largeSizes = ["n", "s", "m", "l"];

  if (hardware === "cuda") {
    // CUDA handles everything well
    return "recommended";
  }

  if (hardware === "metal") {
    // Pose crashes on MPS for larger models
    if (task === "pose") {
      if (smallSizes.includes(sizeKey)) return "recommended";
      if (sizeKey === "m") return "capable";
      return "slow"; // l, x crash
    }
    if (largeSizes.includes(sizeKey)) return "recommended";
    return "capable"; // x is fine but slower
  }

  // CPU
  if (smallSizes.includes(sizeKey)) return "recommended";
  if (medSizes.includes(sizeKey)) return "capable";
  return "slow";
}

const RECOMMENDATION_CONFIG: Record<Recommendation, { label: string; className: string }> = {
  recommended: { label: "Recommended", className: "text-success" },
  capable: { label: "Capable", className: "text-warning" },
  slow: { label: "Slow", className: "text-destructive" },
};

// Structured grouping: Family > Size > Tasks
interface TaskEntry {
  id: string;
  task: string;
}

interface SizeGroup {
  label: string;
  sizeKey: string;
  tasks: TaskEntry[];
}

interface FamilyGroup {
  id: string;
  name: string;
  sizes: SizeGroup[];
}

const SIZE_LABELS: Record<string, string> = {
  n: "Nano", s: "Small", m: "Medium", l: "Large", x: "XLarge",
  b: "Base", c: "Compact", e: "Extended",
};

function getBaseFamily(modelId: string): string {
  // "yolo26-seg-n" -> "yolo26", "yolo11-pose-m" -> "yolo11", "yolov8-n" -> "yolov8"
  const parts = modelId.split("-");
  return parts[0]; // yolo26, yolo11, yolov8, yolov5, yolov9, yolov10
}

function getSizeKey(modelId: string): string {
  const parts = modelId.split("-");
  return parts[parts.length - 1]; // last segment is always size
}

function getFamilyDisplayName(familyId: string): string {
  const map: Record<string, string> = {
    yolo26: "YOLO26", yolo11: "YOLO11", yolov8: "YOLOv8",
    yolov5: "YOLOv5", yolov9: "YOLOv9", yolov10: "YOLOv10", custom: "Custom",
  };
  return map[familyId] || familyId.toUpperCase();
}

export function Sidebar({
  models,
  activeModels,
  onToggleModel,
  onUploadModel,
  loadingModels,
  confidence,
  iou,
  onConfidenceChange,
  onIouChange,
  hardware,
}: ModelSidebarProps) {
  const [search, setSearch] = useState("");
  const [expandedFamilies, setExpandedFamilies] = useState<Set<string>>(new Set());
  const [expandedSizes, setExpandedSizes] = useState<Set<string>>(new Set());

  // Build Family > Size > Task structure
  const familyMap = new Map<string, Map<string, TaskEntry[]>>();
  const familyOrder: string[] = [];

  for (const model of models) {
    if (model.is_custom) {
      if (!familyMap.has("custom")) { familyMap.set("custom", new Map()); familyOrder.push("custom"); }
      const sizeMap = familyMap.get("custom")!;
      const key = model.name;
      if (!sizeMap.has(key)) sizeMap.set(key, []);
      sizeMap.get(key)!.push({ id: model.id, task: model.task });
    } else {
      const familyId = getBaseFamily(model.id);
      const sizeKey = getSizeKey(model.id);

      if (!familyMap.has(familyId)) { familyMap.set(familyId, new Map()); familyOrder.push(familyId); }
      const sizeMap = familyMap.get(familyId)!;
      if (!sizeMap.has(sizeKey)) sizeMap.set(sizeKey, []);
      sizeMap.get(sizeKey)!.push({ id: model.id, task: model.task });
    }
  }

  const families: FamilyGroup[] = familyOrder.map((familyId) => {
    const sizeMap = familyMap.get(familyId)!;
    const sizes: SizeGroup[] = Array.from(sizeMap.entries()).map(([sizeKey, tasks]) => ({
      label: SIZE_LABELS[sizeKey] || sizeKey.toUpperCase(),
      sizeKey,
      tasks,
    }));
    return { id: familyId, name: getFamilyDisplayName(familyId), sizes };
  });

  // Filter
  const lowerSearch = search.toLowerCase();
  const filteredFamilies = !search
    ? families
    : families
        .map((f) => ({
          ...f,
          sizes: f.sizes.filter(
            (s) =>
              f.name.toLowerCase().includes(lowerSearch) ||
              s.label.toLowerCase().includes(lowerSearch) ||
              s.tasks.some((t) => t.task.toLowerCase().includes(lowerSearch))
          ),
        }))
        .filter((f) => f.sizes.length > 0);

  const toggleFamily = (id: string) => {
    setExpandedFamilies((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id); else next.add(id);
      return next;
    });
  };

  const toggleSize = (key: string) => {
    setExpandedSizes((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key); else next.add(key);
      return next;
    });
  };

  // Count active tasks in a size group
  const activeCountForSize = (tasks: TaskEntry[]) =>
    tasks.filter((t) => activeModels.includes(t.id)).length;

  // Count active tasks in a family
  const activeCountForFamily = (family: FamilyGroup) =>
    family.sizes.reduce((sum, s) => sum + activeCountForSize(s.tasks), 0);

  return (
    <aside className="cvd-sidebar flex h-full w-72 flex-col">
      <div className="cvd-panel-heading">
        <div>
          <span className="cvd-eyebrow">Model registry</span>
          <h2>Choose inference models</h2>
        </div>
        <span className="cvd-count">{models.length}</span>
      </div>
      <div className="flex flex-col gap-3 border-b border-sidebar-border p-4">
        <div className="relative">
          <Search className="absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-muted-foreground" />
          <Input
            type="text"
            placeholder="Search models..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="h-8 bg-sidebar-accent pl-8 text-sm"
          />
        </div>
        <label className="cursor-pointer">
          <input
            type="file"
            accept=".pt,.onnx"
            className="hidden"
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) onUploadModel(file);
              e.target.value = "";
            }}
          />
          <Button
            variant="outline"
            size="sm"
            className="w-full justify-start gap-2 border-dashed text-xs pointer-events-none"
          >
            <Upload className="h-3.5 w-3.5" />
            Upload Custom Model
          </Button>
        </label>
      </div>

      <ScrollArea className="flex-1">
        <div className="flex flex-col gap-2 p-3">
          {filteredFamilies.length === 0 && (
            <div className="cvd-empty-state">
              <span>Registry unavailable</span>
              <p>Start the inference service or upload an ONNX model.</p>
            </div>
          )}
          {filteredFamilies.map((family) => {
            const isFamilyOpen = expandedFamilies.has(family.id);
            const familyActiveCount = activeCountForFamily(family);

            return (
              <div key={family.id} className="rounded-lg bg-sidebar-accent/50 border border-sidebar-border/50">
                {/* Family header */}
                <button
                  onClick={() => toggleFamily(family.id)}
                  className="flex w-full items-center justify-between px-3 py-2 text-sm font-medium hover:bg-sidebar-accent transition-colors rounded-lg"
                >
                  <div className="flex items-center gap-2">
                    {isFamilyOpen
                      ? <ChevronDown className="h-3.5 w-3.5 text-muted-foreground" />
                      : <ChevronRight className="h-3.5 w-3.5 text-muted-foreground" />
                    }
                    <span>{family.name}</span>
                  </div>
                  <div className="flex items-center gap-2">
                    {familyActiveCount > 0 && (
                      <span className="flex h-4 min-w-4 items-center justify-center rounded-full bg-primary px-1 text-[10px] font-medium text-primary-foreground">
                        {familyActiveCount}
                      </span>
                    )}
                    <span className="text-[10px] text-muted-foreground">{family.sizes.length} sizes</span>
                  </div>
                </button>

                {/* Sizes */}
                {isFamilyOpen && (
                  <div className="flex flex-col gap-0.5 px-2 pb-2">
                    {family.sizes.map((sizeGroup) => {
                      const sizeKey = `${family.id}-${sizeGroup.sizeKey}`;
                      const isSizeOpen = expandedSizes.has(sizeKey);
                      const sizeActiveCount = activeCountForSize(sizeGroup.tasks);

                      return (
                        <div key={sizeKey}>
                          {/* Size header */}
                          <button
                            onClick={() => toggleSize(sizeKey)}
                            className="flex w-full items-center justify-between rounded-md px-2 py-1.5 text-sm hover:bg-sidebar-accent transition-colors"
                          >
                            <div className="flex items-center gap-2">
                              {isSizeOpen
                                ? <ChevronDown className="h-3 w-3 text-muted-foreground" />
                                : <ChevronRight className="h-3 w-3 text-muted-foreground" />
                              }
                              <span className="font-medium">{sizeGroup.label}</span>
                            </div>
                            <div className="flex items-center gap-1.5">
                              {sizeActiveCount > 0 && (
                                <span className="flex h-3.5 min-w-3.5 items-center justify-center rounded-full bg-primary/20 px-1 text-[9px] font-medium text-primary">
                                  {sizeActiveCount}
                                </span>
                              )}
                              {/* Task dots preview */}
                              <div className="flex gap-0.5">
                                {sizeGroup.tasks.map((t) => {
                                  const taskType = TASK_TYPES[t.task as TaskType];
                                  return taskType ? (
                                    <div
                                      key={t.id}
                                      className={cn(
                                        "h-1.5 w-1.5 rounded-full",
                                        activeModels.includes(t.id) ? taskType.color : "bg-muted-foreground/20"
                                      )}
                                      title={taskType.label}
                                    />
                                  ) : null;
                                })}
                              </div>
                            </div>
                          </button>

                          {/* Task items */}
                          {isSizeOpen && (
                            <div className="ml-5 flex flex-col gap-0.5 py-0.5">
                              {sizeGroup.tasks.map((taskEntry) => {
                                const isSelected = activeModels.includes(taskEntry.id);
                                const isLoading = loadingModels.has(taskEntry.id);
                                const taskType = TASK_TYPES[taskEntry.task as TaskType];
                                const rec = getRecommendation(hardware, sizeGroup.sizeKey, taskEntry.task);
                                const recConfig = RECOMMENDATION_CONFIG[rec];

                                return (
                                  <button
                                    key={taskEntry.id}
                                    onClick={() => !isLoading && onToggleModel(taskEntry.id)}
                                    disabled={isLoading}
                                    className={cn(
                                      "flex items-center justify-between rounded-md px-2 py-1 text-left text-xs transition-colors",
                                      isSelected
                                        ? "bg-primary/10 text-primary"
                                        : "text-muted-foreground hover:bg-sidebar-accent hover:text-foreground"
                                    )}
                                  >
                                    <div className="flex items-center gap-2">
                                      {isLoading ? (
                                        <Loader2 className="h-3 w-3 animate-spin" />
                                      ) : (
                                        <div
                                          className={cn(
                                            "flex h-3.5 w-3.5 items-center justify-center rounded border",
                                            isSelected
                                              ? "border-primary bg-primary"
                                              : "border-muted-foreground/30"
                                          )}
                                        >
                                          {isSelected && <Check className="h-2.5 w-2.5 text-primary-foreground" />}
                                        </div>
                                      )}
                                      <span>{taskType?.label || taskEntry.task}</span>
                                    </div>
                                    <div className="flex items-center gap-1.5">
                                      <span className={cn("text-[9px] font-medium", recConfig.className)}>
                                        {recConfig.label}
                                      </span>
                                      {taskType && (
                                        <div className={cn("h-2 w-2 rounded-full", taskType.color)} />
                                      )}
                                    </div>
                                  </button>
                                );
                              })}
                            </div>
                          )}
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </ScrollArea>

      <div className="cvd-thresholds flex flex-col gap-5 border-t border-sidebar-border p-4">
        <div className="flex flex-col gap-2">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium">Confidence</span>
            <span className="text-xs text-muted-foreground">{Math.round(confidence * 100)}%</span>
          </div>
          <Slider
            value={[confidence]}
            onValueChange={([v]) => onConfidenceChange(v)}
            min={0}
            max={1}
            step={0.01}
          />
        </div>
        <div className="flex flex-col gap-2">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium">IoU Threshold</span>
            <span className="text-xs text-muted-foreground">{Math.round(iou * 100)}%</span>
          </div>
          <Slider
            value={[iou]}
            onValueChange={([v]) => onIouChange(v)}
            min={0}
            max={1}
            step={0.01}
          />
        </div>
      </div>
    </aside>
  );
}
