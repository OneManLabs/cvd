import { Clock, Box, Layers, TrendingUp } from "lucide-react";
import { ScrollArea } from "../ui/ScrollArea";
import { Progress } from "../ui/Progress";
import { cn } from "../../lib/utils";
import { getClassColor } from "../../lib/colors";
import type { InferenceResult } from "../../lib/api";

interface ResultsPanelProps {
  results: InferenceResult[];
  isRunning: boolean;
  activeModels: string[];
}

export function ResultsPanel({ results, isRunning, activeModels }: ResultsPanelProps) {
  // Aggregate stats
  const totalDetections = results.reduce((sum, r) => sum + r.detections.length, 0);
  const avgInferenceTime = results.length > 0
    ? results.reduce((sum, r) => sum + r.inference_time_ms, 0) / results.length
    : 0;
  const fps = avgInferenceTime > 0 ? Math.min(1000 / avgInferenceTime, 60) : 0;

  // Class counts
  const classCounts = new Map<string, { count: number; classId: number; totalConf: number }>();
  for (const result of results) {
    for (const det of result.detections) {
      const existing = classCounts.get(det.class_name);
      if (existing) {
        existing.count++;
        existing.totalConf += det.confidence;
      } else {
        classCounts.set(det.class_name, { count: 1, classId: det.class_id, totalConf: det.confidence });
      }
    }
  }
  const sortedClasses = Array.from(classCounts.entries()).sort((a, b) => b[1].count - a[1].count);

  // Confidence distribution
  let highConf = 0, medConf = 0, lowConf = 0;
  for (const result of results) {
    for (const det of result.detections) {
      if (det.confidence >= 0.9) highConf++;
      else if (det.confidence >= 0.7) medConf++;
      else lowConf++;
    }
  }

  return (
    <aside className="cvd-results flex h-full w-80 flex-col">
      <div className="cvd-panel-heading">
        <div>
          <span className="cvd-eyebrow">Live analysis</span>
          <h2>Results</h2>
        </div>
        <span className={cn("cvd-signal", isRunning ? "is-live" : "")}>{isRunning ? "Live" : "Idle"}</span>
      </div>

      <ScrollArea className="flex-1">
        <div className="results-stack flex flex-col gap-4 p-4">
          {/* Performance */}
          <div className="rounded-lg bg-secondary p-3">
            <div className="mb-3 flex items-center gap-2">
              <Clock className="h-4 w-4 text-primary" />
              <span className="text-sm font-medium">Performance</span>
            </div>
            <div className="grid grid-cols-2 gap-3">
              <div>
                <p className="text-xs text-muted-foreground">FPS</p>
                <p className={cn("text-xl font-semibold tabular-nums", isRunning && results.length > 0 ? "text-foreground" : "text-muted-foreground")}>
                  {isRunning && results.length > 0 ? fps.toFixed(1) : "--"}
                </p>
              </div>
              <div>
                <p className="text-xs text-muted-foreground">Inference</p>
                <p className={cn("text-xl font-semibold tabular-nums", isRunning && results.length > 0 ? "text-foreground" : "text-muted-foreground")}>
                  {isRunning && results.length > 0 ? `${avgInferenceTime.toFixed(1)}ms` : "--"}
                </p>
              </div>
            </div>
          </div>

          {/* Detections */}
          <div className="rounded-lg bg-secondary p-3">
            <div className="mb-3 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Box className="h-4 w-4 text-primary" />
                <span className="text-sm font-medium">Detections</span>
              </div>
              <span className={cn(
                "rounded-full bg-primary/20 px-2 py-0.5 text-xs font-medium",
                results.length > 0 ? "text-primary" : "text-muted-foreground"
              )}>
                {totalDetections}
              </span>
            </div>

            {sortedClasses.length > 0 ? (
              <div className="flex flex-col gap-2">
                {sortedClasses.map(([className, { count, classId, totalConf }]) => (
                  <div key={className} className="flex items-center gap-2">
                    <div
                      className="h-2.5 w-2.5 rounded-full shrink-0"
                      style={{ backgroundColor: getClassColor(classId) }}
                    />
                    <span className="flex-1 text-sm">{className}</span>
                    <span className="text-xs text-muted-foreground">{count}x</span>
                    <span className="w-10 text-right text-xs font-medium">
                      {Math.round((totalConf / count) * 100)}%
                    </span>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-sm text-muted-foreground">
                {isRunning ? "No detections" : "Start inference to see detections"}
              </p>
            )}
          </div>

          {/* Active Models */}
          <div className="rounded-lg bg-secondary p-3">
            <div className="mb-3 flex items-center gap-2">
              <Layers className="h-4 w-4 text-primary" />
              <span className="text-sm font-medium">Active Models</span>
            </div>
            {activeModels.length > 0 ? (
              <div className="flex flex-col gap-2">
                {activeModels.map((modelId) => (
                  <div key={modelId} className="flex items-center justify-between rounded bg-muted/50 px-2 py-1.5">
                    <span className="text-sm font-mono">{modelId}</span>
                    <span className={cn(
                      "h-2 w-2 rounded-full",
                      isRunning ? "animate-pulse bg-success" : "bg-muted-foreground"
                    )} />
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-sm text-muted-foreground">Select models from the sidebar</p>
            )}
          </div>

          {/* Confidence Distribution */}
          {totalDetections > 0 && (
            <div className="rounded-lg bg-secondary p-3">
              <div className="mb-3 flex items-center gap-2">
                <TrendingUp className="h-4 w-4 text-primary" />
                <span className="text-sm font-medium">Confidence</span>
              </div>
              <div className="flex flex-col gap-2">
                <div className="flex items-center justify-between text-xs">
                  <span className="text-muted-foreground">High (90-100%)</span>
                  <span className="font-medium">{highConf}</span>
                </div>
                <Progress value={totalDetections > 0 ? (highConf / totalDetections) * 100 : 0} className="h-1.5" />
                <div className="flex items-center justify-between text-xs">
                  <span className="text-muted-foreground">Medium (70-90%)</span>
                  <span className="font-medium">{medConf}</span>
                </div>
                <Progress value={totalDetections > 0 ? (medConf / totalDetections) * 100 : 0} className="h-1.5" />
                <div className="flex items-center justify-between text-xs">
                  <span className="text-muted-foreground">Low (&lt;70%)</span>
                  <span className="font-medium">{lowConf}</span>
                </div>
                <Progress value={totalDetections > 0 ? (lowConf / totalDetections) * 100 : 0} className="h-1.5" />
              </div>
            </div>
          )}
        </div>
      </ScrollArea>
    </aside>
  );
}
