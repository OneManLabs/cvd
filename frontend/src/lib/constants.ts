export const TASK_TYPES = {
  detect: { label: "Detection", color: "bg-chart-1" },
  segment: { label: "Segmentation", color: "bg-chart-2" },
  pose: { label: "Pose", color: "bg-chart-3" },
  classify: { label: "Classification", color: "bg-chart-4" },
  obb: { label: "Oriented BB", color: "bg-chart-5" },
} as const;

export type TaskType = keyof typeof TASK_TYPES;
export type SourceType = "webcam" | "image" | "video";

export const TRAINING_PRESETS: Record<string, { epochs: number; batchSize: number; lr: number; imgSize: number }> = {
  quick: { epochs: 10, batchSize: 16, lr: 0.01, imgSize: 640 },
  standard: { epochs: 100, batchSize: 16, lr: 0.01, imgSize: 640 },
  thorough: { epochs: 300, batchSize: 8, lr: 0.001, imgSize: 1280 },
};
