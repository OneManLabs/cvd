const API_BASE = '/api';

export interface DeviceInfo {
  id: string;
  name: string;
}

export interface DevicesResponse {
  available: DeviceInfo[];
  recommended: string;
}

export interface ModelInfo {
  id: string;
  name: string;
  task: string;
  size: string | null;
  path: string;
  is_custom: boolean;
  loaded: boolean;
}

export interface Detection {
  class_id: number;
  class_name: string;
  confidence: number;
  bbox: [number, number, number, number];
  mask: number[][] | null;
  keypoints: number[][] | null;
}

export interface InferenceResult {
  model_id: string;
  detections: Detection[];
  inference_time_ms: number;
  image_width: number;
  image_height: number;
}

export async function fetchDevices(): Promise<DevicesResponse> {
  const res = await fetch(`${API_BASE}/devices`);
  if (!res.ok) throw new Error('Failed to fetch devices');
  return res.json();
}

export async function fetchModels(): Promise<ModelInfo[]> {
  const res = await fetch(`${API_BASE}/models`);
  if (!res.ok) throw new Error('Failed to fetch models');
  const data = await res.json();
  return data.models;
}

export async function loadModel(modelId: string, device?: string): Promise<ModelInfo> {
  const url = new URL(`${API_BASE}/models/${modelId}/load`, window.location.origin);
  if (device) url.searchParams.set('device', device);
  const res = await fetch(url.toString(), { method: 'POST' });
  if (!res.ok) throw new Error('Failed to load model');
  return res.json();
}

export async function unloadModel(modelId: string): Promise<void> {
  const res = await fetch(`${API_BASE}/models/${modelId}/unload`, { method: 'POST' });
  if (!res.ok) throw new Error('Failed to unload model');
}

export async function uploadModel(file: File): Promise<ModelInfo> {
  const form = new FormData();
  form.append('file', file);
  const res = await fetch(`${API_BASE}/models/upload`, { method: 'POST', body: form });
  if (!res.ok) throw new Error('Failed to upload model');
  return res.json();
}

export async function inferImage(
  modelId: string,
  file: File | Blob,
  settings?: { confidence?: number; iou?: number; img_size?: number }
): Promise<InferenceResult> {
  const form = new FormData();
  form.append('file', file);
  if (settings?.confidence) form.append('confidence', settings.confidence.toString());
  if (settings?.iou) form.append('iou', settings.iou.toString());
  if (settings?.img_size) form.append('img_size', settings.img_size.toString());

  const res = await fetch(`${API_BASE}/inference/${modelId}`, { method: 'POST', body: form });
  if (!res.ok) throw new Error('Failed to run inference');
  return res.json();
}
