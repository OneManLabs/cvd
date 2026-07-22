const API_BASE = '/api/training';

export interface DatasetInfo {
  id: string;
  name: string;
  path: string;
  num_classes: number;
  class_names: string[];
  train_images: number;
  val_images: number;
  task: string;
}

export interface TrainingMetrics {
  epoch: number;
  total_epochs: number;
  box_loss: number;
  cls_loss: number;
  dfl_loss: number;
  precision: number;
  recall: number;
  mAP50: number;
  mAP50_95: number;
  lr: number;
}

export interface TrainingJob {
  id: string;
  dataset_id: string;
  base_model: string;
  status: string;
  progress: number;
  metrics: TrainingMetrics;
  error: string | null;
  output_model_path: string | null;
  config: Record<string, unknown>;
}

export async function fetchDatasets(): Promise<DatasetInfo[]> {
  const res = await fetch(`${API_BASE}/datasets`);
  if (!res.ok) throw new Error('Failed to fetch datasets');
  return res.json();
}

export async function uploadDataset(file: File): Promise<DatasetInfo> {
  const form = new FormData();
  form.append('file', file);
  const res = await fetch(`${API_BASE}/datasets/upload`, { method: 'POST', body: form });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Upload failed' }));
    throw new Error(err.detail || 'Upload failed');
  }
  return res.json();
}

export async function deleteDataset(datasetId: string): Promise<void> {
  await fetch(`${API_BASE}/datasets/${datasetId}`, { method: 'DELETE' });
}

export async function startTraining(config: {
  dataset_id: string;
  base_model: string;
  epochs: number;
  batch_size: number;
  img_size: number;
  lr0: number;
  device?: string;
  project_name?: string;
}): Promise<TrainingJob> {
  const res = await fetch(`${API_BASE}/start`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Failed to start training' }));
    throw new Error(err.detail || 'Failed to start training');
  }
  return res.json();
}

export async function fetchJobs(): Promise<TrainingJob[]> {
  const res = await fetch(`${API_BASE}/jobs`);
  if (!res.ok) throw new Error('Failed to fetch jobs');
  return res.json();
}

export async function fetchJob(jobId: string): Promise<TrainingJob> {
  const res = await fetch(`${API_BASE}/jobs/${jobId}`);
  if (!res.ok) throw new Error('Failed to fetch job');
  return res.json();
}

export async function stopTraining(jobId: string): Promise<void> {
  await fetch(`${API_BASE}/jobs/${jobId}/stop`, { method: 'POST' });
}
