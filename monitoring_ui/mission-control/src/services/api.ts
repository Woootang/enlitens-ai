import { SnapshotPayload } from '../types';

const API_BASE = import.meta.env.VITE_MONITORING_API ?? '';

async function safeFetch<T>(path: string): Promise<T | null> {
  try {
    const response = await fetch(`${API_BASE}${path}`);
    if (!response.ok) {
      console.warn(`Request to ${path} failed with status`, response.status);
      return null;
    }
    return (await response.json()) as T;
  } catch (error) {
    console.warn('Failed to fetch', path, error);
    return null;
  }
}

export async function fetchSnapshot(): Promise<SnapshotPayload | null> {
  const stats = await safeFetch<SnapshotPayload>('/api/stats');
  return stats;
}
