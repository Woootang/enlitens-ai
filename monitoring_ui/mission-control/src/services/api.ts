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

export async function postChatMessage(payload: { message: string; context: Record<string, unknown> }): Promise<string | null> {
  try {
    const response = await fetch(`${API_BASE}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!response.ok) {
      console.warn('Chat request failed', response.status);
      return null;
    }
    const data = (await response.json()) as { response?: string };
    return data.response ?? null;
  } catch (error) {
    console.warn('Failed to send chat request', error);
    return null;
  }
}

export async function postAction(endpoint: string, body: Record<string, unknown>): Promise<boolean> {
  try {
    const response = await fetch(`${API_BASE}${endpoint}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    return response.ok;
  } catch (error) {
    console.warn('Failed to invoke action', endpoint, error);
    return false;
  }
}
