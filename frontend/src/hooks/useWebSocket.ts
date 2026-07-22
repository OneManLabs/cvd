import { useRef, useCallback, useEffect, useState } from 'react';
import type { InferenceResult } from '../lib/api';

interface WSMessage {
  model_ids: string[];
  frame: string; // base64
  settings: {
    confidence: number;
    iou: number;
    img_size: number;
  };
}

export function useWebSocket() {
  const wsRef = useRef<WebSocket | null>(null);
  const [connected, setConnected] = useState(false);
  const [results, setResults] = useState<InferenceResult[]>([]);
  const pendingRef = useRef(false);
  const connectRef = useRef<() => void>(() => {});
  const shouldReconnectRef = useRef(false);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;
    shouldReconnectRef.current = true;

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const ws = new WebSocket(`${protocol}//${window.location.host}/ws/inference`);

    ws.onopen = () => setConnected(true);
    ws.onclose = () => {
      setConnected(false);
      if (shouldReconnectRef.current) {
        setTimeout(() => connectRef.current(), 2000);
      }
    };
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.results) {
        setResults(data.results);
      }
      pendingRef.current = false;
    };

    wsRef.current = ws;
  }, []);

  useEffect(() => {
    connectRef.current = connect;
  }, [connect]);

  const sendFrame = useCallback((msg: WSMessage) => {
    if (wsRef.current?.readyState !== WebSocket.OPEN) return;
    if (pendingRef.current) return; // Skip if still waiting for previous result
    pendingRef.current = true;
    wsRef.current.send(JSON.stringify(msg));
  }, []);

  const disconnect = useCallback(() => {
    shouldReconnectRef.current = false;
    wsRef.current?.close();
    wsRef.current = null;
  }, []);

  useEffect(() => {
    return () => {
      shouldReconnectRef.current = false;
      wsRef.current?.close();
    };
  }, []);

  return { connect, disconnect, sendFrame, connected, results };
}
