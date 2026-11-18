import { useEffect, useState } from "react";

export function useAsync<T>(fn: () => Promise<T>, deps: unknown[]): { data: T | null; error: string | null; loading: boolean } {
  const [state, setState] = useState<{ data: T | null; error: string | null; loading: boolean }>({ data: null, error: null, loading: true });
  useEffect(() => {
    let alive = true;
    setState((s) => ({ ...s, loading: true }));
    fn().then(
      (data) => alive && setState({ data, error: null, loading: false }),
      (e: Error) => alive && setState({ data: null, error: e.message, loading: false }),
    );
    return () => {
      alive = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
  return state;
}

export function useDebounced<T>(value: T, ms = 300): T {
  const [v, setV] = useState(value);
  useEffect(() => {
    const t = setTimeout(() => setV(value), ms);
    return () => clearTimeout(t);
  }, [value, ms]);
  return v;
}

export const DEFAULT_MODEL = "/path/to/models/MiniCPM-RobotManip";
export const DEFAULT_DATASET = "/path/to/datasets/lerobot_libero_10_image";
export const DEFAULT_VLM = "/path/to/models/MiniCPM-V-4.6";
export const DEFAULT_LLM = "/path/to/models/Qwen2.5-0.5B-Instruct";
export const DEFAULT_WORLD = "/path/to/models/MQDE-TinyDiT";
export const DEFAULT_EWM = "/path/to/models/MQDE-TinyDiT-EWM";
