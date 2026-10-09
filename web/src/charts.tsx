// ECharts wrapper that reads colors from CSS tokens (so light/dark are the validated steps) and
// re-renders on theme change. Chart builders below take plain data and return an option.
import * as echarts from "echarts";
import { useEffect, useRef, useState } from "react";

export function token(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}
export const series = (i: number) => token(`--series-${(i % 8) + 1}`);

function useThemeVersion(): number {
  const [v, setV] = useState(0);
  useEffect(() => {
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    const on = () => setV((x) => x + 1);
    mq.addEventListener("change", on);
    return () => mq.removeEventListener("change", on);
  }, []);
  return v;
}

export function baseOption(): echarts.EChartsOption {
  const muted = token("--text-muted");
  const grid = token("--grid");
  const axis = token("--axis");
  return {
    backgroundColor: "transparent",
    textStyle: { fontFamily: "system-ui, -apple-system, sans-serif", color: token("--text-secondary") },
    grid: { left: 56, right: 20, top: 30, bottom: 36, containLabel: false },
    tooltip: {
      backgroundColor: token("--surface-1"),
      borderColor: token("--border"),
      textStyle: { color: token("--text-primary"), fontSize: 12 },
      confine: true,
    },
    legend: { top: 0, left: 0, itemWidth: 10, itemHeight: 10, icon: "roundRect", textStyle: { color: token("--text-secondary"), fontSize: 12 } },
    xAxis: { axisLine: { lineStyle: { color: axis } }, axisTick: { show: false }, axisLabel: { color: muted }, splitLine: { show: false } },
    yAxis: { axisLine: { show: false }, axisLabel: { color: muted }, splitLine: { lineStyle: { color: grid, width: 1 } } },
  };
}

// Builders return loosely-typed objects: ECharts' option union types reject typed formatter callbacks.
export function Chart({ option, height = 280, onClick }: { option: () => object; height?: number; onClick?: (p: { data?: unknown }) => void }) {
  const ref = useRef<HTMLDivElement>(null);
  const inst = useRef<echarts.ECharts | null>(null);
  const theme = useThemeVersion();
  const clickRef = useRef(onClick);
  clickRef.current = onClick;
  useEffect(() => {
    if (!ref.current) return;
    inst.current = echarts.init(ref.current, undefined, { renderer: "svg" });
    inst.current.on("click", (p) => clickRef.current?.(p as { data?: unknown }));
    const ro = new ResizeObserver(() => inst.current?.resize());
    ro.observe(ref.current);
    return () => {
      ro.disconnect();
      inst.current?.dispose();
    };
  }, []);
  useEffect(() => {
    inst.current?.setOption(option() as echarts.EChartsOption, { notMerge: true });
  });
  void theme;
  return <div ref={ref} style={{ width: "100%", height }} />;
}

/** Recessive markLine for a capacity / threshold, labeled in text ink (not a series color). */
export function refLine(value: number, label: string) {
  return {
    silent: true,
    symbol: "none",
    lineStyle: { color: token("--text-secondary"), type: "dashed" as const, width: 1 },
    label: { formatter: label, color: token("--text-secondary"), position: "insideEndTop" as const, fontSize: 11 },
    data: [{ yAxis: value }],
  };
}

/** Rounded top on the outermost segment only; 2px surface gap between stacked fills. */
export const barItem = (i: number, top: boolean) => ({
  color: series(i),
  borderColor: token("--surface-1"),
  borderWidth: 1,
  borderRadius: top ? [4, 4, 0, 0] : 0,
});
