// T-1307: generic placeholder glyphs for the hardware page -- deliberately not any vendor's real logo
// (confirmed with the user), just a plain shape per device kind so the three tables aren't bare text.
export function KindIcon({ kind }: { kind: string }) {
  const k = kind.toLowerCase();
  const { bg, glyph } = k === "gpu" ? { bg: "#2b6cb0", glyph: "▥" } : k === "npu" ? { bg: "#6b46c1", glyph: "◆" } : k === "cpu" ? { bg: "#4a5568", glyph: "▢" } : { bg: "#718096", glyph: "●" };
  return (
    <span
      aria-hidden="true"
      style={{
        display: "inline-flex",
        alignItems: "center",
        justifyContent: "center",
        width: 22,
        height: 22,
        borderRadius: 5,
        background: bg,
        color: "#fff",
        fontSize: 12,
        lineHeight: 1,
        flexShrink: 0,
      }}
    >
      {glyph}
    </span>
  );
}

// Vendor string -> a stable, distinct hue, so different vendors are visually distinguishable even though
// none of them get a real logo.
export function VendorIcon({ vendor }: { vendor: string }) {
  let h = 0;
  for (const c of vendor) h = (h * 31 + c.charCodeAt(0)) % 360;
  return (
    <span
      aria-hidden="true"
      title={vendor}
      style={{
        display: "inline-flex",
        alignItems: "center",
        justifyContent: "center",
        width: 22,
        height: 22,
        borderRadius: "50%",
        background: `hsl(${h}, 55%, 45%)`,
        color: "#fff",
        fontSize: 11,
        fontWeight: 600,
        lineHeight: 1,
        flexShrink: 0,
      }}
    >
      {vendor ? vendor[0].toUpperCase() : "?"}
    </span>
  );
}
