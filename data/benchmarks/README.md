# Measured baselines

- `local/<hw-slug>/` — written by `mqde bench` on this machine (confidence `measured_local`).
- `community/<hw-slug>/` — imported third-party data; each file **must** carry a top-level `source` with origin and license (confidence `measured_community`).

Matching is identical hardware only (GPU name + CPU name). Format: `schema.json`. See the reference notes A.8.
