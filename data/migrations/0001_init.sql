-- MQDEsys metadata schema, version 1. A merged migration is never edited: change the schema with 0002_*.sql.
CREATE TABLE jobs (
  id          TEXT PRIMARY KEY,
  name        TEXT NOT NULL DEFAULT '',
  kind        TEXT NOT NULL DEFAULT 'llm',          -- llm | vla
  state       TEXT NOT NULL,                        -- QUEUED RUNNING SUCCEEDED FAILED CANCELLED
  spec_json   TEXT NOT NULL,
  report_path TEXT,
  gpu         INTEGER,
  error       TEXT,
  created_at  REAL NOT NULL,
  started_at  REAL,
  finished_at REAL
);
CREATE INDEX idx_jobs_state ON jobs(state);
CREATE INDEX idx_jobs_created ON jobs(created_at);

CREATE TABLE stages (
  job_id      TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
  name        TEXT NOT NULL,
  seq         INTEGER NOT NULL,                     -- order of appearance within the job
  status      TEXT NOT NULL,                        -- running succeeded failed skipped
  started_at  REAL,
  seconds     REAL,
  detail      TEXT,
  PRIMARY KEY (job_id, name, seq)
);

CREATE TABLE events (
  job_id  TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
  seq     INTEGER NOT NULL,
  ts      REAL NOT NULL,
  type    TEXT NOT NULL,
  payload TEXT NOT NULL,
  PRIMARY KEY (job_id, seq)
) WITHOUT ROWID;

CREATE TABLE artifacts (
  id        INTEGER PRIMARY KEY AUTOINCREMENT,
  job_id    TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
  kind      TEXT NOT NULL,                          -- report checkpoint gguf log ...
  name      TEXT NOT NULL,
  cas_hash  TEXT,                                   -- content address when the artifact lives in the cache
  path      TEXT,
  bytes     INTEGER NOT NULL DEFAULT 0,
  created_at REAL NOT NULL
);
CREATE INDEX idx_artifacts_job ON artifacts(job_id);

CREATE TABLE hw_profiles (
  id         TEXT PRIMARY KEY,                      -- e.g. nvidia/rtx4090
  device     TEXT NOT NULL DEFAULT '',
  measured   TEXT NOT NULL,                         -- JSON: bandwidth_gbps, tflops by dtype, source plugin, ...
  updated_at REAL NOT NULL
);

CREATE TABLE bench_history (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  hardware   TEXT NOT NULL,
  backend    TEXT NOT NULL,
  recipe     TEXT NOT NULL,
  model      TEXT NOT NULL,
  metric     TEXT NOT NULL,                         -- peak_mem_bytes ttft_ms tpot_ms tokens_per_s ...
  estimated  REAL,
  measured   REAL NOT NULL,
  created_at REAL NOT NULL
);
CREATE INDEX idx_bench_key ON bench_history(hardware, backend, recipe, metric);

CREATE TABLE calibration (
  hardware  TEXT NOT NULL,
  backend   TEXT NOT NULL,
  recipe    TEXT NOT NULL,
  eta_bw    REAL NOT NULL,
  mfu       REAL NOT NULL,
  overhead_bytes REAL NOT NULL,
  samples   INTEGER NOT NULL DEFAULT 0,
  updated_at REAL NOT NULL,
  PRIMARY KEY (hardware, backend, recipe)
);

CREATE TABLE cache_refs (
  hash        TEXT PRIMARY KEY,
  kind        TEXT NOT NULL,                        -- file | dir
  bytes       INTEGER NOT NULL,
  pins        INTEGER NOT NULL DEFAULT 0,
  created_at  REAL NOT NULL,
  last_access REAL NOT NULL
);
