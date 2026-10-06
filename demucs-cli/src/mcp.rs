//! Minimal MCP server over stdio (newline-delimited JSON-RPC 2.0).
//!
//! stdout carries JSON-RPC only; everything human-facing goes to stderr.
//! One thread reads stdin, one persistent 8 MB-stack worker runs separations
//! (and owns the loaded model), and a mutex serialises writes to stdout.

use std::io::{BufRead, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver};
use std::sync::{Arc, Mutex};

use anyhow::{bail, Result};
use demucs_core::listener::{ForwardEvent, ForwardListener};
use demucs_core::model::metadata::ALL_MODELS;
use demucs_core::provider::fs::FsProvider;
use demucs_core::provider::ModelProvider;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::separate::{default_output_dir, separate_file, ModelCache, Reporter, SeparateRequest};

type InFlight = Arc<Mutex<Option<(Value, Arc<AtomicBool>)>>>;

const LATEST_PROTOCOL: &str = "2025-06-18";
const KNOWN_PROTOCOLS: &[&str] = &["2024-11-05", "2025-03-26", "2025-06-18"];

#[derive(Clone)]
struct Writer(Arc<Mutex<std::io::Stdout>>);

impl Writer {
    fn send(&self, msg: &Value) {
        let mut out = self.0.lock().unwrap_or_else(|e| e.into_inner());
        let _ = writeln!(out, "{msg}");
        let _ = out.flush();
    }

    fn result(&self, id: &Value, result: Value) {
        self.send(&json!({"jsonrpc": "2.0", "id": id, "result": result}));
    }

    fn error(&self, id: &Value, code: i64, message: &str) {
        self.send(
            &json!({"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message}}),
        );
    }
}

struct Job {
    id: Value,
    args: SeparateArgs,
    progress_token: Option<Value>,
    cancel: Arc<AtomicBool>,
}

#[derive(Deserialize)]
struct SeparateArgs {
    input: String,
    model: Option<String>,
    stems: Option<Vec<String>>,
    output_dir: Option<String>,
}

/// Progress is reported as a 0..100 scale so download, warmup and chunks stay monotonic.
struct Progress {
    writer: Writer,
    token: Option<Value>,
    last: AtomicU64,
}

impl Progress {
    fn emit(&self, pct: f64, message: &str) {
        let Some(token) = &self.token else { return };
        // Never go backwards; scale to thousandths for the atomic.
        let v = (pct * 1000.0) as u64;
        let prev = self.last.fetch_max(v, Ordering::SeqCst);
        if v <= prev && prev != 0 {
            return;
        }
        self.writer.send(&json!({
            "jsonrpc": "2.0",
            "method": "notifications/progress",
            "params": {"progressToken": token, "progress": pct, "total": 100, "message": message},
        }));
    }
}

struct McpReporter(Arc<Progress>);

impl Reporter for McpReporter {
    fn status(&self, msg: &str) {
        eprintln!("{msg}");
        let msg = msg.trim();
        if msg.starts_with("Loading model") || msg.starts_with("Pre-compiling") {
            self.0.emit(20.0, msg);
        } else if msg.starts_with("Separating") {
            self.0.emit(25.0, msg);
        }
    }

    fn download_progress(&self, done: u64, total: u64) {
        let frac = done as f64 / total.max(1) as f64;
        self.0.emit(
            (frac * 20.0).min(20.0),
            &format!("Downloading model weights ({} MB)", done / 1_000_000),
        );
    }

    fn show_download_bar(&self) -> bool {
        false
    }
}

struct McpListener {
    progress: Arc<Progress>,
    cancel: Arc<AtomicBool>,
}

impl ForwardListener for McpListener {
    fn on_event(&mut self, event: ForwardEvent) {
        if let ForwardEvent::ChunkDone { index, total } = event {
            let pct = 25.0 + 75.0 * (index + 1) as f64 / total.max(1) as f64;
            self.progress
                .emit(pct, &format!("Separated chunk {}/{}", index + 1, total));
        }
    }

    fn is_cancelled(&self) -> bool {
        self.cancel.load(Ordering::SeqCst)
    }
}

pub fn serve() -> Result<()> {
    let writer = Writer(Arc::new(Mutex::new(std::io::stdout())));
    let (tx, rx) = mpsc::channel::<Value>();

    // Reader thread: a malformed line is answered with a parse error here.
    let err_writer = writer.clone();
    std::thread::spawn(move || {
        for line in std::io::stdin().lock().lines() {
            let Ok(line) = line else { break };
            if line.trim().is_empty() {
                continue;
            }
            match serde_json::from_str::<Value>(&line) {
                Ok(v) => {
                    if tx.send(v).is_err() {
                        break;
                    }
                }
                Err(e) => err_writer.error(&Value::Null, -32700, &format!("Parse error: {e}")),
            }
        }
    });

    let busy = Arc::new(AtomicBool::new(false));
    let in_flight: InFlight = Arc::new(Mutex::new(None));

    let (job_tx, job_rx) = mpsc::channel::<Job>();
    let worker = {
        let writer = writer.clone();
        let busy = busy.clone();
        let in_flight = in_flight.clone();
        std::thread::Builder::new()
            .name("demucs-mcp-worker".into())
            .stack_size(8 * 1024 * 1024)
            .spawn(move || worker_loop(job_rx, writer, busy, in_flight))?
    };

    dispatch_loop(rx, &writer, &busy, &in_flight, &job_tx);

    // stdin closed: stop any running job, then let the worker drain and exit.
    if let Some((_, cancel)) = in_flight.lock().unwrap_or_else(|e| e.into_inner()).as_ref() {
        cancel.store(true, Ordering::SeqCst);
    }
    drop(job_tx);
    let _ = worker.join();
    Ok(())
}

fn dispatch_loop(
    rx: Receiver<Value>,
    writer: &Writer,
    busy: &Arc<AtomicBool>,
    in_flight: &InFlight,
    job_tx: &mpsc::Sender<Job>,
) {
    for msg in rx {
        let method = msg.get("method").and_then(Value::as_str).unwrap_or("");
        let id = msg.get("id").cloned();
        let params = msg.get("params").cloned().unwrap_or(Value::Null);

        // Notifications (no id) and stray responses never get a reply.
        let Some(id) = id else {
            if method == "notifications/cancelled" {
                let guard = in_flight.lock().unwrap_or_else(|e| e.into_inner());
                if let (Some(rid), Some((cur, cancel))) = (params.get("requestId"), guard.as_ref())
                {
                    if cur == rid {
                        cancel.store(true, Ordering::SeqCst);
                    }
                }
            }
            continue;
        };
        if method.is_empty() {
            continue;
        }

        match method {
            "initialize" => {
                let requested = params.get("protocolVersion").and_then(Value::as_str);
                let version = match requested {
                    Some(v) if KNOWN_PROTOCOLS.contains(&v) && v < LATEST_PROTOCOL => v,
                    _ => LATEST_PROTOCOL,
                };
                writer.result(
                    &id,
                    json!({
                        "protocolVersion": version,
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": "demucs", "version": env!("CARGO_PKG_VERSION")},
                    }),
                );
            }
            "ping" => writer.result(&id, json!({})),
            "tools/list" => writer.result(&id, json!({"tools": tool_definitions()})),
            "tools/call" => {
                let name = params.get("name").and_then(Value::as_str).unwrap_or("");
                let args = params.get("arguments").cloned().unwrap_or(json!({}));
                match name {
                    "list_models" => writer.result(&id, list_models()),
                    "separate_stems" => {
                        let parsed = match serde_json::from_value::<SeparateArgs>(args) {
                            Ok(a) => a,
                            Err(e) => {
                                writer.result(&id, tool_error(&format!("Invalid arguments: {e}")));
                                continue;
                            }
                        };
                        if busy.swap(true, Ordering::SeqCst) {
                            writer.result(
                                &id,
                                tool_error("busy: another separation is already running"),
                            );
                            continue;
                        }
                        let cancel = Arc::new(AtomicBool::new(false));
                        *in_flight.lock().unwrap_or_else(|e| e.into_inner()) =
                            Some((id.clone(), cancel.clone()));
                        let job = Job {
                            id: id.clone(),
                            args: parsed,
                            progress_token: params
                                .get("_meta")
                                .and_then(|m| m.get("progressToken"))
                                .cloned(),
                            cancel,
                        };
                        if job_tx.send(job).is_err() {
                            busy.store(false, Ordering::SeqCst);
                            writer.result(&id, tool_error("worker unavailable"));
                        }
                    }
                    other => writer.error(&id, -32602, &format!("Unknown tool: {other}")),
                }
            }
            other => writer.error(&id, -32601, &format!("Method not found: {other}")),
        }
    }
}

fn worker_loop(jobs: Receiver<Job>, writer: Writer, busy: Arc<AtomicBool>, in_flight: InFlight) {
    let mut cache = ModelCache::default();
    for job in jobs {
        let result = run_job(&job, &mut cache, &writer);
        *in_flight.lock().unwrap_or_else(|e| e.into_inner()) = None;
        busy.store(false, Ordering::SeqCst);
        writer.result(&job.id, result);
    }
}

fn run_job(job: &Job, cache: &mut ModelCache, writer: &Writer) -> Value {
    let outcome = (|| -> Result<Value> {
        let input = PathBuf::from(&job.args.input);
        if !input.is_absolute() {
            bail!("input must be an absolute path: {}", job.args.input);
        }
        if !input.is_file() {
            bail!("input file not found: {}", input.display());
        }
        let output = match &job.args.output_dir {
            Some(d) => {
                let p = PathBuf::from(d);
                if !p.is_absolute() {
                    bail!("output_dir must be an absolute path: {d}");
                }
                p
            }
            None => default_output_dir(&input),
        };
        let req = SeparateRequest {
            input,
            model: job.args.model.clone().unwrap_or_else(|| "htdemucs".into()),
            stems: job.args.stems.clone(),
            output,
        };
        if job.cancel.load(Ordering::SeqCst) {
            bail!("cancelled");
        }
        let progress = Arc::new(Progress {
            writer: writer.clone(),
            token: job.progress_token.clone(),
            last: AtomicU64::new(0),
        });
        let reporter = McpReporter(progress.clone());
        let cancel = job.cancel.clone();
        let out = separate_file(
            &req,
            cache,
            |_plan| McpListener { progress, cancel },
            &reporter,
        )?;

        let stems: Vec<Value> = out
            .stems
            .iter()
            .map(|(id, p)| json!({"id": id.as_str(), "path": p.to_string_lossy()}))
            .collect();
        let summary = format!(
            "Separated {:.1}s of audio with {} into {} stems:\n{}",
            out.duration_secs,
            out.model,
            stems.len(),
            out.stems
                .iter()
                .map(|(id, p)| format!("- {}: {}", id.as_str(), p.display()))
                .collect::<Vec<_>>()
                .join("\n")
        );
        Ok(json!({
            "content": [{"type": "text", "text": summary}],
            "structuredContent": {
                "model": out.model,
                "sample_rate": out.sample_rate,
                "duration_secs": out.duration_secs,
                "stems": stems,
            },
            "isError": false,
        }))
    })();
    match outcome {
        Ok(v) => v,
        Err(_) if job.cancel.load(Ordering::SeqCst) => tool_error("cancelled"),
        Err(e) => tool_error(&format!("{e:#}")),
    }
}

fn tool_error(msg: &str) -> Value {
    json!({"content": [{"type": "text", "text": msg}], "isError": true})
}

fn list_models() -> Value {
    let provider = FsProvider::new().ok();
    let models: Vec<Value> = ALL_MODELS
        .iter()
        .map(|m| {
            json!({
                "id": m.id,
                "label": m.label,
                "description": m.description,
                "size_mb": m.size_mb,
                "stems": m.stems.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
                "cached": provider.as_ref().is_some_and(|p| p.is_cached(m)),
            })
        })
        .collect();
    let text = models
        .iter()
        .map(|m| {
            format!(
                "- {} ({} MB, {}){}",
                m["id"].as_str().unwrap_or_default(),
                m["size_mb"],
                m["description"].as_str().unwrap_or_default(),
                if m["cached"].as_bool() == Some(true) {
                    " [cached]"
                } else {
                    " [download required]"
                }
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    json!({
        "content": [{"type": "text", "text": text}],
        "structuredContent": {"models": models},
        "isError": false,
    })
}

fn tool_definitions() -> Value {
    json!([
        {
            "name": "separate_stems",
            "description": "Separate a music file into stems (drums, bass, vocals, other; guitar and piano with htdemucs_6s) and write them as WAV files. Runs locally and can take minutes. Downloads model weights on first use of a model (see list_models).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "input": {"type": "string", "description": "Absolute path to the input audio file (WAV, AIFF, FLAC, MP3, OGG, M4A/AAC)."},
                    "model": {"type": "string", "enum": ["htdemucs", "htdemucs_6s", "htdemucs_ft"], "default": "htdemucs"},
                    "stems": {"type": "array", "items": {"type": "string", "enum": ["drums", "bass", "other", "vocals", "guitar", "piano"]}, "description": "Stems to write. Defaults to all stems of the model."},
                    "output_dir": {"type": "string", "description": "Absolute output directory. Defaults to '<input name>_stems' beside the input."}
                },
                "required": ["input"],
                "additionalProperties": false
            }
        },
        {
            "name": "list_models",
            "description": "List available separation models with size, stems and whether the weights are already cached locally (uncached models are downloaded on first use).",
            "inputSchema": {"type": "object", "properties": {}, "additionalProperties": false}
        }
    ])
}
