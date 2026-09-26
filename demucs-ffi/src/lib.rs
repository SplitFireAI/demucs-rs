//! UniFFI surface for the demucs-swift package.
//!
//! Everything here is a thin translation layer over `demucs-core`: records and
//! enums prefixed `Ffi` mirror the core types, and the hand-written Swift in
//! demucs-swift wraps them again in an idiomatic API. Keep behaviour in the
//! core crate, not here.

// Proving `Demucs<Wgpu>: Send` for the uniffi object walks burn's whole
// fusion/wgpu type graph and overflows the default limit of 128.
#![recursion_limit = "256"]

use std::collections::HashMap;
use std::fs;
use std::io::{Read, Write};
use std::path::PathBuf;
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, RwLock};
use std::thread;
use std::time::{Duration, Instant};

use burn::backend::wgpu::{graphics::AutoGraphicsApi, init_setup, RuntimeOptions, WgpuDevice};
use demucs_core::listener::{ForwardEvent, ForwardListener};
use demucs_core::model::metadata::{self, ModelInfo, StemId};
use demucs_core::provider::fs::FsProvider;
use demucs_core::provider::ModelProvider;
use demucs_core::{Demucs, DemucsError, ModelOptions, Stem, TRAINING_LENGTH};

type B = burn::backend::wgpu::Wgpu;

/// The CLI needed an 8 MB main thread (demucs-rs#3); inference runs on its own
/// worker threads here, so they get the same.
const WORKER_STACK_SIZE: usize = 8 * 1024 * 1024;

const DOWNLOAD_CONNECT_TIMEOUT: Duration = Duration::from_secs(30);
const DOWNLOAD_READ_TIMEOUT: Duration = Duration::from_secs(60);

/// Sample rate the models were trained at. Input at any other rate is
/// resampled by demucs-core on the way in and on the way out.
const MODEL_SAMPLE_RATE: u64 = 44_100;

// ─── Records and enums ──────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, uniffi::Enum)]
pub enum FfiStemKind {
    Drums,
    Bass,
    Other,
    Vocals,
    Guitar,
    Piano,
}

impl From<StemId> for FfiStemKind {
    fn from(value: StemId) -> Self {
        match value {
            StemId::Drums => Self::Drums,
            StemId::Bass => Self::Bass,
            StemId::Other => Self::Other,
            StemId::Vocals => Self::Vocals,
            StemId::Guitar => Self::Guitar,
            StemId::Piano => Self::Piano,
        }
    }
}

impl From<FfiStemKind> for StemId {
    fn from(value: FfiStemKind) -> Self {
        match value {
            FfiStemKind::Drums => Self::Drums,
            FfiStemKind::Bass => Self::Bass,
            FfiStemKind::Other => Self::Other,
            FfiStemKind::Vocals => Self::Vocals,
            FfiStemKind::Guitar => Self::Guitar,
            FfiStemKind::Piano => Self::Piano,
        }
    }
}

/// Which model variant to download or load.
///
/// `FineTuned` runs one sub-model per requested stem, so asking for fewer
/// stems is proportionally faster. It only knows drums, bass, other and
/// vocals; an empty list or any other stem is rejected at load time.
#[derive(Debug, Clone, PartialEq, Eq, uniffi::Enum)]
pub enum FfiDemucsModel {
    FourStem,
    SixStem,
    FineTuned { stems: Vec<FfiStemKind> },
}

impl FfiDemucsModel {
    fn info(&self) -> &'static ModelInfo {
        match self {
            Self::FourStem => &metadata::HTDEMUCS,
            Self::SixStem => &metadata::HTDEMUCS_6S,
            Self::FineTuned { .. } => &metadata::HTDEMUCS_FT,
        }
    }

    fn options(&self) -> Result<ModelOptions, FfiDemucsError> {
        match self {
            Self::FourStem => Ok(ModelOptions::FourStem),
            Self::SixStem => Ok(ModelOptions::SixStem),
            Self::FineTuned { stems } => {
                let supported = metadata::HTDEMUCS_FT.stems;
                let mut selected: Vec<StemId> = Vec::with_capacity(stems.len());
                for &stem in stems {
                    let id = StemId::from(stem);
                    if !supported.contains(&id) {
                        return Err(FfiDemucsError::InvalidInput {
                            reason: format!("the fine-tuned model has no {} stem", id.as_str()),
                        });
                    }
                    if !selected.contains(&id) {
                        selected.push(id);
                    }
                }
                if selected.is_empty() {
                    return Err(FfiDemucsError::InvalidInput {
                        reason: "the fine-tuned model needs at least one stem".to_string(),
                    });
                }
                Ok(ModelOptions::FineTuned(selected))
            }
        }
    }
}

#[derive(Debug, Clone, uniffi::Record)]
pub struct FfiModelMetadata {
    pub model: FfiDemucsModel,
    pub id: String,
    pub label: String,
    pub description: String,
    pub size_mb: u32,
    pub stems: Vec<FfiStemKind>,
}

impl FfiModelMetadata {
    fn new(model: FfiDemucsModel) -> Self {
        let info = model.info();
        Self {
            model,
            id: info.id.to_string(),
            label: info.label.to_string(),
            description: info.description.to_string(),
            size_mb: info.size_mb,
            stems: info.stems.iter().copied().map(Into::into).collect(),
        }
    }
}

#[derive(Debug, Clone, uniffi::Record)]
pub struct FfiDownloadProgress {
    pub downloaded_bytes: u64,
    /// From `Content-Length`, or the model's nominal size when the server
    /// does not send one.
    pub total_bytes: u64,
    pub fraction: f64,
}

#[derive(Debug, Clone, uniffi::Record)]
pub struct FfiSeparationProgress {
    pub fraction: f64,
    pub chunk_index: u64,
    pub total_chunks: u64,
}

/// One separated source. `left` and `right` are at the input sample rate and
/// have the same length as the input channels.
#[derive(Debug, Clone, uniffi::Record)]
pub struct FfiStem {
    pub kind: FfiStemKind,
    pub left: Vec<f32>,
    pub right: Vec<f32>,
}

impl From<Stem> for FfiStem {
    fn from(value: Stem) -> Self {
        Self {
            kind: value.id.into(),
            left: value.left,
            right: value.right,
        }
    }
}

#[derive(Debug, thiserror::Error, uniffi::Error)]
pub enum FfiDemucsError {
    #[error("operation cancelled")]
    Cancelled,
    #[error("no model is loaded")]
    NotLoaded,
    #[error("model {model_id} is not downloaded")]
    NotCached { model_id: String },
    #[error("{reason}")]
    InvalidInput { reason: String },
    #[error("download failed: {reason}")]
    Download { reason: String },
    #[error("i/o error: {reason}")]
    Io { reason: String },
    #[error("{reason}")]
    Inference { reason: String },
}

impl From<DemucsError> for FfiDemucsError {
    fn from(value: DemucsError) -> Self {
        match value {
            DemucsError::Cancelled => Self::Cancelled,
            other => Self::Inference {
                reason: other.to_string(),
            },
        }
    }
}

impl From<std::io::Error> for FfiDemucsError {
    fn from(value: std::io::Error) -> Self {
        Self::Io {
            reason: value.to_string(),
        }
    }
}

// ─── Callbacks ──────────────────────────────────────────────────────────────

/// Called from a background thread. Return `false` to cancel the download.
#[uniffi::export(callback_interface)]
pub trait FfiDownloadListener: Send + Sync {
    fn on_progress(&self, progress: FfiDownloadProgress) -> bool;
}

/// Called from a background thread. Return `false` to cancel; cancellation
/// takes effect at the next chunk boundary (every ~7.8 s of audio), so audio
/// short enough to fit in one chunk always runs to completion.
#[uniffi::export(callback_interface)]
pub trait FfiSeparationListener: Send + Sync {
    fn on_progress(&self, progress: FfiSeparationProgress) -> bool;
}

/// Turns demucs-core's forward-pass events into a single monotonic fraction.
struct ProgressBridge {
    listener: Box<dyn FfiSeparationListener>,
    chunk: usize,
    total_chunks: usize,
    cancelled: bool,
}

impl ProgressBridge {
    fn report(&mut self, completed: f64) {
        let total = self.total_chunks.max(1);
        let progress = FfiSeparationProgress {
            fraction: (completed / total as f64).clamp(0.0, 1.0),
            chunk_index: self.chunk as u64,
            total_chunks: total as u64,
        };
        if !self.listener.on_progress(progress) {
            self.cancelled = true;
        }
    }
}

impl ForwardListener for ProgressBridge {
    fn on_event(&mut self, event: ForwardEvent) {
        match event {
            ForwardEvent::ChunkStarted { index, total } => {
                self.chunk = index;
                self.total_chunks = total;
            }
            ForwardEvent::StemDone { index, total } => {
                let within = (index + 1) as f64 / total.max(1) as f64;
                self.report(self.chunk as f64 + within);
            }
            ForwardEvent::ChunkDone { index, total } => {
                self.chunk = index;
                self.total_chunks = total;
                self.report((index + 1) as f64);
            }
            _ => {}
        }
    }

    fn is_cancelled(&self) -> bool {
        self.cancelled
    }
}

// ─── Engine ─────────────────────────────────────────────────────────────────

/// Holds at most one loaded model.
///
/// burn modules are `Send` but not `Sync` (parameters sit in a `OnceCell`), so
/// the model lives behind a `Mutex` and each operation holds it for its whole
/// run: loads, warmups and separations on one engine happen one at a time.
/// The loaded model's identity is kept apart so status queries never wait on
/// a separation.
#[derive(uniffi::Object)]
pub struct FfiDemucsEngine {
    state: Arc<Mutex<Option<Demucs<B>>>>,
    current: Arc<Mutex<Option<FfiDemucsModel>>>,
}

impl Default for FfiDemucsEngine {
    fn default() -> Self {
        Self {
            state: Arc::new(Mutex::new(None)),
            current: Arc::new(Mutex::new(None)),
        }
    }
}

#[uniffi::export]
impl FfiDemucsEngine {
    #[uniffi::constructor]
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// Loads a downloaded model, replacing any model already loaded. Returns
    /// the load time in seconds. Fails with `NotCached` if the weights have
    /// not been downloaded yet.
    pub async fn load_model(&self, model: FfiDemucsModel) -> Result<f64, FfiDemucsError> {
        let state = Arc::clone(&self.state);
        let current = Arc::clone(&self.current);
        run_blocking("demucs-load", move || {
            let started = Instant::now();
            let options = model.options()?;
            let info = model.info();
            let provider = FsProvider::with_dir(cache_dir()?);
            if !provider.is_cached(info) {
                return Err(FfiDemucsError::NotCached {
                    model_id: info.id.to_string(),
                });
            }
            let mut slot = lock(&state);
            // Free the old weights before reading the new ones so peak memory
            // is one model, not two.
            *slot = None;
            *lock(&current) = None;
            let bytes = provider.load_cached(info).map_err(|e| FfiDemucsError::Io {
                reason: e.to_string(),
            })?;
            let demucs = Demucs::<B>::from_bytes(options, &bytes, gpu_device())?;
            drop(bytes);
            *lock(&current) = Some(model);
            *slot = Some(demucs);
            Ok(started.elapsed().as_secs_f64())
        })
        .await?
    }

    /// Runs a dummy separation so the first real one does not pay for shader
    /// compilation.
    pub async fn warmup(&self) -> Result<(), FfiDemucsError> {
        let state = Arc::clone(&self.state);
        run_blocking("demucs-warmup", move || {
            let guard = lock(&state);
            let demucs = guard.as_ref().ok_or(FfiDemucsError::NotLoaded)?;
            pollster::block_on(demucs.warmup());
            Ok(())
        })
        .await?
    }

    /// Separates stereo PCM at any sample rate. Both channels must have the
    /// same, non-zero length; pass the same buffer twice for mono.
    pub async fn separate(
        &self,
        left: Vec<f32>,
        right: Vec<f32>,
        sample_rate: u32,
        listener: Box<dyn FfiSeparationListener>,
    ) -> Result<Vec<FfiStem>, FfiDemucsError> {
        validate_input(&left, &right, sample_rate)?;
        let state = Arc::clone(&self.state);
        run_blocking("demucs-separate", move || {
            let guard = lock(&state);
            let demucs = guard.as_ref().ok_or(FfiDemucsError::NotLoaded)?;
            let mut bridge = ProgressBridge {
                listener,
                chunk: 0,
                total_chunks: num_chunks(left.len() as u64, sample_rate) as usize,
                cancelled: false,
            };
            let stems = pollster::block_on(demucs.separate_with_listener(
                &left,
                &right,
                sample_rate,
                &mut bridge,
            ))?;
            Ok(stems.into_iter().map(Into::into).collect())
        })
        .await?
    }

    /// Frees the loaded model. Returns `false` if nothing was loaded.
    pub async fn unload_model(&self) -> bool {
        let state = Arc::clone(&self.state);
        let current = Arc::clone(&self.current);
        run_blocking("demucs-unload", move || {
            let mut slot = lock(&state);
            *lock(&current) = None;
            slot.take().is_some()
        })
        .await
        .unwrap_or(false)
    }

    pub fn is_loaded(&self) -> bool {
        lock(&self.current).is_some()
    }

    pub fn loaded_model(&self) -> Option<FfiDemucsModel> {
        lock(&self.current).clone()
    }
}

// ─── Model cache ────────────────────────────────────────────────────────────

static CACHE_DIR: RwLock<Option<PathBuf>> = RwLock::new(None);

/// Sets where model weights are stored. Sandboxed apps should call this once
/// at launch with a directory inside their container. Without it, weights go
/// to the platform cache directory under `demucs-rs/`.
#[uniffi::export]
pub fn configure_cache_dir(path: String) {
    *CACHE_DIR.write().unwrap_or_else(|e| e.into_inner()) = Some(PathBuf::from(path));
}

fn cache_dir() -> Result<PathBuf, FfiDemucsError> {
    if let Some(dir) = CACHE_DIR.read().unwrap_or_else(|e| e.into_inner()).clone() {
        return Ok(dir);
    }
    dirs::cache_dir()
        .map(|base| base.join("demucs-rs"))
        .ok_or_else(|| FfiDemucsError::Io {
            reason: "could not determine the cache directory; call configure_cache_dir".to_string(),
        })
}

fn model_path(model: &FfiDemucsModel) -> Option<PathBuf> {
    cache_dir().ok().map(|dir| dir.join(model.info().filename))
}

#[uniffi::export]
pub fn all_models() -> Vec<FfiModelMetadata> {
    // Listed explicitly rather than derived from `ALL_MODELS` ids, so a model
    // added to demucs-core without an `FfiDemucsModel` variant fails the
    // `all_models_matches_core_metadata` test instead of being mislabelled.
    [
        FfiDemucsModel::FourStem,
        FfiDemucsModel::SixStem,
        FfiDemucsModel::FineTuned {
            stems: metadata::HTDEMUCS_FT
                .stems
                .iter()
                .copied()
                .map(Into::into)
                .collect(),
        },
    ]
    .into_iter()
    .map(FfiModelMetadata::new)
    .collect()
}

#[uniffi::export]
pub fn model_metadata(model: FfiDemucsModel) -> FfiModelMetadata {
    FfiModelMetadata::new(model)
}

/// All fine-tuned stem selections share one weights file, so this answers the
/// same for every `FineTuned` value.
#[uniffi::export]
pub fn is_model_cached(model: FfiDemucsModel) -> bool {
    model_path(&model).is_some_and(|path| path.exists())
}

#[uniffi::export]
pub fn cached_model_path(model: FfiDemucsModel) -> Option<String> {
    model_path(&model)
        .filter(|path| path.exists())
        .map(|path| path.to_string_lossy().into_owned())
}

/// Returns `true` if a file was removed.
#[uniffi::export]
pub fn delete_cached_model(model: FfiDemucsModel) -> bool {
    model_path(&model).is_some_and(|path| fs::remove_file(path).is_ok())
}

/// Downloads the model's weights from HuggingFace into the cache directory.
/// Returns immediately if they are already there.
#[uniffi::export]
pub async fn download_model(
    model: FfiDemucsModel,
    listener: Box<dyn FfiDownloadListener>,
) -> Result<(), FfiDemucsError> {
    run_blocking("demucs-download", move || {
        download(model.info(), &*listener)
    })
    .await?
}

fn download(info: &ModelInfo, listener: &dyn FfiDownloadListener) -> Result<(), FfiDemucsError> {
    // Two downloads of the same file would share one `.part` path and
    // interleave their writes. The second caller waits here instead, then
    // finds the weights already cached.
    let file_lock = download_lock(info.filename);
    let _downloading = lock(&file_lock);
    let dir = cache_dir()?;
    let target = dir.join(info.filename);
    if target.exists() {
        return Ok(());
    }
    fs::create_dir_all(&dir)?;

    let download_error = |reason: String| FfiDemucsError::Download { reason };
    let url = metadata::download_url(info);
    let tls = ureq::native_tls::TlsConnector::new().map_err(|e| download_error(e.to_string()))?;
    // No overall timeout: the fine-tuned weights are 333 MB. The connect and
    // per-read timeouts turn a stalled connection into an error instead of a
    // read that blocks forever, where the cancel callback can never run.
    let agent = ureq::AgentBuilder::new()
        .tls_connector(Arc::new(tls))
        .timeout_connect(DOWNLOAD_CONNECT_TIMEOUT)
        .timeout_read(DOWNLOAD_READ_TIMEOUT)
        .build();
    let response = agent
        .get(&url)
        .call()
        .map_err(|e| download_error(e.to_string()))?;
    let total_bytes = response
        .header("Content-Length")
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(info.size_mb as u64 * 1_000_000);

    // Write next to the target and rename at the end, so an interrupted
    // download never leaves a truncated file that `is_model_cached` trusts.
    let partial = dir.join(format!("{}.part", info.filename));
    let result = (|| {
        let mut reader = response.into_reader();
        let mut file = fs::File::create(&partial)?;
        let mut buffer = vec![0u8; 256 * 1024];
        let mut downloaded = 0u64;
        loop {
            let n = reader.read(&mut buffer)?;
            if n == 0 {
                break;
            }
            file.write_all(&buffer[..n])?;
            downloaded += n as u64;
            let progress = FfiDownloadProgress {
                downloaded_bytes: downloaded,
                total_bytes,
                fraction: (downloaded as f64 / total_bytes.max(1) as f64).min(1.0),
            };
            if !listener.on_progress(progress) {
                return Err(FfiDemucsError::Cancelled);
            }
        }
        file.sync_all()?;
        fs::rename(&partial, &target)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&partial);
    }
    result
}

/// How many chunks `separate` will split this many input samples into, for
/// sizing progress UI up front. `sample_rate` is the input rate; audio is
/// chunked after resampling to 44.1 kHz.
#[uniffi::export]
pub fn num_chunks(n_samples: u64, sample_rate: u32) -> u64 {
    let resampled = if sample_rate == 0 || sample_rate as u64 == MODEL_SAMPLE_RATE {
        n_samples
    } else {
        (n_samples as u128 * MODEL_SAMPLE_RATE as u128).div_ceil(sample_rate as u128) as u64
    };
    demucs_core::num_chunks(usize::try_from(resampled).unwrap_or(usize::MAX)) as u64
}

/// Samples per chunk at 44.1 kHz.
#[uniffi::export]
pub fn chunk_length() -> u64 {
    TRAINING_LENGTH as u64
}

// ─── Helpers ────────────────────────────────────────────────────────────────

fn validate_input(left: &[f32], right: &[f32], sample_rate: u32) -> Result<(), FfiDemucsError> {
    let reason = if left.is_empty() {
        "audio is empty"
    } else if left.len() != right.len() {
        "left and right channels have different lengths"
    } else if sample_rate == 0 {
        "sample rate must be greater than zero"
    } else {
        return Ok(());
    };
    Err(FfiDemucsError::InvalidInput {
        reason: reason.to_string(),
    })
}

/// Initializes the wgpu runtime once per process and returns its device.
fn gpu_device() -> WgpuDevice {
    static INIT: OnceLock<()> = OnceLock::new();
    let device = WgpuDevice::default();
    INIT.get_or_init(|| {
        let options = RuntimeOptions {
            tasks_max: 128,
            ..Default::default()
        };
        init_setup::<AutoGraphicsApi>(&device, options);
    });
    device
}

/// Runs blocking work on a dedicated thread and awaits it without needing an
/// async runtime, so the foreign side's executor never blocks on inference.
async fn run_blocking<T, F>(name: &str, work: F) -> Result<T, FfiDemucsError>
where
    T: Send + 'static,
    F: FnOnce() -> T + Send + 'static,
{
    let (tx, rx) = futures_channel::oneshot::channel();
    thread::Builder::new()
        .name(name.to_string())
        .stack_size(WORKER_STACK_SIZE)
        .spawn(move || {
            let _ = tx.send(work());
        })?;
    rx.await.map_err(|_| FfiDemucsError::Inference {
        reason: format!("{name} worker panicked"),
    })
}

/// One lock per weights file. The fine-tuned stem selections share a file, so
/// this is keyed by filename rather than by model.
fn download_lock(filename: &'static str) -> Arc<Mutex<()>> {
    type Locks = Mutex<HashMap<&'static str, Arc<Mutex<()>>>>;
    static LOCKS: OnceLock<Locks> = OnceLock::new();
    let mut locks = lock(LOCKS.get_or_init(Default::default));
    Arc::clone(locks.entry(filename).or_default())
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|e| e.into_inner())
}

uniffi::setup_scaffolding!();

#[cfg(test)]
mod tests {
    use super::*;
    use demucs_core::model::metadata::ALL_MODELS;

    #[test]
    fn fine_tuned_rejects_unsupported_and_empty_selections() {
        let guitar = FfiDemucsModel::FineTuned {
            stems: vec![FfiStemKind::Guitar],
        };
        assert!(matches!(
            guitar.options(),
            Err(FfiDemucsError::InvalidInput { .. })
        ));
        let empty = FfiDemucsModel::FineTuned { stems: vec![] };
        assert!(matches!(
            empty.options(),
            Err(FfiDemucsError::InvalidInput { .. })
        ));
    }

    #[test]
    fn fine_tuned_deduplicates_stems() {
        let model = FfiDemucsModel::FineTuned {
            stems: vec![FfiStemKind::Vocals, FfiStemKind::Vocals, FfiStemKind::Drums],
        };
        match model.options() {
            Ok(ModelOptions::FineTuned(stems)) => {
                assert_eq!(stems, vec![StemId::Vocals, StemId::Drums]);
            }
            _ => panic!("expected a fine-tuned selection"),
        }
    }

    #[test]
    fn all_models_matches_core_metadata() {
        let models = all_models();
        assert_eq!(models.len(), ALL_MODELS.len());
        for (ffi, core) in models.iter().zip(ALL_MODELS) {
            assert_eq!(ffi.id, core.id);
            assert_eq!(ffi.model.info().id, core.id);
            assert_eq!(ffi.stems.len(), core.stems.len());
        }
    }

    #[test]
    fn num_chunks_accounts_for_resampling() {
        let one_chunk = TRAINING_LENGTH as u64;
        assert_eq!(num_chunks(one_chunk, 44_100), 1);
        // The same sample count at 22.05 kHz doubles after resampling.
        assert!(num_chunks(one_chunk, 22_050) > 1);
        assert_eq!(num_chunks(0, 0), 1);
    }

    #[test]
    fn validate_input_rejects_bad_buffers() {
        assert!(validate_input(&[], &[], 44_100).is_err());
        assert!(validate_input(&[0.0], &[0.0, 0.0], 44_100).is_err());
        assert!(validate_input(&[0.0], &[0.0], 0).is_err());
        assert!(validate_input(&[0.0], &[0.0], 44_100).is_ok());
    }

    #[derive(Clone, Default)]
    struct Recorder {
        fractions: Arc<Mutex<Vec<f64>>>,
        cancel_after: Option<usize>,
    }

    impl FfiSeparationListener for Recorder {
        fn on_progress(&self, progress: FfiSeparationProgress) -> bool {
            let mut fractions = lock(&self.fractions);
            fractions.push(progress.fraction);
            self.cancel_after.is_none_or(|n| fractions.len() < n)
        }
    }

    #[test]
    fn progress_bridge_is_monotonic_and_reaches_one() {
        let recorder = Recorder::default();
        let mut bridge = ProgressBridge {
            listener: Box::new(recorder.clone()),
            chunk: 0,
            total_chunks: 2,
            cancelled: false,
        };
        for chunk in 0..2 {
            bridge.on_event(ForwardEvent::ChunkStarted {
                index: chunk,
                total: 2,
            });
            for stem in 0..4 {
                bridge.on_event(ForwardEvent::StemDone {
                    index: stem,
                    total: 4,
                });
            }
            bridge.on_event(ForwardEvent::ChunkDone {
                index: chunk,
                total: 2,
            });
        }
        let fractions = lock(&recorder.fractions).clone();
        assert!(fractions.windows(2).all(|w| w[0] <= w[1]), "{fractions:?}");
        assert_eq!(fractions.last().copied(), Some(1.0));
        assert!(!bridge.is_cancelled());
    }

    #[test]
    fn progress_bridge_cancels_when_listener_returns_false() {
        let mut bridge = ProgressBridge {
            listener: Box::new(Recorder {
                cancel_after: Some(1),
                ..Default::default()
            }),
            chunk: 0,
            total_chunks: 1,
            cancelled: false,
        };
        bridge.on_event(ForwardEvent::StemDone { index: 0, total: 4 });
        assert!(bridge.is_cancelled());
    }
}
