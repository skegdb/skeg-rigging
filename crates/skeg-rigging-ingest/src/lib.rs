//! `skeg-rigging-ingest` — take files, embed them, write them into a
//! tenant.
//!
//! This is the missing "load my data into skeg" step, placed at the
//! rigging layer on purpose: it writes through the
//! [`TenantWrite`](skeg_rigging::TenantWrite) capability, so the very
//! same routine fills a **local on-disk vault** (`skeg-rigging-skeg`) or
//! a **remote skeg server** (`skeg-rigging-net-resp3`) without caring
//! which. Consumers like `hansa-cli` and `skeg-cli` just pick the writer.
//!
//! The pipeline is: walk a path → [`chunk`] each file → [`Embed`] each
//! chunk (caller-side, e.g. Ollama) → `TenantWrite::insert`. Each record
//! stores the chunk text as its payload and a `src:<relpath>:<lines>`
//! tag for provenance.
//!
//! With the `watch` feature, [`watch_tree`] keeps a directory live:
//! files created or modified after start are re-ingested as they change.

mod chunk;
mod embed;

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use skeg_rigging::{RecordId, TenantWrite};

pub use chunk::{Chunk, ChunkMode, chunk, word_count};
pub use embed::{Embed, OllamaEmbed};

/// Knobs for an ingest run.
#[derive(Debug, Clone)]
pub struct IngestOptions {
    /// Allowed file extensions (without the dot). Empty = accept all.
    pub exts: Vec<String>,
    /// Skip any path containing one of these substrings.
    pub excludes: Vec<String>,
    /// Mark inserted records as shareable with federation peers.
    pub shareable: bool,
    /// Extra tags attached to every record (on top of the `src:` tag).
    pub tags: Vec<String>,
    /// First record id to allocate; ids increase by one per chunk.
    pub start_id: u64,
}

impl Default for IngestOptions {
    fn default() -> Self {
        IngestOptions {
            exts: Vec::new(),
            excludes: vec![
                "/.git/".into(),
                "/target/".into(),
                "/node_modules/".into(),
                "/.venv/".into(),
                "/dist/".into(),
            ],
            shareable: false,
            tags: Vec::new(),
            start_id: 0,
        }
    }
}

/// What an ingest run produced.
#[derive(Debug, Clone, Default)]
pub struct IngestStats {
    /// Files read and chunked.
    pub files: usize,
    /// Chunks embedded and inserted.
    pub chunks: usize,
    /// Total words across all chunks.
    pub words: usize,
    /// Next free record id after this run (persist this for the next run).
    pub next_id: u64,
}

/// Ingest a file or directory tree into `writer`.
///
/// `progress` is called once per file with the cumulative stats so far,
/// so a CLI can render a live line; pass `&mut |_| {}` to ignore it.
pub fn ingest_tree<W: TenantWrite, E: Embed>(
    writer: &mut W,
    embed: &E,
    root: &Path,
    opts: &IngestOptions,
    progress: &mut dyn FnMut(&IngestStats),
) -> Result<IngestStats> {
    if embed.dim() != writer.embedding_dim() {
        anyhow::bail!(
            "embedding dim {} does not match tenant dim {}",
            embed.dim(),
            writer.embedding_dim()
        );
    }
    let files = collect_files(root, opts)?;
    let mut stats = IngestStats {
        next_id: opts.start_id,
        ..Default::default()
    };
    for file in &files {
        ingest_file(writer, embed, root, file, opts, &mut stats)?;
        stats.files += 1;
        progress(&stats);
    }
    writer
        .flush()
        .map_err(|e| anyhow::anyhow!("flush writer: {e}"))?;
    Ok(stats)
}

/// Ingest one file, advancing `stats` (id counter, counts).
fn ingest_file<W: TenantWrite, E: Embed>(
    writer: &mut W,
    embed: &E,
    root: &Path,
    file: &Path,
    opts: &IngestOptions,
    stats: &mut IngestStats,
) -> Result<()> {
    let text = std::fs::read_to_string(file).with_context(|| format!("read {}", file.display()))?;
    let ext = file.extension().and_then(|s| s.to_str()).unwrap_or("");
    let rel = relpath(root, file);
    for c in chunk::chunk(&text, ChunkMode::for_ext(ext)) {
        let v = embed.passage(&c.text)?;
        let mut tags = opts.tags.clone();
        tags.push(format!("src:{rel}:{}-{}", c.line_lo, c.line_hi));
        writer
            .insert(
                RecordId(stats.next_id),
                &v,
                opts.shareable,
                tags,
                c.text.as_bytes().to_vec(),
            )
            .map_err(|e| anyhow::anyhow!("insert record {}: {e}", stats.next_id))?;
        stats.next_id += 1;
        stats.chunks += 1;
        stats.words += word_count(&c.text);
    }
    Ok(())
}

/// Path of `file` relative to `root` (or the file name if `root` is the
/// file itself / not a prefix).
fn relpath(root: &Path, file: &Path) -> String {
    file.strip_prefix(root)
        .ok()
        .map(|p| p.to_string_lossy().into_owned())
        .or_else(|| file.file_name().map(|n| n.to_string_lossy().into_owned()))
        .unwrap_or_else(|| file.to_string_lossy().into_owned())
}

/// True if `file` should be ingested under `opts`.
fn accepts(file: &Path, opts: &IngestOptions) -> bool {
    let p = file.to_string_lossy();
    if opts.excludes.iter().any(|e| p.contains(e.as_str())) {
        return false;
    }
    if opts.exts.is_empty() {
        return true;
    }
    match file.extension().and_then(|s| s.to_str()) {
        Some(ext) => opts.exts.iter().any(|e| e == ext),
        None => false,
    }
}

/// Collect ingestable files under `root` (which may be a single file).
fn collect_files(root: &Path, opts: &IngestOptions) -> Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    if root.is_file() {
        if accepts(root, opts) {
            out.push(root.to_path_buf());
        }
        return Ok(out);
    }
    walk(root, opts, &mut out)?;
    out.sort();
    Ok(out)
}

fn walk(dir: &Path, opts: &IngestOptions, out: &mut Vec<PathBuf>) -> Result<()> {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return Ok(()), // unreadable dir: skip, don't fail the run
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let s = path.to_string_lossy();
        if opts.excludes.iter().any(|e| s.contains(e.as_str())) {
            continue;
        }
        if path.is_dir() {
            walk(&path, opts, out)?;
        } else if path.is_file() && accepts(&path, opts) {
            out.push(path);
        }
    }
    Ok(())
}

/// Watch `root` and re-ingest files as they are created or modified.
///
/// Blocks until the process is interrupted. `on_batch` is called with the
/// cumulative stats after each file is ingested. New ids are allocated
/// from `opts.start_id` onward; a re-edited file appends fresh records
/// (this v1 does not delete the file's previous chunks — stable
/// per-chunk ids are a future refinement).
#[cfg(feature = "watch")]
pub fn watch_tree<W: TenantWrite, E: Embed>(
    writer: &mut W,
    embed: &E,
    root: &Path,
    mut opts: IngestOptions,
    mut on_batch: impl FnMut(&IngestStats),
) -> Result<()> {
    use notify::{EventKind, RecursiveMode, Watcher};
    use std::collections::HashMap;
    use std::sync::mpsc::channel;
    use std::time::{Duration, Instant};

    let (tx, rx) = channel();
    let mut watcher = notify::recommended_watcher(move |res| {
        let _ = tx.send(res);
    })
    .context("create filesystem watcher")?;
    watcher
        .watch(root, RecursiveMode::Recursive)
        .with_context(|| format!("watch {}", root.display()))?;

    // A single save emits several fs events (create + data + metadata).
    // Debounce per path so one edit ingests once, not three times.
    let debounce = Duration::from_millis(800);
    let mut last_seen: HashMap<PathBuf, Instant> = HashMap::new();
    let mut next_id = opts.start_id;
    for res in rx {
        let event = match res {
            Ok(e) => e,
            Err(_) => continue,
        };
        if !matches!(event.kind, EventKind::Create(_) | EventKind::Modify(_)) {
            continue;
        }
        for path in event.paths {
            if !path.is_file() || !accepts(&path, &opts) {
                continue;
            }
            let now = Instant::now();
            if let Some(prev) = last_seen.get(&path)
                && now.duration_since(*prev) < debounce
            {
                continue;
            }
            last_seen.insert(path.clone(), now);
            opts.start_id = next_id;
            let mut stats = IngestStats {
                next_id,
                ..Default::default()
            };
            // A file mid-write can fail to read; skip and let the next
            // event pick it up.
            if ingest_file(writer, embed, root, &path, &opts, &mut stats).is_err() {
                continue;
            }
            stats.files = 1;
            let _ = writer.flush();
            next_id = stats.next_id;
            on_batch(&stats);
        }
    }
    Ok(())
}
