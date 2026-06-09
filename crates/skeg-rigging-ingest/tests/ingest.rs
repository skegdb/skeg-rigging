//! Integration tests for the ingest pipeline.
//!
//! These use a deterministic [`StubEmbed`] and an in-memory `MockWriter`,
//! so they run in CI with no model server and no skeg backend.

use std::path::PathBuf;

use skeg_rigging::{RecordId, TenantWrite};
use skeg_rigging_ingest::{ChunkMode, IngestOptions, StubEmbed, chunk, ingest_tree};

const DIM: u32 = 64;

/// A `TenantWrite` that just records what it was asked to insert.
#[derive(Default)]
struct MockWriter {
    rows: Vec<Row>,
    flushed: usize,
}

#[derive(Clone)]
struct Row {
    id: u64,
    dim: usize,
    shareable: bool,
    tags: Vec<String>,
    payload: String,
}

#[derive(Debug)]
struct MockErr;
impl std::fmt::Display for MockErr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "mock error")
    }
}
impl std::error::Error for MockErr {}

impl TenantWrite for MockWriter {
    type Error = MockErr;

    fn insert(
        &mut self,
        record_id: RecordId,
        embedding: &[f32],
        shareable: bool,
        tags: Vec<String>,
        payload: Vec<u8>,
    ) -> Result<(), MockErr> {
        self.rows.push(Row {
            id: record_id.0,
            dim: embedding.len(),
            shareable,
            tags,
            payload: String::from_utf8_lossy(&payload).into_owned(),
        });
        Ok(())
    }

    fn flush(&mut self) -> Result<(), MockErr> {
        self.flushed += 1;
        Ok(())
    }

    fn embedding_dim(&self) -> u32 {
        DIM
    }
}

/// Make a temp dir under the OS temp root, unique to `tag`.
fn tmpdir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("skeg-ingest-test-{tag}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn write(dir: &PathBuf, name: &str, body: &str) {
    let p = dir.join(name);
    if let Some(parent) = p.parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    std::fs::write(p, body).unwrap();
}

#[test]
fn ingest_dir_counts_and_payloads() {
    let dir = tmpdir("dir");
    write(
        &dir,
        "a.md",
        "first paragraph here.\n\nsecond paragraph here.\n",
    );
    write(&dir, "b.md", "lonely paragraph.\n");

    let mut w = MockWriter::default();
    let stats = ingest_tree(
        &mut w,
        &StubEmbed::new(DIM),
        &dir,
        &IngestOptions {
            exts: vec!["md".into()],
            ..Default::default()
        },
        &mut |_| {},
    )
    .unwrap();

    assert_eq!(stats.files, 2, "two files");
    assert_eq!(stats.chunks, 3, "three paragraphs total");
    assert_eq!(stats.next_id, 3, "ids 0,1,2 allocated");
    assert_eq!(w.rows.len(), 3);
    assert_eq!(w.flushed, 1, "flushed once at the end");

    // Ids are sequential from 0.
    assert_eq!(
        w.rows.iter().map(|r| r.id).collect::<Vec<_>>(),
        vec![0, 1, 2]
    );
    // Every vector matches the writer dim.
    assert!(w.rows.iter().all(|r| r.dim == DIM as usize));
    // Payload is the chunk text.
    assert!(w.rows.iter().any(|r| r.payload.contains("first paragraph")));
    // Each record carries a src: provenance tag.
    assert!(
        w.rows
            .iter()
            .all(|r| r.tags.iter().any(|t| t.starts_with("src:")))
    );
}

#[test]
fn shareable_and_extra_tags_propagate() {
    let dir = tmpdir("tags");
    write(&dir, "n.md", "alpha beta gamma.\n");

    let mut w = MockWriter::default();
    ingest_tree(
        &mut w,
        &StubEmbed::new(DIM),
        &dir,
        &IngestOptions {
            exts: vec!["md".into()],
            shareable: true,
            tags: vec!["corpus".into()],
            ..Default::default()
        },
        &mut |_| {},
    )
    .unwrap();

    assert_eq!(w.rows.len(), 1);
    assert!(w.rows[0].shareable);
    assert!(w.rows[0].tags.iter().any(|t| t == "corpus"));
}

#[test]
fn ext_filter_excludes_other_files() {
    let dir = tmpdir("ext");
    write(&dir, "keep.md", "keep me.\n");
    write(&dir, "skip.rs", "fn skipped() {}\n");

    let mut w = MockWriter::default();
    let stats = ingest_tree(
        &mut w,
        &StubEmbed::new(DIM),
        &dir,
        &IngestOptions {
            exts: vec!["md".into()],
            ..Default::default()
        },
        &mut |_| {},
    )
    .unwrap();
    assert_eq!(stats.files, 1, "only the .md file");
    assert!(w.rows[0].payload.contains("keep me"));
}

#[test]
fn excludes_skip_matching_paths() {
    let dir = tmpdir("excl");
    write(&dir, "src/keep.md", "keep this.\n");
    write(&dir, "target/skip.md", "build artifact.\n");

    let mut w = MockWriter::default();
    let stats = ingest_tree(
        &mut w,
        &StubEmbed::new(DIM),
        &dir,
        &IngestOptions {
            exts: vec!["md".into()],
            ..Default::default()
        },
        &mut |_| {},
    )
    .unwrap();
    // Default excludes contain "/target/".
    assert_eq!(stats.files, 1);
    assert!(w.rows.iter().all(|r| !r.payload.contains("build artifact")));
}

#[test]
fn start_id_offsets_allocation() {
    let dir = tmpdir("startid");
    write(&dir, "x.md", "one.\n\ntwo.\n");

    let mut w = MockWriter::default();
    let stats = ingest_tree(
        &mut w,
        &StubEmbed::new(DIM),
        &dir,
        &IngestOptions {
            exts: vec!["md".into()],
            start_id: 100,
            ..Default::default()
        },
        &mut |_| {},
    )
    .unwrap();
    assert_eq!(
        w.rows.iter().map(|r| r.id).collect::<Vec<_>>(),
        vec![100, 101]
    );
    assert_eq!(stats.next_id, 102);
}

#[test]
fn dim_mismatch_is_rejected() {
    let dir = tmpdir("dim");
    write(&dir, "x.md", "hello.\n");

    let mut w = MockWriter::default(); // embedding_dim = DIM (64)
    let err = ingest_tree(
        &mut w,
        &StubEmbed::new(32), // wrong dim
        &dir,
        &IngestOptions {
            exts: vec!["md".into()],
            ..Default::default()
        },
        &mut |_| {},
    );
    assert!(err.is_err(), "dim mismatch must error before inserting");
    assert!(w.rows.is_empty());
}

#[test]
fn chunker_modes() {
    // Paragraphs split on blank lines.
    let paras = chunk("a\nb\n\nc\n", ChunkMode::Paragraph);
    assert_eq!(paras.len(), 2);
    assert_eq!(paras[0].text, "a\nb");
    assert_eq!(paras[0].line_lo, 1);
    assert_eq!(paras[0].line_hi, 2);
    assert_eq!(paras[1].text, "c");

    // Line windows slide by step.
    let text = (1..=10)
        .map(|n| n.to_string())
        .collect::<Vec<_>>()
        .join("\n");
    let wins = chunk(&text, ChunkMode::Lines { win: 4, step: 3 });
    assert!(wins.len() >= 3);
    assert_eq!(wins[0].line_lo, 1);
    assert_eq!(wins[0].line_hi, 4);
}
