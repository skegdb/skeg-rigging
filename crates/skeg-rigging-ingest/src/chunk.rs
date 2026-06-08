//! Splitting a file into embeddable chunks.
//!
//! Embedding a whole file at once blurs everything together; embedding
//! single lines is too granular. The two modes here cover the common
//! cases: paragraph chunks for prose/markdown, and a sliding line window
//! for code.

/// One chunk of a file, with the line range it came from (1-based,
/// inclusive) for provenance.
#[derive(Debug, Clone)]
pub struct Chunk {
    /// The chunk text.
    pub text: String,
    /// First source line (1-based).
    pub line_lo: usize,
    /// Last source line (1-based).
    pub line_hi: usize,
}

/// How to split a file.
#[derive(Debug, Clone, Copy)]
pub enum ChunkMode {
    /// Split on blank lines (prose, markdown).
    Paragraph,
    /// Sliding window of `win` lines advancing by `step` (code).
    Lines { win: usize, step: usize },
}

impl ChunkMode {
    /// A sensible default for a file extension.
    pub fn for_ext(ext: &str) -> ChunkMode {
        match ext {
            "md" | "mdx" | "txt" | "rst" => ChunkMode::Paragraph,
            _ => ChunkMode::Lines { win: 40, step: 30 },
        }
    }
}

/// Count whitespace-delimited words.
pub fn word_count(s: &str) -> usize {
    s.split_whitespace().count()
}

/// Split `text` into chunks under `mode`. Empty/whitespace-only chunks
/// are dropped.
pub fn chunk(text: &str, mode: ChunkMode) -> Vec<Chunk> {
    match mode {
        ChunkMode::Paragraph => paragraphs(text),
        ChunkMode::Lines { win, step } => line_windows(text, win.max(1), step.max(1)),
    }
}

fn paragraphs(text: &str) -> Vec<Chunk> {
    let mut out = Vec::new();
    let mut buf: Vec<&str> = Vec::new();
    let mut start = 1usize; // line where the current paragraph began
    for (i, line) in text.lines().enumerate() {
        let lineno = i + 1;
        if line.trim().is_empty() {
            if !buf.is_empty() {
                push_para(&mut out, &buf, start, lineno - 1);
                buf.clear();
            }
            start = lineno + 1;
        } else {
            if buf.is_empty() {
                start = lineno;
            }
            buf.push(line);
        }
    }
    if !buf.is_empty() {
        let end = text.lines().count();
        push_para(&mut out, &buf, start, end);
    }
    out
}

fn push_para(out: &mut Vec<Chunk>, lines: &[&str], lo: usize, hi: usize) {
    let text = lines.join("\n");
    if text.trim().is_empty() {
        return;
    }
    out.push(Chunk {
        text,
        line_lo: lo,
        line_hi: hi,
    });
}

fn line_windows(text: &str, win: usize, step: usize) -> Vec<Chunk> {
    let lines: Vec<&str> = text.lines().collect();
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < lines.len() {
        let end = (i + win).min(lines.len());
        let slice = &lines[i..end];
        let body = slice.join("\n");
        if !body.trim().is_empty() {
            out.push(Chunk {
                text: body,
                line_lo: i + 1,
                line_hi: end,
            });
        }
        if end == lines.len() {
            break;
        }
        i += step;
    }
    out
}
