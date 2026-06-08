//! Caller-side embeddings.
//!
//! skeg stores vectors, never text — so turning text into a vector is the
//! caller's job. This module defines the [`Embed`] capability and an
//! [`OllamaEmbed`] backend that talks to a local Ollama-compatible
//! `/api/embed` endpoint (the same GPU path the rest of skeg uses). The
//! trait keeps the backend swappable: a different provider only has to
//! implement [`Embed`].

use anyhow::{Context, Result};

/// mxbai-family models want this prefix on the *query* side only; it is
/// a no-op for models that were not trained with it.
const QUERY_PREFIX: &str = "Represent this sentence for searching relevant passages: ";

/// Turn text into vectors. Passage vs query is distinguished because some
/// retrieval models embed the two asymmetrically.
pub trait Embed {
    /// Embedding dimension produced by this backend.
    fn dim(&self) -> u32;
    /// Embed a stored passage.
    fn passage(&self, text: &str) -> Result<Vec<f32>>;
    /// Embed a search query. Defaults to [`Self::passage`].
    fn query(&self, text: &str) -> Result<Vec<f32>> {
        self.passage(text)
    }
}

/// An [`Embed`] backed by an Ollama-compatible HTTP endpoint.
pub struct OllamaEmbed {
    url: String,
    model: String,
    dim: u32,
}

impl OllamaEmbed {
    /// Connect and probe the model for its embedding dimension.
    pub fn connect(url: impl Into<String>, model: impl Into<String>) -> Result<Self> {
        let url = url.into();
        let model = model.into();
        let dim = call(&url, &model, "dimension probe")?.len() as u32;
        if dim == 0 {
            anyhow::bail!("embedder '{model}' returned a zero-length vector");
        }
        Ok(Self { url, model, dim })
    }
}

impl Embed for OllamaEmbed {
    fn dim(&self) -> u32 {
        self.dim
    }

    fn passage(&self, text: &str) -> Result<Vec<f32>> {
        call(&self.url, &self.model, text)
    }

    fn query(&self, text: &str) -> Result<Vec<f32>> {
        call(&self.url, &self.model, &format!("{QUERY_PREFIX}{text}"))
    }
}

/// One `/api/embed` round-trip.
fn call(url: &str, model: &str, input: &str) -> Result<Vec<f32>> {
    let resp = ureq::post(&format!("{url}/api/embed"))
        .timeout(std::time::Duration::from_secs(60))
        .send_json(serde_json::json!({ "model": model, "input": [input] }))
        .map_err(|e| {
            anyhow::anyhow!(
                "embedder unreachable at {url} ({e}). Is Ollama running and is '{model}' pulled?"
            )
        })?;
    let body: serde_json::Value = resp.into_json().context("decode embed response")?;
    let arr = body["embeddings"][0]
        .as_array()
        .context("embed response missing embeddings[0]")?;
    Ok(arr
        .iter()
        .filter_map(|x| x.as_f64().map(|f| f as f32))
        .collect())
}
