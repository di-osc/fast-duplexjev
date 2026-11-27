use std::{
    collections::BTreeSet,
    fs::File,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, ensure};
use modelhub::{Backend, DownloadOptions, DownloadedRepo, RepoKind};
use serde::Deserialize;

pub const MODEL_ID: &str = "adventists-ai/DuplexJev-4B-Para";
const MODEL_REVISION: &str = "a0aea2e705b2503695d61629f543d0fc9d30f8a7";
const INDEX_FILE: &str = "model.safetensors.index.json";

#[derive(Deserialize)]
struct Index {
    weight_map: std::collections::BTreeMap<String, String>,
}

pub async fn resolve(
    model: &str,
    revision: Option<&str>,
    cache_dir: Option<&Path>,
) -> Result<PathBuf> {
    let path = PathBuf::from(model);
    if path.is_dir() {
        ensure!(
            revision.is_none(),
            "--revision applies to Hugging Face model IDs"
        );
        return Ok(path);
    }
    ensure!(
        model.split_once('/').is_some_and(|(owner, name)| {
            !owner.is_empty()
                && !name.is_empty()
                && !name.contains('/')
                && !matches!(owner, "." | "..")
                && !matches!(name, "." | "..")
        }),
        "--model must be an existing directory or a Hugging Face owner/model ID"
    );

    let revision = revision.unwrap_or(if model == MODEL_ID {
        MODEL_REVISION
    } else {
        "main"
    });
    let mut options = DownloadOptions::new(model);
    if let Some(cache_dir) = cache_dir {
        options.cache_root = cache_dir.to_owned();
    }
    options.kind = Some(RepoKind::Model);
    options.backend = Some(Backend::HuggingFace);
    options.revision = Some(revision.to_owned());
    options.progress = true;

    let download = download_file(&options, INDEX_FILE).await?;
    let index = download
        .file
        .context("modelhub did not return the index file")?;
    let hub = download
        .huggingface_root
        .context("modelhub did not return its Hugging Face cache root")?;
    let checkpoint = index
        .parent()
        .context("modelhub index has no parent directory")?;
    let weights: Index = serde_json::from_reader(File::open(&index)?)
        .with_context(|| format!("invalid model index {}", index.display()))?;
    ensure!(
        !weights.weight_map.is_empty(),
        "checkpoint index contains no weights"
    );
    let mut files = BTreeSet::from(["config.json", "tokenizer.json"]);
    for name in weights.weight_map.values() {
        ensure!(
            Path::new(name)
                .file_name()
                .is_some_and(|basename| basename == name.as_str()),
            "invalid checkpoint shard name {name:?}"
        );
        files.insert(name);
    }

    // A preexisting repository download may resolve a branch to a commit snapshot.
    // Fetch any missing files using the same snapshot as the index.
    let resolved = checkpoint
        .strip_prefix(hub.join("snapshots"))?
        .to_str()
        .context("invalid modelhub snapshot directory")?;
    options.revision = Some(resolved.to_owned());
    let mut downloads = tokio::task::JoinSet::new();
    for file in files {
        let options = options.clone();
        let file = file.to_owned();
        downloads.spawn(async move { download_file(&options, &file).await });
        if downloads.len() == 4 {
            check_download(&mut downloads, checkpoint).await?;
        }
    }
    while !downloads.is_empty() {
        check_download(&mut downloads, checkpoint).await?;
    }
    Ok(checkpoint.to_owned())
}

async fn check_download(
    downloads: &mut tokio::task::JoinSet<Result<DownloadedRepo>>,
    checkpoint: &Path,
) -> Result<()> {
    let download = downloads
        .join_next()
        .await
        .context("no pending checkpoint download")???;
    let file = download
        .file
        .context("modelhub did not return the requested file")?;
    ensure!(
        file.parent() == Some(checkpoint),
        "checkpoint files resolved to different modelhub snapshots"
    );
    Ok(())
}

async fn download_file(options: &DownloadOptions, file: &str) -> Result<DownloadedRepo> {
    let mut options = options.clone();
    options.file = Some(file.to_owned());
    modelhub::download(&options)
        .await
        .with_context(|| format!("download {file} from {}", options.repo_id))
}
