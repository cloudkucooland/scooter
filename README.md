# scooter

`scooter` is a local-first AI agent designed for the Rockchip RK3588 platform (tested on the 32GB FriendlyElec CM3588), leveraging hardware NPU acceleration via `librkllmrt.so` (LLMs) and `librknnrt.so` (image generation / diffusion). Built in Go, it features an interactive terminal user interface (TUI) powered by Bubble Tea and provides modular toolsets for software development, local image generation, and workflow automation.

## Features

- **RK3588 NPU Acceleration:** Direct execution of local `.rkllm` language models and `.rknn` diffusion/vision models (such as LCM-SSD-1B) using `ebitengine/purego` with thread-safe NPU core scheduling and mutex synchronization.
- **Local Image Generation (`image`):** Generate images locally on the NPU from text prompts using CLIP tokenization and UNet/VAE pipelines via the `generate_image` tool, `/imagine` TUI command, or `--imagine` (`--image`) CLI flag. All entry points support output path, denoising steps, seed, and guidance; CLI and TUI also accept a negative prompt.
- **Interactive TUI:** Built with Charm Bracelet (`bubbletea`, `bubbles`, `lipgloss`) for a rich terminal experience with slash commands (`/toolset`, `/imagine`, etc.).
- **Modular Toolsets:**
  - **Development (`dev`):** Workspace tools for reading/writing files, searching (`grep`), running shell commands, and managing codebases.
  - **Web (`web`):** URL fetching and web content retrieval tool (`web_fetch`).
  - **Image (`image` / `imagine`):** Local NPU image generation.
  - **Research (`research`):** Research-oriented toolsets.
  - **All (`all`):** Combines development, web, and image toolsets.
- **Context Management & Auto-Compaction:** Intelligent context window handling with automatic history summarization and JSONL session caching per workspace.
- **Hybrid / Fallback Support:** Optional integration with Google GenAI SDK (`google.golang.org/genai`).

## Requirements

- **Hardware:** Rockchip RK3588 SoC (e.g., FriendlyElec CM3588, Orange Pi 5, Rock 5) with NPU support.
- **Software:**
  - Go 1.26+
  - RKLLM runtime library (`librkllmrt.so`)
  - RKNN runtime library (`librknnrt.so`)
  - RKLLM model files & RKNN diffusion model directory (e.g., LCM-SSD-1B with CLIP tokenizer and UNet/VAE `.rknn` files)

## Installation & Building

```bash
git clone https://codeberg.org/indievisible/scooter.git
cd scooter
go build -o scooter .
```

## Configuration

Scooter looks for configuration in your user config directory (`$XDG_CONFIG_HOME/scooter/config.json` or custom path via `SCOOTER_CONFIG_FILE`).

Supported configuration options:
- `default_context_tokens`
- `max_context_tokens`
- `default_model`
- `default_rknn_model`
- `rkllm_models_dir`
- `rkllm_lib`
- `rknn_models_dir`
- `rknn_lib`
- `default_toolset`
- `google_api_key`
- `google_genai_model`

Environment variables:
- `SCOOTER_CONFIG_FILE`: Path to custom configuration file.
- `SCOOTER_CONFIG_DIR`: Path to configuration directory.
- `SCOOTER_CACHE_FILE`: Path to session cache file.
- `SCOOTER_CACHE_DIR`: Path to session cache directory.
- `SCOOTER_RKNN_MODEL` / `RKNN_MODEL`: Active RKNN model name or directory path.
- `SCOOTER_RKNN_MODELS_DIR` / `RKNN_MODEL_DIR`: Path to RKNN models directory.
- `RKNN_LIB`: Path to `librknnrt.so`.

## License

See Codeberg repository for license and contributions.
