# scooter

`scooter` is a local-first AI agent designed for the Rockchip RK3588 platform (tested on the 32GB FriendlyElec CM3588), leveraging hardware NPU acceleration via `librkllmrt.so`. Built in Go, it features an interactive terminal user interface (TUI) powered by Bubble Tea and provides modular toolsets for software development and smart home IoT automation.

## Features

- **RK3588 NPU Acceleration:** Direct execution of local `.rkllm` models using `ebitengine/purego` for high-performance edge inference.
- **Interactive TUI:** Built with Charm Bracelet (`bubbletea`, `bubbles`, `lipgloss`) for a rich terminal experience.
- **Modular Toolsets:**
  - **Development (`dev`):** Workspace tools for reading/writing files, searching (`grep`), running shell commands, and managing codebases.
  - **Web (`web`):** URL fetching and web content retrieval tool (`web_fetch`).
  - **All (`all`):** Combines development and web toolsets.
- **Context Management & Auto-Compaction:** Intelligent context window handling with automatic history summarization and JSONL session caching per workspace.
- **Hybrid / Fallback Support:** Optional integration with Google GenAI SDK (`google.golang.org/genai`).

## Requirements

- **Hardware:** Rockchip RK3588 SoC (e.g., FriendlyElec CM3588, Orange Pi 5, Rock 5) with NPU support.
- **Software:**
  - Go 1.26+
  - RKLLM runtime library (`librkllmrt.so`)
  - RKLLM model files (e.g., Qwen3-4b or compatible `.rkllm` models)

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
- `rkllm_models_dir`
- `rkllm_lib`
- `default_toolset`
- `google_api_key`
- `google_genai_model`

Environment variables:
- `SCOOTER_CONFIG_FILE`: Path to custom configuration file.
- `SCOOTER_CONFIG_DIR`: Path to configuration directory.
- `SCOOTER_CACHE_FILE`: Path to session cache file.
- `SCOOTER_CACHE_DIR`: Path to session cache directory.

## License

See Codeberg repository for license and contributions.
