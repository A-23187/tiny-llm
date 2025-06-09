# [Setting Up the Environment](https://skyzh.github.io/tiny-llm/setup.html)

1. 运行 `pdm install -v` 安装依赖时报错：

    ```
    ✖ Install mlx 0.25.2 failed
    pdm.termui: Error occurs adding mlx:
    ...
    pdm.exceptions.CandidateNotFound: No candidate is found for `mlx` that matches the environment or hashes

    ✖ Install torch 2.7.0 failed
    pdm.termui: Error occurs adding torch:
    ...
    pdm.exceptions.CandidateNotFound: No candidate is found for `torch` that matches the environment or hashes
    ```

    报错信息并不清晰。尝试使用 uv 作为 pdm 的 installer

    ```shell
    pdm config use_uv true
    ```

    报错信息会更清楚：

    ```
    error: Distribution `mlx==0.25.2 @ registry+https://pypi.org/simple` can't be installed because it doesn't have a source distribution or wheel for the current platform

    hint: You're on macOS (`macosx_13_0_x86_64`), but `mlx` (v0.25.2) only has wheels for the following platforms: `manylinux_2_31_x86_64`, `macosx_13_0_arm64`, `macosx_14_0_arm64`, `macosx_15_0_arm64`
    ```

    但我的 macOS 就是苹果芯片。根据 [MLX Installation Troubleshooting](https://ml-explore.github.io/mlx/build/html/install.html#troubleshooting)，发现 pdm 自动下载的 python 是 x86_64 的，因为我的 brew 是 x86_64 的，而 pdm 是通过 brew 安装的，所以 pdm 是 x86_64 的，在 pdm 根据项目的 requires-python 下载符合的 python 时就会下载 x86_64 的！

    ```bash
    $ file ~/Library/Application\ Support/pdm/python/cpython\@3.12.10/bin/python
    /Users/a23187/Library/Application Support/pdm/python/cpython@3.12.10/bin/python: Mach-O 64-bit executable x86_64

    $ file `which pdm`
    /usr/local/bin/pdm: a /usr/local/Cellar/pdm/2.24.2/libexec/bin/python script text executable, ASCII text
    $ file /usr/local/Cellar/pdm/2.24.2/libexec/bin/python
    /usr/local/Cellar/pdm/2.24.2/libexec/bin/python: Mach-O 64-bit executable x86_64

    $ brew config
    HOMEBREW_VERSION: 4.5.4
    ORIGIN: https://github.com/Homebrew/brew
    HEAD: 54c8b127ea2263fbbaf1354e3d8d86025e387ea6
    Last commit: 9 days ago
    Branch: stable
    Core tap JSON: 09 Jun 06:44 UTC
    Core cask tap JSON: 09 Jun 06:45 UTC
    HOMEBREW_PREFIX: /usr/local
    HOMEBREW_CASK_OPTS: []
    HOMEBREW_EDITOR: vim
    HOMEBREW_MAKE_JOBS: 10
    Homebrew Ruby: 3.4.4 => /usr/local/Homebrew/Library/Homebrew/vendor/portable-ruby/3.4.4/bin/ruby
    CPU: deca-core 64-bit westmere
    Clang: 14.0.3 build 1403
    Git: 2.39.2 => /Library/Developer/CommandLineTools/usr/bin/git
    Curl: 8.1.2 => /usr/bin/curl
    macOS: 13.5.1-x86_64
    CLT: 14.3.1.0.1.1683849156
    Xcode: N/A
    Rosetta 2: true
    ```

    // macOS 通过 rosetta 翻译 x86_64 指令到对应的 arm64/aarch64 指令

    解决方法：

    ```bash
    # 通过 uv 安装 arm64/aarch64 的 python
    # uv 提供了各平台和架构的 standalone python，可运行 uv python list --all-platforms --all-arches --all-versions 查看
    $ uv python install cpython-3.12.9-macos-aarch64-none
    $ file ~/.local/share/uv/python/cpython-3.12.9-macos-aarch64-none/bin/python
    Users/a23187/.local/share/uv/python/cpython-3.12.9-macos-aarch64-none/bin/python: Mach-O 64-bit executable arm64

    # 然后在 pdm 里链接
    $ pdm py link ~/.local/share/uv/python/cpython-3.12.9-macos-aarch64-none/bin/python
    Successfully linked cpython@3.12 to /Users/a23187/.local/share/uv/python/cpython-3.12.9-macos-aarch64-none

    # 注意先卸载 pdm 自动下载的 x86_64 python
    $ pdm py remove 3.12.10

    $ pdm install -v
    ```

2. download model from modelscope instead of huggingface

    下载模型 Qwen/Qwen2-7B-Instruct-MLX

    ```shell
    pip3 install modelscope
    modelscope download --model Qwen/Qwen2-7B-Instruct-MLX
    # or
    python3 -c 'from modelscope import snapshot_download; snapshot_download("Qwen/Qwen2-7B-Instruct-MLX")'
    ```

    配置环境变量使 mlxlm 使用从 modelscope 下载的模型

    ```shell
    $ export MLXLM_USE_MODELSCOPE=true
    $ pdm run main --solution ref --loader week1
    ```
