#!/usr/bin/env bash
# setup_mac.sh — Mac 1からセットアップ & 動作確認スクリプト
# 使い方:
#   bash scripts/setup_mac.sh
#   bash scripts/setup_mac.sh --model-root ~/autoware_models   # モデルがある場合

set -euo pipefail

# ---------------------------------------------------------------------------
# カラー出力
# ---------------------------------------------------------------------------
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

info()    { echo -e "${CYAN}▶ $*${RESET}"; }
success() { echo -e "${GREEN}✅ $*${RESET}"; }
warn()    { echo -e "${YELLOW}⚠️  $*${RESET}"; }
fail()    { echo -e "${RED}❌ $*${RESET}"; exit 1; }
header()  { echo -e "\n${BOLD}━━━ $* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"; }

# ---------------------------------------------------------------------------
# 引数処理
# ---------------------------------------------------------------------------
MODEL_ROOT=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-root) MODEL_ROOT="$2"; shift 2 ;;
        -h|--help)
            echo "使い方: bash scripts/setup_mac.sh [--model-root <path>]"
            echo "  --model-root  モデルディレクトリ（省略時はパイプラインテストをスキップ）"
            exit 0 ;;
        *) fail "不明なオプション: $1" ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

header "Autoware ONNX Test UI — Mac セットアップ"
echo "リポジトリ: $REPO_ROOT"

# ---------------------------------------------------------------------------
# 1. macOS 確認
# ---------------------------------------------------------------------------
header "1. 環境チェック"

if [[ "$(uname)" != "Darwin" ]]; then
    fail "このスクリプトは macOS 専用です。Linux の場合は uv sync を直接実行してください。"
fi

ARCH="$(uname -m)"
if [[ "$ARCH" == "arm64" ]]; then
    success "Apple Silicon (arm64) を検出"
else
    success "Intel Mac (x86_64) を検出"
fi

OS_VER="$(sw_vers -productVersion)"
success "macOS $OS_VER"

# ---------------------------------------------------------------------------
# 2. uv インストール確認
# ---------------------------------------------------------------------------
header "2. uv パッケージマネージャー"

if command -v uv &>/dev/null; then
    UV_VER="$(uv --version)"
    success "uv は既にインストール済み: $UV_VER"
else
    info "uv をインストールします..."
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # シェルに PATH を反映
    export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"

    if ! command -v uv &>/dev/null; then
        fail "uv のインストールに失敗しました。https://docs.astral.sh/uv/getting-started/installation/ を参照してください。"
    fi
    success "uv をインストールしました: $(uv --version)"
fi

# ---------------------------------------------------------------------------
# 3. Python 依存関係インストール
# ---------------------------------------------------------------------------
header "3. 依存関係インストール (uv sync)"

cd "$REPO_ROOT"
info "uv sync を実行中..."
uv sync

success "依存関係インストール完了"

# インストールされた主要パッケージを確認
echo ""
echo "  インストール済みの主要パッケージ:"
uv run python -c "
import importlib, sys
pkgs = [
    ('streamlit',       'streamlit'),
    ('onnxruntime',     'onnxruntime'),
    ('cv2',             'opencv-python'),
    ('numpy',           'numpy'),
    ('PIL',             'Pillow'),
    ('mss',             'mss'),
    ('streamlit_cropper','streamlit-cropper'),
]
for mod, name in pkgs:
    try:
        m = importlib.import_module(mod)
        ver = getattr(m, '__version__', '?')
        print(f'    ✅  {name:<28} {ver}')
    except ImportError:
        print(f'    ❌  {name:<28} (インポート失敗)', file=sys.stderr)
"

# ---------------------------------------------------------------------------
# 4. サンプル画像のダウンロード
# ---------------------------------------------------------------------------
header "4. サンプル画像のダウンロード"

info "公開サンプル画像をダウンロード中..."
if uv run python scripts/download_models.py --samples; then
    success "サンプル画像の準備完了 → samples/"
else
    warn "一部のサンプル画像がダウンロードできませんでした（ネットワーク確認してください）"
fi

# ---------------------------------------------------------------------------
# 5. パイプラインスモークテスト
# ---------------------------------------------------------------------------
header "5. パイプラインテスト"

if [[ -n "$MODEL_ROOT" ]]; then
    info "モデルルート: $MODEL_ROOT"
    info "スモークテストを実行中..."
    echo ""
    if uv run python scripts/test_pipeline.py --model-root "$MODEL_ROOT"; then
        success "スモークテスト全パス"
    else
        warn "一部モデルでエラーが発生しました（詳細は上記ログを確認）"
    fi
else
    warn "モデルがないためパイプラインテストはスキップします"
    echo "  モデルをダウンロードしてから再実行:"
    echo ""
    echo "    # モデルダウンロード（数GB、時間がかかります）"
    echo "    uv run python scripts/download_models.py --model-root ~/autoware_models"
    echo ""
    echo "    # その後スモークテスト"
    echo "    bash scripts/setup_mac.sh --model-root ~/autoware_models"
fi

# ---------------------------------------------------------------------------
# 6. 画面キャプチャ権限の案内（Mac 固有）
# ---------------------------------------------------------------------------
header "6. Mac 権限の設定（Screen Capture 使用時）"

cat <<'NOTICE'
  スクリーンキャプチャ機能を使う場合、以下の権限が必要です:

  [システム設定] → [プライバシーとセキュリティ] → [画面収録]
    → Terminal（または使用しているターミナルアプリ）を許可

  ※ アプリ起動後に初回使用時にダイアログが表示される場合もあります。
NOTICE

# ---------------------------------------------------------------------------
# 完了 & 起動案内
# ---------------------------------------------------------------------------
header "セットアップ完了"

cat <<DONE

  ${GREEN}アプリを起動するには:${RESET}

    cd $REPO_ROOT
    uv run streamlit run app.py

  ${CYAN}便利なオプション:${RESET}

    # モデルをダウンロードする
    uv run python scripts/download_models.py --model-root ~/autoware_models

    # 利用可能なモデル一覧
    uv run python scripts/download_models.py --list

    # パイプラインのみテスト（UIなし）
    uv run python scripts/test_pipeline.py --model-root ~/autoware_models

    # サンプル画像のみ再ダウンロード
    uv run python scripts/download_models.py --samples

  ${YELLOW}起動後はブラウザで http://localhost:8501 が開きます。${RESET}

DONE
