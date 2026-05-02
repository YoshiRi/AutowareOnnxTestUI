import streamlit as st

st.set_page_config(page_title="Autoware ONNX Test UI", layout="wide")

st.title("Autoware ONNX Test UI")
st.markdown("""
PythonからAutowareのMLモデルをONNXで実行し、推論結果を可視化するツール。
ROS2 / Autowareのフルセットアップ不要で動作確認ができる。

---

### 利用可能なページ

| ページ | モデル | 入力 |
|--------|--------|------|
| **Image Detection** | YOLOX（2D物体検出） | 画像ファイル |
| **Segmentation** | Semantic Segmentation | 画像ファイル |
| **Traffic Light** | Traffic Light Classifier | クロップ画像ファイル |
| **Point Cloud Detection** | CenterPoint / PointPillars *(coming soon)* | `.pcd` / `.bin` |

左のサイドバーからページを選択してください。

---

### 必要なファイル

各モデルには以下が必要です。パスは `config/model_registry.yaml` で管理しています。

- `.onnx` — モデル重み
- `label.txt` — クラス名（1行1クラス）
- 追加パラメータファイル（アンカーCSV、カラーマップCSV、config YAML等）→ `param_files` キーで指定

モデルパスは各ページのサイドバーから直接上書きできます。
""")
