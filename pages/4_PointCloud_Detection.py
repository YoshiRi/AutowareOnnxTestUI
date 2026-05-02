import streamlit as st

st.set_page_config(page_title="Point Cloud Detection", layout="wide")
st.title("Point Cloud Detection — Coming Soon")

st.info(
    "Phase 2: LiDAR 点群モデル (CenterPoint, PointPillars) は今後実装予定です。"
)

st.markdown("""
### 予定モデル
- **CenterPoint** — voxelベースの3D物体検出
- **PointPillars** — pillarベースの3D物体検出

### 予定入力フォーマット
- `.pcd` — PCL形式
- `.bin` — Velodyne binary形式（N×4 float32: x, y, z, intensity）

### 予定可視化
- Plotly 3Dスキャッタープロットで点群 + 3Dバウンディングボックスをインタラクティブ表示

### 設計上の注意点
これらのモデルは ONNX ファイル単体では動作せず、以下の追加ファイルが必要です:
- `voxel_config.yaml` / `anchors.csv` など（`param_files` で管理）
- voxelization / pillarization の前処理実装

詳細は `DESIGN.md` および `config/model_registry.yaml` を参照してください。
""")
