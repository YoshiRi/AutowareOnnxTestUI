# Autoware ONNX Test UI

AutowareがC++で配布するMLモデルをPythonから実行し、推論結果を可視化するStreamlitアプリ。  
ROS2 / Autowareのフルセットアップなしに、ONNXモデルの動作確認が誰でもできる。

## 対応モデル

| ページ | モデル | タスク |
|--------|--------|--------|
| Image Detection | YOLOX | 2D物体検出 |
| Segmentation | Semantic Segmentation | ピクセル分類 |
| Traffic Light | Traffic Light Classifier | 信号状態分類 |
| Point Cloud Detection | CenterPoint / PointPillars | 3D物体検出 *(coming soon)* |

## セットアップ

```bash
# リポジトリのクローン
git clone https://github.com/YoshiRi/AutowareOnnxTestUI.git
cd AutowareOnnxTestUI

# 依存パッケージのインストール（uv が必要）
uv sync
```

uv のインストールは [公式ドキュメント](https://docs.astral.sh/uv/getting-started/installation/) を参照。

## モデルの配置

### 1. model_registry.yaml を確認する

`config/model_registry.yaml` にモデルファイルのパスが定義されています。

```yaml
image:
  yolox:
    onnx: /opt/autoware/mlmodels/yolox/yolox-sPlus-T4-960x960-pseudo-finetune.onnx
    label: /opt/autoware/mlmodels/yolox/label.txt
    ...
```

### 2. パスを変更する（2通り）

**方法A: yaml を直接編集する**

```yaml
image:
  yolox:
    onnx: /your/path/to/model.onnx
    label: /your/path/to/label.txt
```

**方法B: アプリ起動後にサイドバーから上書きする**

各ページの左サイドバーにパス入力欄があり、実行時に変更できます。

### 各モデルに必要なファイル

| モデル | 必須ファイル | 追加ファイル |
|--------|------------|------------|
| YOLOX | `.onnx`, `label.txt` | — |
| Semantic Seg | `.onnx` | `semseg_color_map.csv`（省略時は自動生成） |
| Traffic Light | `.onnx` | `label.txt`（省略時はGREEN/YELLOW/RED/UNKNOWN） |

> **param_files について**  
> CenterPoint / PointPillars のように ONNX 以外のパラメータファイル（アンカーCSV、voxel config YAML 等）が必要なモデルは `param_files` キーで管理します。詳細は `DESIGN.md` を参照してください。

## 起動

```bash
uv run streamlit run app.py
```

ブラウザで `http://localhost:8501` が開きます。

## 使い方

1. 左サイドバーのページリストからモデルを選択
2. サイドバーでモデルパスを確認・変更
3. 画像ファイルをアップロード
4. 推論結果と可視化が表示される

## ディレクトリ構成

```
AutowareOnnxTestUI/
├── app.py                        # エントリーポイント（ホーム画面）
├── pages/
│   ├── 1_Image_Detection.py      # YOLOX 2D検出
│   ├── 2_Segmentation.py         # セマンティックセグメンテーション
│   ├── 3_Traffic_Light.py        # 信号分類
│   └── 4_PointCloud_Detection.py # 点群検出（coming soon）
├── models/
│   ├── base.py                   # ModelRunner 抽象基底クラス
│   └── image/
│       ├── yolox.py
│       ├── semantic_seg.py
│       └── traffic_light.py
├── utils/
│   └── visualization.py          # 共通描画ユーティリティ
└── config/
    ├── model_registry.yaml       # モデルパス・パラメータ設定
    └── semseg_color_map.csv      # セグメンテーション用カラーマップ
```

## 新しいモデルを追加する

`models/base.py` の `ModelRunner` を継承して5つのメソッドを実装するだけです。

```python
from models.base import ModelRunner

class MyModelRunner(ModelRunner):
    def load(self, config: dict) -> None:
        # config["onnx"], config["label"],
        # config["param_files"]["my_anchor_csv"] 等を読み込む
        ...

    def preprocess(self, image_bgr: np.ndarray) -> dict:
        # {"tensor": ..., "orig_w": ..., ...} を返す
        ...

    def infer(self, preprocess_result: dict) -> dict:
        # ONNX推論、"outputs" キーを追加して返す
        ...

    def postprocess(self, infer_result: dict, **params) -> list:
        # 結果のリストを返す
        ...

    def visualize(self, image_rgb: np.ndarray, results: list, **kwargs) -> np.ndarray:
        # アノテーション済み画像を返す
        ...
```

その後 `config/model_registry.yaml` にエントリを追加し、`pages/` に Streamlit ページを作成します。

## 設計ドキュメント

詳細なアーキテクチャ・前後処理仕様・実装フェーズは [`DESIGN.md`](DESIGN.md) を参照してください。
