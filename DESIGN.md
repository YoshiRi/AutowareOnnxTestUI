# AutowareOnnxTestUI 設計ドキュメント

## 概要

AutowareがC++で配布するMLモデルをPythonから実行し、誰でも簡単にモデルの挙動を確認できるStreamlit UIを提供する。

---

## 対象モデル

### 画像処理系

| モデル | タスク | 入力 | 出力 | 優先度 |
|--------|--------|------|------|--------|
| YOLOX | 2D物体検出 | RGB画像 (960x960) | BBox + クラス + スコア | Phase 1 |
| Semantic Segmentation | ピクセル分類 | RGB画像 | ピクセルごとクラスマップ | Phase 1 |
| Traffic Light Classifier | 信号状態分類 | クロップ画像 | クラス確率 | Phase 1 |
| Traffic Light Fine Detector | 信号位置検出 | RGB画像 | BBox + クラス | Phase 1 |

### 点群処理系

| モデル | タスク | 入力 | 出力 | 優先度 |
|--------|--------|------|------|--------|
| CenterPoint | 3D物体検出 | voxel化点群 (複数tensor) | 3D BBox + クラス + スコア | Phase 2 |
| PointPillars | 3D物体検出 | pillar形式点群 | 3D BBox + クラス + スコア | Phase 2 |

---

## ディレクトリ構造

```
AutowareOnnxTestUI/
├── app.py                            # エントリーポイント（ランチャー）
├── DESIGN.md                         # 本ドキュメント
│
├── models/
│   ├── base.py                       # ModelRunner 抽象基底クラス
│   ├── image/
│   │   ├── __init__.py
│   │   ├── yolox.py                  # 2D物体検出
│   │   ├── semantic_seg.py           # セマンティックセグメンテーション
│   │   └── traffic_light.py         # 信号検出・分類
│   └── pointcloud/
│       ├── __init__.py
│       ├── centerpoint.py            # CenterPoint 3D検出
│       └── pointpillars.py           # PointPillars 3D検出
│
├── pages/
│   ├── 1_Image_Detection.py          # 2D検出UI (YOLOX / Traffic Light)
│   ├── 2_Segmentation.py             # セグメンテーションUI
│   └── 3_PointCloud_Detection.py     # 3D検出UI (LiDAR)
│
├── utils/
│   ├── visualization.py              # 2D描画共通ユーティリティ
│   └── pointcloud_viz.py             # 点群可視化 (Plotly)
│
└── config/
    └── model_registry.yaml           # モデルパス・設定の外部化
```

---

## コアインターフェース

### `ModelRunner` 抽象基底クラス (`models/base.py`)

新モデルを追加する際はこのクラスを継承し、5つのメソッドを実装する。

```python
class ModelRunner(ABC):
    @abstractmethod
    def load(self, onnx_path: str) -> None:
        """ONNXモデルをロードする"""

    @abstractmethod
    def preprocess(self, input_data) -> np.ndarray:
        """入力データをモデル入力形式に変換する"""

    @abstractmethod
    def run(self, preprocessed) -> Any:
        """ONNXランタイムで推論を実行する"""

    @abstractmethod
    def postprocess(self, outputs, **kwargs) -> Any:
        """推論結果を人間が読める形式に変換する"""

    @abstractmethod
    def visualize(self, original_input, results) -> Any:
        """結果を可視化した画像やFigureを返す"""
```

### `model_registry.yaml` の構造

```yaml
image:
  yolox:
    onnx: /opt/autoware/mlmodels/yolox/yolox-sPlus-T4-960x960-pseudo-finetune.onnx
    label: /opt/autoware/mlmodels/yolox/label.txt
    input_size: 960
  semantic_seg:
    onnx: /opt/autoware/mlmodels/semantic_seg/model.onnx
    color_map: config/semseg_color_map.csv
  traffic_light:
    classifier_onnx: /opt/autoware/mlmodels/traffic_light/classifier.onnx
    detector_onnx: /opt/autoware/mlmodels/traffic_light/detector.onnx
    label: /opt/autoware/mlmodels/traffic_light/label.txt

pointcloud:
  centerpoint:
    onnx: /opt/autoware/mlmodels/centerpoint/model.onnx
    label: /opt/autoware/mlmodels/centerpoint/label.txt
  pointpillars:
    onnx: /opt/autoware/mlmodels/pointpillars/model.onnx
    label: /opt/autoware/mlmodels/pointpillars/label.txt
```

---

## データ入力UI（将来追加予定）

現状はファイルアップロードのみ対応。以下を将来的に追加する。

- ROSBag からの画像・点群フレーム抽出
- ROS2 トピックのリアルタイム購読
- データセットディレクトリの一括スキャン
- サンプルデータのバンドル（デモ用）

---

## 画像処理系の前後処理仕様

### YOLOX

**前処理:**
1. アスペクト比を保ったまま `input_size x input_size` にリサイズ
2. グレーパディング (114, 114, 114) で余白埋め
3. BGR → RGB 変換
4. `(1, 3, H, W)` のfloat32テンソルに変換（正規化なし）

**後処理:**
1. grid座標とstrideを使いBBoxをデコード
2. objectness × class scoreでfinal scoreを計算
3. confidence閾値でフィルタ
4. NMS適用
5. パディング・スケールを逆変換して元画像座標に戻す

### Semantic Segmentation

**前処理:**
1. モデル入力サイズにリサイズ
2. ImageNet正規化 (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
3. `(1, 3, H, W)` のfloat32テンソルに変換

**後処理:**
1. argmaxでクラスマップ取得
2. `semseg_color_map.csv` を使いRGBカラーマップに変換
3. 元画像と半透明でオーバーレイ表示

### Traffic Light Classifier

**前処理:**
1. 固定サイズ (例: 224x224) にリサイズ
2. ImageNet正規化
3. `(1, 3, H, W)` のfloat32テンソルに変換

**後処理:**
1. softmaxでクラス確率を計算
2. 信号状態（red/yellow/green/unknown）を表示

---

## 点群処理系の前後処理仕様（Phase 2）

### CenterPoint / PointPillars 共通フロー

```
.pcd / .bin ファイルアップロード
    ↓
点群パース (x, y, z, intensity)
    ↓
Range filter (ROI内の点のみ残す)
    ↓
Voxel化 / Pillar化 (モデル依存)
    ↓
ONNX推論 (複数入力テンソル)
    ↓
3D BBox デコード (center, size, yaw)
    ↓
Plotly 3D scatterで点群 + BBoxを可視化
```

---

## 技術スタック

| 用途 | ライブラリ | 理由 |
|------|-----------|------|
| UI | Streamlit | 既存踏襲、プロトタイピングに最適 |
| ONNX推論 | onnxruntime | Autoware準拠 |
| 画像処理 | OpenCV, NumPy | 既存踏襲 |
| 点群可視化 | Plotly | Open3D不要でインタラクティブ3D表示可能 |
| 設定管理 | PyYAML | モデルパスのハードコード排除 |

---

## 実装フェーズ

### Phase 1 — 画像処理系（優先実施）

- [ ] ディレクトリ構造の作成
- [ ] `ModelRunner` ABC実装
- [ ] `model_registry.yaml` 実装
- [ ] YOLOXを新構造にリファクタ（`sample.py`/`visualizer.py` を統合）
- [ ] Semantic Segmentation対応（`semseg_color_map.csv` を活用）
- [ ] Traffic Light Classifier / Detector対応
- [ ] 各モデルのStreamlitページ実装

### Phase 2 — 点群処理系

- [ ] `.pcd` / `.bin` パーサー実装
- [ ] CenterPoint前処理（voxelization）実装
- [ ] PointPillars前処理（pillarization）実装
- [ ] Plotlyによる3D BBox可視化
- [ ] 点群モデルのStreamlitページ実装

### Phase 3 — データ入力拡張

- [ ] ROSBagからのフレーム抽出
- [ ] ROS2トピックリアルタイム購読
- [ ] サンプルデータバンドル
