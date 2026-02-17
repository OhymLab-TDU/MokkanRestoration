# 深層学習による削屑木簡の画像復元

Image-based Restoration of Mokkan Fragments Using Deep Learning

------------------------------------------------------------------------

## 📌 概要

本リポジトリは，研究プロジェクト\
**「深層学習による削屑木簡の画像復元」**\
のために開発されたソースコードおよび関連資料を公開するものです。

本研究では，深層学習を用いて断片化した削屑木簡画像を対象に復元処理を行い，形状や文字情報の再構成を試みます。本リポジトリには，学習・推論コード，前処理スクリプト，および実験設定が含まれています。

------------------------------------------------------------------------

## 📄 論文

本プロジェクトの詳細は以下の論文を参照ください：

https://ipsj.ixsq.nii.ac.jp/records/2007003

------------------------------------------------------------------------

## 📁 ディレクトリ構成（例）

    .
    ├── edgeconnect/        # GAN（EdgeConnect）による復元
    ├── strdiffusion/       # 拡散モデル（strdiffution）による復元
    ├── README.md           # 本ファイル
    └── requirements.txt    # 動作環境定義

------------------------------------------------------------------------

## ✏ 引用について

本リポジトリを研究目的で利用される場合は，上記論文を適切に引用してください。

------------------------------------------------------------------------

## ライセンス / License

本リポジトリには、異なるライセンスの複数コンポーネントが含まれています。

### 独自コード（本研究プロジェクト）

本リポジトリ内の独自に開発したコードは MIT License の下で公開されています。

Copyright (c) 2026 Pattern and Media Informatics Laboratory, TDU

---

### Third-party software

本研究では以下の外部ソフトウェアを軽微に改変した上で利用しています。  
それぞれ元のライセンス条件に従います。

#### StrDiffusion

- License: Apache License 2.0  
- Modifications: Minor modifications for research purposes

See `StrDiffusion/LICENSE.txt`.

---

#### EdgeConnect

- License: Creative Commons Attribution–NonCommercial 4.0 International (CC BY-NC 4.0)  
- Modifications: Minor modifications for research purposes  
- Note: This component is restricted to **non-commercial use only**.

See `edgeconnect/LICENSE.md`.

---

### Important Notice

EdgeConnect is licensed under CC BY-NC 4.0.  
Therefore, any use involving this component or its derivatives is limited to **non-commercial purposes**.
