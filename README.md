# uec-internship-2026-2nd-team2
2026年度電通大インターン向けサンプルコード

## 概要
フォトロン社製ハイスピードカメラであるINFINICAMを用いてチーム開発を行うにあたって必要な知識を身に付ける。
基本的な画像処理の仕方や、AIを用いた画像処理、GPUやNPUを用いた高速化とリアルタイム処理のサンプルコードをまとめている。

動作環境はWindows。AI処理のサンプルはIntel製のGPU(内蔵グラフィックス)とNPUを使うため、Core Ultra世代のPCを想定している。

## インストール方法
pip installもしくはpy -m pip installで必要なライブラリをインストールする

・pypuclib(INFINICAMを制御するライブラリ)
  ```
  pip install pypuclib
  ```
  ※Windows専用。インストールする前に、INFINICAM SDK(PUCLIB)、カメラドライバ、Visual C++ 再頒布可能パッケージ2019を先に入れておく必要がある。
  「import pypuclibでエラーが出る」場合は、たいていこのどれかが未インストール。

・numpy(数値計算を高速に行うライブラリ)
  ```
  pip install numpy
  ```

・OpenCV(画像処理を行うライブラリ)

  ```
  pip install opencv-python
  ```

・pillow(画像処理を行うライブラリ)

  ```
  pip install pillow
  ```

・ffmpeg(動画の処理に特化したライブラリ)

  ```
  pip install static-ffmpeg ffmpeg-python
  ```

・OpenVINO(Intel製ハードウェアでAI処理を高速化する推論エンジン)
  ```
  pip install openvino
  ```
  ※NPU用のプラグインは同梱されているので別途インストールは不要。ただしNPUドライバは必要で、
  デバイスマネージャーの「ニューラル プロセッサ」に Intel(R) AI Boost が出ているか確認する。

  ※使えるハードウェアは次のコマンドで確認できる。['CPU', 'GPU', 'NPU'] と出れば準備完了。
  ```
  python -c "import openvino as ov; print(ov.Core().available_devices)"
  ```

・AI推論で用いるonnxファイルの配置

onnxフォルダーにddcolor_512.onnxとyolov8m_640.onnxを配置する。
必要なファイル名は各サンプルの先頭にあるMODEL(またはCOLOR_MODEL、YOLO_MODEL)の設定に書かれているので、それに合わせる。

yoloのonnxはultralyticsから書き出せる。
  ```
  pip install ultralytics onnx onnxslim
  ```
  ```python
  from ultralytics import YOLO
  YOLO("yolov8m.pt").export(format="onnx", imgsz=640, opset=17,
                            dynamic=False, batch=1, simplify=True)
  ```
※NPUは動的シェイプに対応していないため、dynamic=False と batch=1 で入力サイズを固定して書き出すこと。

## サンプルコードの内容

### ・基本(web-camera-sample/)

* [hello-opencv.py](web-camera-sample/hello-opencv.py)

OpenCVで画像を読み込んで表示する
1. hello-opencv.pyを実行

* [web-camera-sample.py](web-camera-sample/web-camera-sample.py)

webカメラから取得した画像を1枚だけ表示する
1. web-camera-sample.pyを実行


### ・INFINICAM周りのサンプル(pypuclib-main/pypuclib/pypuclib_sample/)
* [hello_world.py](pypuclib-main/pypuclib/pypuclib_sample/hello_world.py)

INFINICAMで取得した画像をリアルタイムで表示するサンプルコード
1. カメラを接続し、hello_world.pyを実行

* [gui_sample.py](pypuclib-main/pypuclib/pypuclib_sample/gui_sample.py)

INFINICAMをGUIで制御するサンプルコード。解像度や撮影速度、シャッター速度をGUIベースで切り替えられる。
1. カメラを接続し、gui_sample.pyを実行
2. 右側のコンボボックスから撮影速度、シャッター速度、解像度を調整
3. 録画はAcquisition Modeをcontinuous、Save Fileをbinaryに設定し、RECで録画開始/停止できる。
4. 録画したものはFileタブから同じ階層にある test.json を開くと再生可能。

※画面下部のFrameに数値を入力すると、指定フレームへ移動可能。

* [create_movie.py](pypuclib-main/pypuclib/pypuclib_sample/create_movie.py)

INFINICAMで取得した映像をmp4形式で録画する
1. カメラを接続し、create_movie.pyのスクリプトを実行。
2. 起動時にデコードスレッド数を問われるので「12」と入力する。
3. ライブ表示のウィンドウが表示されたら、「s」キーを押して録画を開始、画面左上にRECと表示され、再度「s」キーを押すとエンコードを終了。同階層に create_movie.mp4 が生成される。
4. 通常のCPUエンコードだと重く、ドロップフレームが多発するのでvcodecをh264_qsvに設定するとGPUエンコードが可能になる。GPUエンコードを使用するとほぼドロップせずにリアルタイムエンコードが可能になる。

※ MAX_SAVE_FRAME_COUNT で最大録画枚数を変更可能。

* [tracking_sample.py](pypuclib-main/pypuclib/pypuclib_sample/tracking_sample.py)

テンプレートマッチングを用いたオブジェクト追跡用サンプル
1. カメラを接続してtracking_sample.pyを実行
2. 静止画上で追跡したいターゲットをドラッグして選択
3. 選択後、Enterキーを押してトラッキングを開始

その後ウィンドウが立ち上がり、指定したオブジェクトが追跡される。

※テンプレートマッチングは選択した見た目をそのまま探す方式なので、対象が回転したり大きさが変わったりすると見失う。うまく追えないときは、模様がはっきりした部分を選び直す。

### ・カラー化サンプル(ddcolor_sample/)

AIを用いてモノクロ画像をカラー化するサンプル。処理が重いので、GPUやNPUに計算させて高速化する。

* [image_colorize_openvino.py](ddcolor_sample/image_colorize_openvino.py)

DDColorモデルで読み込んだモノクロ画像のカラー化を行う。
1. imageフォルダに白黒画像を入れる
2. image_colorize_openvino.py を開き、コード内のMODEで使用するハードウェアを指定する。(cpu_fp32、gpu_fp32、gpu_fp16、npu_fp16)
3. 実行する
4. 同じ階層のoutputフォルダーにカラー化された画像が格納される

* [infinicam_colorize_openvino.py](ddcolor_sample/infinicam_colorize_openvino.py)
* [webcam_colorize_openvino.py](ddcolor_sample/webcam_colorize_openvino.py)

INFINICAM及びwebカメラの画像をカラー化する。必要に応じて使い分け。
webカメラはカラー→モノクロにしてからカラー化している。

1. INFINICAMを繋いで、infinicam_colorize_openvino.pyを実行。
2. コード内のMODEでそれぞれ使用するハードウェアを指定する。(cpu_fp32、gpu_fp32、gpu_fp16、npu_fp16)
3. カラー化されたINFINICAMの画像が表示される
4. タスクマネージャーでCPU、GPUのCompute/3DやNPUの項目で使用率を確認

※Sでoutputフォルダにbmp保存
※Cキーでモノクロ/カラー切り替え

※NPUとGPUは初回だけモデルのコンパイルに10〜30秒かかる。固まったわけではないので待つこと。2回目以降はcacheフォルダから読むのですぐ起動する。
※npu_fp32という選択肢が無いのは、NPUがf16での計算にしか対応していないため。
※MODEを4通り試して、画面左上のfpsがどれだけ変わるか比べてみるとよい。

* [infinicam_colirize_yolo_realtime.py](ddcolor_sample/infinicam_colirize_yolo_realtime.py)
* [webcam_colirize_yolo_realtime.py](ddcolor_sample/webcam_colirize_yolo_realtime.py)

INFINICAM及びwebカメラの画像をカラー化しつつyoloで物体検出をリアルタイムで行う。必要に応じて使い分け。
webカメラはカラー→モノクロにしてからカラー化している。

カラー化と物体検出という2つのAIモデルを、別々のハードウェアに割り当てて同時に動かしている。
1フレームずつずらして流すことで両方のデバイスが同時に働くため、処理時間は2つの足し算にはならず、遅い方のモデルで決まる。

1. INFINICAMを接続後、infinicam_colorize_yolo_realtime.py を実行。
2. コード内のCOLOR_MODEとYOLO_MODEで各モデルで使用するハードウェアを指定する。(cpu_fp32、gpu_fp32、gpu_fp16、npu_fp16)
3. カラー化する場合とモノクロの場合で検出精度を見比べる。Cキーでカラー化ありなしを切り替え
4. タスクマネージャーでCPU、GPUのCompute/3DやNPUの項目で使用率を確認

※COLOR_MODEとYOLO_MODEには別のハードウェアを指定する。両方を同じにすると1つのデバイスの取り合いになり、同時に動かした意味が無くなる。
※検出にかけているのは常に画面に出ている画像そのもの。カラー化ONならカラー画像を、OFFならモノクロ画像を検出している。

### ・yoloリアルタイムサンプル(yolo_sample/)

GPUやNPUを用いてリアルタイムに物体検出を行うサンプル。

* [infinicam_yolo_realtime.py](yolo_sample/infinicam_yolo_realtime.py)
* [webcam_yolo_realtime.py](yolo_sample/webcam_yolo_realtime.py)

INFINICAM及びwebカメラの画像でGPUやNPUを用いてリアルタイムにyoloの検出を行う。
1. INFINICAMを接続後、infinicam_yolo_realtime.py を実行。
2. コード内のMODEで使用するハードウェアを指定する。(cpu_fp32、gpu_fp32、gpu_fp16、npu_fp16)
3. タスクマネージャーでCPU、GPUのCompute/3DやNPUの項目で使用率を確認

※検出結果が出ない、枠がずれる場合は、onnxを書き出したときのimgszとコード内の入力サイズが一致しているか確認する。