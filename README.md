# Team Prism 足あと検出

### 今後やること
- 接地位置推定
    1. OpenPoseの関節位置を使って2クラス分類問題として推定
    2. 関節位置と直線の距離のピークから計算
    
- パノラマ画像生成
    1. 画像中直線の角度からトランスレーションを補正
        並行移動だけでなく回転も行う
        トランスレーション推定がうまくいかない原因
    2. OpticalFlow
    3. トラック内はHスケール、トラック外はgrayスケール
    
- トラッキング(歩数推定の精度に大きく影響)
    1. OpenPoseの関節位置
    2. 複数パラメータ
        1. カラーヒストグラム
        2. 関節位置と直線の距離
        3. 前フレーム人物との距離
        4. フレーム情報
        
- OpenPose検出
    - 矩形を入力