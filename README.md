# Team Prism 足あと検出

### 今後やること
- Structure from Motion で部屋の3次元点取得 OK
- 平面フィッティングで床の3次元位置取得 OK
- 前の足あと検出アルゴリズムで足あと計測
- openPoseの関節位置を3次元復元

- キャリブレーション
- 実行時間短縮
    1. 点群再投影
    2. Searching rectの大きさに画像を分割
    3. 再投影点群を枠ごとに分割保持
    
##### 11/25
- TODO
    - homography 変換
    - HeatMap 完成
    - 長時間撮影
    - 行動認識 
    
##### 11/29
- TODO
    - WebCamMode実装
        - WebCamCalibration
        - Re-Identification
        - 歩行予測
        
##### 12/14
gait & posture idea
- 様々なシーン
    - 廊下で不特定多数の人
    - cornerで計測
    - 対象人物のトラッキング(一度画面外に出てもok)
    - 坂道
- 出力
    - 俯瞰した足あと
    - HeatMap
- 精度
    - optiTrackとの比較実験
    
TODO
    - キャリブレーションボード作成(poster印刷)
    - 床に敷いてキャリブレーション
    - goProで撮影し接地位置の正解取っておく
    - 簡単な精度評価
    - 足向き実装

##### 12/15
TODO
- vote range も実際のスケールに合わせる
- 足のタイミング直す
- キャリブレーションボード印刷
- 評価実験