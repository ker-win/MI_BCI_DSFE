好，我來把「Phase 1：只做 F×T segmentation + 單一 FTS wrapper」完整展開，包含**時間窗口怎麼切、頻段怎麼切、每個 FTS 怎麼抽 FTA+PTC、怎麼做 wrapper 評估**，一步一步講清楚。

我會假設你在做的是「**單一 subject、單次實驗的 within-subject CV**」，你可以之後自己套到你原本的 5-fold / 10-fold CV 流程裡。

---

## 0. 大方向再說一次（一句話）

> 把每個 trial 的 MI epoch 切成很多「時間 × 頻率」的小片段（FTS），
> 每一片段都：**全通道 + FTA + PTC → 練一個小小的 linear SVM → 看這塊片段自己可以分對幾成**，
> 然後看哪一個 FTS 的 accuracy 最高，再用它去跑外層 CV 的 test，看看「單一最佳 FTS」能做到多少準確度。

Phase 1 不用 ReliefF、不用多個 FTS combo，只是要回答：

> 「如果我只用時間 × 頻帶網格裡某一塊最有辨識力的片段，它自己能做到多少？」

---

## 1. 前處理設定（所有 FTS 共用）

這一段是「對整個 trial 做一次」，之後每個 FTS 只是取子段用，不要一個 FTS 重做一次前處理，會爆算力。

### 1.1 先決定分析 epoch

以 BCI-IV 2a 為例（你可以照你現在 pipeline 微調）：

* cue 出現時間假設是 `t = 0`
* MI 真正開始執行大概是 0.5 s 之後，持續到 3–4 s 左右

這裡給一個建議設定：

* **分析 epoch**：`[0.5 s, 4.0 s]`（長度 3.5 秒）

如果你的原 pipeline 是別的範圍（例如 3–6 s），就把下面所有時間相對應改掉就好，邏輯一樣。

### 1.2 前處理步驟（每個 trial）

對原始 EEG（22 ch）：

1. **去 DC / 去漂移**（可選）：高通 0.5 Hz。
2. **帶通 4–40 Hz**（IIR/FIR 皆可，但整個實驗固定一種）。
3. **50 Hz notch**（如果你原本有做）。
4. **re-reference**：平均參考（CAR）。
5. **downsample**：例如到 250 Hz（或你現在在用的值）。
6. **切 epoch**：每個 trial 取 `t ∈ [0.5, 4.0]` 這段，得到 shape：
   [
   X_{\text{trial}} \in \mathbb{R}^{n_{\text{ch}} \times n_{\text{time}}}
   ]

> **之後所有的時間窗都是在這個 [0.5, 4.0] 的 epoch 裡切出來的子片段。**

---

## 2. 定義 T（時間）grid：多尺度時間窗 + overlap

我們要做的是：**對每個 trial，在 [0.5, 4.0] 之內用多種窗長滑動，建立一堆時間窗**。

### 2.1 選幾種窗長

建議先不要太多，Phase 1 用這三種就好：

* ( L_1 = 0.5 ) 秒
* ( L_2 = 1.0 ) 秒
* ( L_3 = 1.5 ) 秒

之後如果算力足夠，再加 0.7 s / 1.25 s 那些。

### 2.2 窗的滑動步長（overlap）

每一種窗長，都用 **50% overlap**：

* 對 (L) ：step = ( L / 2 )

### 2.3 具體例子

分析 epoch 長度：3.5 秒（從 t=0.5 到 4.0）。

#### (1) (L_1 = 0.5 s)，step = 0.25 s

時間窗中心/起訖大約會是：

* 0.50–1.00
* 0.75–1.25
* 1.00–1.50
* …
* 3.50–4.00

概略個數大約 = ((3.5 - 0.5) / 0.25 ≈ 12) 個窗口。

#### (2) (L_2 = 1.0 s)，step = 0.5 s

* 0.50–1.50
* 1.00–2.00
* 1.50–2.50
* 2.00–3.00
* 2.50–3.50
* 3.00–4.00

大約 6 個。

#### (3) (L_3 = 1.5 s)，step = 0.75 s

* 0.50–2.00
* 1.25–2.75
* 2.00–3.50
* 2.75–4.25（超出 4.0 就不要）

大約 3 個。

所以總共時間窗數：

* (N_T ≈ 12 + 6 + 3 = 21) 個左右（精確數字依你實作的邊界處理）。

我們給每個時間窗一個索引 ( j = 1,\dots,N_T )。
你可以用一個 list 存：

```python
time_windows = [
    (t_start_j, t_end_j),
    ...
]
```

---

## 3. 定義 F（頻率）grid：固定 filter bank

Phase 1 先「不用 FDCC」，而是用一組固定的 filter bank，讓每個 F 候選頻段都被公平評估一次。

### 3.1 總頻率範圍

* `f_min = 4 Hz`
* `f_max = 40 Hz`

### 3.2 頻帶寬度與 overlap

先建議用 **4 Hz 寬度 + 50% overlap**，粒度夠細，也不會多到爆炸。

* band width = 4 Hz
* step = 2 Hz（因為 50% overlap）

### 3.3 具體頻段

從 4 Hz 開始，一路往上：

* [4–8]
* [6–10]
* [8–12]
* [10–14]
* [12–16]
* …
* [32–36]
* [34–38]
* [36–40]

你可以寫程式生成：

```python
freq_bands = []
f = 4
while f + 4 <= 40:
    freq_bands.append((f, f + 4))
    f += 2
```

這樣大約會有 (N_F ≈ (40-4-4)/2 + 1 = 16) 個 band 左右。

---

## 4. F×T segmentation：定義每一個 FTS

現在我們有：

* 時間窗：`time_windows[j]`，j = 1..N_T
* 頻帶：`freq_bands[i]`，i = 1..N_F
* 空間：先固定為「所有 EEG 通道」（例如 22 ch）

每一個 FTS（Fine-grained Time×Frequency Segment）定義為：

> FTS(i, j) = { all channels, freq_bands[i], time_windows[j] }

總數大約：

* (N_{\text{FTS}} = N_F \times N_T ≈ 16 × 21 = 336) 個 FTS

這個數字是可接受的（Phase 1 只要跑一次 wrapper，不用 combo），之後如果覺得太多可以縮減窗長或減少 band。

---

## 5. 對單一 FTS 抽 FTA + PTC 特徵

這一段是在「**內部**：給定一個 FTS(i,j)，對全體 trials 抽特徵」。

### 5.1 從 preprocessed epoch 截出時間子段

對某一 trial 的 preprocessed epoch (X_{\text{trial}} \in \mathbb{R}^{C \times T})：

1. 找到時間點落在 `time_windows[j] = (t_start, t_end)` 的 sample index 範圍。
2. 截出這段：
   [
   X_{\text{seg}} \in \mathbb{R}^{C \times T_j}
   ]

### 5.2 對這段做 band-pass 到 `freq_bands[i]`

最好是先對整段 epoch 做 4–40 Hz 的 broad bandpass（已做），
然後在這裡再對 `X_seg` 做窄帶 band-pass [f_low, f_high]（可以用 IIR/FIR，order 不必太高）。

得到：

[
X_{\text{FTS}}(i,j) \in \mathbb{R}^{C \times T_j}
]

### 5.3 抽 FTA 特徵（對這個 FTS）

對每個 channel 的這段訊號做 FFT（或 rFFT）：

1. 對時間長度 (T_j) 做 FFT：
   [
   X_c(f),\ c=1..C
   ]
2. 取對應於 `freq_bands[i]` 的頻率 bins（只要正頻率）。
3. 取 **振幅** (|X_c(f)|) 作為 FTA 特徵。
4. 對所有 channels & 所有選到的 freq bins 做 flatten：
   [
   \text{FTA_vec}*{\text{FTS(i,j)}} \in \mathbb{R}^{d*{\text{FTA},ij}}
   ]

你也可以做一些簡單降維（例如對 band 內的 amplitude 平均成一個數值），那維度會少很多，不過 Phase 1 也可以先用原始頻點。

### 5.4 抽 PTC 特徵（對這個 FTS）

這部分就是你自己的 PTC 定義：

* 在 (X_{\text{FTS}}(i,j)) 這段窄頻多通道訊號上，
* 用你原本的方法計算 PTC 向量：
  [
  \text{PTC_vec}*{\text{FTS(i,j)}} \in \mathbb{R}^{d*{\text{PTC},ij}}
  ]

關鍵只有：

* 要對 **同樣的時間段 + 同樣的頻帶** 計算 PTC，
* 確保跟 FTA 是 align 在同一個 FTS 上。

### 5.5 合併成單個 FTS 的特徵向量

對每個 trial：

[
\mathbf{z}^{(\text{trial})}*{ij} = [\text{FTA_vec}*{ij},\ \text{PTC_vec}_{ij}]
]

對所有 trial 疊起來：

[
Z_{ij} \in \mathbb{R}^{N_{\text{trials}} \times d_{ij}}
]

其中 (d_{ij} = d_{\text{FTA},ij} + d_{\text{PTC},ij})。

---

## 6. 外層交叉驗證 + FTS-level wrapper 評估

接下來是關鍵：**FTS-level wrapper**
也就是：對每一個 FTS(i,j)，只用它自己的特徵 Z_{ij} 訓練一個 linear SVM，
看他自己在 train set 的 CV accuracy 是多少。

### 6.1 外層 CV：跟你現在一樣

假設你用 10-fold CV：

```text
for outer_fold in 1..10:
    train_trials, test_trials = split_trials(...)
```

我們所有的 FTS 選擇、wrapper 都只用 `train_trials` 做，
最後才用「最佳的 FTS」在 `test_trials` 上評估。

### 6.2 對 train fold 的每個 FTS 做 wrapper

對於第 `outer_fold`，在 `train_trials` 上：

```text
for each FTS(i,j):
    # 1. 抽 feature: Z_ij_train  (train_trials × d_ij)
    # 2. z-score normalize
    # 3. linear SVM + inner CV → 得到 Acc_ij
```

更細節：

1. **建立特徵矩陣**：

   * 對 train_trials 裡每個 trial，計算這個 FTS 的 feature vector `z_trial_ij`，
   * 疊成 `Z_ij_train ∈ R^{N_train × d_ij}`。

2. **標準化（很重要）**：

   * 用 `Z_ij_train` 計算每一維的 mean / std，
   * 對 `Z_ij_train` 做 z-score，
   * （test 時也要用一樣的 mean/std）。

3. **Inner CV 評估**：

   * 在 `Z_ij_train` 上做一個簡單的內層 CV（例如 5-fold StratifiedKFold），
   * classifier 用 **linear SVM**（`sklearn.svm.LinearSVC` 或 `SVC(kernel='linear')`）。
   * 每個 inner fold：

     * 用 inner-train 訓練，
     * 用 inner-val 測 accuracy。
   * 所有 inner fold 的 accuracy 平均，就是這個 FTS 的 `Acc_ij`。

4. **存結果**：

```python
FTS_scores.append({
    "i": i,
    "j": j,
    "Acc": Acc_ij,
    "dim": d_ij,
    "time_window": (t_start_j, t_end_j),
    "freq_band": (f_low_i, f_high_i)
})
```

對 336 個 FTS 都做完，你就得到一份 score list。

---

## 7. 找出「單一最佳 FTS」，在外層 test 上驗證

現在對第 `outer_fold`：

1. 把 `FTS_scores` 按 Acc 排序：

```python
best_fts = max(FTS_scores, key=lambda x: x["Acc"])
i_star = best_fts["i"]
j_star = best_fts["j"]
```

2. **在整個 train_trials 上，用這個最佳 FTS 的特徵 Z_{i*, j*} 再訓練一次 classifier**：

   * 重新用 `train_trials` 抽一次 `Z_best_train`；
   * z-score（用 training 的 mean/std）；
   * 用同樣的 linear SVM 訓練，得到 `clf_best`.

3. **在 test_trials 上用同樣 FTS 特徵做預測**

   * 對 test_trials，每個 trial 抽 FTS(i*, j*) 特徵 → `Z_best_test`。
   * 用 training 時的 mean/std 做 z-score。
   * 丟進 `clf_best.predict` → 得到 test fold 的 accuracy。

4. 把這個 outer_fold 的 test accuracy 存起來。

對所有 outer folds 重複這一整套流程，你最後就會有：

* 一個「Phase 1：單一最佳 FTS」的平均 accuracy（跨 folds 平均），
* 同時你也會知道每個 subject、每個 fold 最常出現的 best FTS 大概落在哪些時間窗 + 頻段。

---

## 8. 這個 Phase 1 你可以得到什麼資訊？

完成 Phase 1 之後，你可以回答很多關鍵問題：

1. **單一最佳 FTS 的表現能到幾 %？**

   * 如果有些 subject 在某個 FTS 上可以衝到 80% 左右，
   * 但你現在整體方法（把所有時間窗、頻段都塞一起再 ReliefF）只有 70%，
     ⇒ 那很可能是「好片段被一堆沒用片段稀釋掉」。

2. **最佳 FTS 多半出現在哪些時間窗？**

   * 例如你發現很多 subject 的最佳 FTS 都在：

     * 時間 1.0–2.0 s、頻率 8–12 Hz
       ⇒ 代表你的 ERD/ERS 可能在那附近最穩定，
       ⇒ 你之後就可以把 FDCC / FTA 的設計偏重那一帶。

3. **PTC 是否真的有幫助？**

   * 你可以在 Phase 1 再跑一個對照：只用 FTA，不加 PTC，看 FTS-level accuracy 差多少。
   * 如果差異很小，表示 PTC 對「局部時間×頻帶片段」沒什麼加成，
   * 可以考慮縮減 PTC 維度或只在某些 FTS 上用。

4. **線性 SVM vs 你原本的 classifier**

   * Phase 1 全程都用 linear SVM；
   * 你也可以在最佳 FTS 上試試 RBF SVM / RF 看有沒有比較好。
   * 如果 linear 本身就不錯，代表你後面可以放心用「高維 + linear」，不一定要 RBF。

---

## 9. 實作時的小技巧與建議

* **先在少數 subject 上試跑**：例如先挑 1–2 位受試者，把整套 Phase 1 跑通，確認程式 / FTS 數量 / 時間都可接受，再擴展到全部 subject。
* **記錄每個 best FTS 的 (time, freq)**：之後 Phase 2 / Phase 3 做 combo / ReliefF 時，這些資訊會很有用。
* **確保內外層 CV 不混**：outer fold 的 test trial 絕對不能參與到 FTS wrapper 的 inner CV。
* **z-score 一定只用 train**：避免 leakage。
