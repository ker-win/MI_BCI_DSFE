好，我先照你的要求「**先不做通道優化＝全部通道一起用**」，幫你把一個 **FGSFT 思想 + 你現有 FTA / PTC / ReliefF** 的完整流程設計出來。你可以把它想成是：

> 「把你現在的一條線，拆成很多 Time×Freq 小積木（SFT segment），
> 替每塊積木算 FTA+PTC 特徵，
> 用 wrapper 先選出『有用的積木』，
> 再在裡面用 ReliefF 做細粒度特徵選擇。」

我會用「實作順序」寫，這樣你可以直接照著一步步實現、A/B test。

---

## 0. 設定與前提

* 資料集：BCI-IV 2a
* 通道：**全部 EEG 通道**（先不做 channel-set）
* 類別：你目前如果是 4 類，就先維持 4 類；之後要比 FGSFT 再改成左右手二分類。
* 主要特徵家族：

  * FTA（頻域振幅）
  * PTC（你自定義的 phase/time 特徵）
  * 之後 *可以* 再加 CSP / RG，但這版先不強迫。

---

## 1. 前處理流程（整體一致的基礎）

每個 subject 的前處理固定，不要在後面 SFTS 裡重複做：

1. 讀取原始 EEG（22 ch），去除 EOG 通道或保留視需求。
2. **帶通濾波：4–40 Hz**（可以用 5th-order Butterworth、FIR，都可以，保持一致就好）。
3. 50 Hz notch（如果你原本有做就保留）。
4. 重新參考：平均參考（CAR）。
5. downsample → 250 Hz 或 128 Hz（視你原本 pipeline）。
6. 切 epoch：相對 cue 的時間窗，例如：

   * `t = 0` 為 cue on
   * 抓 `0.5–4.0 s` 的資料當 MI epoch（你可依自己慣例調整）。

後續的所有時間窗、SFT segment，都只在這個 epoch 內取子片段。

---

## 2. 設計 SFT grid（現在先只有 F+T，S = 全通道）

你說先不做 channel 優化，所以這一版 S 維是「單一值：all channels」。
我們只在 **時間 + 頻率** 兩個維度做 fine-grained segmentation。

### 2.1 時間窗（多尺度 + overlap）

在 MI epoch（例如 0.5–4.0 s）中定義多種長度 + 滑動窗，例：

* 窗長集合：`L = {0.5, 1.0, 1.5} 秒`
* 步長：每個窗 **50% overlap**，所以 step = L/2

對每個 L：

* 從 `t_start = 0.5 s` 開始，
* 每次往後滑 `L/2`，直到窗尾不超過 4.0 s。

這樣你會得到一堆時間窗：

* 短窗：0.5–1.0，0.75–1.25，…
* 中窗：0.5–1.5，1.0–2.0，…
* 長窗：0.5–2.0，1.25–2.75，…

### 2.2 頻帶（固定 filter bank，先不 FDCC）

這一階段**先不用 FDCC**，避免一開始複雜度爆炸。
用固定寬度的 filter bank，像這樣：

* 寬度 4 Hz，50% overlap，例如：

  * [4–8], [6–10], [8–12], [10–14], …, [28–32]
* 或者寬度 8 Hz，也可以，端看你想多細。

每一個頻帶就是一個 F 候選。

### 2.3 SFT segments（現在其實是 F×T，每一段都用全通道）

每一個 SFT segment（你可以叫它 `FTS`）定義為：

> `FTS(i, j) = 全通道 × freq_band_i × time_window_j`

後面所有操作都是對「每一個 FTS」做特徵、評估。

---

## 3. 每個 FTS 的基本特徵：FTA + PTC

對於某個特定 FTS(i, j)：
（= 某一個時間窗 + 某一個頻帶 + 全通道）

### 3.1 取子訊號

對每個 trial：

1. 從已前處理好的 epoch 中，切出 `time_window_j` 的範圍。
2. 對這段再做 band-pass 到 `freq_band_i`（可以 reuse 濾波器，減少運算）。
3. 得到形狀 `(n_channels, n_times_segment)` 的子訊號。

### 3.2 抽 FTA 特徵

對這個 segment 的 trial 整段做 FFT：

* 對每個 channel：FFT → |X(f)|
* 只保留 `freq_band_i` 的振幅值（你可以在頻率軸上只選那些 bins）。
* 把所有 channel × freq bins flatten → `FTA_vec`。

### 3.3 抽 PTC 特徵

用你自己的 PTC 算法：

* 直接在該 segment 的多通道訊號上計算 PTC → 得到一個 `PTC_vec`。

（這裡你最清楚自己的定義，我只要求 **在同一個 FTS 的子時域+子頻帶上計算**，保持一致）

### 3.4 合併為 FTS-level 特徵向量

單一 FTS 的 feature vector：

[
\mathbf{z}_{\text{FTS(i,j)}} = [\text{FTA_vec},\ \text{PTC_vec}]
]

所以對整個資料集、這一個 FTS，你會有：

* `Z_ij`：形狀 `(n_trials, d_ij)` 的矩陣。

（之後還要對每個 i,j 做評估）

---

## 4. 在「FTS 層級」做 wrapper selection

這是把 FGSFT 的核心搬過來的部分。

### 4.1 外層 vs 內層 CV（避免 leakage）

你要有一個**外層 CV**來報最終 accuracy（例如 10-fold CV 或 LOBO by run）。
**所有的 FTS 選擇、ReliefF、超參數調整，都必須在「外層 train fold 裡面」做。**

以下都假設是在某一個外層 fold 的 `train trials` 上進行。

### 4.2 單一 FTS 的 wrapper 評估

對每一個 FTS(i,j)：

1. 取出該 FTS 的 `Z_ij_train`（只看 train fold 的 trials）。
2. 對 `Z_ij_train` 做 standardize（z-score）。
3. 用一個 **簡單的 linear classifier**（建議 linear SVM 或 logistic regression）

   * 做 **內層 k-fold CV 或 LOBO**，
   * 得到這個 FTS 的平均 accuracy：`Acc_ij`。
4. 把 `Acc_ij` 記下，對所有 i,j 重複。

結果：
你會得到一份 list：

```text
FTS_list = [
  {id: (i,j),  Acc: Acc_ij,  dim: d_ij},
  ...
]
```

### 4.3 排序 + 選前 M 個 FTS

* 依 `Acc_ij` 由高到低排序。
* 保留前 M 個 FTS（例如 M=60，視總數和算力調整）。
* 其餘的 FTS 整塊丟掉（這比在 feature 維度上修修補補更乾脆）。

這一步就是：

> 「先選出『單獨就有辨識力』的時間窗×頻帶片段」。

---

## 5. 將多個 FTS 合併 + ReliefF 進行 feature-level 融合

這邊有兩個層次：

1. **FTS 組合的 wrapper（看多少個 FTS 一起用比較好）**
2. **組合內的 ReliefF（對高維特徵再做 fine-grained selection）**

### 5.1 建立逐步擴充的 FTS 組合

依照排序後的前 M 個 FTS：

* 組合 1：Top 5 個 FTS
* 組合 2：Top 10 個 FTS
* 組合 3：Top 15 個 FTS
* …
* 每次增加 D 個 FTS（D=5 或 10），直到用完 M 個。

對每一個組合 C_k：

1. 把此組合裡所有 FTS 的特徵向量 concatenate 起來：

   ```text
   Z_combo_k_train = [Z_FTS_1, Z_FTS_2, ..., Z_FTS_m]  # m = 該組合 FTS 數
   ```

2. 對 `Z_combo_k_train` 做 z-score。

3. 用 linear SVM + 內層 CV 評估這一個「組合整體」的 accuracy：`Acc_combo_k`。

得到：

```text
Combo_list = [
  {id: k,  FTS_ids: [...], Acc: Acc_combo_k},
  ...
]
```

### 5.2 選出前 K 個最佳組合（子模型）

* 依 `Acc_combo_k` 排序，
* 選出前 K 個組合（例如 K=3），
* 之後這 K 個會變成 **ensemble 裡的 K 個子模型**。

這就對應到 FGSFT 的「三個最強 SFTS 組合」。

### 5.3 對每個組合再做 ReliefF（feature-level 融合）

現在針對每個被選中的組合 k：

1. 取 `Z_combo_k_train`，先 z-score。
2. 在 train fold 上跑 **ReliefF**：

   * `ReliefF.fit(Z_combo_k_train, y_train)`
   * 得到 `weights_k`。
3. 選出前 `p%` 的特徵（例如 25%） → 得到一個 index set `idx_k_keep`。
4. 產生壓縮後的 `Z_combo_k_train_reduced = Z_combo_k_train[:, idx_k_keep]`。
5. 在這個 reduced 特徵上訓練一個 **最終版 linear SVM**：`clf_k`。

對每個組合 k，你都要存下：

* 這個組合用到的 FTS list
* 每個 FTS 的位置（time, freq）
* ReliefF 保留的 feature index `idx_k_keep`
* 標準化參數（mean/std）
* final classifier `clf_k`

---

## 6. 測試階段（在外層 test fold 上）

對外層某一個 fold 的 `test trials`，預測流程如下：

1. 對每個 **被選中的 FTS**：

   * 取相同的時間窗 + 頻帶 + 全通道 → 得到 segment 資料。
   * 抽同樣的 FTA+PTC 特徵 → `z_ij_test`。

2. 對每個「選中的組合 k」：

   * 把此組合裡所有 FTS 的 features 串起來 → `z_combo_k_test`。
   * 用訓練時的 z-score 參數 normalize。
   * 用 `idx_k_keep` 選出 ReliefF 保留的特徵 → `z_combo_k_test_reduced`。
   * 丟進 `clf_k.predict_proba`（或 decision_function），得到每個類別的分數 `p_k(y|trial)`。

3. 對 K 個子模型做 average / 加權平均：

   [
   p(y | x) = \frac{1}{K} \sum_{k=1}^K p_k(y | x)
   ]

4. 取 argmax → 最終預測類別。

這樣你每個 outer fold 的 test trials 都有預測，
平均起來就是你新的準確率。

---

## 7. 實作上的「務實版本」建議（避免一次爆炸）

這整套流程蠻重的，我建議你分段實作：

### Phase 1：只做 F×T segmentation + 單一 FTS wrapper，不用 ReliefF、不用 combo

1. 定義 F×T grid（全通道）。
2. 對每個 FTS 抽 FTA+PTC。
3. 在 train fold 上做 FTS-level wrapper（linear SVM + CV → Acc_ij）。
4. **先看：用「單一最佳 FTS」的特徵，accuracy 能到多少？**

這一步可以讓你知道：

* 你的 FTA+PTC 特徵在某些時間窗×頻帶上是不是其實很強，
* 70% 是否因為你把一堆沒用的 segment 混進去稀釋掉了。

### Phase 2：加 FTS combo（多個 FTS concat），仍然不加 ReliefF

在 Phase 1 的基礎上：

1. 取前 M 個 FTS
2. 建立 Combo 5, 10, 15, …
3. 看哪一個 Combo 的 linear SVM accuracy 最好。

這時你就能看到：

* 「多個有用 FTS 一起用」是否比「單一最佳 FTS」好多少。
* 很接近 FGSFT 的核心。

### Phase 3：在最佳幾個 combo 上再加 ReliefF

當你確認 Combo 真的能把 accuracy 拉上去，再：

1. 在 Combo 特徵上跑 ReliefF，
2. 比較有無 ReliefF 的差異。

---

## 8. FDCC 要放在哪裡？

目前這版流程 **沒有用 FDCC**，原因是：

* 你已經在 F 維度上用一堆固定 filter bank + wrapper 做 selection，
* 再加 FDCC 會大幅增加複雜度與 leakage 風險。

比較合理的做法是：

* 先用這套 SFT wrapper（不 FDCC）看能不能把 70% 推高。
* 如果有明顯提升，再考慮兩種 FDCC 用法：

  1. 在某些「高分 FTS」附近，用 FDCC 微調頻帶邊界。
  2. 或者只在最終 Combo 上，用 FDCC + ReliefF 做微幅 fine-tune。

但那是之後的事，先把這個「無 FDCC 的 SFT + wrapper + ReliefF」版本做穩，比較能看出結構性的改善。

---

### 總結一句話

> 你現在的 70% 很可能是因為「把一整段時間 + 一整段頻帶 + 全通道的特徵全部丟進去，再用 ReliefF 在超大空間裡硬挑」，
> FGSFT 的核心其實是：**先把時間×頻率切得很細，用 wrapper 找出哪幾塊局部片段本身就有辨識力，再在這些片段內做特徵層次的融合與選擇**。
> 我幫你規劃的流程，就是把這個邏輯換成「FTS（全通道） + FTA + PTC + ReliefF」的版本。

如果你願意，你可以把你目前的程式架構（例如 `preprocess.py`, `features.py`, `fusion.py`, `train_eval.py` 的主要流程）貼個簡化版，我可以直接幫你改寫成「外層 CV + FTS wrapper + combo + ReliefF」的 pseudo-code，對應到你現有的 function 名稱，會更容易實作。
