先講一句白話總結：
這篇 FGSFT 論文之所以在 BCI-IV 2a 做到 8x% 的準確率，**不是因為用了一個神奇的 classifier**，而是整個 pipeline 從「切資料的方式」「特徵抽法」「選哪些片段」到「怎麼 ensemble」都非常針對 MI 的空間–頻率–時間特性在設計。

你現在的做法（FTA 時間窗 + FDCC + PTC + ReliefF）其實已經有「頻率 + 時間 + 特徵融合」的精神，但有幾個關鍵層級跟 FGSFT 還差一截，我會把它拆開來說，最後給你一個「如何把 FGSFT 的底層邏輯移植到你方法」的具體改版建議。

---

## 一、FGSFT 為什麼準？核心設計拆解

### 1. 真正細緻的 S–F–T 切割（SFTS）

它不是只做「多時間窗」或「filter bank」，而是**同時**在三個維度做多尺度切割：

* **時間**：六種窗長 0.2, 0.4, 0.7, 1.0, 1.25, 1.5 s，彼此 50% overlap。
* **頻率**：4 / 8 / 16 / 32 Hz 的多尺度頻帶，彼此 50% overlap（最窄可以到 4 Hz，如 30–34 Hz）。
* **空間**：為 22 個電極設計了 10 組通道子集（多組左右對稱 sensorimotor、額葉、枕葉 + 自動 correlation-based selection）。

一個「SFT segment (SFTS)」就是：

> 某一組 channel set × 某一個頻帶 × 某一個時間窗。

所以它不是在整段 0–3 s 上做一組特徵，而是把 EEG 切成**大量的小積木**，每塊積木都只看「某一個時刻附近 + 某個 band + 某一群電極」。

這直接對應到 MI 的特性：

* ERD/ERS 在不同 subject 的**時間**和**頻帶**不一樣
* activation 也不一定只在 C3/C4，很多人會有 prefrontal 或 occipital 的參與

多尺度 SFT 的好處就是：**不假設「整段 0.5–2.5 s、8–30 Hz、所有 channels」都重要**，而是讓演算法自己在 fine-grained 空間裡找「哪幾塊」真的是 discriminative。

---

### 2. 每一塊 SFTS 都先做「wrapper 型」評估，而不是只看統計指標

這篇的 feature selection 完全是**wrapper-based**：

1. 對每一個 SFTS：

   * 用這組 SFTS 的資料抽 divCSP 特徵（d=4）。
   * 用 **linear SVM + LOBO cross-validation** 評估這個 SFTS 自己的分類準確率。
2. 把所有 SFTS 依照 accuracy 排序（Acc*)。
3. 再用步進合併：

   * 先用 top 5 SFTS concat 特徵 → SVM LOBO → 得到一個 accuracy
   * 再用 top 10、top 15、…（步長 D=5）一路加，對每個「合併組合」再算一次 accuracy。
4. 最後選出 top-K=3 個表現最好的「SFTS 組合」，作為三個 sub-classifier，testing 時用 **機率平均 ensemble**。

這跟你現在的 FDCC + ReliefF 差別非常大：

* **FDCC / ReliefF 都是「filter 型 feature selection」**：

  * 用 correlation / 鄰近樣本距離來評估 feature 維度的重要性，
  * 但不真正看「放進 classifier 以後整體 accuracy 會怎樣」。
* FGSFT 則是：

  * **先在「SFTS 層級」直接看分類表現（wrapper）**，
  * 只讓「單獨就有分類力的片段」進入下一階段，
  * 再用 merged SFTS 形成高維特徵 + ensemble。

這一點在論文的 Algorithm 2 寫得很清楚：先對每個 SFTS 跑 divCSP+SVM+LOBO 得到 Acc[i]，排序後再逐步 merge。

**關鍵：它選的是「哪幾個時間×頻帶×空間 patch 的整體表現最好」，而不是單一特徵維度的權重。**

---

### 3. 強力的空間特徵：divCSP + regularization

整個框架的核心特徵是 **divergence-based CSP (divCSP)**，不是 FTA：

* 目標函數同時：

  * 最大化兩類平均 covariance 的 KL divergence（class mean 間差異大）
  * 最小化同類 trials covariance 的 KL divergence（類內一致）
* 用在預白化後的 covariance 上，在正交流形上做 gradient descent，得到 rotation matrix R，再算 spatial filter W_sf。
* 每個 SFTS 只取 d=4 個 log-variance 特徵。

直覺上，這比「逐 channel 的 FTA 振幅 / PSD」更強，因為：

* CSP 類方法直接在「兩類 covariance 差異」上學習空間濾波器，
* 很適合 MI 這種「左/右半球 ERD/ERS 差異」的任務。

你現在的 FTA + PTC 比較偏向 **頻譜 / 相位** 特徵，空間的處理相對弱（除非你有另外做 RG 或 covariance 類特徵）。

---

### 4. 實驗設計＆證據：SFT 組合真的比單一維度 segmentation 好

論文在 Table I 做了各種 ablation（只 S、只 F、只 T、SF、ST、FT、SFT），在 BCI-IV 2a（左/右手）上：

* 單純時間 segmentation（T）就已經比「只頻帶」或「只空間」好，說明**時間窗是最大的 boost**。
* SFT（三維一起）在 test set 上的平均準確率大約是 **83.68%**，顯著高於 ST 與 SF。

也就是說，它的高表現不是「某一個 trick」，而是：

> **多尺度時間窗是主力，
> 再加上頻帶×空間 segmentation + wrapper 選段 + divCSP 空間特徵，疊起來的效果。**

---

## 二、你的方法 vs FGSFT 的落差在哪？

你現在的 pipeline（我整理一下）大概是：

* 前處理 →
* 多個時間窗上的 **FTA 特徵**
* 用 **FDCC** 做頻帶選擇
* 再加上 **PTC 特徵**（第三組）
* 所有特徵丟進 **ReliefF** 做特徵選擇與融合
* 再交給 SVM / RF / NB (ensemble)

對照 FGSFT，有幾個層級不太一樣：

1. **時間維度：**

   * 你：有時間窗，但通常是少數幾個、固定窗長。
   * FGSFT：**多尺度 + 50% overlap 的 dense 覆蓋**（0.2–1.5s 各種窗長）。
     ⇒ 你可能**沒有涵蓋到每個 subject 真正 ERD/ERS 最強的區間**。

2. **頻率維度：**

   * 你：用 FDCC 在較大的頻段內選一小段 optimal band，然後整個時間窗都用那個 band。
   * FGSFT：多種 bandwidth (4/8/16/32 Hz) 的 filter bank，所有 band 都當候選 SFTS，最後由 wrapper 選出**在哪個時間窗+channel set 的哪個 band** 真的有用。

3. **空間維度：**

   * 你：主要靠 FTA / PTC（可能 per-channel + 後面 ReliefF），空間上沒有「多個 electrode set」的 segmentation。
   * FGSFT：明確建立了多個對稱 channel group（sensorimotor、prefrontal、occipital 等）+ 自動 selection，讓「哪塊頭皮區域重要」也進入分段與選擇流程。

4. **特徵選擇層級：**

   * 你：**一次把所有時間窗×頻段×特徵都 concat → ReliefF 挑 feature 維度**（filter）。
   * FGSFT：**先在 SFTS 的「區塊層級」做 wrapper selection**（看 LOBO accuracy），只保留真正有辨識力的 patch，再考慮 merge & ensemble。

5. **特徵型態：**

   * 你：頻譜 amplitude (FTA) + PTC（相位/時間耦合）。
   * FGSFT：**CSP 類空間 variance 特徵 (divCSP)**。對 MI 而言通常比較「粗暴有效」。

6. **classifier 設計：**

   * 你：SVM (常用 RBF)、RF、NB 的 weighted ensemble。
   * FGSFT：非常「乾淨」的 **linear SVM ensemble**，每個 SFTS 組合一個 linear SVM，最後做機率平均。
     在高維但樣本不多的狀況下，linear + 好的 feature selection 通常比 RBF 更穩。

此外還有一個可能的差別：

* FGSFT 在 BCI-IV 2a 是只做 **左/右手二分類**，而不是四類（左/右/腳/舌）。
* 如果你現在的 70% 是做四分類，那本來就比較難，不能直接拿來跟 83% 的二分類比。

---

## 三、有哪些「底層邏輯」可以直接移植到你的方法？

我會分成幾個可以逐步加上去的「設計原則」，不建議一次全改，這樣你也比較知道是哪一步真正有幫助。

---

### 1. 把你的 pipeline 也「SFTS 化」

**想法：**
現在你是「多時間窗 + FDCC 頻帶 + 全通道 / 固定通道」，可以改成：

> 定義 candidate SFTS ＝ (channel group, frequency band, time window)，
> 然後在每個 SFTS 上算你自己的特徵（FTA + PTC），再做 wrapper selection。

**實作方向：**

1. **先定義幾組 channel groups**（不用一開始就跟論文一樣多）：

   * 例如：

     * G1: C3, Cz, C4
     * G2: FC3, FCz, FC4
     * G3: CP3, CPz, CP4
     * G4: 左半球 (FC3,C3,CP3,…)
     * G5: 右半球 (FC4,C4,CP4,…)
   * 甚至可以加一組 occipital (O1,O2,PO3,PO4)。

2. 頻率：

   * 用 FDCC 找到「候選頻帶集合」，或直接照 FGSFT 的 4/8/16/32Hz filter bank，再讓 FDCC/ReliefF 之後去篩。

3. 時間：

   * 借用他們的多尺度窗長（0.2–1.5 s, 50% overlap），套到你現在的 epoch 上。

4. 這樣就得到一堆 SFTS，對每個 SFTS：

   * 取該 channel group 的資料；
   * band-pass 到某個頻帶；
   * 截某個時間窗；
   * 算 FTA、PTC（甚至再加一個 CSP/log-var 支線也可以）。

---

### 2. 在「SFTS 層級」先做一次 wrapper selection，再丟 ReliefF

模仿 Algorithm 2，但換成你的特徵：

1. 對每個 SFTS：

   * 用這個 SFTS 的特徵（FTA+PTC），訓練一個非常簡單的 linear classifier（linear SVM / LR 均可），用 LOBO 或 k-fold 在 **train set** 上估 accuracy。
   * 存下 Acc[i]。

2. 依 Acc[i] 排序，留下 top M 個 SFTS（例如 M=50 或 100），先把完全沒用的時間窗×頻帶×通道組整塊丟掉。

3. 再來有兩種選擇：

   * **版本 A（比較接近 FGSFT）：**

     * 以步長 D（例如 5 或 10）逐步合併 top SFTS，對每個「合併組合」訓練一個 linear SVM，記錄 accuracy，最後選 top-K 組合做 ensemble。
   * **版本 B（跟你現在架構比較相容）：**

     * 把「所有保留下來的 SFTS」的特徵 concat 起來（FTA+PTC），再跑一次 ReliefF，做 dimension-level 的 feature selection。

這樣你的 ReliefF 就不是在「一大堆可能完全沒用的時間×頻率×通道片段」上硬挑，而是只在「已經證明單獨就有點分類力的 SFTS 片段」上做更細緻的 feature-level 過濾。

這一層「先選片段，再選 feature 維度」其實就是 FGSFT 最重要的底層邏輯之一。

---

### 3. 把 CSP 類空間特徵加成一個新 feature family

你現在有：FTA + PTC (+ 可能 RG) → ReliefF。
可以考慮再加一個「簡化版 divCSP / RCSP」：

* 在每個 SFTS 上：

  * 只用那個 channel group 的 trials，算兩類 covariance；
  * 做 standard CSP 或 RCSP，取前後各 2 個 filter；
  * 算 log-variance 當特徵。

這樣每個 SFTS 的 feature vector 就變成：

> [FTA feature, PTC feature, CSP-logvar feature]

然後再走「SFTS wrapper → 保留 top SFTS → ReliefF / ensemble」的流程。

FGSFT 是完全靠 divCSP 在每個 SFTS 上抽特徵才有現在的表現；這樣做，你等於把它的空間優勢「借用」回來，又不必完全重寫你整個系統。

---

### 4. 調整 classifier：在高維特徵上優先試 linear SVM

在 FGSFT 裡，他們用的 classifier 很克制：**linear SVM（Matlab fitclinear 的預設參數）**，搭配很精挑細選的特徵。

你現在的 RBF SVM / RF / NB ensemble，可能在高維但樣本數不大的情況下：

* RBF SVM 很容易 overfit / 對 hyperparameters 敏感；
* RF / NB 對「高相關且高維」特徵也不一定友善。

建議你至少做一組實驗：

1. 用「SFTS wrapper + 保留 top SFTS + ReliefF」得到的特徵；
2. 分別試：

   * Linear SVM（C 做個小 grid）
   * RBF SVM（同樣的 CV）
   * RF
3. 用同一種 CV（例如 LOBO）比較 accuracy。

如果 linear SVM 可以馬上把你的 70% 拉高幾個百分點，就代表「特徵本身其實不差，只是 classifier 不夠 match」。

---

### 5. 把 FGSFT 的 ablation 思路搬過來做診斷

你可以照他們 Table I 的方式，針對你自己的方法做幾組：

* 只做 **時間 segmentation（T）** + 你的 FTA/PTC
* 只做 **頻段 segmentation（F）**（FDCC or filter bank）
* 只做 **空間 segmentation（S）**（多個 channel groups）
* 再試 SF / ST / FT / SFT
* 每一個都用同一種 classifier & CV

這樣你可以很清楚看到：

* 是不是和 FGSFT 一樣，「時間 segmentation」是最抬分的？
* 你的 FDCC 是否真的帶來明顯改善？
* 加 PTC 之前 vs 之後，哪一段真的有幫助？

做完這一輪，你就能比較有信心地說：「我的 pipeline 主要卡在時間窗設計？還是頻帶選法？還是空間資訊不夠？」

---

## 四、最後整理一個「實作優先順序」

如果要我幫你排優先順序，大概會是：

1. **先對齊實驗條件**

   * 只做 BCI-IV 2a 的左/右手二分類；
   * 用接近他們的 epoch（例如 0.5–2.5 s 或 0–1.5 s）和 LOBO。

2. **把你的 pipeline SFTS 化 + SFTS wrapper selection（用你現有的 FTA+PTC）**

   * 先不加 CSP，只用 wrapper 來選時間×頻帶×通道組。

3. **在保留的 SFTS 上再跑 ReliefF / linear SVM，看看 accuracy 有沒有明顯上來。**

4. **如果有進步，再加 CSP/RCSP 空間特徵，變成 (FTA + PTC + CSP) 的 multi-feature SFTS。**

這樣你可以一步一步接近 FGSFT 的「底層邏輯」，又保留你在 DSFE / FDCC / PTC 上已經建立好的架構。

如果你願意，把你現在程式裡「時間窗設計 + feature concat + ReliefF」的大致流程用 pseudo-code 打給我，我可以幫你改寫一個「加入 SFTS wrapper selection」的版本，讓你比較容易直接實作。
