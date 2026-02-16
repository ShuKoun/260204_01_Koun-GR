
---

# 英文檔名

resolver-ptc-newton-krylov-globalization-architecture-v1

---

# 中文標題

解析器架構論文：基於偽時間延拓之 PTC-Newton-Krylov 全局化求解框架

---

# 英文標題

Resolver Architecture Paper: A Pseudo-Transient Newton–Krylov Globalization Framework for Nonlinear Elliptic Systems

---

# 摘要

本研究提出一種面向強非線性橢圓型偏微分方程之全局化求解框架。該框架以 Jacobian-free Newton–Krylov 方法為基礎，結合偽時間延拓機制（Pseudo-Transient Continuation, PTC）、尺度自適應對角預條件，以及幀內重試策略，構成一個閉環的可解性治理結構。不同於傳統靜態牛頓法假設局部線性化在全域範圍內有效，本框架在高度各向異性與譜不定區域中優先維持數值穩定性與單調可接受性。本文系統性描述該求解器之架構設計原則、動態步長調度策略與失敗恢復機制，並展示其在強拉伸網格下非線性哈密頓約束模型中的行為特徵。研究表明，全局可解性在某些極端 regime 下不再是連續可調參量，而需透過動態正則化才能維持。

---

# 1. 引言

## 1.1 問題背景

考慮如下半線性橢圓型偏微分方程：

$$
F(\psi) = \mathcal{L}(\psi) - S(\psi) = 0
$$

其中 $\mathcal{L}$ 為幾何拉普拉斯型算子，$S(\psi)$ 為非線性源項。在廣義相對論初始資料構造中，該形式對應於哈密頓約束方程。在其他物理與工程場景中，亦出現在反應–擴散系統與非線性勢場問題。

在高度各向異性網格（例如經由雙曲正弦映射產生之拉伸網格）下，算子係數退化，Jacobian 條件數急劇惡化，靜態牛頓法常出現如下行為：

* 線性化方向失真
* 信任域急劇縮小
* 步長反覆縮減後停滯
* 殘差無法持續下降

## 1.2 傳統求解範式

傳統 Newton–Raphson 迭代可寫為：

$$
J(\psi_k),\delta_k = -F(\psi_k)
$$

$$
\psi_{k+1} = \psi_k + \delta_k
$$

為避免發散，常配合 line search 或 trust region 技術。然而在極端各向異性 regime 下，即便縮步亦可能陷入凍結區域。

---

# 2. 求解器架構總覽

本求解器採用如下三層結構：

1. Jacobian-free Newton–Krylov 主體
2. 偽時間延拓全局化
3. 失敗恢復與尺度自適應穩定化

---

# 3. 偽時間延拓（PTC）機制

將靜態問題嵌入動態形式：

$$
\frac{\partial \psi}{\partial \tau} = -F(\psi)
$$

離散後得到修正線性系統：

$$
\left(J + \frac{1}{\Delta t} I\right)\delta = -F
$$

其中 $\Delta t$ 為偽時間步長。

### 3.1 譜平移效應

若 $J$ 存在接近零或負特徵值，則加入 $\frac{1}{\Delta t}I$ 將產生譜平移：

$$
\lambda_i^{\text{shifted}} = \lambda_i + \frac{1}{\Delta t}
$$

當 $\Delta t$ 足夠小時，可強制算子恢復正定性，提升可解性。

---

# 4. 對角預條件與尺度自適應

定義 Jacobian 對角近似：

$$
J_{\text{diag}} \approx \text{diag}(\mathcal{L}) - S'(\psi)
$$

為避免局部退化，採用絕對值穩定化：

$$
J_{\text{abs}} = \max\left(|J_{\text{diag}}|,\ \varepsilon_{\text{local}}\right)
$$

其中：

$$
\varepsilon_{\text{local}} = 10^{-8}\left(|\text{diag}(\mathcal{L})| + |S'(\psi)|\right) + 10^{-15}
$$

最終預條件形式為：

$$
M^{-1} r = \frac{r}{J_{\text{abs}} + \frac{1}{\Delta t}}
$$

---

# 5. 幀內重試與步長調度

外層非線性迭代中，對於每次嘗試更新：

* 若殘差滿足：

$$
|F_{\text{new}}| < |F_{\text{old}}|(1 + 10^{-6})
$$

則接受更新，並令：

$$
\Delta t \leftarrow \min(1.1,\Delta t,\ \Delta t_{\max})
$$

* 若失敗，則立即縮減步長並重試：

$$
\Delta t \leftarrow \max(0.5,\Delta t,\ \Delta t_{\min})
$$

該策略避免將失敗推遲至下一外層迭代，減少空轉。

---

# 6. 行為特徵

在中等拉伸參數 $\gamma$ 下，觀察到：

* 初期殘差單調下降
* 進入困難區後 $\Delta t$ 自動縮小
* 殘差可能出現平台，但系統保持數值穩定
* 不產生爆炸或不可逆發散

此現象表明該架構優先維持可解性，而非追求局部最快下降。

---

# 7. 與主流方法對比

| 方法           | 局部收斂 | 全局穩定性 | 結構治理 |
| ------------ | ---- | ----- | ---- |
| 靜態 Newton    | 高    | 弱     | 無    |
| Trust Region | 中    | 中     | 部分   |
| 多重網格         | 高    | 依問題   | 依實作  |
| 本架構          | 中    | 強     | 明確閉環 |

---

# 8. 結論

本文提出之求解器並未發明新數學算子，而是在高度各向異性與強非線性條件下，構建了一種簡潔而穩健的全局化架構。其核心價值在於將失敗恢復與步長調度整合為閉環控制機制，使非線性橢圓系統在困難 regime 下仍具可推進性。未來工作包括更高 $\gamma$ 區間驗證、雙黑洞初始資料應用，以及理論收斂性條件之形式化分析。

---

