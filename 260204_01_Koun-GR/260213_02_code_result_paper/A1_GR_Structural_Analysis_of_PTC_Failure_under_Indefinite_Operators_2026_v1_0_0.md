


---

## 摘要

偽暫態延拓（Pseudo-Transient Continuation, PTC）常被視為 Newton 方法之全域化技術，透過引入 shift 結構 $(J+\frac1{dt}I)$ 以提升數值穩定性。然而，當 Jacobian $J$ 為不定算子時，此 shift 結構可能改變候選方向在 merit 度量 $M=\frac12|F|^2$ 下之幾何性質。本文從線性代數與幾何動力學角度出發，分析 PTC 方向與梯度之對齊關係，並指出存在條件使得幾何對齊指標 $\mathcal{C}_{align}=\frac{\langle g,dx\rangle}{|g||dx|}>0$，即候選方向為上坡方向。

我們進一步證明，當 $J$ 具有跨越零點之特徵值結構時，shift 項 $\frac1{dt}I$ 雖可改變光譜分佈，卻不保證方向在 $M$ 度量下為下降方向。在嚴格接受條件 $M_{new}\le M_{old}(1+10^{-12})+10^{-15}$ 下，此幾何失配可能導致連續拒絕與 $dt$ 收縮，最終形成 dt-stagnation。此現象並非線性求解誤差所致，而為 globalization 結構在不定算子下的內在機制。

本文建立一個形式化框架，分析 Newton 方向 $Jdx_N=-F$ 與 PTC 方向 $(J+\frac1{dt}I)dx_P=-F$ 在 merit 動力學下之差異，並給出幾何失配產生之充分條件。透過理論推導與結構分類，本文揭示 PTC 失效的光譜來源，並說明為何僅依賴時間步長調整無法從根本上消除該缺口。

本文所建立之分析框架不僅適用於 Hamiltonian constraint 類型方程，亦可推廣至其他不定非線性系統之 globalization 設計，為求解器治理提供理論基礎。




## 1. 引言

### 1.1 全域化技術與不定算子問題

在求解非線性方程組 $F(u)=0$ 時，Newton 方法因其局部二次收斂性而成為標準工具。然而，當初始點遠離吸引域或 Jacobian $J(u)$ 具不定性時，直接使用 Newton 可能導致震盪或發散。為改善穩定性，常引入全域化技術，其中偽暫態延拓（Pseudo-Transient Continuation, PTC）是一種廣泛使用的方法。

PTC 的核心思想為解修正後的線性子問題
$$
(J(u_k)+\frac1{dt_k}I)dx_k=-F(u_k)
$$
並透過調整 $dt_k$ 以平衡穩定性與收斂速度。當 $dt_k$ 較小時，系統行為接近梯度流；當 $dt_k$ 較大時，則逐步回到 Newton 方向。

在 Jacobian 為正定或近正定情形下，PTC 通常能有效提升收斂穩定性。然而，當 $J$ 為不定算子時，shift 結構可能改變候選方向在 merit 度量下的幾何性質，使其不再為下降方向。

---

### 1.2 Merit 動力學視角

本文採用 merit function
$$
M(u)=\frac12|F(u)|^2
$$
作為分析框架。其梯度為
$$
g=\nabla M(u)
$$

方向 $dx$ 是否為下降方向，可由內積 $\langle g,dx\rangle$ 判定。定義幾何對齊指標
$$
\mathcal{C}*{align}=\frac{\langle g,dx\rangle}{|g||dx|}
$$
則 $\mathcal{C}*{align}<0$ 表示下降方向，$\mathcal{C}_{align}>0$ 表示上坡方向。

在此框架下，Newton 方向由
$$
Jdx_N=-F
$$
給出；PTC 方向由
$$
(J+\frac1{dt}I)dx_P=-F
$$
給出。本文的核心問題為：在何種條件下，$dx_P$ 可能滿足 $\mathcal{C}_{align}>0$？

---

### 1.3 本文貢獻

本文從光譜結構與幾何旋轉角度，形式化分析 PTC 在不定算子下的失效機制，並提出以下貢獻：

1. 推導在不定 $J$ 下，PTC 方向可能成為上坡方向的條件。
2. 解釋 shift 項 $\frac1{dt}I$ 為何無法保證下降性質。
3. 分析嚴格接受條件下 dt-stagnation 的動力學形成機制。
4. 建立一個可推廣至一般不定非線性系統的 globalization 結構分析框架。

本文不涉及新的數值協議或實驗實作，而專注於理論結構的形式化與分類。

---

## 2. Newton 與 PTC 方向的線性代數分析

### 2.1 Newton 方向之下降性質

設 $F(u)$ 在當前點之 Jacobian 為 $J$，則 Newton 方向滿足
$$
Jdx_N=-F
$$

在 merit 度量下，若 $J$ 與 $g$ 具有一致結構，則可得
$$
\langle g,dx_N\rangle < 0
$$
此性質源於 Newton 方向近似解決最速下降方向與曲率校正的結合，並解釋其在吸引域內的快速收斂行為。

---

### 2.2 PTC 方向之光譜變換

PTC 方向滿足
$$
(J+\frac1{dt}I)dx_P=-F
$$

若 $J$ 具有特徵分解 $J=Q\Lambda Q^{-1}$，則
$$
J+\frac1{dt}I=Q(\Lambda+\frac1{dt}I)Q^{-1}
$$

當 $J$ 存在正負特徵值混合時，shift 項僅將所有特徵值平移 $\frac1{dt}$，但不改變其符號分佈的相對結構。若原系統存在接近零或跨越零點的特徵值，則方向可能產生顯著旋轉。

此旋轉意味著 $dx_P$ 與 $g$ 之間的夾角可能進入上半平面，即
$$
\langle g,dx_P\rangle>0
$$

此即幾何失配現象。

---





