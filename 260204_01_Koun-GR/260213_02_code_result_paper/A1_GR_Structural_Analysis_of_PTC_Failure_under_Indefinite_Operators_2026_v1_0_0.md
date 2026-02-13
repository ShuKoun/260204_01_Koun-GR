


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





## 3. 幾何失配條件之形式化分析

### 3.1 幾何對齊與方向旋轉

在 merit 度量 $M=\frac12|F|^2$ 下，梯度為 $g=\nabla M$。對任意候選方向 $dx$，其瞬時變化率可表示為
$$
\frac{d}{d\alpha} M(u+\alpha dx)\big|_{\alpha=0}=\langle g,dx\rangle
$$
因此 $\langle g,dx\rangle<0$ 為下降方向之必要條件。

對 Newton 方向 $dx_N$ 而言，若 $J$ 與 $g$ 在局部具有一致的曲率結構，則 $dx_N$ 在 $M$ 度量下通常為下降方向。然而，對 PTC 方向 $dx_P$，由於解的是
$$
(J+\frac1{dt}I)dx_P=-F
$$
其方向由改變後之算子決定。當 $J$ 為不定算子時，$J$ 的正負特徵子空間可能對 $dx_P$ 產生非對稱縮放，使其偏離原本下降方向。

若存在特徵值 $\lambda_i$ 使得 $\lambda_i \approx -\frac1{dt}$，則對應特徵子空間之分量將被放大，導致方向顯著旋轉。此時即可能出現
$$
\mathcal{C}_{align}=\frac{\langle g,dx_P\rangle}{|g||dx_P|}>0
$$
即 PTC 方向成為上坡方向。

---

### 3.2 幾何失配的充分條件

設 $J$ 之特徵值包含正負混合，且存在 $\lambda_{min}<0<\lambda_{max}$。當 $dt$ 滿足
$$
-\lambda_{min} < \frac1{dt} < \lambda_{max}
$$
則 shift 後之算子 $(J+\frac1{dt}I)$ 仍可能保持不定性。於此情形下，方向 $dx_P$ 可能跨越 $M$ 度量之下降區域，進入 $\langle g,dx_P\rangle>0$ 的區域。

此條件說明，單純增加或減少 $dt$ 並不保證下降性質。當 $J$ 之光譜跨越零點時，PTC 方向可能在特定區間內產生幾何失配。

---

### 3.3 與線性審計無關性

值得注意的是，幾何失配並不意味線性子問題未充分收斂。若 $dx_P$ 精確滿足 $(J+\frac1{dt}I)dx_P=-F$，則線性殘差指標
$$
LinErr=\frac{|A(dx_P)+F|}{|F|}
$$
可極小。

因此，$\mathcal{C}_{align}>0$ 與 $LinErr$ 極小可同時成立。此現象顯示幾何失配為 globalization 結構問題，而非線性求解誤差。

---

## 4. dt-stagnation 的動力學模型

### 4.1 嚴格接受條件下的迭代行為

在嚴格接受條件
$$
M_{new}\le M_{old}(1+10^{-12})+10^{-15}
$$
下，若 $\mathcal{C}_{align}>0$，則更新幾乎必然被拒絕。對 Blind PTC 而言，拒絕後將縮小 $dt$，再重新計算方向。

若縮小 $dt$ 仍落在幾何失配區間內，則 $\mathcal{C}_{align}>0$ 之情況將重複出現，形成連續拒絕。

---

### 4.2 dt 收縮與固定點

設 $dt_{k+1}=\beta dt_k$，其中 $0<\beta<1$ 為縮減比例。若在某區間內所有 $dt$ 均導致 $\mathcal{C}*{align}>0$，則迭代將使 $dt$ 收斂至下限 $dt*{min}$。

當 $dt=dt_{min}$ 且更新仍被拒絕時，系統進入固定點狀態，即 dt-stagnation。此時殘差停滯，迭代無法前進。

---

### 4.3 結構性解釋

dt-stagnation 並非隨機現象，而是由以下三個條件共同導致：

1. Jacobian 不定且光譜跨越零點。
2. shift 結構未消除不定性。
3. 嚴格接受條件拒絕上坡方向。

此三條件構成 globalization 的結構缺口。只調整 $dt$ 並不足以跨越該缺口，必須改變方向幾何性質。

---

## 5. 推廣與一般性意義

### 5.1 不限於 Hamiltonian Constraint

上述分析不依賴特定方程形式。只要非線性系統之 Jacobian 具不定性，且採用 PTC 型全域化策略，皆可能出現類似幾何失配。

因此，該機制具有一般性，適用於廣泛的不定非線性問題。

---

### 5.2 對求解器設計的啟示

傳統 globalization 多聚焦於步長控制與殘差監測。然而本文分析顯示，方向幾何性質本身應成為顯性監測對象。

透過審計 $\mathcal{C}_{align}$，可提前識別潛在上坡方向，避免單純依賴 $dt$ 調整而陷入死鎖。此觀點為求解器治理提供新的設計原則。

---

## 6. 結論

本文從光譜與幾何角度分析 PTC 在不定算子下的失效機制，證明存在條件使得 PTC 方向滿足 $\mathcal{C}_{align}>0$，即在 merit 度量下為上坡方向。此現象可在 $LinErr$ 極小的情況下發生，顯示其並非線性誤差所致。

在嚴格接受條件下，幾何失配可能導致 dt-stagnation。此機制為 globalization 結構之內在缺口，而非偶發數值現象。

本文建立之理論框架為後續治理協議提供基礎，並為不定非線性系統之全域化設計提供結構性分析工具。
