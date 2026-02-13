

## 摘要

偽暫態延拓（Pseudo-Transient Continuation, PTC）常被用作 Newton 方法之全域化技術，以提升非線性橢圓型方程在遠離吸引域時的穩定性。然而，在 Jacobian 不定之情形下，PTC 之 shift 結構可能改變候選方向之幾何性質，使其在以 $M=\frac12|F|^2$ 為度量之框架下產生上坡方向。本文指出，對於由 $Jdx_N=-F$ 所定義之 Newton 候選方向，於該度量下通常具下降性質；但對於由 $(J+\frac1{dt}I)dx_P=-F$ 所定義之 PTC 候選方向，當算子不定時，可能出現幾何對齊指標 $ \mathcal{C}_{align}=\frac{\langle g,dx\rangle}{|g||dx|} > 0 $，即方向與梯度同向。此現象可在同時滿足線性審計指標 $ LinErr=\frac{|A(dx)+F|}{|F|} $ 極小之條件下發生，顯示其並非線性子問題失效，而是 globalization 機制之結構性缺口。

在嚴格 dual tolerance 接受條件 $ M_{new}\le M_{old}(1+10^{-12})+10^{-15} $ 下，Blind PTC 可能因連續拒絕而導致時間步長 $dt$ 收縮至下限，形成 dt-stagnation。為處理此治理缺口，本文提出 Koun-A1-GR 協議，透過幾何對齊審計與救援機制（Sentinel 與 Rescue），在不改變接受法律與線性求解容忍度之前提下，識別並修正上坡候選方向，使殘差恢復單調下降。

我們以三維拉伸網格下之 Hamiltonian constraint 方程為測試案例，進行 A/B/C 三模式對照實驗：Pure Newton 作為健康基準快速收斂；Blind PTC 出現幾何上坡並導致 dt-stagnation；Koun-A1-GR 則成功解除死鎖並持續推進系統。結果顯示，本文所提出之治理層並非替代 Newton，而是在不定算子下為 PTC globalization 提供一種結構化之審計與修復機制，為非線性方程之全域化分析提供新的方法論視角。




## 1. 引言

### 1.1 非線性橢圓方程與全域化問題

在求解非線性橢圓型偏微分方程時，Newton 方法因其局部二次收斂性而廣泛使用。對於形如 $F(u)=0$ 之離散系統，Newton 迭代可表示為解線性子問題 $J(u_k)dx_k=-F(u_k)$，並更新 $u_{k+1}=u_k+dx_k$。當初始猜測已位於吸引域內時，Newton 方法通常展現快速且穩定的收斂行為。然而，若初始點遠離吸引域，或 Jacobian 結構具不定性，則直接應用 Newton 可能出現震盪或發散，因而需要引入全域化（globalization）技術以改善穩定性。

偽暫態延拓（Pseudo-Transient Continuation, PTC）是一種常見的全域化策略，其核心思想為引入時間步長參數 $dt$，將線性子問題修正為 $(J+\frac1{dt}I)dx=-F$，並藉由控制 $dt$ 以調節更新方向的穩定性。當 $dt$ 較小時，修正項 $\frac1{dt}I$ 使系統趨近於梯度流行為；當 $dt$ 增大時，則逐漸回到 Newton 方向。此方法在多數定號問題中能有效提升收斂穩定性。

然而，對於 Jacobian 不定之問題，PTC 之 shift 結構可能改變候選方向與梯度之幾何關係。當以 $M=\frac12|F|^2$ 作為 merit function 時，其梯度為 $g=\nabla M$。在此度量下，方向 $dx$ 是否為下降方向，可由幾何對齊指標 $ \mathcal{C}*{align}=\frac{\langle g,dx\rangle}{|g||dx|} $ 判定。若 $\mathcal{C}*{align}<0$，則方向與梯度反向，為下降方向；若 $\mathcal{C}_{align}>0$，則為上坡方向。

本文的核心觀察在於：在不定算子條件下，PTC 候選方向可能出現 $\mathcal{C}_{align}>0$，即在 merit 度量下為幾何上坡方向，即使其線性審計指標 $ LinErr=\frac{|A(dx)+F|}{|F|} $ 仍然極小。此現象表明，問題並非源於線性求解誤差，而是 globalization 機制本身在不定結構下產生的幾何扭曲。

### 1.2 嚴格接受條件與死鎖現象

為避免以放寬接受條件掩蓋幾何問題，本文採用固定且嚴格的 dual tolerance 接受法律 $ M_{new}\le M_{old}(1+10^{-12})+10^{-15} $。在此條件下，若候選方向為上坡方向，則更新必然被拒絕。對 Blind PTC 而言，連續拒絕將導致時間步長 $dt$ 不斷縮小，最終達到下限並形成 dt-stagnation。此種死鎖並非物理模型失效，而是 globalization 結構在不定算子下的治理缺口。

### 1.3 本文貢獻與定位

本文的貢獻並非替代 Newton 方法，也非追求更快收斂，而是揭示並形式化一種在不定算子下可能出現的 globalization 幾何缺口，並提出對應之治理協議 Koun-A1-GR。該協議透過幾何對齊審計與救援機制，在不改變接受法律與線性求解容忍度之前提下，識別並修正上坡候選方向，使系統恢復單調下降。

為驗證此主張，我們設計 A/B/C 三模式對照實驗：Pure Newton 作為健康基準；Blind PTC 作為失效對照；Koun-A1-GR 作為治理層驗證。結果顯示，在相同模型、相同初始條件與相同接受法律下，治理層能有效解除 dt-stagnation 並持續推進殘差下降。

本文因此提供一種從幾何審計角度理解 globalization 結構失效的分析框架，並為非線性橢圓方程之全域化設計提出一種可操作的治理方法。







## 2. 問題設定與數學框架

### 2.1 Hamiltonian Constraint 方程

為具體呈現 globalization 在不定算子下之結構行為，本文選取一類具有非線性源項之橢圓型方程作為測試模型，其形式為
$$
\Delta \psi - 2\pi \rho \psi^5 = 0
$$
其中 $\psi$ 為未知函數，$\rho$ 為給定之密度分佈。該方程可視為一類 Hamiltonian constraint 的簡化模型，在高密度區域與強非線性耦合下，Jacobian 可能呈現不定結構。

設離散後之非線性系統為 $F(\psi)=0$，其 Jacobian 記為 $J(\psi)$。本文之分析與實驗均在固定離散與固定物理參數條件下進行，不隨模式改變。

邊界條件採用 $\psi=1$ 之 Dirichlet 型條件，並透過殘差遮罩方式將邊界點之殘差設為零，使內部區域主導非線性結構行為。

---

### 2.2 三維拉伸網格與離散化

為增加問題之結構張力，本文採用三維拉伸網格。設一維映射為
$$
x = L_{max} \frac{\sinh(\gamma \xi)}{\sinh(\gamma)}
$$
其中 $\xi \in [-1,1]$，$\gamma$ 為拉伸參數。當 $\gamma>0$ 時，網格點於中心區域更為密集，使高梯度區域之非線性行為更為顯著。

Laplacian 採二階差分離散，並透過張量映射方式構成三維算子。此離散方式雖非最高階精度，但足以呈現 globalization 在不定算子下的幾何現象，且保持實驗之可重現性與可控性。

---

### 2.3 Merit Function 與梯度結構

為統一分析框架，本文以
$$
M(\psi)=\frac12 |F(\psi)|^2
$$
作為 merit function。其梯度可表示為
$$
g = \nabla M(\psi)
$$
在理想精確線性解條件下，Newton 方向由
$$
J(\psi) dx_N = -F(\psi)
$$
給出。若 Jacobian 於當前點為適當條件，則 $dx_N$ 在 $M$ 度量下通常為下降方向。

對於 PTC，全域化修正後之候選方向由
$$
(J(\psi)+\frac1{dt}I) dx_P = -F(\psi)
$$
決定。當 $dt$ 變小時，$\frac1{dt}I$ 可能改變算子之特徵結構，進而改變方向之幾何性質。

---

### 2.4 幾何對齊指標與線性審計指標

為分析方向與梯度之關係，本文引入幾何對齊指標
$$
\mathcal{C}*{align} = \frac{\langle g, dx \rangle}{|g||dx|}
$$
當 $\mathcal{C}*{align}<0$ 時，方向為下降方向；當 $\mathcal{C}*{align}>0$ 時，方向為上坡方向；當 $\mathcal{C}*{align}\approx0$ 時，方向接近等高線切向。

為排除線性子問題未充分收斂之影響，另引入線性審計指標
$$
LinErr = \frac{|A(dx)+F|}{|F|}
$$
其中 $A$ 表示對應之線性算子。當 $LinErr$ 極小時，表示候選方向確實滿足線性子問題，幾何異常並非源於線性解不精確。

上述兩項指標共同構成本文所稱之幾何審計框架，用以區分 globalization 之結構問題與數值求解誤差。

---

## 3. Newton 與 PTC 候選方向的幾何差異

### 3.1 Newton 方向之下降性質

在 merit 度量 $M=\frac12|F|^2$ 下，Newton 方向滿足 $J dx_N=-F$。若 $J$ 在該點具適當性質，則可證明 $\langle g, dx_N\rangle<0$，即 $\mathcal{C}_{align}<0$。此性質解釋了 Newton 在吸引域內之快速下降行為。

實驗結果顯示，在本文所設定之高密度與拉伸條件下，Pure Newton 模式於五步內將殘差降低八個數量級，且對應 $\mathcal{C}_{align}$ 始終為負。此結果作為健康基準，排除了模型或離散結構失效之可能。

---

### 3.2 PTC 方向之幾何扭曲

對 PTC 而言，候選方向由 $(J+\frac1{dt}I) dx_P=-F$ 給出。當 $J$ 不定時，shift 項 $\frac1{dt}I$ 可能改變方向與梯度之幾何關係。即使 $LinErr$ 極小，亦可能出現 $\mathcal{C}_{align}>0$。

此現象意味著，候選方向在 merit 度量下為幾何上坡方向。若採用嚴格接受條件，則該方向必然被拒絕。若連續產生此類方向，則 $dt$ 將持續縮小並最終達到下限，形成 dt-stagnation。

本文將此現象定義為 globalization 在不定算子下的幾何缺口，而非線性求解誤差。

---





