

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








