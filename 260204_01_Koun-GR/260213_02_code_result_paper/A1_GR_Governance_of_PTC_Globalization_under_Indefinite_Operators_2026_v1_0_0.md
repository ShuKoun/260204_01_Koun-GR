

## 摘要

偽暫態延拓（Pseudo-Transient Continuation, PTC）常被用作 Newton 方法之全域化技術，以提升非線性橢圓型方程在遠離吸引域時的穩定性。然而，在 Jacobian 不定之情形下，PTC 之 shift 結構可能改變候選方向之幾何性質，使其在以 $M=\frac12|F|^2$ 為度量之框架下產生上坡方向。本文指出，對於由 $Jdx_N=-F$ 所定義之 Newton 候選方向，於該度量下通常具下降性質；但對於由 $(J+\frac1{dt}I)dx_P=-F$ 所定義之 PTC 候選方向，當算子不定時，可能出現幾何對齊指標 $ \mathcal{C}_{align}=\frac{\langle g,dx\rangle}{|g||dx|} > 0 $，即方向與梯度同向。此現象可在同時滿足線性審計指標 $ LinErr=\frac{|A(dx)+F|}{|F|} $ 極小之條件下發生，顯示其並非線性子問題失效，而是 globalization 機制之結構性缺口。

在嚴格 dual tolerance 接受條件 $ M_{new}\le M_{old}(1+10^{-12})+10^{-15} $ 下，Blind PTC 可能因連續拒絕而導致時間步長 $dt$ 收縮至下限，形成 dt-stagnation。為處理此治理缺口，本文提出 Koun-A1-GR 協議，透過幾何對齊審計與救援機制（Sentinel 與 Rescue），在不改變接受法律與線性求解容忍度之前提下，識別並修正上坡候選方向，使殘差恢復單調下降。

我們以三維拉伸網格下之 Hamiltonian constraint 方程為測試案例，進行 A/B/C 三模式對照實驗：Pure Newton 作為健康基準快速收斂；Blind PTC 出現幾何上坡並導致 dt-stagnation；Koun-A1-GR 則成功解除死鎖並持續推進系統。結果顯示，本文所提出之治理層並非替代 Newton，而是在不定算子下為 PTC globalization 提供一種結構化之審計與修復機制，為非線性方程之全域化分析提供新的方法論視角。




