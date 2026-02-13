


---

## 摘要

偽暫態延拓（Pseudo-Transient Continuation, PTC）常被視為 Newton 方法之全域化技術，透過引入 shift 結構 $(J+\frac1{dt}I)$ 以提升數值穩定性。然而，當 Jacobian $J$ 為不定算子時，此 shift 結構可能改變候選方向在 merit 度量 $M=\frac12|F|^2$ 下之幾何性質。本文從線性代數與幾何動力學角度出發，分析 PTC 方向與梯度之對齊關係，並指出存在條件使得幾何對齊指標 $\mathcal{C}_{align}=\frac{\langle g,dx\rangle}{|g||dx|}>0$，即候選方向為上坡方向。

我們進一步證明，當 $J$ 具有跨越零點之特徵值結構時，shift 項 $\frac1{dt}I$ 雖可改變光譜分佈，卻不保證方向在 $M$ 度量下為下降方向。在嚴格接受條件 $M_{new}\le M_{old}(1+10^{-12})+10^{-15}$ 下，此幾何失配可能導致連續拒絕與 $dt$ 收縮，最終形成 dt-stagnation。此現象並非線性求解誤差所致，而為 globalization 結構在不定算子下的內在機制。

本文建立一個形式化框架，分析 Newton 方向 $Jdx_N=-F$ 與 PTC 方向 $(J+\frac1{dt}I)dx_P=-F$ 在 merit 動力學下之差異，並給出幾何失配產生之充分條件。透過理論推導與結構分類，本文揭示 PTC 失效的光譜來源，並說明為何僅依賴時間步長調整無法從根本上消除該缺口。

本文所建立之分析框架不僅適用於 Hamiltonian constraint 類型方程，亦可推廣至其他不定非線性系統之 globalization 設計，為求解器治理提供理論基礎。






