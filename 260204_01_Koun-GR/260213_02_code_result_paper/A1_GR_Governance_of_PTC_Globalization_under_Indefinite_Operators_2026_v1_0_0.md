
---

論文英文文件名：A1_GR_Governance_of_PTC_Globalization_under_Indefinite_Operators_2026_v1_0_0.md

中文標題：不定算子下偽暫態延拓全域化的治理缺口及其結構修復機制
英文標題：Governance of Pseudo-Transient Continuation Globalization under Indefinite Operators: Geometric Alignment Audit and Deadlock Resolution via Koun-A1-GR

作者（Author）：Shu Koun
作者拼音（Pinyin）：Shu Koun
作者姓名音標（Pronunciation）：/ʃu koʊn/
通訊作者（Corresponding Author）：Shu Koun

日期（Date）：2026-02-13
版本（Version）：v1.0.0
語言（Language）：zh-Hant（主文），en（標題與關鍵欄位）

領域（Fields）：Numerical Methods；Computational Physics；Nonlinear Solvers；Scientific Computing

摘要（Abstract, zh-Hant）：本文研究偽暫態延拓（Pseudo-Transient Continuation, PTC）作為 Newton 方法全域化策略時，在 Jacobian 不定條件下可能出現的幾何治理缺口。本文指出 PTC shift 方向由 $(J+\frac1{dt}I)dx=-F$ 定義時，可能在以 $M=\frac12|F|^2$ 為度量下產生幾何上坡方向，其幾何對齊指標為 $\mathcal{C}*{align}=\frac{\langle g,dx\rangle}{|g||dx|}>0$，即使線性審計指標 $LinErr=\frac{|A(dx)+F|}{|F|}$ 仍極小。於嚴格 dual tolerance 接受條件 $M*{new}\le M_{old}(1+10^{-12})+10^{-15}$ 下，Blind PTC 可能因連續拒絕導致 $dt$ 收縮至下限並形成 dt-stagnation。本文提出 Koun-A1-GR 治理協議，透過幾何審計與救援機制在不更改接受法律與線性求解容忍度之前提下解除死鎖並恢復單調下降。A/B/C 三模式對照實驗顯示 Pure Newton 快速收斂，Blind PTC 出現 dt-stagnation，而 Koun-A1-GR 能解除死鎖並持續推進殘差下降，提供不定算子下 globalization 結構分析與修復的可操作框架。

關鍵詞（Keywords, en）：pseudo-transient continuation；globalization；indefinite operator；Hamiltonian constraint；merit function；geometric alignment；deadlock；solver governance；Koun-A1-GR

關鍵詞（Keywords, zh-Hant）：偽暫態延拓；全域化；不定算子；Hamiltonian constraint；merit function；幾何對齊；死鎖；求解器治理；Koun-A1-GR

分類（Subjects）：65N30；65H10；65F10；35J60

核心術語與符號（Core Terms and Notation）：

1. 殘差：$F(\psi)$
2. Jacobian：$J(\psi)$
3. Merit function：$M(\psi)=\frac12|F(\psi)|^2$
4. 梯度：$g=\nabla M(\psi)$
5. Newton 候選方向：$Jdx_N=-F$
6. PTC 候選方向：$(J+\frac1{dt}I)dx_P=-F$
7. 幾何對齊指標：$\mathcal{C}_{align}=\frac{\langle g,dx\rangle}{|g||dx|}$
8. 線性審計指標：$LinErr=\frac{|A(dx)+F|}{|F|}$
9. 嚴格接受法律：$M_{new}\le M_{old}(1+10^{-12})+10^{-15}$
10. 失效型態：dt-stagnation

程式與可重現性（Code and Reproducibility）：
參考實作檔名：koun_a1_gr_final.py
運行模式：Mode A（Pure Newton），Mode B（Blind PTC），Mode C（Koun-A1-GR）
輸出建議圖表：ResNorm vs Iter；$\mathcal{C}_{align}$ vs Iter；$dt$ vs Iter

授權（License）：All rights reserved by the author unless explicitly stated otherwise.

引用建議（Suggested Citation, en）：Shu Koun. Governance of Pseudo-Transient Continuation Globalization under Indefinite Operators: Geometric Alignment Audit and Deadlock Resolution via Koun-A1-GR. Version v1.0.0, 2026-02-13.

---



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




## 4. Dual Tolerance 接受法律與實驗凍結原則

### 4.1 嚴格接受條件的形式

為避免以放寬條件掩蓋幾何問題，本文在所有模式中採用固定且嚴格的 dual tolerance 接受法律。其形式為
$$
M_{new} \le M_{old}(1+10^{-12})+10^{-15}
$$
其中 $M=\frac12|F|^2$ 為 merit function。此條件允許極小量級的數值浮動，但不容許真正的上升行為。

在此法律下，若候選方向滿足 $\mathcal{C}*{align}>0$，則 $M*{new}>M_{old}$ 幾乎不可避免，更新將被拒絕。此設計確保實驗結果反映方向之幾何性質，而非接受條件的鬆動。

---

### 4.2 為何不放寬接受條件

在實務計算中，常見作法是當更新被拒絕時放寬接受條件或加入額外阻尼。然而，此種策略可能掩蓋 globalization 在不定算子下之結構缺口，使幾何問題被誤判為數值震盪。

本文刻意凍結接受法律，不調整參數，不改變線性求解容忍度，目的在於純粹呈現幾何扭曲本身的影響。所有模式均在相同接受法律下進行，確保 A/B/C 對照具可比性。

---

### 4.3 dt-stagnation 的形成機制

在 Blind PTC 模式下，當候選方向為上坡方向時，更新將被拒絕，並導致時間步長 $dt$ 縮小。若縮小後仍產生 $\mathcal{C}_{align}>0$，則更新再次被拒絕，形成連續拒絕循環。當 $dt$ 達到下限時，演算法無法繼續推進，形成 dt-stagnation。

此停滯並非因物理模型無解，也非因線性子問題未收斂，而是 globalization shift 在不定算子下導致的方向幾何異常。此點為本文之核心觀察。

---

## 5. 治理協議：Koun-A1-GR

### 5.1 幾何審計層（Sentinel）

Koun-A1-GR 在每次更新前進行幾何審計，計算 $\mathcal{C}_{align}$ 並判定方向類型：

* 當 $\mathcal{C}_{align}<0$ 時，方向為下降方向，維持標準更新。
* 當 $\mathcal{C}_{align}>0$ 時，判定為 Type-1，上坡方向。
* 當 $\mathcal{C}_{align}\approx0$ 時，判定為近等高線方向。

此審計層不改變線性子問題本身，而僅對候選方向進行幾何分類。

---

### 5.2 Rescue 機制

當檢測到 Type-1 上坡方向時，Koun-A1-GR 不直接使用該方向，而改採梯度流方向作為修正更新，即使用與 $g$ 反向之方向進行小步長更新。

此修正不依賴放寬接受條件，亦不改變 dual tolerance 法律，而是透過幾何修正恢復下降趨勢。當殘差重新進入下降區域後，系統可逐步回到 PTC 行為。

---

### 5.3 與 Blind PTC 的差異

Blind PTC 在檢測到更新失敗時僅透過縮小 $dt$ 嘗試修正，但不審計方向幾何性質，因此可能在不定算子區域反覆生成上坡方向。

Koun-A1-GR 的核心差異在於增加幾何審計層，使 globalization 不再僅依賴步長調整，而具備結構化的方向治理能力。此治理層並非替代 Newton，而是在 PTC globalization 框架中補上幾何缺口。

---




## 6. A/B/C 三模式對照實驗

### 6.1 實驗設計與固定條件

為驗證前述幾何分析與治理協議之有效性，本文設計三種模式於相同物理與數值條件下進行對照實驗。所有模式共用下列固定條件：

* 相同方程與離散方式
* 相同三維拉伸網格與拉伸參數
* 相同初始猜測
* 相同線性求解容忍度
* 相同 dual tolerance 接受法律
* 相同最大迭代次數

此設計確保差異僅來自於 globalization 策略本身，而非其他數值因素。

三種模式定義如下：

* Mode A：Pure Newton
* Mode B：Blind PTC
* Mode C：Koun-A1-GR

---

### 6.2 Mode A：Pure Newton 作為健康基準

Pure Newton 模式解線性子問題 $J dx_N=-F$，並於更新時採用線搜尋以保證接受條件。

實驗結果顯示，殘差於五步內由初始量級 $10^{-1}$ 降至 $10^{-11}$，共下降約八個數量級。對應之 $\mathcal{C}_{align}$ 始終為負，顯示方向在 merit 度量下為穩定下降方向。

此結果證明：

1. 物理模型與離散結構並無內在病態。
2. Jacobian 雖可能不定，但系統仍存在吸引域。
3. Newton 在吸引域內表現強勁且穩定。

因此，後續失效現象不可歸因於模型本身。

---

### 6.3 Mode B：Blind PTC 與 dt-stagnation

Blind PTC 模式解 $(J+\frac1{dt}I)dx_P=-F$，並於更新失敗時縮小 $dt$。

實驗顯示：

* $\mathcal{C}_{align}$ 約為正值且顯著大於零。
* $LinErr$ 維持極小，表示線性子問題解精確。
* 每次更新均違反接受條件而被拒絕。
* $dt$ 持續縮小至下限。

最終形成 dt-stagnation，殘差無法下降。

此結果證明，在不定算子條件下，PTC shift 可能生成幾何上坡方向，即使線性子問題本身無誤。此現象為 globalization 結構之缺口，而非數值誤差。

---

### 6.4 Mode C：Koun-A1-GR 治理結果

Koun-A1-GR 模式在每次更新前進行幾何審計，當檢測到 $\mathcal{C}_{align}>0$ 時啟動 Rescue 機制。

實驗結果顯示：

* Type-1 上坡方向被識別。
* Rescue 更新恢復下降趨勢。
* 殘差自初始值約 $0.8$ 持續下降至約 $0.25$。
* 系統未出現 dt-stagnation。

值得注意的是，在某些迭代中仍出現拒絕與 $dt$ 調整，但整體殘差呈現單調下降趨勢，顯示治理層成功解除死鎖。

---

### 6.5 圖像化證據

為清楚呈現三模式差異，本文繪製下列圖像：

1. ResNorm 與迭代步數之關係圖。
2. $\mathcal{C}_{align}$ 與迭代步數之關係圖。
3. $dt$ 與迭代步數之關係圖。

圖像顯示：

* Mode A 快速下降。
* Mode B 在高正 $\mathcal{C}_{align}$ 區域停滯。
* Mode C 在識別 Type-1 後恢復下降。

此三圖構成完整證據鏈，支持本文所提出之幾何缺口與治理修復主張。

---

## 7. 討論

### 7.1 Newton 之角色

本文結果並未否定 Newton 方法之有效性。相反地，Pure Newton 作為基準證明系統於吸引域內具有穩定收斂行為。本文所揭示之問題並非 Newton 本身，而是 PTC globalization 在不定算子下之結構行為。

---

### 7.2 治理層之意義

Koun-A1-GR 並非替代 Newton，也非追求更快收斂，而是提供 globalization 的幾何審計與修復層。其意義在於：

* 將方向幾何性質顯性化。
* 區分線性誤差與幾何失效。
* 在不改變接受法律下解除死鎖。

此觀點為 globalization 設計提供新的方法論框架。

---

### 7.3 限制與未來方向

本文實驗基於單一方程與特定參數設定。未來工作可延伸至：

* 不同物理模型。
* 更高解析度網格。
* 其他不定算子問題。

此外，治理層思想亦可能應用於其他全域化策略之分析與改進。

---

## 8. 結論

本文指出，在不定算子條件下，PTC globalization 可能生成幾何上坡方向，即 $\mathcal{C}_{align}>0$，即使線性審計指標 $LinErr$ 極小。在嚴格接受條件下，此現象可能導致 dt-stagnation。

透過引入幾何審計與 Rescue 機制，Koun-A1-GR 能在不改變接受法律與線性容忍度之前提下解除死鎖並恢復下降趨勢。

本文因此提供一種結構化的 globalization 治理方法，並為不定算子下之非線性方程求解提供新的分析視角。
