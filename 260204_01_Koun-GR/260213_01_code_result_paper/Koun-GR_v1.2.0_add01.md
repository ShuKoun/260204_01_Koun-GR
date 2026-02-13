
### 3.3 A/B/C 三模式對照實驗：PTC 治理的必要性

為驗證 Koun-A1 框架在極端條件下的治理效能，我們設計了針對 Hamiltonian 約束的高壓測試場景（$\rho_{\text{peak}}=10.0, \Gamma=2.0$, Init='dip'）。在此場景下，標準 Newton 法（Mode A）因處於吸引域內而表現優異（5步收斂），這證實了物理模型本身的良態性。然而，當我們強制使用偽瞬態延拓（PTC）以模擬工程中常見的時間演化需求或惡劣初值情境時，未受治理的 PTC（Mode B）與受治理的 PTC（Mode C, Koun-A1）展現出截然不同的動力學行為。

**實驗結果摘要：**

| 模式 | 策略描述 | 最終殘差 (ResNorm) | 狀態 | 結構特徵 |
| :--- | :--- | :--- | :--- | :--- |
| **Mode A** | Pure Newton + Line Search | $2.59 \times 10^{-11}$ | **CONVERGED** | $N_{\text{Cos}} < 0$ (幾何一致) |
| **Mode B** | Blind PTC (No Sentinel) | $8.06 \times 10^{-1}$ | **STAGNATION** | $P_{\text{Cos}} > 0$ (幾何逆行), DT 死鎖 |
| **Mode C** | Governed PTC (Koun-A1) | $1.78 \times 10^{-1}$ | **EVOLVING** | Sentinel 觸發 Rescue，突破平台 |

**現象學分析：**

1.  **PTC 的幾何陷阱 (The PTC Trap)：**
    Mode B 的失敗並非源於步長過大，而是源於方向的結構性錯誤。日誌顯示，在 $dt$ 縮小過程中，PTC 算子 $(J + \frac{1}{dt}I)$ 產生了與目標函數幾何逆行的方向（$P_{\text{Cos}} > 0$），且線性殘差極小（$\text{LinErr} \approx 10^{-16}$）。這證明了該逆行並非數值誤差，而是 PTC 全局化策略在不定系統（Indefinite System）中固有的幾何扭曲。盲目接受此方向導致算法陷入死鎖。

2.  **治理的價值 (The Value of Governance)：**
    Mode C 在同樣的物理條件與接受法則下，透過結構型哨兵（Structural Sentinel）識別出了上述幾何逆行，並自動切換至伴隨矯正（Adjoint Rescue, $d \propto -g$）。雖然 Sniper 步的收斂速度低於 Newton（一階 vs 二階），但它成功地將系統殘差從 $0.806$ 壓低至 $0.178$，打破了 Mode B 的死鎖狀態，使系統重回可演化軌道。

**結論：**
本實驗證明，雖然 Newton 法在理想條件下效率最高，但在必須採用 PTC 策略的場景中（如初值極差或需路徑演化），Koun-A1 提供的結構治理是防止算法崩潰的必要條件。它將「死鎖」轉化為「可控的慢速推進」，為後續接入高階加速器提供了可能性。

