# 機器學習課程作業：多元與簡單線性回歸分析工作坊 (ML Linear Regression Workshop)

本專案是一個遵循 **CRISP-DM (Cross-Industry Standard Process for Data Mining)** 資料科學標準流程開發的互動式機器學習回歸分析教學網站。專案具備純前端執行能力，使用純原生 JavaScript 實現最小平方法 (Ordinary Least Squares, OLS)、矩陣運算與嚴謹的 Student's t 分布統計推論，**無須 Python 後端伺服器或任何 API Key**，即可在 `localhost` 與 `GitHub Pages` 上流暢運行。

---

## 🌟 核心功能特色

### 1. 兩種 Problem 模型架構切換
- **簡單線性回歸 (Simple Linear Regression)**：
  - 探討單一自變數 $x$ 與目標變數 $y$ 的線性關係：$y = \beta_0 + \beta_1 x + \epsilon$。
  - 特徵選擇區自動限制為單一特徵。
  - 圖表呈現 2D 散佈圖、迴歸線、**95% 信賴帶 (Confidence Band)** 與 **95% 預測帶 (Prediction Band)**。
- **多元線性回歸 (Multiple Linear Regression)**：
  - 探討多個自變數對目標變數的綜合影響：$y = \beta_0 + \beta_1 x_1 + \dots + \beta_p x_p + \epsilon$。
  - 支援多特徵勾選、類別變數 One-Hot Dummy Encoding（自動排除基準類別避免虛擬變數陷阱）。
  - 圖表呈現實際值 vs 預測值 (Actual vs Predicted) 散佈圖（含 45° 完美預測線）與殘差圖 (Residual Plot)。

### 2. 自訂資料集 (Custom Dataset) 與 CORS 友善防護
- **內建經典資料集**：
  - `50 Startups`：50 筆新創公司研發、行政、行銷費用與利潤，包含類別特徵 State。
  - `Advertising`：200 筆廣告預算 (TV, Radio, Newspaper) 與銷量 (Sales)。
- **本機檔案上傳**：支援點擊與拖放 (Drag & Drop) 上傳任意 CSV 檔案。
- **公開 CSV 網址載入**：支援輸入 Direct URL 載入資料。若遭遇瀏覽器跨來源資源共享限制 (CORS)，系統會自動攔截並顯示清楚的錯誤解說與下載引導，**絕不偽裝載入成功**。
- **自動探索與摘要**：即時分析資料筆數、欄位總數、數值欄位數、缺失值總數，並預覽前 5 筆資料。

### 3. 模型訓練與嚴格防止資料洩漏 (Zero Data Leakage)
- **固定隨機數種子 (Random Seed)**：內建 Mulberry32 確定性虛擬隨機數生成器，固定資料切分，避免每次隨機跳動。
- **資料前處理**：支援均值填補 (Mean Imputation) 或剔除空值。所有統計均值與編碼規則**嚴格僅在 Training Set 上計算**，杜絕資料洩漏。
- **特徵選取 (Feature Selection)**：
  - 手動選取 (Manual)。
  - 相關係數篩選 (Pearson Correlation Filter - Top K)：僅以 Training Set 計算 $|r|$。
  - 向前逐步選取 (Forward Stepwise Selection)：在 Training Set 上以 $R^2$ 增益逐步加入特徵。
- **奇異矩陣與共線性防護**：若矩陣不可逆（如完全共線性或樣本數 $n \le p + 1$），系統提供具備統計學教育意義的錯誤提示。

### 4. 完整的模型評估指標 (Model Evaluation)
- 同步展示**訓練集 (Train)** 與**測試集 (Test)** 的 $R^2$、Adjusted $R^2$、RMSE、MAE，方便檢驗過擬合 (Overfitting)。
- 輸出完整迴歸方程式與係數摘要表：迴歸係數 $\beta_j$、標準誤 (Std Error)、t-統計量、$p$-值（雙尾 t 檢定）、顯著性星號標記 ($***, **, *$) 與 95% 係數信賴區間。
- 殘差診斷：繪製殘差分佈散佈圖與常態性直方圖，檢驗均值為 0、同方差性與常態性假設。

### 5. 95% 信賴區間 (CI) vs 95% 預測區間 (PI) 推論模擬器
- **統計理論嚴謹區分**：
  - **95% 信賴區間 (Confidence Interval for Mean Response)**：
    $$\hat{y}_0 \pm t_{crit, n-p-1} \cdot s_e \sqrt{h_{00}}$$
    衡量母體「平均反應值回歸線 $E(Y|X=x_0)$」的估計誤差。隨樣本數 $n \to \infty$，$h_{00} \to 0$，信賴區間收斂至 0 寬度。
  - **95% 預測區間 (Prediction Interval for Individual Observation)**：
    $$\hat{y}_0 \pm t_{crit, n-p-1} \cdot s_e \sqrt{1 + h_{00}}$$
    衡量「未來單一新觀測值 $Y_{new}$」的落點範圍，額外包含個體不可化約的隨機誤差 ($\epsilon \sim N(0, \sigma^2)$ 中的 $1$)。因此**預測區間必然比信賴區間更寬**。
- **互動式推論器**：自訂特徵數值，動態滑動即時呈現 $\hat{y}_0$、95% CI、95% PI 及視覺化長條圖。
- **測試集區間覆蓋率**：自動統計測試集樣本落於 95% 預測區間的覆蓋比例。

### 6. CRISP-DM 六大階段深度整合
1. **商業理解 (Business Understanding)**：定義預測目標與投資決策效益。
2. **資料理解 (Data Understanding)**：結構分析、型態判斷與缺失值探索。
3. **資料準備 (Data Preparation)**：切分集、均值填補、One-Hot 編碼與特徵選取。
4. **模型建置 (Modeling)**：最小平方法 OLS 正規方程式求解 $\hat{\beta} = (X^T X)^{-1} X^T y$。
5. **模型評估 (Evaluation)**：殘差診斷與指標對照。
6. **模型部署 (Deployment)**：推論計算機、分析報告 Markdown 下載與預測結果 CSV 匯出。

---

## 📂 專案檔案結構

```text
├── index.html          # 主頁面結構與 CRISP-DM 儀表板版面
├── style.css           # 現代響應式 Dashboard 樣式表 (CSS 自訂變數)
├── model.js            # 核心數學模組 (矩陣運算、OLS、Student-t、CI/PI、特徵篩選)
├── model_data.js       # 資料模組 (內建 50 Startups & Advertising 資料集、CSV 解析、URL 載入)
├── app.js              # 前端應用邏輯 (狀態管理、事件監聽、Chart.js 繪圖、互動模擬)
├── 50_Startups.csv     # 50 Startups 實體 CSV 檔案 (可直接用於上傳測試)
├── Advertising.csv     # Advertising 實體 CSV 檔案 (可直接用於上傳測試)
└── lib/
    ├── papaparse.min.js # 本地離線備份 PapaParse CSV 解析庫
    └── chart.umd.min.js # 本地離線備份 Chart.js 圖表庫
```

---

## 🚀 本地端 (localhost) 啟動方式

本專案為純靜態網站，任何靜態網頁伺服器皆可直接執行：

### 方法一：使用 Python 快速啟動 (推薦)
在專案根目錄開啟終端機執行：
```bash
python3 -m http.server 8000
```
啟動後，使用瀏覽器打開：[http://localhost:8000](http://localhost:8000)

### 方法二：使用 Node.js / npx
```bash
npx serve .
```

### 方法三：使用 VS Code Live Server 擴充套件
在 VS Code 中開啟專案目錄，右鍵點擊 `index.html`，選擇 **「Open with Live Server」** 即可。

---

## 🌐 GitHub Pages 部署步驟

將本專案發布至 GitHub Pages 非常簡單，只需 1 分鐘：

1. **建立 GitHub Repository**：
   在 GitHub 上建立一個新的公開儲存庫（例如 `ml-linear-regression-workshop`）。

2. **推送專案程式碼**：
   在專案目錄下執行以下指令：
   ```bash
   git init
   git add .
   git commit -m "feat: complete ML linear regression course assignment website"
   git branch -M main
   git remote add origin https://github.com/<您的GitHub帳號>/<您的儲存庫名稱>.git
   git push -u origin main
   ```

3. **啟用 GitHub Pages**：
   - 進入該 GitHub 專案頁面，點選上方 **Settings**（設定）。
   - 在左側選單點選 **Pages**。
   - 在 **Build and deployment** 下方的 **Source** 選擇 `Deploy from a branch`。
   - 在 **Branch** 選取 `main` 分支，資料夾選擇 `/ (root)`，點擊 **Save**（儲存）。

4. **瀏覽部署完成的網站**：
   等待約 1~2 分鐘，GitHub Pages 將提供公開網址，格式如下：
   `https://<您的GitHub帳號>.github.io/<您的儲存庫名稱>/`
   打開該網址即可全功能正常運作！
