/**
 * app.js
 * 機器學習線性回歸課程專用前端控制器 (UI Controller & Event Handling)
 * 負責掌控 Problem 切換、自訂 Dataset 載入、模型訓練流程、Chart.js 繪圖、
 * 95% CI/PI 互動推論器以及 CRISP-DM 專題卡片連動與分析報告匯出
 */

// ==========================================
// 全域應用程式狀態 (Application State)
// ==========================================

const AppState = {
  problemType: 'multiple', // 'simple' 或 'multiple'
  datasetId: 'startups', // 'startups', 'advertising', 'custom_file', 'custom_url'
  rawRows: [],
  summary: null,
  targetCol: '',
  selectedFeatures: [],
  trainRatio: 0.8,
  randomSeed: 42,
  missingStrategy: 'mean', // 'mean' 或 'drop'
  featureSelectionMode: 'manual', // 'manual', 'correlation', 'stepwise'
  featureSelectionK: 3,

  // 訓練後儲存狀態
  trainedModel: null,
  trainCleanRows: [],
  testCleanRows: [],
  trainMetrics: null,
  testMetrics: null,
  testPredictions: [], // 包含實際值、預測值、殘差、CI、PI
  activeEncodedFeatures: [], // 包含 Dummy 變數後的所有特徵名稱

  // Chart.js 實例管理
  charts: {
    mainScatter: null,
    residualScatter: null,
    residualHist: null
  }
};

// ==========================================
// 初始化與生命週期 (Initialization)
// ==========================================

document.addEventListener('DOMContentLoaded', () => {
  initEventListeners();
  loadPresetDataset('startups');
});

/**
 * 綁定所有使用者互動事件
 */
function initEventListeners() {
  // 1. Problem 切換 (Simple vs Multiple)
  const selectProblem = document.getElementById('selectProblem');
  if (selectProblem) {
    selectProblem.addEventListener('change', (e) => {
      setProblemType(e.target.value);
    });
  }

  // 2. 資料集來源切換頁籤
  const btnPresets = document.querySelectorAll('.dataset-tab-btn');
  btnPresets.forEach(btn => {
    btn.addEventListener('click', () => {
      const source = btn.dataset.source;
      switchDatasetSourceTab(source);
    });
  });

  // 預設資料集下拉選擇
  const selectPreset = document.getElementById('selectPresetDataset');
  if (selectPreset) {
    selectPreset.addEventListener('change', (e) => {
      loadPresetDataset(e.target.value);
    });
  }

  // 3. 本機檔案上傳
  const csvFileInput = document.getElementById('csvFileInput');
  if (csvFileInput) {
    csvFileInput.addEventListener('change', (e) => {
      if (e.target.files && e.target.files[0]) {
        handleFileUpload(e.target.files[0]);
      }
    });
  }

  // 拖放檔案上傳
  const dropZone = document.getElementById('fileDropZone');
  if (dropZone) {
    dropZone.addEventListener('dragover', (e) => {
      e.preventDefault();
      dropZone.classList.add('dragover');
    });
    dropZone.addEventListener('dragleave', () => {
      dropZone.classList.remove('dragover');
    });
    dropZone.addEventListener('drop', (e) => {
      e.preventDefault();
      dropZone.classList.remove('dragover');
      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        handleFileUpload(e.dataTransfer.files[0]);
      }
    });
  }

  // 4. URL 載入按鈕
  const btnLoadUrl = document.getElementById('btnLoadUrl');
  if (btnLoadUrl) {
    btnLoadUrl.addEventListener('click', handleUrlLoad);
  }

  // 5. Target 欄位切換
  const selectTarget = document.getElementById('selectTarget');
  if (selectTarget) {
    selectTarget.addEventListener('change', (e) => {
      AppState.targetCol = e.target.value;
      renderFeatureSelectionOptions();
    });
  }

  // 6. 切分比例滑桿
  const splitRatioInput = document.getElementById('splitRatioInput');
  const splitRatioVal = document.getElementById('splitRatioVal');
  if (splitRatioInput && splitRatioVal) {
    splitRatioInput.addEventListener('input', (e) => {
      AppState.trainRatio = Number(e.target.value) / 100;
      splitRatioVal.textContent = `${e.target.value}% / ${(100 - e.target.value)}%`;
    });
  }

  // 7. 隨機數種子
  const randomSeedInput = document.getElementById('randomSeedInput');
  if (randomSeedInput) {
    randomSeedInput.addEventListener('change', (e) => {
      AppState.randomSeed = parseInt(e.target.value, 10) || 42;
    });
  }

  const btnRandomSeed = document.getElementById('btnRandomSeed');
  if (btnRandomSeed) {
    btnRandomSeed.addEventListener('click', () => {
      const newSeed = Math.floor(Math.random() * 10000);
      AppState.randomSeed = newSeed;
      if (randomSeedInput) randomSeedInput.value = newSeed;
    });
  }

  // 8. 缺失值處理選項
  const missingInputs = document.querySelectorAll('input[name="missingStrategy"]');
  missingInputs.forEach(radio => {
    radio.addEventListener('change', (e) => {
      AppState.missingStrategy = e.target.value;
    });
  });

  // 9. 特徵選取模式
  const featureSelectionMode = document.getElementById('featureSelectionMode');
  if (featureSelectionMode) {
    featureSelectionMode.addEventListener('change', (e) => {
      AppState.featureSelectionMode = e.target.value;
      const kContainer = document.getElementById('featureSelectionKContainer');
      if (kContainer) {
        kContainer.style.display = e.target.value === 'manual' ? 'none' : 'block';
      }
    });
  }

  const featureSelectionK = document.getElementById('featureSelectionK');
  if (featureSelectionK) {
    featureSelectionK.addEventListener('input', (e) => {
      AppState.featureSelectionK = parseInt(e.target.value, 10) || 1;
    });
  }

  // 10. 全選 / 全部清除特徵按鈕
  const btnSelectAll = document.getElementById('btnSelectAllFeatures');
  if (btnSelectAll) {
    btnSelectAll.addEventListener('click', () => {
      selectAllFeatures(true);
    });
  }
  const btnDeselectAll = document.getElementById('btnDeselectAllFeatures');
  if (btnDeselectAll) {
    btnDeselectAll.addEventListener('click', () => {
      selectAllFeatures(false);
    });
  }

  // 11. 模型訓練按鈕
  const btnTrain = document.getElementById('btnTrainModel');
  if (btnTrain) {
    btnTrain.addEventListener('click', trainModel);
  }

  // 12. 匯出預測結果 CSV
  const btnExportCsv = document.getElementById('btnExportCsv');
  if (btnExportCsv) {
    btnExportCsv.addEventListener('click', exportPredictionsCSV);
  }

  // 13. 匯出分析報告 Markdown
  const btnExportReport = document.getElementById('btnExportReport');
  if (btnExportReport) {
    btnExportReport.addEventListener('click', exportMarkdownReport);
  }

  // 14. 模態框關閉
  const modalClose = document.getElementById('modalCloseBtn');
  const modalOverlay = document.getElementById('modalOverlay');
  if (modalClose && modalOverlay) {
    modalClose.addEventListener('click', () => modalOverlay.classList.remove('show'));
    modalOverlay.addEventListener('click', (e) => {
      if (e.target === modalOverlay) modalOverlay.classList.remove('show');
    });
  }

  // 15. CRISP-DM Stepper 點擊跳轉
  const stepTabs = document.querySelectorAll('.step-tab');
  stepTabs.forEach(tab => {
    tab.addEventListener('click', () => {
      stepTabs.forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      const targetId = tab.dataset.targetSection;
      const section = document.getElementById(targetId);
      if (section) {
        section.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    });
  });
}

// ==========================================
// 功能一：切換 Problem 模式 (Simple vs Multiple)
// ==========================================

function setProblemType(type) {
  AppState.problemType = type;

  // 更新模型說明卡片
  const titleEl = document.getElementById('problemDescTitle');
  const mathEl = document.getElementById('problemDescMath');
  const noteEl = document.getElementById('problemDescNote');

  if (type === 'simple') {
    if (titleEl) titleEl.textContent = '簡單線性回歸 (Simple Linear Regression)';
    if (mathEl) mathEl.innerHTML = '$$y = \\beta_0 + \\beta_1 x + \\epsilon$$';
    if (noteEl) {
      noteEl.textContent = '探討單一自變數（單特徵 x）與目標變數 y 之間的線性關聯性。系統將展示 2D 散佈圖、真實回歸線、95% 信賴帶 (Confidence Band) 與 95% 預測帶 (Prediction Band)。';
    }
  } else {
    if (titleEl) titleEl.textContent = '多元線性回歸 (Multiple Linear Regression)';
    if (mathEl) mathEl.innerHTML = '$$y = \\beta_0 + \\beta_1 x_1 + \\beta_2 x_2 + \\dots + \\beta_p x_p + \\epsilon$$';
    if (noteEl) {
      noteEl.textContent = '探討多個自變數對目標變數的綜合線性影響。支援特徵選取 (相關係數/向前逐步) 及類別變數 One-Hot Dummy Encoding (自動排除基準組防共線性)。展示實際值 vs 預測值與殘差診斷圖。';
    }
    // 切換回多元回歸時，若僅選中一個特徵，自動展開預設多特徵以利觀察
    if (AppState.selectedFeatures.length <= 1 && AppState.summary) {
      const candidateCols = AppState.summary.columns.filter(c => c !== AppState.targetCol);
      if (candidateCols.length > 1) {
        AppState.selectedFeatures = [...candidateCols];
      }
    }
  }

  // 重新渲染特徵選取元件
  renderFeatureSelectionOptions();

  // 若已有訓練結果，自動重新訓練以更新視圖
  if (AppState.rawRows.length > 0) {
    trainModel();
  }
}

// ==========================================
// 功能二：載入與切換 Dataset
// ==========================================

function switchDatasetSourceTab(source) {
  const tabs = document.querySelectorAll('.dataset-tab-btn');
  tabs.forEach(t => t.classList.remove('active'));
  const activeTab = document.querySelector(`.dataset-tab-btn[data-source="${source}"]`);
  if (activeTab) activeTab.classList.add('active');

  const containerPreset = document.getElementById('sourceContainerPreset');
  const containerUpload = document.getElementById('sourceContainerUpload');
  const containerUrl = document.getElementById('sourceContainerUrl');

  if (containerPreset) containerPreset.style.display = source === 'preset' ? 'block' : 'none';
  if (containerUpload) containerUpload.style.display = source === 'upload' ? 'block' : 'none';
  if (containerUrl) containerUrl.style.display = source === 'url' ? 'block' : 'none';
}

/**
 * 載入預設資料集
 */
function loadPresetDataset(presetKey) {
  const preset = window.DatasetManager.PRESETS[presetKey];
  if (!preset) return;

  try {
    const rows = window.DatasetManager.parseCSVString(preset.csv);
    applyLoadedDataset(rows, preset.name, preset.defaultTarget, preset.defaultFeatures, preset.simpleFeature);
  } catch (err) {
    showErrorModal('載入預設資料集失敗', err.message);
  }
}

/**
 * 處理使用者上傳的 CSV 檔案
 */
async function handleFileUpload(file) {
  if (!file.name.endsWith('.csv') && file.type !== 'text/csv') {
    showErrorModal('檔案格式錯誤', '請上傳標準 .csv 格式之資料檔。');
    return;
  }

  try {
    const rows = await window.DatasetManager.loadCSVFromFile(file);
    applyLoadedDataset(rows, `自訂檔案: ${file.name}`);
  } catch (err) {
    showErrorModal('讀取 CSV 檔案失敗', err.message);
  }
}

/**
 * 處理使用者輸入的公開 CSV URL
 */
async function handleUrlLoad() {
  const urlInput = document.getElementById('csvUrlInput');
  const url = urlInput ? urlInput.value.trim() : '';

  if (!url) {
    showErrorModal('請輸入網址', '請填寫欲載入之公開 CSV 檔案網址。');
    return;
  }

  const btnLoad = document.getElementById('btnLoadUrl');
  if (btnLoad) {
    btnLoad.disabled = true;
    btnLoad.textContent = '載入中...';
  }

  try {
    const rows = await window.DatasetManager.loadCSVFromURL(url);
    applyLoadedDataset(rows, `網址來源: ${url}`);
  } catch (err) {
    if (err.message === 'CORS_RESTRICTION') {
      showErrorModal('無法讀取外部 CSV 網址（CORS 跨來源限制）', err.customMessage, true);
    } else {
      showErrorModal('載入 URL 資料失敗', err.message);
    }
  } finally {
    if (btnLoad) {
      btnLoad.disabled = false;
      btnLoad.textContent = '載入資料';
    }
  }
}

/**
 * 將解析完成的資料列套用至應用程式並進行資料理解與分析
 */
function applyLoadedDataset(rows, datasetName, defaultTarget = null, defaultFeatures = null, simpleFeature = null) {
  AppState.rawRows = rows;
  AppState.summary = window.DatasetManager.summarizeDataset(rows);

  // 更新目前資料集名稱指示徽章
  const badgeName = document.getElementById('currentDatasetBadge');
  if (badgeName) badgeName.textContent = datasetName;

  // 更新資料統計儀表盤 (Data Understanding)
  renderDatasetSummaryCards();

  // 更新 Target 欄位下拉選單
  populateTargetSelect(defaultTarget);

  // 設定預設 Features
  if (defaultFeatures && defaultFeatures.length > 0) {
    AppState.selectedFeatures = [...defaultFeatures];
  } else {
    // 預設選取所有非 Target 的數值欄位
    AppState.selectedFeatures = AppState.summary.numericColumns.filter(c => c !== AppState.targetCol);
  }

  if (AppState.problemType === 'simple') {
    if (simpleFeature && AppState.summary.numericColumns.includes(simpleFeature)) {
      AppState.selectedFeatures = [simpleFeature];
    } else if (AppState.selectedFeatures.length > 0) {
      AppState.selectedFeatures = [AppState.selectedFeatures[0]];
    }
  }

  // 渲染特徵選擇區塊
  renderFeatureSelectionOptions();

  // 渲染前 5 筆資料預覽表
  renderDataPreviewTable();

  // 自動執行訓練
  trainModel();
}

function renderDatasetSummaryCards() {
  const s = AppState.summary;
  if (!s) return;

  const rowCountEl = document.getElementById('statTotalRows');
  const colCountEl = document.getElementById('statTotalCols');
  const numCountEl = document.getElementById('statNumericCols');
  const missingCountEl = document.getElementById('statMissingVals');

  if (rowCountEl) rowCountEl.textContent = s.rowCount;
  if (colCountEl) colCountEl.textContent = s.columns.length;
  if (numCountEl) numCountEl.textContent = s.numericColumns.length;
  if (missingCountEl) {
    missingCountEl.textContent = s.totalMissing;
    missingCountEl.style.color = s.totalMissing > 0 ? 'var(--warning)' : 'var(--success)';
  }
}

function populateTargetSelect(preferredTarget = null) {
  const selectTarget = document.getElementById('selectTarget');
  if (!selectTarget || !AppState.summary) return;

  selectTarget.innerHTML = '';
  const numCols = AppState.summary.numericColumns;

  if (numCols.length === 0) {
    showErrorModal('資料集無數值欄位', '所載入之資料集找不到可作為目標變數 (y) 的數值欄位。');
    return;
  }

  numCols.forEach(col => {
    const opt = document.createElement('option');
    opt.value = col;
    opt.textContent = `${col} (數值)`;
    selectTarget.appendChild(opt);
  });

  if (preferredTarget && numCols.includes(preferredTarget)) {
    selectTarget.value = preferredTarget;
  } else {
    // 預設選最後一個數值欄位作為 Target (經典慣例)
    selectTarget.value = numCols[numCols.length - 1];
  }

  AppState.targetCol = selectTarget.value;
}

function renderDataPreviewTable() {
  const container = document.getElementById('dataPreviewTableContainer');
  if (!container || !AppState.rawRows || AppState.rawRows.length === 0) return;

  const cols = AppState.summary.columns;
  const previewRows = AppState.rawRows.slice(0, 5);

  let html = `<table class="data-table"><thead><tr>`;
  cols.forEach(c => {
    const isNum = AppState.summary.numericColumns.includes(c);
    const isTarget = c === AppState.targetCol;
    let badge = isTarget ? '<span class="card-badge" style="background:#fee2e2;color:#dc2626;">Target</span>' :
                (isNum ? '<span class="card-badge">數值</span>' : '<span class="card-badge" style="background:#fef3c7;color:#b45309;">類別</span>');
    html += `<th>${c} ${badge}</th>`;
  });
  html += `</tr></thead><tbody>`;

  previewRows.forEach(row => {
    html += `<tr>`;
    cols.forEach(c => {
      const isNum = AppState.summary.numericColumns.includes(c);
      const val = row[c] !== undefined ? row[c] : '';
      html += `<td class="${isNum ? 'num-cell' : ''}">${val}</td>`;
    });
    html += `</tr>`;
  });

  html += `</tbody></table>`;
  container.innerHTML = html;
}

// ==========================================
// 功能三：特徵工程與訓練配置 UI
// ==========================================

function renderFeatureSelectionOptions() {
  const container = document.getElementById('featureChipsContainer');
  if (!container || !AppState.summary) return;

  container.innerHTML = '';
  const candidateCols = AppState.summary.columns.filter(c => c !== AppState.targetCol);

  if (AppState.problemType === 'simple') {
    // 簡單線性回歸：僅允許單選一個數值特徵
    const numericCandidates = candidateCols.filter(c => AppState.summary.numericColumns.includes(c));

    if (AppState.selectedFeatures.length === 0 || !numericCandidates.includes(AppState.selectedFeatures[0])) {
      AppState.selectedFeatures = numericCandidates.length > 0 ? [numericCandidates[0]] : [];
    } else if (AppState.selectedFeatures.length > 1) {
      AppState.selectedFeatures = [AppState.selectedFeatures[0]];
    }

    numericCandidates.forEach(col => {
      const isChecked = AppState.selectedFeatures.includes(col);
      const label = document.createElement('label');
      label.className = `chip-item ${isChecked ? 'selected' : ''}`;
      label.innerHTML = `
        <input type="radio" name="simpleFeatureRadio" value="${col}" ${isChecked ? 'checked' : ''}>
        <span>${col}</span>
      `;
      label.querySelector('input').addEventListener('change', () => {
        AppState.selectedFeatures = [col];
        renderFeatureSelectionOptions();
        trainModel();
      });
      container.appendChild(label);
    });

    // 隱藏全選與進階特徵篩選區
    const multiControls = document.getElementById('multiFeatureControls');
    if (multiControls) multiControls.style.display = 'none';
  } else {
    // 多元線性回歸：支援多選、類別欄位 One-Hot 與特徵選取
    const multiControls = document.getElementById('multiFeatureControls');
    if (multiControls) multiControls.style.display = 'flex';

    candidateCols.forEach(col => {
      const isNum = AppState.summary.numericColumns.includes(col);
      const isChecked = AppState.selectedFeatures.includes(col);
      const label = document.createElement('label');
      label.className = `chip-item ${isChecked ? 'selected' : ''}`;
      label.innerHTML = `
        <input type="checkbox" value="${col}" ${isChecked ? 'checked' : ''}>
        <span>${col} ${!isNum ? '(類別 OHE)' : ''}</span>
      `;
      label.querySelector('input').addEventListener('change', (e) => {
        if (e.target.checked) {
          if (!AppState.selectedFeatures.includes(col)) AppState.selectedFeatures.push(col);
        } else {
          AppState.selectedFeatures = AppState.selectedFeatures.filter(f => f !== col);
        }
        renderFeatureSelectionOptions();
      });
      container.appendChild(label);
    });
  }
}

function selectAllFeatures(select) {
  if (AppState.problemType === 'simple') return;
  const candidateCols = AppState.summary.columns.filter(c => c !== AppState.targetCol);
  AppState.selectedFeatures = select ? [...candidateCols] : [];
  renderFeatureSelectionOptions();
}

// ==========================================
// 功能三：模型訓練核心流程 (Train Model)
// ==========================================

function trainModel() {
  if (!AppState.rawRows || AppState.rawRows.length === 0) {
    showErrorModal('尚未載入資料', '請先載入或上傳資料集。');
    return;
  }

  if (!AppState.targetCol) {
    showErrorModal('未選擇目標變數', '請指定目標變數 Target (y)。');
    return;
  }

  if (AppState.selectedFeatures.length === 0) {
    showErrorModal('未選擇特徵變數', '請至少勾選一個特徵變數 (Feature)。');
    return;
  }

  try {
    // 1. 固定 Seed 切分 Training Set 與 Test Set
    const { trainRows, testRows } = window.ModelUtils.splitTrainTest(
      AppState.rawRows,
      1.0 - AppState.trainRatio,
      AppState.randomSeed
    );

    // 2. 缺失值處理 (僅從 Training Set 學習均值填補，杜絕資料洩漏)
    const activeNumericFeatures = AppState.selectedFeatures.filter(f =>
      AppState.summary.numericColumns.includes(f)
    );
    const colsToClean = [...activeNumericFeatures, AppState.targetCol];

    const { trainClean, testClean } = window.ModelUtils.processMissingValues(
      trainRows,
      testRows,
      colsToClean,
      AppState.missingStrategy
    );

    if (trainClean.length === 0 || testClean.length === 0) {
      throw new Error('在剔除缺失值後，訓練集或測試集為空。建議改用「均值填補」策略。');
    }

    // 3. 類別欄位 One-Hot Dummy Encoding (僅由 Training Set 建立 Dummy mapping，且 Drop First Dummy 防共線性)
    const activeCategoricalFeatures = AppState.selectedFeatures.filter(f =>
      AppState.summary.categoricalColumns.includes(f)
    );

    const { trainEncoded, testEncoded, dummyFeatures } = window.ModelUtils.encodeCategoricals(
      trainClean,
      testClean,
      activeCategoricalFeatures
    );

    // 合併所有最終特徵欄位
    let finalFeatureCols = [...activeNumericFeatures, ...dummyFeatures];

    // 4. 特徵選取 (Feature Selection - 嚴格基於 Training Set 進行，防止 Data Leakage)
    if (AppState.problemType === 'multiple' && AppState.featureSelectionMode !== 'manual') {
      if (AppState.featureSelectionMode === 'correlation') {
        const res = window.FeatureSelector.filterByCorrelation(
          trainEncoded,
          AppState.targetCol,
          finalFeatureCols,
          AppState.featureSelectionK
        );
        finalFeatureCols = res.selectedFeatures;
      } else if (AppState.featureSelectionMode === 'stepwise') {
        const res = window.FeatureSelector.forwardStepwise(
          trainEncoded,
          AppState.targetCol,
          finalFeatureCols,
          AppState.featureSelectionK
        );
        finalFeatureCols = res.selectedFeatures;
      }
    }

    if (finalFeatureCols.length === 0) {
      throw new Error('特徵選取後無有效特徵可供訓練。');
    }

    AppState.activeEncodedFeatures = finalFeatureCols;
    AppState.trainCleanRows = trainEncoded;
    AppState.testCleanRows = testEncoded;

    // 5. 最小平方法 (OLS) 模型建置
    const model = new window.LinearRegressionModel();
    model.fitFromRows(trainEncoded, AppState.targetCol, finalFeatureCols);
    AppState.trainedModel = model;

    // 6. 模型在訓練集與測試集之評估
    const yTestTrue = testEncoded.map(r => Number(r[AppState.targetCol]));
    const XTestRaw = testEncoded.map(r => finalFeatureCols.map(f => Number(r[f])));
    const yTestPred = model.predictRaw(XTestRaw);

    AppState.trainMetrics = model.metrics;
    AppState.testMetrics = model.evaluate(yTestTrue, yTestPred);

    // 7. 計算測試集每個點之 95% 信賴區間與預測區間
    AppState.testPredictions = testEncoded.map((row, idx) => {
      const x0 = finalFeatureCols.map(f => Number(row[f]));
      const interval = model.predictInterval(x0, 0.95);
      const actualY = Number(row[AppState.targetCol]);
      const isCoveredByPI = actualY >= interval.pi.lower && actualY <= interval.pi.upper;

      return {
        id: idx + 1,
        actual: actualY,
        predicted: interval.yHat,
        residual: actualY - interval.yHat,
        ciLower: interval.ci.lower,
        ciUpper: interval.ci.upper,
        piLower: interval.pi.lower,
        piUpper: interval.pi.upper,
        isCoveredByPI: isCoveredByPI,
        rawFeatures: x0
      };
    });

    // 8. 更新所有 UI 元件
    renderModelSummaryAndCoefficients();
    renderEvaluationMetricsCards();
    renderVisualizations();
    setupIntervalSimulator();
    renderTestSetIntervalTable();
    updateCRISPSections();

  } catch (err) {
    console.error('訓練錯誤:', err);
    showErrorModal('模型訓練失敗', err.message);
  }
}

// ==========================================
// 功能四：模型評估結果與統計摘要 UI
// ==========================================

function renderModelSummaryAndCoefficients() {
  const model = AppState.trainedModel;
  if (!model) return;

  // 1. 迴歸方程式
  const eqEl = document.getElementById('modelEquationDisplay');
  if (eqEl) {
    eqEl.textContent = model.getEquationString(4);
  }

  // 2. 係數統計檢定表
  const tbody = document.getElementById('coefficientsTableBody');
  if (!tbody) return;

  tbody.innerHTML = '';
  model.coeffDetails.forEach(c => {
    let sigTag = '';
    if (c.pValue < 0.001) sigTag = '<span class="badge-sig">*** (p < 0.001)</span>';
    else if (c.pValue < 0.01) sigTag = '<span class="badge-sig">** (p < 0.01)</span>';
    else if (c.pValue < 0.05) sigTag = '<span class="badge-sig">* (p < 0.05)</span>';
    else sigTag = '<span style="color:var(--text-muted);font-size:0.75rem;">不顯著</span>';

    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td><strong>${c.name}</strong></td>
      <td class="num-cell" style="font-weight:700;">${c.coefficient.toFixed(4)}</td>
      <td class="num-cell">${c.stdError.toFixed(4)}</td>
      <td class="num-cell">${c.tStatistic.toFixed(3)}</td>
      <td class="num-cell">${isNaN(c.pValue) ? 'N/A' : (c.pValue < 0.0001 ? '< 0.0001' : c.pValue.toFixed(4))}</td>
      <td style="text-align:center;">${sigTag}</td>
      <td class="num-cell">[ ${c.ciLower.toFixed(2)}, ${c.ciUpper.toFixed(2)} ]</td>
    `;
    tbody.appendChild(tr);
  });
}

function renderEvaluationMetricsCards() {
  const train = AppState.trainMetrics;
  const test = AppState.testMetrics;
  if (!train || !test) return;

  const setCard = (prefix, trainVal, testVal, decimals = 4) => {
    const valEl = document.getElementById(`metric${prefix}Val`);
    const trainSubEl = document.getElementById(`metric${prefix}TrainSub`);
    if (valEl) valEl.textContent = Number(testVal).toFixed(decimals);
    if (trainSubEl) trainSubEl.textContent = `Train: ${Number(trainVal).toFixed(decimals)}`;
  };

  setCard('R2', train.r2, test.r2, 4);
  setCard('AdjR2', train.adjR2, test.adjR2, 4);
  setCard('RMSE', train.rmse, test.rmse, 2);
  setCard('MAE', train.mae, test.mae, 2);
}

// ==========================================
// 功能四 & 功能五：視覺化圖表 (Chart.js)
// ==========================================

function renderVisualizations() {
  if (AppState.problemType === 'simple') {
    renderSimpleRegressionScatterChart();
  } else {
    renderActualVsPredictedChart();
  }
  renderResidualPlotChart();
  renderResidualHistogramChart();
}

/**
 * 簡單線性回歸專屬圖表：2D 散佈圖、回歸線、95% 信賴帶 (CI) 與 95% 預測帶 (PI)
 */
function renderSimpleRegressionScatterChart() {
  const canvas = document.getElementById('mainScatterChart');
  if (!canvas) return;

  if (AppState.charts.mainScatter) {
    AppState.charts.mainScatter.destroy();
  }

  const model = AppState.trainedModel;
  const featName = AppState.activeEncodedFeatures[0];
  const trainRows = AppState.trainCleanRows;
  const testRows = AppState.testCleanRows;

  // 取得散佈點
  const trainPoints = trainRows.map(r => ({ x: Number(r[featName]), y: Number(r[AppState.targetCol]) }));
  const testPoints = testRows.map(r => ({ x: Number(r[featName]), y: Number(r[AppState.targetCol]) }));

  // 取得 x 範圍以產生光滑的 CI/PI 曲線
  const allX = [...trainPoints, ...testPoints].map(p => p.x);
  const minX = Math.min(...allX);
  const maxX = Math.max(...allX);
  const padding = (maxX - minX) * 0.05 || 1;

  const curveData = model.generateSimpleRegressionCurve(minX - padding, maxX + padding, 70);

  const fitLineData = curveData.map(p => ({ x: p.x, y: p.yHat }));
  const ciUpperData = curveData.map(p => ({ x: p.x, y: p.ciUpper }));
  const ciLowerData = curveData.map(p => ({ x: p.x, y: p.ciLower }));
  const piUpperData = curveData.map(p => ({ x: p.x, y: p.piUpper }));
  const piLowerData = curveData.map(p => ({ x: p.x, y: p.piLower }));

  AppState.charts.mainScatter = new Chart(canvas, {
    type: 'scatter',
    data: {
      datasets: [
        {
          label: '測試集樣本 (Test Data)',
          data: testPoints,
          backgroundColor: '#ef4444',
          borderColor: '#dc2626',
          pointRadius: 6,
          pointHoverRadius: 8,
          order: 1
        },
        {
          label: '訓練集樣本 (Train Data)',
          data: trainPoints,
          backgroundColor: '#94a3b8',
          pointRadius: 4,
          order: 2
        },
        {
          type: 'line',
          label: '迴歸擬合線 (Fitted Line)',
          data: fitLineData,
          borderColor: '#2563eb',
          borderWidth: 3,
          pointRadius: 0,
          fill: false,
          order: 3
        },
        {
          type: 'line',
          label: '95% 信賴帶 (Confidence Band)',
          data: ciUpperData,
          borderColor: 'rgba(22, 163, 74, 0.8)',
          borderWidth: 1.5,
          borderDash: [5, 5],
          pointRadius: 0,
          fill: false,
          order: 4
        },
        {
          type: 'line',
          label: '95% 信賴帶下界',
          data: ciLowerData,
          borderColor: 'rgba(22, 163, 74, 0.8)',
          borderWidth: 1.5,
          borderDash: [5, 5],
          pointRadius: 0,
          fill: false,
          order: 5
        },
        {
          type: 'line',
          label: '95% 預測帶 (Prediction Band)',
          data: piUpperData,
          borderColor: 'rgba(217, 119, 6, 0.8)',
          borderWidth: 1.5,
          borderDash: [2, 2],
          pointRadius: 0,
          fill: false,
          order: 6
        },
        {
          type: 'line',
          label: '95% 預測帶下界',
          data: piLowerData,
          borderColor: 'rgba(217, 119, 6, 0.8)',
          borderWidth: 1.5,
          borderDash: [2, 2],
          pointRadius: 0,
          fill: false,
          order: 7
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        title: {
          display: true,
          text: `簡單線性回歸擬合與 95% 信賴帶 / 預測帶 (X: ${featName}, Y: ${AppState.targetCol})`,
          font: { size: 14, weight: 'bold' }
        },
        tooltip: {
          callbacks: {
            label: (ctx) => `${ctx.dataset.label}: (${ctx.parsed.x.toFixed(2)}, ${ctx.parsed.y.toFixed(2)})`
          }
        }
      },
      scales: {
        x: {
          title: { display: true, text: featName, font: { weight: 'bold' } },
          grid: { color: '#f1f5f9' }
        },
        y: {
          title: { display: true, text: AppState.targetCol, font: { weight: 'bold' } },
          grid: { color: '#f1f5f9' }
        }
      }
    }
  });
}

/**
 * 多元線性回歸專屬圖表：實際值 vs 預測值 (Actual vs Predicted) 散佈圖與 45 度理想線
 */
function renderActualVsPredictedChart() {
  const canvas = document.getElementById('mainScatterChart');
  if (!canvas) return;

  if (AppState.charts.mainScatter) {
    AppState.charts.mainScatter.destroy();
  }

  const testPreds = AppState.testPredictions;
  const scatterData = testPreds.map(p => ({ x: p.predicted, y: p.actual }));

  const allVals = testPreds.flatMap(p => [p.actual, p.predicted]);
  const minVal = Math.min(...allVals);
  const maxVal = Math.max(...allVals);
  const pad = (maxVal - minVal) * 0.05 || 1;

  const idealLine = [
    { x: minVal - pad, y: minVal - pad },
    { x: maxVal + pad, y: maxVal + pad }
  ];

  AppState.charts.mainScatter = new Chart(canvas, {
    type: 'scatter',
    data: {
      datasets: [
        {
          label: '測試集樣本 (Actual vs Predicted)',
          data: scatterData,
          backgroundColor: '#2563eb',
          borderColor: '#1d4ed8',
          pointRadius: 6,
          pointHoverRadius: 8
        },
        {
          type: 'line',
          label: '45° 完美預測線 (y = x)',
          data: idealLine,
          borderColor: '#ef4444',
          borderWidth: 2,
          borderDash: [6, 6],
          pointRadius: 0,
          fill: false
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        title: {
          display: true,
          text: `預測值 vs 實際值散佈圖 (R² = ${AppState.testMetrics.r2.toFixed(4)})`,
          font: { size: 14, weight: 'bold' }
        },
        tooltip: {
          callbacks: {
            label: (ctx) => `預測: ${ctx.parsed.x.toFixed(2)}, 實際: ${ctx.parsed.y.toFixed(2)}`
          }
        }
      },
      scales: {
        x: {
          title: { display: true, text: '模型預測值 (Predicted Value)', font: { weight: 'bold' } },
          grid: { color: '#f1f5f9' }
        },
        y: {
          title: { display: true, text: '真實觀測值 (Actual Value)', font: { weight: 'bold' } },
          grid: { color: '#f1f5f9' }
        }
      }
    }
  });
}

/**
 * 殘差圖 (Residual Plot: 預測值 vs 殘差 e_i)
 */
function renderResidualPlotChart() {
  const canvas = document.getElementById('residualScatterChart');
  if (!canvas) return;

  if (AppState.charts.residualScatter) {
    AppState.charts.residualScatter.destroy();
  }

  const testPreds = AppState.testPredictions;
  const residualData = testPreds.map(p => ({ x: p.predicted, y: p.residual }));

  const preds = testPreds.map(p => p.predicted);
  const minPred = Math.min(...preds);
  const maxPred = Math.max(...preds);
  const pad = (maxPred - minPred) * 0.05 || 1;

  const zeroLine = [
    { x: minPred - pad, y: 0 },
    { x: maxPred + pad, y: 0 }
  ];

  AppState.charts.residualScatter = new Chart(canvas, {
    type: 'scatter',
    data: {
      datasets: [
        {
          label: '測試集殘差 (Residuals)',
          data: residualData,
          backgroundColor: '#0d9488',
          borderColor: '#0f766e',
          pointRadius: 5
        },
        {
          type: 'line',
          label: '殘差基準線 (Residual = 0)',
          data: zeroLine,
          borderColor: '#94a3b8',
          borderWidth: 2,
          borderDash: [4, 4],
          pointRadius: 0,
          fill: false
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        title: {
          display: true,
          text: '殘差分佈散佈圖 (檢驗線性、獨立性與同方差性假設)',
          font: { size: 14, weight: 'bold' }
        },
        tooltip: {
          callbacks: {
            label: (ctx) => `預測值: ${ctx.parsed.x.toFixed(2)}, 殘差: ${ctx.parsed.y.toFixed(2)}`
          }
        }
      },
      scales: {
        x: {
          title: { display: true, text: '模型預測值 (Predicted)', font: { weight: 'bold' } },
          grid: { color: '#f1f5f9' }
        },
        y: {
          title: { display: true, text: '殘差 (Residual: e = y - ŷ)', font: { weight: 'bold' } },
          grid: { color: '#f1f5f9' }
        }
      }
    }
  });
}

/**
 * 殘差分佈長條圖 (檢驗常態分佈假設)
 */
function renderResidualHistogramChart() {
  const canvas = document.getElementById('residualHistChart');
  if (!canvas) return;

  if (AppState.charts.residualHist) {
    AppState.charts.residualHist.destroy();
  }

  const residuals = AppState.testPredictions.map(p => p.residual);
  const minR = Math.min(...residuals);
  const maxR = Math.max(...residuals);

  const binCount = 8;
  const binWidth = (maxR - minR) / binCount || 1;
  const bins = new Array(binCount).fill(0);
  const labels = [];

  for (let i = 0; i < binCount; i++) {
    const start = minR + i * binWidth;
    const end = start + binWidth;
    labels.push(`${start.toFixed(0)} ~ ${end.toFixed(0)}`);
  }

  residuals.forEach(r => {
    let b = Math.floor((r - minR) / binWidth);
    if (b >= binCount) b = binCount - 1;
    bins[b]++;
  });

  AppState.charts.residualHist = new Chart(canvas, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [
        {
          label: '殘差頻次 (Residual Frequency)',
          data: bins,
          backgroundColor: '#6366f1',
          borderRadius: 4
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        title: {
          display: true,
          text: '殘差常態性直方圖 (Residual Histogram)',
          font: { size: 14, weight: 'bold' }
        }
      },
      scales: {
        x: {
          title: { display: true, text: '殘差區間 (Residual Bins)', font: { weight: 'bold' } }
        },
        y: {
          title: { display: true, text: '樣本數量 (Count)', font: { weight: 'bold' } },
          ticks: { stepSize: 1 }
        }
      }
    }
  });
}

// ==========================================
// 功能五：95% 信賴區間與預測區間推論器 (Simulator)
// ==========================================

function setupIntervalSimulator() {
  const container = document.getElementById('simulatorInputsGrid');
  if (!container || !AppState.trainedModel) return;

  container.innerHTML = '';
  const features = AppState.activeEncodedFeatures;
  const trainRows = AppState.trainCleanRows;

  // 計算每個特徵的最小值、最大值與平均值作為初始輸入值
  const currentInputs = [];

  features.forEach((feat, idx) => {
    const vals = trainRows.map(r => Number(r[feat]));
    const minV = Math.min(...vals);
    const maxV = Math.max(...vals);
    const avgV = vals.reduce((a, b) => a + b, 0) / vals.length;

    currentInputs.push(avgV);

    const isBinary = minV === 0 && maxV === 1;

    const div = document.createElement('div');
    div.className = 'form-group';
    div.innerHTML = `
      <label class="form-label">${feat} ${isBinary ? '(虛擬變數 0/1)' : ''}</label>
      <input type="number" step="${isBinary ? '1' : '0.01'}" class="input-text sim-input" 
             data-index="${idx}" value="${avgV.toFixed(2)}" min="${minV}" max="${maxV}">
    `;

    div.querySelector('input').addEventListener('input', (e) => {
      currentInputs[idx] = parseFloat(e.target.value) || 0;
      updateSimulatorOutput(currentInputs);
    });

    container.appendChild(div);
  });

  updateSimulatorOutput(currentInputs);
}

function updateSimulatorOutput(featureVals) {
  const model = AppState.trainedModel;
  if (!model) return;

  try {
    const res = model.predictInterval(featureVals, 0.95);

    const predValEl = document.getElementById('simPredictedY');
    const ciRangeEl = document.getElementById('simCIRange');
    const piRangeEl = document.getElementById('simPIRange');
    const leverageEl = document.getElementById('simLeverage');

    if (predValEl) predValEl.textContent = res.yHat.toFixed(2);
    if (ciRangeEl) ciRangeEl.textContent = `[ ${res.ci.lower.toFixed(2)},  ${res.ci.upper.toFixed(2)} ] (± ${res.ci.margin.toFixed(2)})`;
    if (piRangeEl) piRangeEl.textContent = `[ ${res.pi.lower.toFixed(2)},  ${res.pi.upper.toFixed(2)} ] (± ${res.pi.margin.toFixed(2)})`;
    if (leverageEl) leverageEl.textContent = `槓桿值 h₀₀ = ${res.leverage.toFixed(4)}，殘差標準誤 sₑ = ${model.residualSE.toFixed(2)}`;

    // 更新長條視覺比較帶
    updateIntervalVisualBar(res);
  } catch (err) {
    console.warn('推論器計算錯誤:', err);
  }
}

function updateIntervalVisualBar(res) {
  const piBand = document.getElementById('visualBandPI');
  const ciBand = document.getElementById('visualBandCI');
  const pinPoint = document.getElementById('visualPinPoint');

  if (!piBand || !ciBand || !pinPoint) return;

  const widthPI = res.pi.upper - res.pi.lower;
  if (widthPI <= 0) return;

  // 以 PI 寬度為 100% 進行相對定位
  piBand.style.left = '5%';
  piBand.style.width = '90%';

  const ciLeftRatio = (res.ci.lower - res.pi.lower) / widthPI;
  const ciWidthRatio = (res.ci.upper - res.ci.lower) / widthPI;

  ciBand.style.left = `${5 + ciLeftRatio * 90}%`;
  ciBand.style.width = `${ciWidthRatio * 90}%`;

  const pinRatio = (res.yHat - res.pi.lower) / widthPI;
  pinPoint.style.left = `${5 + pinRatio * 90}%`;
}

function renderTestSetIntervalTable() {
  const tbody = document.getElementById('testIntervalTableBody');
  const coverageBadge = document.getElementById('testCoverageBadge');
  if (!tbody || !AppState.testPredictions) return;

  tbody.innerHTML = '';
  const list = AppState.testPredictions;
  let coveredCount = 0;

  list.forEach(p => {
    if (p.isCoveredByPI) coveredCount++;
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>#${p.id}</td>
      <td class="num-cell">${p.actual.toFixed(2)}</td>
      <td class="num-cell" style="font-weight:700;color:var(--primary);">${p.predicted.toFixed(2)}</td>
      <td class="num-cell">[ ${p.ciLower.toFixed(2)}, ${p.ciUpper.toFixed(2)} ]</td>
      <td class="num-cell">[ ${p.piLower.toFixed(2)}, ${p.piUpper.toFixed(2)} ]</td>
      <td style="text-align:center;">
        ${p.isCoveredByPI ? '<span class="card-badge" style="background:#dcfce7;color:#15803d;">落在預測區間內 (Covered)</span>' : 
                           '<span class="card-badge" style="background:#fee2e2;color:#b91c1c;">落於區間外</span>'}
      </td>
    `;
    tbody.appendChild(tr);
  });

  if (coverageBadge && list.length > 0) {
    const rate = ((coveredCount / list.length) * 100).toFixed(1);
    coverageBadge.textContent = `95% 預測區間覆蓋率：${coveredCount} / ${list.length} (${rate}%)`;
  }
}

// ==========================================
// 功能六：CRISP-DM 深度對應即時狀態更新
// ==========================================

function updateCRISPSections() {
  // 1. Business Understanding
  const buEl = document.getElementById('crispBUContent');
  if (buEl) {
    buEl.innerHTML = `
      <strong>目前分析目標：</strong>預測目標變數 <code>${AppState.targetCol}</code>。<br>
      <strong>商業價值：</strong>透過統計迴歸係數，量化各項特徵對 ${AppState.targetCol} 的邊際貢獻度（偏迴歸係數），辨識最具投資效益或關鍵影響之驅動變數，協助資源分配決策。
    `;
  }

  // 2. Data Understanding
  const duEl = document.getElementById('crispDUContent');
  if (duEl && AppState.summary) {
    duEl.innerHTML = `
      <strong>資料維度：</strong>共 ${AppState.summary.rowCount} 筆樣本，${AppState.summary.columns.length} 個欄位。<br>
      <strong>欄位型態：</strong>數值欄位 ${AppState.summary.numericColumns.length} 個、類別欄位 ${AppState.summary.categoricalColumns.length} 個。<br>
      <strong>缺失值統計：</strong>共發現 ${AppState.summary.totalMissing} 處缺失資料。
    `;
  }

  // 3. Data Preparation
  const dpEl = document.getElementById('crispDPContent');
  if (dpEl) {
    dpEl.innerHTML = `
      <strong>資料切分：</strong>訓練集 ${(AppState.trainRatio * 100).toFixed(0)}% (${AppState.trainCleanRows.length} 筆) / 測試集 ${((1 - AppState.trainRatio) * 100).toFixed(0)}% (${AppState.testCleanRows.length} 筆)，固定隨機數種子 Seed = ${AppState.randomSeed}。<br>
      <strong>缺失值策略：</strong>採用 ${AppState.missingStrategy === 'mean' ? '均值填補 (Mean Imputation，純由 Training Set 計算)' : '剔除空值列'}。<br>
      <strong>特徵工程：</strong>對類別欄位進行 Dummy Encoding 並排除基準類別防止共線性陷阱；特徵篩選模式為 <code>${AppState.featureSelectionMode}</code>，嚴格防範 Data Leakage。
    `;
  }

  // 4. Modeling
  const modEl = document.getElementById('crispModContent');
  if (modEl && AppState.trainedModel) {
    modEl.innerHTML = `
      <strong>演算法：</strong>最小平方法 Ordinary Least Squares (OLS)，計算 $\\hat{\\beta} = (X^T X)^{-1} X^T y$。<br>
      <strong>模型架構：</strong>${AppState.problemType === 'simple' ? '簡單線性回歸 (Simple Linear Regression)' : '多元線性回歸 (Multiple Linear Regression)'}。<br>
      <strong>自由度：</strong>$df = n - p - 1 = ${AppState.trainedModel.df}$，殘差標準誤 $s_e = ${AppState.trainedModel.residualSE.toFixed(2)}$。
    `;
  }

  // 5. Evaluation
  const evEl = document.getElementById('crispEvContent');
  if (evEl && AppState.testMetrics && AppState.trainMetrics) {
    evEl.innerHTML = `
      <strong>測試集表現：</strong>$R^2 = ${AppState.testMetrics.r2.toFixed(4)}$，RMSE = ${AppState.testMetrics.rmse.toFixed(2)}，MAE = ${AppState.testMetrics.mae.toFixed(2)}。<br>
      <strong>過擬合診斷：</strong>訓練集 $R^2 = ${AppState.trainMetrics.r2.toFixed(4)}$ vs 測試集 $R^2 = ${AppState.testMetrics.r2.toFixed(4)}$。<br>
      <strong>殘差檢驗：</strong>已繪製殘差分佈圖與常態性直方圖，檢驗均值為 0、同方差與常態分佈假設。
    `;
  }

  // 6. Deployment
  const depEl = document.getElementById('crispDepContent');
  if (depEl) {
    depEl.innerHTML = `
      <strong>部署方式：</strong>純靜態客戶端架構，無後端依賴，完全相容 GitHub Pages 與 localhost。<br>
      <strong>推論應用：</strong>提供互動式推論器，即時計算 95% Confidence Interval (信賴區間) 與 95% Prediction Interval (預測區間)，並支援完整報表與測試預測 CSV 匯出。
    `;
  }
}

// ==========================================
// 匯出功能 (Export Features)
// ==========================================

function exportPredictionsCSV() {
  if (!AppState.testPredictions || AppState.testPredictions.length === 0) {
    showErrorModal('無預測資料', '請先完成模型訓練以產生測試集預測資料。');
    return;
  }

  const exportRows = AppState.testPredictions.map(p => ({
    '樣本序號': p.id,
    '實際目標值': p.actual,
    '模型預測值': p.predicted.toFixed(4),
    '預測殘差': p.residual.toFixed(4),
    '95%信賴區間_下界': p.ciLower.toFixed(4),
    '95%信賴區間_上界': p.ciUpper.toFixed(4),
    '95%預測區間_下界': p.piLower.toFixed(4),
    '95%預測區間_上界': p.piUpper.toFixed(4),
    '落在95%預測區間內': p.isCoveredByPI ? 'YES' : 'NO'
  }));

  window.DatasetManager.downloadCSV(exportRows, `ML_Regression_Predictions_${Date.now()}.csv`);
}

function exportMarkdownReport() {
  if (!AppState.trainedModel || !AppState.testMetrics) {
    showErrorModal('尚未訓練模型', '請先完成模型訓練以產生完整分析報告。');
    return;
  }

  const m = AppState.trainedModel;
  const tm = AppState.testMetrics;
  const trm = AppState.trainMetrics;

  let md = `# 機器學習線性回歸分析專題報告
*產生時間：${new Date().toLocaleString()}*

---

## 1. CRISP-DM 階段總覽

### 1.1 商業理解 (Business Understanding)
- **分析主題**：${AppState.problemType === 'simple' ? '簡單線性回歸 (Simple Linear Regression)' : '多元線性回歸 (Multiple Linear Regression)'}
- **預測目標 (Target)**：\`${AppState.targetCol}\`
- **選取特徵 (Features)**：${AppState.activeEncodedFeatures.map(f => `\`${f}\``).join(', ')}

### 1.2 資料理解 (Data Understanding)
- 原始資料筆數：${AppState.summary.rowCount} 筆
- 欄位數量：${AppState.summary.columns.length} 欄 (數值欄位: ${AppState.summary.numericColumns.length}, 類別欄位: ${AppState.summary.categoricalColumns.length})
- 原始缺失值：${AppState.summary.totalMissing} 處

### 1.3 資料準備 (Data Preparation)
- 訓練/測試切分比：${(AppState.trainRatio * 100).toFixed(0)}% / ${((1 - AppState.trainRatio) * 100).toFixed(0)}%
- 固定隨機數種子 (Random Seed)：${AppState.randomSeed}
- 缺失值策略：${AppState.missingStrategy}
- 特徵工程：One-Hot Dummy Encoding (Drop First)，特徵選取模式：${AppState.featureSelectionMode} (嚴格於 Training Set 執行，杜絕 Data Leakage)

---

## 2. 模型方程式與係數估計 (Modeling)

### 2.1 迴歸方程式
\`\`\`
${m.getEquationString(4)}
\`\`\`

### 2.2 係數統計檢定摘要表
| 特徵變數 | 迴歸係數 (β) | 標準誤 (SE) | t-統計量 | p-value | 95% 信賴區間 |
| :--- | :--- | :--- | :--- | :--- | :--- |
${m.coeffDetails.map(c => `| ${c.name} | ${c.coefficient.toFixed(4)} | ${c.stdError.toFixed(4)} | ${c.tStatistic.toFixed(3)} | ${isNaN(c.pValue) ? 'N/A' : (c.pValue < 0.0001 ? '< 0.0001' : c.pValue.toFixed(4))} | [${c.ciLower.toFixed(2)}, ${c.ciUpper.toFixed(2)}] |`).join('\n')}

---

## 3. 模型評估 (Evaluation)

| 評估指標 | 訓練集 (Training Set) | 測試集 (Test Set) | 說明 |
| :--- | :--- | :--- | :--- |
| **R² (判定係數)** | ${trm.r2.toFixed(4)} | **${tm.r2.toFixed(4)}** | 解釋變異百分比 |
| **Adjusted R²** | ${trm.adjR2.toFixed(4)} | **${tm.adjR2.toFixed(4)}** | 懲罰特徵數之調整判定係數 |
| **RMSE (均方根誤差)** | ${trm.rmse.toFixed(2)} | **${tm.rmse.toFixed(2)}** | 預測誤差標準差 |
| **MAE (平均絕對誤差)** | ${trm.mae.toFixed(2)} | **${tm.mae.toFixed(2)}** | 實際值與預測值平均絕對差 |

---

## 4. 統計區間理論探討 (Confidence vs Prediction Interval)
- **95% 信賴區間 (Confidence Interval for Mean)**：估計特定特徵條件下「母體平均反應值 E(Y|X)」的可能範圍。隨樣本數增多收斂趨近於 0 寬度。
- **95% 預測區間 (Prediction Interval for Individual)**：估計「單一未來觀測值 Y_new」的可能落點，包含模型估計誤差加上未來的隨機隨機誤差 ε ~ N(0, σ²)。其寬度永遠大於信賴區間。
- **測試集覆蓋率**：共 ${AppState.testPredictions.filter(p => p.isCoveredByPI).length} / ${AppState.testPredictions.length} 筆落於 95% 預測區間內。

---
*本報告由純前端機器學習分析工作坊自動產生，完全支援 GitHub Pages 靜態部署。*
`;

  const blob = new Blob([md], { type: 'text/markdown;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `Linear_Regression_Report_${Date.now()}.md`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

// ==========================================
// 錯誤彈窗輔助函式
// ==========================================

function showErrorModal(title, message, isHTML = false) {
  const overlay = document.getElementById('modalOverlay');
  const titleEl = document.getElementById('modalTitle');
  const bodyEl = document.getElementById('modalBody');

  if (!overlay || !titleEl || !bodyEl) {
    alert(`${title}: ${message}`);
    return;
  }

  titleEl.textContent = title;
  if (isHTML) {
    bodyEl.innerHTML = message;
  } else {
    bodyEl.textContent = message;
  }
  overlay.classList.add('show');
}
