/**
 * model_data.js
 * 機器學習線性回歸課程專用資料模組
 * 包含內建 50 Startups 與 Advertising 資料集、CSV 解析、URL 載入與資料統計分析
 */

// 50 Startups 預設資料 (真實完整 50 筆紀錄)
const STARTUPS_CSV = `R&D Spend,Administration,Marketing Spend,State,Profit
165349.2,136897.8,471784.1,New York,192261.83
162597.7,151377.59,443898.53,California,191792.06
153441.51,101145.55,407934.54,Florida,191050.39
144372.41,118671.85,383199.62,New York,182901.99
142107.34,91391.77,366168.42,Florida,166187.94
131876.9,99814.71,362861.36,New York,156991.12
134615.46,147198.87,127716.82,California,156122.51
130298.13,145530.06,323876.68,Florida,155752.6
120542.52,148718.95,311613.29,New York,152211.77
123334.88,108679.17,304981.62,California,149759.96
101913.08,110594.11,229160.95,Florida,146121.95
100671.96,91790.61,249744.55,California,144259.4
93863.75,127320.38,249839.44,Florida,141585.52
91992.39,135495.07,252664.93,California,134307.35
119943.24,156547.42,256512.92,Florida,132602.65
114523.61,122616.84,261776.23,New York,129917.04
78013.11,121597.55,264346.06,California,126992.93
94657.16,145077.58,282574.31,New York,125370.37
91749.16,114175.79,294919.57,Florida,124266.9
86419.7,153514.11,0,New York,122776.86
76253.86,113867.3,298664.47,California,118474.03
78389.47,153773.43,299737.29,New York,111313.02
73994.56,122782.75,303319.26,Florida,110352.25
67532.53,105751.03,304768.73,Florida,108733.99
77044.01,99281.34,140574.81,New York,108552.04
64664.71,139553.16,137962.62,California,107404.34
75328.87,144135.98,134050.07,Florida,105733.54
72107.6,127864.55,353183.81,New York,105008.31
66051.52,182645.56,118148.2,Florida,103282.38
65605.48,153032.06,107138.38,New York,101004.64
61994.48,115641.28,91131.24,Florida,99937.59
61136.38,152701.92,88218.23,New York,97483.56
63408.86,129219.61,46085.25,California,97427.84
55493.95,103057.49,214634.81,Florida,96778.92
46426.07,157693.92,210797.67,California,96712.8
46014.02,85047.44,205517.64,New York,96479.51
28663.76,127056.21,201126.82,Florida,90708.19
44069.95,51283.14,197029.42,California,89949.14
20229.59,65947.93,185265.1,New York,81229.06
38558.51,82982.09,174999.3,California,81005.76
28754.33,118546.05,172795.67,California,78239.91
27892.92,84710.77,164470.71,Florida,77798.83
23640.93,96189.63,148001.11,California,71498.49
15505.73,127382.3,35534.17,New York,69758.98
22177.74,154806.14,28334.72,California,65200.33
1000.23,124153.04,1903.93,New York,64926.08
1315.46,115816.21,297114.46,Florida,49490.75
0,135426.92,0,California,42559.73
542.05,51743.15,0,New York,35673.41
0,116983.8,45173.06,California,14681.4`;

// Advertising 預設資料 (精選前 50 筆)
const ADVERTISING_CSV = `TV,Radio,Newspaper,Sales
230.1,37.8,69.2,22.1
44.5,39.3,45.1,10.4
17.2,45.9,69.3,9.3
151.5,41.3,58.5,18.5
180.8,10.8,58.4,12.9
8.7,48.9,75.0,7.2
57.5,32.8,23.5,11.8
120.2,19.6,11.6,13.2
8.6,2.1,1.0,4.8
199.8,2.6,21.2,10.6
66.1,5.8,24.2,8.6
214.7,24.0,4.0,17.4
23.8,35.1,65.9,9.2
97.5,7.6,7.2,9.7
204.1,32.9,46.0,19.0
195.4,47.7,52.9,22.4
67.8,36.6,114.0,12.5
281.4,39.6,55.8,24.4
69.2,20.5,18.3,11.3
147.3,23.9,19.1,14.6
218.4,27.7,53.4,18.0
237.4,5.1,23.5,12.5
13.2,15.9,49.6,5.6
228.3,16.9,26.2,15.5
62.3,12.6,18.3,9.7
262.9,3.5,19.5,12.0
142.9,29.3,12.6,15.0
240.1,16.7,22.9,15.9
248.8,27.1,22.9,18.9
70.6,16.0,40.8,10.5
292.9,28.3,43.2,21.4
112.9,17.4,38.6,11.9
97.2,1.5,30.0,9.6
265.6,20.0,0.3,17.4
95.7,1.4,7.4,9.5
290.7,4.1,8.5,12.8
266.9,43.8,5.0,25.4
74.7,49.4,45.7,14.7
43.1,26.7,35.1,10.1
228.0,37.7,32.0,21.5
59.0,31.6,44.7,12.8
177.0,33.4,38.7,17.1
293.6,27.7,1.8,20.7
206.9,8.4,26.4,12.9
25.1,25.7,43.3,8.5
175.1,22.5,31.5,14.9
89.7,9.9,35.7,10.6
239.9,41.5,18.5,23.2
227.2,15.8,49.9,14.8
202.5,22.3,31.6,16.6`;

// 資料集清單設定
const DATASET_PRESETS = {
  startups: {
    id: 'startups',
    name: '50 Startups (新創公司獲利預測)',
    csv: STARTUPS_CSV,
    source: '經典機器學習回歸教材 (Kaggle / Machine Learning A-Z)',
    defaultTarget: 'Profit',
    defaultFeatures: ['R&D Spend', 'Administration', 'Marketing Spend', 'State'],
    simpleFeature: 'R&D Spend',
    description: '記錄 50 家新創公司在研發 (R&D Spend)、行政管理 (Administration)、行銷 (Marketing Spend) 的費用與所在州 (State)，以及最終實現的淨利潤 (Profit)。適合探討多元線性回歸與虛擬變數。'
  },
  advertising: {
    id: 'advertising',
    name: 'Advertising (廣告預算與產品銷量)',
    csv: ADVERTISING_CSV,
    source: 'An Introduction to Statistical Learning (ISLR Gareth James et al.)',
    defaultTarget: 'Sales',
    defaultFeatures: ['TV', 'Radio', 'Newspaper'],
    simpleFeature: 'TV',
    description: '記載不同市場在電視 (TV)、廣播 (Radio) 與報紙 (Newspaper) 的廣告支出（單位：千美元），以及相應產生的產品銷售量 (Sales，單位：千件)。經典的多重迴歸教學資料。'
  }
};

/**
 * 使用 Papa Parse 或正規表示法解析 CSV 字串
 * @param {string} csvText 
 * @returns {Array<Object>} 原始資料物件陣列
 */
function parseCSVString(csvText) {
  if (!csvText || typeof csvText !== 'string' || !csvText.trim()) {
    throw new Error('CSV 內容為空或格式無效。');
  }

  // 優先使用 PapaParse（若已載入）
  if (typeof Papa !== 'undefined') {
    const parsed = Papa.parse(csvText.trim(), {
      header: true,
      skipEmptyLines: true,
      dynamicTyping: false // 自行進行型態偵測以精準辨識缺失值
    });

    if (parsed.errors && parsed.errors.length > 0) {
      console.warn('PapaParse 警告:', parsed.errors);
    }

    if (!parsed.data || parsed.data.length === 0) {
      throw new Error('無法從 CSV 解析出任何有效資料列。');
    }

    // 清理欄位標題中的空格
    return cleanParsedRows(parsed.data);
  }

  // 簡易 Fallback 解析器
  return simpleCSVParser(csvText);
}

/**
 * 簡易 CSV 解析器（PapaParse 備用方案）
 */
function simpleCSVParser(text) {
  const lines = text.trim().split(/\r?\n/).filter(line => line.trim().length > 0);
  if (lines.length < 2) {
    throw new Error('CSV 檔案至少需包含標題行與一筆資料。');
  }

  const parseLine = (line) => {
    const result = [];
    let cur = '';
    let inQuotes = false;
    for (let i = 0; i < line.length; i++) {
      const c = line[i];
      if (c === '"') {
        inQuotes = !inQuotes;
      } else if (c === ',' && !inQuotes) {
        result.push(cur.trim());
        cur = '';
      } else {
        cur += c;
      }
    }
    result.push(cur.trim());
    return result;
  };

  const headers = parseLine(lines[0]);
  const rows = [];
  for (let i = 1; i < lines.length; i++) {
    const vals = parseLine(lines[i]);
    const obj = {};
    headers.forEach((h, idx) => {
      obj[h] = vals[idx] !== undefined ? vals[idx] : '';
    });
    rows.push(obj);
  }

  return cleanParsedRows(rows);
}

/**
 * 清理並標準化欄位名稱與資料
 */
function cleanParsedRows(data) {
  if (!data || data.length === 0) return [];
  const rawHeaders = Object.keys(data[0]);

  // 過濾無名欄位 (例如 index column ,TV,Radio...)
  const validHeaders = rawHeaders.map(h => h.trim());

  return data.map(row => {
    const cleanRow = {};
    rawHeaders.forEach((origH, idx) => {
      const cleanH = validHeaders[idx] || `Column_${idx}`;
      let val = row[origH];
      if (typeof val === 'string') {
        val = val.trim();
      }
      cleanRow[cleanH] = val;
    });
    return cleanRow;
  });
}

/**
 * 資料集特徵分析與摘要
 * @param {Array<Object>} rows 
 * @returns {Object} 包含欄位型態、統計、缺失值的分析報告
 */
function summarizeDataset(rows) {
  if (!rows || rows.length === 0) {
    return {
      rowCount: 0,
      columns: [],
      numericColumns: [],
      categoricalColumns: [],
      missingCounts: {},
      totalMissing: 0
    };
  }

  const columns = Object.keys(rows[0]);
  const rowCount = rows.length;
  const numericColumns = [];
  const categoricalColumns = [];
  const missingCounts = {};
  let totalMissing = 0;

  columns.forEach(col => {
    let numCount = 0;
    let missingCount = 0;
    const distinctVals = new Set();

    rows.forEach(r => {
      const v = r[col];
      if (v === null || v === undefined || v === '' || v === 'NA' || v === 'NaN' || v === 'null') {
        missingCount++;
      } else {
        distinctVals.add(v);
        const parsed = Number(v);
        if (!isNaN(parsed) && isFinite(parsed)) {
          numCount++;
        }
      }
    });

    missingCounts[col] = missingCount;
    totalMissing += missingCount;

    // 若非空值中有 90% 以上可轉為數值，判定為數值欄位
    const validCount = rowCount - missingCount;
    if (validCount > 0 && (numCount / validCount) >= 0.85) {
      numericColumns.push(col);
    } else {
      categoricalColumns.push(col);
    }
  });

  return {
    rowCount,
    columns,
    numericColumns,
    categoricalColumns,
    missingCounts,
    totalMissing
  };
}

/**
 * 從公開 URL 載入 CSV 資料（具備完整 CORS 錯誤攔截與友善導引）
 * @param {string} url 
 * @returns {Promise<Array<Object>>}
 */
async function loadCSVFromURL(url) {
  if (!url || typeof url !== 'string' || !url.trim()) {
    throw new Error('請輸入有效的 CSV 檔案網址。');
  }

  const trimmedUrl = url.trim();

  // 嘗試透過 fetch 載入
  try {
    const response = await fetch(trimmedUrl, {
      method: 'GET',
      mode: 'cors'
    });

    if (!response.ok) {
      throw new Error(`伺服器回應錯誤 HTTP ${response.status}: ${response.statusText}`);
    }

    const csvText = await response.text();
    if (!csvText || !csvText.trim()) {
      throw new Error('從該網址取得的檔案內容為空。');
    }

    return parseCSVString(csvText);
  } catch (err) {
    console.error('URL Fetch Error:', err);

    // 判斷是否為典型 CORS 或網路失敗
    const isCorsOrNetwork = err.name === 'TypeError' || 
                            err.message.includes('Failed to fetch') || 
                            err.message.includes('NetworkError') || 
                            err.message.includes('CORS');

    if (isCorsOrNetwork) {
      const error = new Error('CORS_RESTRICTION');
      error.customMessage = `因瀏覽器跨網域安全機制 (CORS) 限制，該伺服器未開放跨域讀取權限。<br><br>` +
        `<strong>建議解決方案：</strong><br>` +
        `1. 請在瀏覽器另開分頁打開該網址或點此下載：<a href="${trimmedUrl}" target="_blank" rel="noopener noreferrer" class="link-download">點擊下載 CSV</a><br>` +
        `2. 下載完成後，請點擊上方<strong>「上傳 CSV 檔案」</strong>按鈕進行載入即可！`;
      error.url = trimmedUrl;
      throw error;
    }

    throw err;
  }
}

/**
 * 從本機上傳的 File 物件讀取 CSV
 * @param {File} file 
 * @returns {Promise<Array<Object>>}
 */
function loadCSVFromFile(file) {
  return new Promise((resolve, reject) => {
    if (!file) {
      reject(new Error('未選擇任何檔案。'));
      return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const text = e.target.result;
        const rows = parseCSVString(text);
        resolve(rows);
      } catch (err) {
        reject(err);
      }
    };
    reader.onerror = () => {
      reject(new Error('讀取本機檔案時發生錯誤。'));
    };
    reader.readAsText(file);
  });
}

/**
 * 匯出資料為 CSV 檔案並觸發下載
 */
function downloadCSV(dataArray, filename = 'regression_analysis.csv') {
  if (!dataArray || dataArray.length === 0) return;

  const headers = Object.keys(dataArray[0]);
  const lines = [headers.join(',')];

  dataArray.forEach(row => {
    const line = headers.map(h => {
      let val = row[h] !== undefined ? String(row[h]) : '';
      if (val.includes(',') || val.includes('"') || val.includes('\n')) {
        val = `"${val.replace(/"/g, '""')}"`;
      }
      return val;
    }).join(',');
    lines.push(line);
  });

  const blob = new Blob([lines.join('\n')], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

// 匯出至全域
window.DatasetManager = {
  PRESETS: DATASET_PRESETS,
  parseCSVString,
  summarizeDataset,
  loadCSVFromURL,
  loadCSVFromFile,
  downloadCSV
};
