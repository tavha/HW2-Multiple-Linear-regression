/**
 * model.js
 * 機器學習線性回歸核心運算模組
 * 包含矩陣運算、t 分布推論、訓練/測試集切分 (含隨機數種子)、特徵選取 (無資料洩漏)、
 * 最小平方法 (OLS)、模型評估指標、95% 信賴區間 (CI) 與 95% 預測區間 (PI)
 */

// ==========================================
// 1. 矩陣運算庫 (Matrix Algebra)
// ==========================================

const MatrixOps = {
  /**
   * 矩陣乘法 A (m x k) * B (k x n) = C (m x n)
   */
  multiply(A, B) {
    const rowsA = A.length;
    const colsA = A[0].length;
    const rowsB = B.length;
    const colsB = B[0].length;

    if (colsA !== rowsB) {
      throw new Error(`矩陣乘法維度不合：A 是 ${rowsA}x${colsA}，B 是 ${rowsB}x${colsB}`);
    }

    const C = new Array(rowsA);
    for (let i = 0; i < rowsA; i++) {
      C[i] = new Float64Array(colsB);
      for (let k = 0; k < colsA; k++) {
        const aik = A[i][k];
        for (let j = 0; j < colsB; j++) {
          C[i][j] += aik * B[k][j];
        }
      }
    }
    return C;
  },

  /**
   * 矩陣轉置 A^T
   */
  transpose(A) {
    const rows = A.length;
    const cols = A[0].length;
    const AT = new Array(cols);
    for (let j = 0; j < cols; j++) {
      AT[j] = new Float64Array(rows);
      for (let i = 0; i < rows; i++) {
        AT[j][i] = A[i][j];
      }
    }
    return AT;
  },

  /**
   * 高斯-約當消去法求逆矩陣 (帶部分主元選取 Partial Pivoting)
   */
  inverse(A) {
    const n = A.length;
    if (n !== A[0].length) {
      throw new Error('只有方陣才能求逆矩陣。');
    }

    // 建立增廣矩陣 [A | I]
    const M = new Array(n);
    for (let i = 0; i < n; i++) {
      M[i] = new Float64Array(2 * n);
      for (let j = 0; j < n; j++) {
        M[i][j] = A[i][j];
      }
      M[i][n + i] = 1.0;
    }

    for (let i = 0; i < n; i++) {
      // 部分主元選取
      let maxRow = i;
      let maxVal = Math.abs(M[i][i]);
      for (let k = i + 1; k < n; k++) {
        const val = Math.abs(M[k][i]);
        if (val > maxVal) {
          maxVal = val;
          maxRow = k;
        }
      }

      if (maxVal < 1e-12) {
        throw new Error('矩陣為奇異矩陣 (Singular Matrix) 或接近不可逆。可能原因：特徵間存在完全共線性 (Multicollinearity)、包含常數特徵、或樣本數少於特徵數。');
      }

      if (maxRow !== i) {
        const temp = M[i];
        M[i] = M[maxRow];
        M[maxRow] = temp;
      }

      const pivot = M[i][i];
      for (let j = 0; j < 2 * n; j++) {
        M[i][j] /= pivot;
      }

      for (let k = 0; k < n; k++) {
        if (k !== i) {
          const factor = M[k][i];
          if (factor !== 0) {
            for (let j = 0; j < 2 * n; j++) {
              M[k][j] -= factor * M[i][j];
            }
          }
        }
      }
    }

    // 取出右半邊逆矩陣
    const inv = new Array(n);
    for (let i = 0; i < n; i++) {
      inv[i] = new Float64Array(n);
      for (let j = 0; j < n; j++) {
        inv[i][j] = M[i][j + n];
      }
    }
    return inv;
  }
};

// ==========================================
// 2. 隨機數生成器與統計推論 (PRNG & Stats)
// ==========================================

/**
 * 確定性隨機數生成器 (Mulberry32 PRNG)
 * 確保在指定隨機數種子 (Seed) 下，資料切分完全一致可重現
 */
function createPRNG(seed = 42) {
  let s = Math.floor(seed) >>> 0;
  return function () {
    s |= 0;
    s = (s + 0x6D2B79F5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * 連分數法計算不完全 Beta 函數 I_x(a, b)
 */
function incompleteBeta(a, b, x) {
  if (x <= 0.0) return 0.0;
  if (x >= 1.0) return 1.0;

  // 對數 Gamma 函數近似 (Lanczos Approximation)
  function logGamma(z) {
    const p = [
      676.5203681218851,
      -1259.1392167224028,
      771.32342877765313,
      -176.61502916214059,
      12.507343278686905,
      -0.138571095836526,
      9.9843695780195716e-6,
      1.5056327351493116e-7
    ];
    let g = 7;
    if (z < 0.5) {
      return Math.log(Math.PI / Math.sin(Math.PI * z)) - logGamma(1 - z);
    }
    z -= 1;
    let x_val = 0.99999999999980993;
    for (let i = 0; i < p.length; i++) {
      x_val += p[i] / (z + i + 1);
    }
    let t = z + g + 0.5;
    return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(x_val);
  }

  // Beta 函數連續連分數 (Lentz's method)
  function betacf(a_in, b_in, x_in) {
    const MAXIT = 200;
    const EPS = 3e-14;
    const FPMIN = 1e-30;

    let qab = a_in + b_in;
    let qap = a_in + 1.0;
    let qam = a_in - 1.0;
    let c = 1.0;
    let d = 1.0 - (qab * x_in) / qap;
    if (Math.abs(d) < FPMIN) d = FPMIN;
    d = 1.0 / d;
    let h = d;

    for (let m = 1; m <= MAXIT; m++) {
      let m2 = 2 * m;
      let aa = (m * (b_in - m) * x_in) / ((qam + m2) * (a_in + m2));
      d = 1.0 + aa * d;
      if (Math.abs(d) < FPMIN) d = FPMIN;
      c = 1.0 + aa / c;
      if (Math.abs(c) < FPMIN) c = FPMIN;
      d = 1.0 / d;
      h *= d * c;

      aa = -((a_in + m) * (qab + m) * x_in) / ((a_in + m2) * (qap + m2));
      d = 1.0 + aa * d;
      if (Math.abs(d) < FPMIN) d = FPMIN;
      c = 1.0 + aa / c;
      if (Math.abs(c) < FPMIN) c = FPMIN;
      d = 1.0 / d;
      let del = d * c;
      h *= del;
      if (Math.abs(del - 1.0) < EPS) break;
    }
    return h;
  }

  const lbeta = logGamma(a) + logGamma(b) - logGamma(a + b);
  const bt = Math.exp(Math.log(x) * a + Math.log(1.0 - x) * b - lbeta);

  if (x < (a + 1.0) / (a + b + 2.0)) {
    return (bt * betacf(a, b, x)) / a;
  } else {
    return 1.0 - (bt * betacf(b, a, 1.0 - x)) / b;
  }
}

/**
 * Student's t 分布累積機率函數 CDF
 */
function studentTCDF(t, df) {
  if (df <= 0) return NaN;
  const x = df / (df + t * t);
  const prob = 0.5 * incompleteBeta(df / 2.0, 0.5, x);
  return t > 0 ? 1.0 - prob : prob;
}

/**
 * Student's t 分布百分位數 (反函數 / Quantile)
 * @param {number} p - 累積機率 (例如 0.975 對應 95% 雙尾)
 * @param {number} df - 自由度 (Degrees of freedom)
 */
function studentTInv(p, df) {
  if (df <= 0) return NaN;
  if (p <= 0.5) return -studentTInv(1.0 - p, df);
  if (p >= 1.0) return Infinity;

  // 大自由度時快速近似 (漸進常態)
  if (df > 1500) {
    return 1.95996; // 97.5% quantile of standard normal
  }

  // 二分搜尋法求解 t_cdf(t, df) = p
  let low = 0.0;
  let high = 100.0;
  for (let iter = 0; iter < 60; iter++) {
    const mid = (low + high) / 2.0;
    const cdfVal = studentTCDF(mid, df);
    if (cdfVal < p) {
      low = mid;
    } else {
      high = mid;
    }
  }
  return (low + high) / 2.0;
}

/**
 * 取得雙尾 95% 統計檢定臨界值 t_crit
 */
function getTCritical(df, confidence = 0.95) {
  const alpha = 1.0 - confidence;
  const p = 1.0 - alpha / 2.0;
  return studentTInv(p, df);
}

/**
 * 計算係數 t 檢定雙尾 p-value
 */
function computePValue(tStat, df) {
  if (df <= 0 || isNaN(tStat)) return NaN;
  const absT = Math.abs(tStat);
  const cdf = studentTCDF(absT, df);
  return 2.0 * (1.0 - cdf);
}

// ==========================================
// 3. 資料前處理與資料防洩漏機制 (Preprocessing)
// ==========================================

/**
 * 依指定種子固定切分 Train / Test Set
 */
function splitTrainTest(rows, testRatio = 0.2, seed = 42) {
  if (!rows || rows.length < 5) {
    throw new Error('資料筆數過少，無法切分訓練與測試集。');
  }

  const prng = createPRNG(seed);
  // 建立索引並以 Fisher-Yates 洗牌
  const indices = rows.map((_, i) => i);
  for (let i = indices.length - 1; i > 0; i--) {
    const j = Math.floor(prng() * (i + 1));
    const temp = indices[i];
    indices[i] = indices[j];
    indices[j] = temp;
  }

  const testCount = Math.max(1, Math.round(rows.length * testRatio));
  const trainIndices = indices.slice(testCount);
  const testIndices = indices.slice(0, testCount);

  const trainRows = trainIndices.map(i => ({ ...rows[i] }));
  const testRows = testIndices.map(i => ({ ...rows[i] }));

  return { trainRows, testRows, trainIndices, testIndices };
}

/**
 * 缺失值處理 (嚴格只從 Training Set 計算統計參數，杜絕資料洩漏)
 */
function processMissingValues(trainRows, testRows, numericCols, strategy = 'mean') {
  if (strategy === 'drop') {
    // 剔除任何含有空值的列
    const isClean = (r) => {
      return numericCols.every(col => {
        const v = r[col];
        return v !== null && v !== undefined && v !== '' && !isNaN(Number(v));
      });
    };
    return {
      trainClean: trainRows.filter(isClean),
      testClean: testRows.filter(isClean)
    };
  }

  // 均值填補：僅從 trainRows 計算各欄均值
  const trainMeans = {};
  numericCols.forEach(col => {
    let sum = 0;
    let count = 0;
    trainRows.forEach(r => {
      const v = Number(r[col]);
      if (!isNaN(v) && isFinite(v)) {
        sum += v;
        count++;
      }
    });
    trainMeans[col] = count > 0 ? sum / count : 0;
  });

  const impute = (rows) => {
    return rows.map(r => {
      const newR = { ...r };
      numericCols.forEach(col => {
        const v = Number(newR[col]);
        if (isNaN(v) || !isFinite(v)) {
          newR[col] = trainMeans[col];
        } else {
          newR[col] = v;
        }
      });
      return newR;
    });
  };

  return {
    trainClean: impute(trainRows),
    testClean: impute(testRows)
  };
}

/**
 * 類別特徵 One-Hot Dummy Encoding
 * 自動扣除第一個基準類別 (Drop First Dummy)，避免虛擬變數陷阱 (Dummy Variable Trap)
 */
function encodeCategoricals(trainRows, testRows, categoricalCols) {
  if (!categoricalCols || categoricalCols.length === 0) {
    return { trainEncoded: trainRows, testEncoded: testRows, dummyFeatures: [] };
  }

  // 從 trainRows 中學習所有出現的類別值 (避免測試集洩漏)
  const categoryMap = {};
  categoricalCols.forEach(col => {
    const categories = Array.from(new Set(trainRows.map(r => String(r[col] || '').trim()))).filter(Boolean);
    categories.sort();
    // 排除第一個作為基準類別 (Baseline)
    categoryMap[col] = categories.slice(1);
  });

  const dummyFeatures = [];
  Object.keys(categoryMap).forEach(col => {
    categoryMap[col].forEach(catVal => {
      dummyFeatures.push(`${col}_${catVal}`);
    });
  });

  const transform = (rows) => {
    return rows.map(r => {
      const newRow = { ...r };
      categoricalCols.forEach(col => {
        const rowVal = String(r[col] || '').trim();
        categoryMap[col].forEach(catVal => {
          const dummyName = `${col}_${catVal}`;
          newRow[dummyName] = rowVal === catVal ? 1 : 0;
        });
      });
      return newRow;
    });
  };

  return {
    trainEncoded: transform(trainRows),
    testEncoded: transform(testRows),
    dummyFeatures
  };
}

// ==========================================
// 4. 特徵選取 (Feature Selection - 僅基於 Training Set)
// ==========================================

const FeatureSelector = {
  /**
   * 皮爾森相關係數篩選 (Pearson Correlation Filter)
   * 僅計算 Training Set 上各特徵與 Target 的相關係數絕對值 |r|，由高至低選取 Top K
   */
  filterByCorrelation(trainRows, targetCol, candidateCols, topK = null) {
    const n = trainRows.length;
    const yVals = trainRows.map(r => Number(r[targetCol]));
    const yMean = yVals.reduce((a, b) => a + b, 0) / n;
    let yVar = 0;
    yVals.forEach(y => { yVar += (y - yMean) ** 2; });

    const correlations = [];

    candidateCols.forEach(col => {
      const xVals = trainRows.map(r => Number(r[col]));
      const xMean = xVals.reduce((a, b) => a + b, 0) / n;
      let cov = 0;
      let xVar = 0;

      for (let i = 0; i < n; i++) {
        const dx = xVals[i] - xMean;
        const dy = yVals[i] - yMean;
        cov += dx * dy;
        xVar += dx * dx;
      }

      const denom = Math.sqrt(xVar * yVar);
      const r = denom === 0 ? 0 : cov / denom;

      correlations.push({
        feature: col,
        r: r,
        absR: Math.abs(r)
      });
    });

    // 依 |r| 降冪排序
    correlations.sort((a, b) => b.absR - a.absR);

    const k = topK ? Math.min(topK, correlations.length) : correlations.length;
    const selected = correlations.slice(0, k).map(c => c.feature);

    return {
      selectedFeatures: selected,
      correlations: correlations
    };
  },

  /**
   * 向前逐步選取 (Forward Stepwise Selection)
   * 從空集合出發，在 Training Set 上貪婪加入能帶來最大 R^2 增益的特徵
   */
  forwardStepwise(trainRows, targetCol, candidateCols, maxFeatures = null) {
    const selected = [];
    const remaining = [...candidateCols];
    const steps = [];
    const maxK = maxFeatures ? Math.min(maxFeatures, candidateCols.length) : candidateCols.length;

    let bestR2 = -Infinity;

    for (let step = 0; step < maxK; step++) {
      let stepBestFeat = null;
      let stepBestR2 = -Infinity;

      for (const feat of remaining) {
        const testFeatures = [...selected, feat];
        try {
          const model = new LinearRegressionModel();
          model.fitFromRows(trainRows, targetCol, testFeatures);
          if (model.metrics.r2 > stepBestR2) {
            stepBestR2 = model.metrics.r2;
            stepBestFeat = feat;
          }
        } catch (e) {
          // 奇異矩陣或 collinearity 略過
        }
      }

      if (stepBestFeat && stepBestR2 > bestR2 + 1e-4) {
        selected.push(stepBestFeat);
        remaining.splice(remaining.indexOf(stepBestFeat), 1);
        bestR2 = stepBestR2;
        steps.push({
          step: step + 1,
          addedFeature: stepBestFeat,
          r2: bestR2
        });
      } else {
        // 無顯著改善提前停止
        break;
      }
    }

    return {
      selectedFeatures: selected.length > 0 ? selected : [candidateCols[0]],
      steps
    };
  }
};

// ==========================================
// 5. 線性回歸核心模型 (LinearRegressionModel)
// ==========================================

class LinearRegressionModel {
  constructor() {
    this.coefficients = []; // [beta_0 (intercept), beta_1, ..., beta_p]
    this.featureNames = [];
    this.targetName = '';
    this.xtxInv = null; // (X^T X)^-1 逆矩陣，用於區間估計
    this.p = 0; // 特徵數
    this.n = 0; // 訓練樣本數
    this.df = 0; // n - p - 1
    this.residualVariance = 0; // s_e^2
    this.residualSE = 0; // s_e
    this.tCrit95 = 1.96;
    this.metrics = null;
    this.coeffDetails = []; // 包含標準誤、t 值、p-value、95% CI
  }

  /**
   * 使用特徵矩陣 X (n x p) 與目標向量 y (n x 1) 進行最小平方法 (OLS) 訓練
   */
  fit(X_raw, y_raw, featureNames = [], targetName = 'Target') {
    this.n = X_raw.length;
    this.p = X_raw[0].length;
    this.featureNames = featureNames;
    this.targetName = targetName;
    this.df = this.n - this.p - 1;

    if (this.n <= this.p + 1) {
      throw new Error(`訓練樣本數不足！目前樣本數 n = ${this.n}，特徵數 p = ${this.p}。最小平方法自由度 (df = n - p - 1 = ${this.df}) 必須大於 0。請增加資料筆數或減少特徵。`);
    }

    // 建立增廣矩陣 X (首欄為 1 代表截距項 beta_0)
    const X = new Array(this.n);
    for (let i = 0; i < this.n; i++) {
      X[i] = new Float64Array(this.p + 1);
      X[i][0] = 1.0;
      for (let j = 0; j < this.p; j++) {
        X[i][j + 1] = X_raw[i][j];
      }
    }

    // y 矩陣 (n x 1)
    const y = new Array(this.n);
    for (let i = 0; i < this.n; i++) {
      y[i] = new Float64Array([y_raw[i]]);
    }

    // 1. 計算 X^T
    const XT = MatrixOps.transpose(X);

    // 2. 計算 X^T X
    const XTX = MatrixOps.multiply(XT, X);

    // 3. 計算 (X^T X)^-1
    try {
      this.xtxInv = MatrixOps.inverse(XTX);
    } catch (e) {
      throw new Error(`矩陣運算失敗：${e.message}`);
    }

    // 4. 計算 X^T y
    const XTY = MatrixOps.multiply(XT, y);

    // 5. 計算 beta = (X^T X)^-1 (X^T y)
    const betaMatrix = MatrixOps.multiply(this.xtxInv, XTY);
    this.coefficients = new Array(this.p + 1);
    for (let i = 0; i < this.p + 1; i++) {
      this.coefficients[i] = betaMatrix[i][0];
    }

    // 6. 計算殘差與統計指標
    const yHat = this.predictRaw(X_raw);
    let rss = 0;
    let ySum = 0;
    for (let i = 0; i < this.n; i++) {
      const e = y_raw[i] - yHat[i];
      rss += e * e;
      ySum += y_raw[i];
    }

    const yMean = ySum / this.n;
    let tss = 0;
    for (let i = 0; i < this.n; i++) {
      tss += (y_raw[i] - yMean) ** 2;
    }

    this.residualVariance = rss / this.df;
    this.residualSE = Math.sqrt(this.residualVariance);
    this.tCrit95 = getTCritical(this.df, 0.95);

    // R^2 與 調整後 R^2
    const r2 = tss === 0 ? 0 : 1.0 - rss / tss;
    const adjR2 = tss === 0 ? 0 : 1.0 - ((1.0 - r2) * (this.n - 1)) / this.df;

    this.metrics = {
      n: this.n,
      p: this.p,
      df: this.df,
      rss,
      tss,
      residualSE: this.residualSE,
      r2,
      adjR2,
      mae: this.computeMAE(y_raw, yHat),
      mse: rss / this.n,
      rmse: Math.sqrt(rss / this.n)
    };

    // 7. 計算回歸係數統計檢定 (標準誤、t-stat、p-value、95% CI)
    const allNames = ['截距 (Intercept)', ...this.featureNames];
    this.coeffDetails = this.coefficients.map((b, idx) => {
      const varB = this.residualVariance * this.xtxInv[idx][idx];
      const seB = Math.sqrt(Math.max(0, varB));
      const tStat = seB === 0 ? 0 : b / seB;
      const pVal = computePValue(tStat, this.df);
      const margin = this.tCrit95 * seB;

      return {
        name: allNames[idx],
        coefficient: b,
        stdError: seB,
        tStatistic: tStat,
        pValue: pVal,
        ciLower: b - margin,
        ciUpper: b + margin
      };
    });

    return this;
  }

  /**
   * 便利方法：直接傳入物件陣列訓練
   */
  fitFromRows(rows, targetCol, featureCols) {
    const X = rows.map(r => featureCols.map(f => Number(r[f])));
    const y = rows.map(r => Number(r[targetCol]));
    return this.fit(X, y, featureCols, targetCol);
  }

  /**
   * 對特徵二維陣列進行點預測
   */
  predictRaw(X_raw) {
    const m = X_raw.length;
    const preds = new Float64Array(m);
    const b0 = this.coefficients[0];

    for (let i = 0; i < m; i++) {
      let val = b0;
      const row = X_raw[i];
      for (let j = 0; j < this.p; j++) {
        val += this.coefficients[j + 1] * row[j];
      }
      preds[i] = val;
    }
    return preds;
  }

  /**
   * 對單筆觀測值 x0 (特徵值陣列) 計算 95% 信賴區間與 95% 預測區間
   * 
   * 統計理論基礎：
   * 1. 槓桿值 (Leverage) h_00 = x0^T (X^T X)^-1 x0
   * 2. 平均反應估計標準誤 SE(Mean) = s_e * sqrt(h_00)
   *    -> 95% 信賴區間 (Confidence Interval for Mean Response):
   *       y_hat +- t_crit * SE(Mean)
   * 3. 單一新預測值估計標準誤 SE(Pred) = s_e * sqrt(1 + h_00)
   *    -> 95% 預測區間 (Prediction Interval for Individual Observation):
   *       y_hat +- t_crit * SE(Pred)
   */
  predictInterval(x0_features, confidence = 0.95) {
    if (x0_features.length !== this.p) {
      throw new Error(`輸入特徵長度 (${x0_features.length}) 與模型特徵數 (${this.p}) 不符。`);
    }

    // 向量 x0 包含開頭截距項 1.0 (長度 p + 1)
    const x0 = new Float64Array(this.p + 1);
    x0[0] = 1.0;
    for (let j = 0; j < this.p; j++) {
      x0[j + 1] = Number(x0_features[j]);
    }

    // 1. 點預測 y_hat
    let yHat = 0;
    for (let j = 0; j <= this.p; j++) {
      yHat += this.coefficients[j] * x0[j];
    }

    // 2. 槓桿值 h_00 = x0^T (X^T X)^-1 x0
    // 先算 temp = (X^T X)^-1 * x0
    const temp = new Float64Array(this.p + 1);
    for (let i = 0; i <= this.p; i++) {
      let sum = 0;
      for (let j = 0; j <= this.p; j++) {
        sum += this.xtxInv[i][j] * x0[j];
      }
      temp[i] = sum;
    }

    let h00 = 0;
    for (let i = 0; i <= this.p; i++) {
      h00 += x0[i] * temp[i];
    }
    h00 = Math.max(0, h00);

    const se = this.residualSE;
    const tCrit = getTCritical(this.df, confidence);

    // 3. 信賴區間 (Mean Response Confidence Interval)
    const seMean = se * Math.sqrt(h00);
    const ciMargin = tCrit * seMean;

    // 4. 預測區間 (Individual Prediction Interval)
    const sePred = se * Math.sqrt(1.0 + h00);
    const piMargin = tCrit * sePred;

    return {
      yHat,
      leverage: h00,
      seMean,
      sePred,
      tCrit,
      confidence,
      ci: {
        lower: yHat - ciMargin,
        upper: yHat + ciMargin,
        margin: ciMargin
      },
      pi: {
        lower: yHat - piMargin,
        upper: yHat + piMargin,
        margin: piMargin
      }
    };
  }

  /**
   * 計算評估指標
   */
  evaluate(yTrue, yPred) {
    const n = yTrue.length;
    let rss = 0;
    let ySum = 0;
    let maeSum = 0;

    for (let i = 0; i < n; i++) {
      const err = yTrue[i] - yPred[i];
      rss += err * err;
      maeSum += Math.abs(err);
      ySum += yTrue[i];
    }

    const yMean = ySum / n;
    let tss = 0;
    for (let i = 0; i < n; i++) {
      tss += (yTrue[i] - yMean) ** 2;
    }

    const mse = rss / n;
    const rmse = Math.sqrt(mse);
    const mae = maeSum / n;
    const r2 = tss === 0 ? 0 : 1.0 - rss / tss;
    const df = n - this.p - 1;
    const adjR2 = (tss === 0 || df <= 0) ? 0 : 1.0 - ((1.0 - r2) * (n - 1)) / df;

    return { n, p: this.p, df, mae, mse, rmse, r2, adjR2, rss, tss };
  }

  computeMAE(yTrue, yPred) {
    let sum = 0;
    for (let i = 0; i < yTrue.length; i++) {
      sum += Math.abs(yTrue[i] - yPred[i]);
    }
    return sum / yTrue.length;
  }

  /**
   * 針對簡單線性回歸產生光滑擬合線與 CI / PI 曲線資料點 (供 Chart.js 畫帶狀圖)
   */
  generateSimpleRegressionCurve(minX, maxX, steps = 80) {
    if (this.p !== 1) {
      throw new Error('此功能僅適用於簡單線性回歸 (p = 1)。');
    }

    const stepSize = (maxX - minX) / steps;
    const curvePoints = [];

    for (let i = 0; i <= steps; i++) {
      const x = minX + i * stepSize;
      const interval = this.predictInterval([x]);
      curvePoints.push({
        x: x,
        yHat: interval.yHat,
        ciLower: interval.ci.lower,
        ciUpper: interval.ci.upper,
        piLower: interval.pi.lower,
        piUpper: interval.pi.upper
      });
    }

    return curvePoints;
  }

  /**
   * 格式化輸出迴歸方程式字串
   */
  getEquationString(digits = 4) {
    const b0 = this.coefficients[0].toFixed(digits);
    let str = `${this.targetName} = ${b0}`;

    for (let j = 0; j < this.p; j++) {
      const b = this.coefficients[j + 1];
      const sign = b >= 0 ? ' + ' : ' - ';
      str += `${sign}${Math.abs(b).toFixed(digits)} × (${this.featureNames[j]})`;
    }
    return str;
  }
}

// 匯出至全域
window.LinearRegressionModel = LinearRegressionModel;
window.MatrixOps = MatrixOps;
window.FeatureSelector = FeatureSelector;
window.ModelUtils = {
  splitTrainTest,
  processMissingValues,
  encodeCategoricals,
  studentTCDF,
  studentTInv,
  getTCritical,
  computePValue,
  createPRNG
};
