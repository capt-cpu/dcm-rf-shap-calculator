(function () {
  "use strict";

  const model = window.RF_MODEL_DATA;
  const engine = window.RFEngine.create(model);
  const featureLabels = [
    "症状持续时间", "脚踩棉花感", "手部精细运动障碍",
    "Hoffmann征", "术前mJOA", "MLR", "WBC"
  ];
  const featureLongLabels = [
    "症状持续时间", "脚踩棉花感", "手部笨拙/精细运动障碍",
    "Hoffmann征", "术前mJOA评分", "单核细胞/淋巴细胞比值（MLR）",
    "白细胞计数（WBC）"
  ];
  const inputIds = model.features;
  let lastResult = null;
  let busy = false;

  const byId = (id) => document.getElementById(id);
  const calculateButton = byId("calculate");
  const calculateLabel = byId("calculate-label");
  const calculateProgress = byId("calculate-progress");

  function formatValue(index, value) {
    if (index === 0) return `${value.toFixed(1)}个月`;
    if (index === 1 || index === 2) return value === 1 ? "有" : "无";
    if (index === 3) return value === 1 ? "阳性" : "阴性";
    if (index === 4) return `${value.toFixed(0)}分`;
    if (index === 5) return value.toFixed(3);
    if (index === 6) return `${value.toFixed(2)} ×10⁹/L`;
    return String(value);
  }

  function setDefaults() {
    inputIds.forEach((id, index) => {
      const input = byId(id);
      const range = model.ranges[index];
      input.min = range[0];
      input.max = range[1];
      input.value = model.defaults[index];
    });
    byId("cotton").value = "0";
    byId("fine").value = "0";
    byId("Hoffman").value = "0";
  }

  function readPatient() {
    return inputIds.map((id) => Number(byId(id).value));
  }

  function validatePatient(values) {
    for (let index = 0; index < values.length; index += 1) {
      const value = values[index];
      const range = model.ranges[index];
      if (!Number.isFinite(value)) throw new Error(`请填写${featureLongLabels[index]}。`);
      if (value < range[0] || value > range[1]) {
        throw new Error(`${featureLongLabels[index]}应位于开发队列范围 ${range[0]}–${range[1]} 内。`);
      }
    }
    for (const index of [1, 2, 3]) {
      if (![0, 1].includes(values[index])) throw new Error(`${featureLongLabels[index]}必须选择“无/有”或“阴性/阳性”。`);
    }
    if (!Number.isInteger(values[4])) throw new Error("术前mJOA评分应为整数。")
  }

  function showError(message) {
    const banner = byId("error-banner");
    banner.textContent = message;
    banner.hidden = false;
  }

  function clearError() {
    byId("error-banner").hidden = true;
  }

  function setBusy(isBusy, progress) {
    busy = isBusy;
    calculateButton.disabled = isBusy;
    byId("reset").disabled = isBusy;
    calculateProgress.style.width = `${Math.max(0, Math.min(1, progress || 0)) * 100}%`;
    calculateLabel.textContent = isBusy ? "正在计算精确SHAP…" : "计算风险并生成SHAP";
  }

  function nextFrame() {
    return new Promise((resolve) => window.requestAnimationFrame(resolve));
  }

  function directionText(value) {
    return value >= 0 ? "推高风险" : "降低风险";
  }

  function renderChart(rows, baseline, probability) {
    const chart = byId("shap-chart");
    chart.replaceChildren();
    const maxAbs = Math.max(...rows.map((row) => Math.abs(row.shap)), 0.000001);

    rows.forEach((row) => {
      const wrapper = document.createElement("div");
      wrapper.className = "shap-row";

      const label = document.createElement("div");
      label.className = "shap-label";
      label.innerHTML = `<strong>${row.label}</strong><span>${row.displayValue}</span>`;

      const track = document.createElement("div");
      track.className = "shap-track";
      const axis = document.createElement("span");
      axis.className = "shap-axis";
      const bar = document.createElement("span");
      bar.className = `shap-bar ${row.shap >= 0 ? "positive" : "negative"}`;
      bar.style.width = `${Math.max(1.2, Math.abs(row.shap) / maxAbs * 47)}%`;
      track.append(axis, bar);

      const value = document.createElement("div");
      value.className = `shap-value ${row.shap >= 0 ? "positive" : "negative"}`;
      value.textContent = `${row.shap >= 0 ? "+" : ""}${(100 * row.shap).toFixed(2)}`;

      wrapper.append(label, track, value);
      chart.append(wrapper);
    });

    byId("shap-caption").textContent =
      `合成参考基线风险 ${(100 * baseline).toFixed(1)}% + SHAP贡献 = 个体预测风险 ${(100 * probability).toFixed(1)}%。横轴单位为预测概率百分点。`;
  }

  function renderTable(rows) {
    const body = byId("shap-table-body");
    body.replaceChildren();
    rows.forEach((row) => {
      const tr = document.createElement("tr");
      const signClass = row.shap >= 0 ? "positive" : "negative";
      tr.innerHTML = `
        <td>${row.label}</td>
        <td>${row.displayValue}</td>
        <td class="contribution-${signClass}">${row.shap >= 0 ? "+" : ""}${(100 * row.shap).toFixed(2)}百分点</td>
        <td><span class="direction-tag ${signClass}">${directionText(row.shap)}</span></td>`;
      body.append(tr);
    });
  }

  function renderResult(values, shapResult, elapsedMs) {
    const probability = shapResult.probability;
    const above = probability >= model.threshold;
    const rows = shapResult.shapValues.map((shap, index) => ({
      index,
      feature: model.features[index],
      label: featureLabels[index],
      rawLabel: featureLongLabels[index],
      value: values[index],
      displayValue: formatValue(index, values[index]),
      shap
    })).sort((a, b) => Math.abs(b.shap) - Math.abs(a.shap));

    lastResult = { values, ...shapResult, rows, elapsedMs };
    byId("prediction-section").hidden = false;
    byId("shap-section").hidden = false;
    byId("risk-content").hidden = false;
    byId("shap-chart").hidden = false;
    byId("shap-caption").hidden = false;
    byId("shap-table-wrap").hidden = false;
    byId("download-png").disabled = false;
    byId("download-csv").disabled = false;

    const riskColour = above ? "#c84e42" : "#177e78";
    const ring = byId("risk-ring");
    ring.style.setProperty("--risk-angle", `${360 * probability}deg`);
    ring.style.setProperty("--risk-color", riskColour);
    byId("risk-number").textContent = `${(100 * probability).toFixed(1)}%`;

    const pill = byId("classification-pill");
    pill.className = `classification-pill ${above ? "above" : "below"}`;
    pill.textContent = above
      ? "达到预设阈值：模型分类为不良恢复"
      : "低于预设阈值：模型分类为有利恢复";
    byId("risk-heading").textContent = above
      ? "建议加强术前风险沟通与术后随访"
      : "预测风险低于研究分类阈值";
    byId("risk-description").textContent =
      `模型概率为 ${(100 * probability).toFixed(1)}%，研究预设分类阈值为 ${(100 * model.threshold).toFixed(0)}%。` +
      `计算用时 ${(elapsedMs / 1000).toFixed(2)}秒；该概率用于风险分层，不代表患者必然发生或不发生不良恢复。`;

    byId("baseline-value").textContent = `${(100 * shapResult.baseline).toFixed(1)}%`;
    byId("shap-sum-value").textContent = `${shapResult.shapSum >= 0 ? "+" : ""}${(100 * shapResult.shapSum).toFixed(1)}百分点`;
    byId("formula-risk-value").textContent = `${(100 * probability).toFixed(1)}%`;
    renderChart(rows, shapResult.baseline, probability);
    renderTable(rows);
  }

  async function calculate() {
    if (busy) return;
    clearError();
    const values = readPatient();

    try {
      validatePatient(values);
      setBusy(true, 0.02);
      const directProbability = engine.predict(values);
      const start = performance.now();
      const result = await engine.exactShap(values, {
        yieldEvery: 4,
        yieldFunction: nextFrame,
        onProgress: (progress) => setBusy(true, 0.05 + progress * 0.93)
      });
      const elapsed = performance.now() - start;

      if (Math.abs(directProbability - result.probability) > 1e-12) {
        throw new Error("网页RF与SHAP计算结果不一致，已停止显示。")
      }
      if (Math.abs(result.additivityError) > 1e-10) {
        throw new Error("SHAP加和校验未通过，已停止显示。")
      }
      renderResult(values, result, elapsed);
    } catch (error) {
      showError(error.message || "计算失败，请刷新页面后重试。")
    } finally {
      setBusy(false, 0);
    }
  }

  function csvEscape(value) {
    const text = String(value);
    return /[",\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
  }

  function downloadBlob(filename, blob) {
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    document.body.append(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
  }

  function timestamp() {
    return new Date().toISOString().replace(/[-:]/g, "").replace(/\..+/, "");
  }

  function downloadCsv() {
    if (!lastResult) return;
    const header = ["Feature", "Display_feature", "Patient_value", "SHAP_probability", "SHAP_percentage_points", "Direction", "Predicted_probability", "Baseline_probability", "Threshold"];
    const lines = [header.join(",")];
    lastResult.rows.forEach((row) => {
      lines.push([
        row.feature, row.rawLabel, row.displayValue, row.shap, 100 * row.shap,
        directionText(row.shap), lastResult.probability, lastResult.baseline, model.threshold
      ].map(csvEscape).join(","));
    });
    downloadBlob(`RF_SHAP_patient_${timestamp()}.csv`, new Blob(["\ufeff" + lines.join("\r\n")], { type: "text/csv;charset=utf-8" }));
  }

  function downloadPng() {
    if (!lastResult) return;
    const canvas = document.createElement("canvas");
    canvas.width = 2100;
    canvas.height = 1450;
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#172033";
    ctx.font = 'bold 58px "Microsoft YaHei", sans-serif';
    ctx.fillText("个体RF–SHAP解释", 110, 105);
    ctx.fillStyle = "#64748b";
    ctx.font = '30px "Microsoft YaHei", sans-serif';
    ctx.fillText("正值推高不良恢复风险，负值降低风险", 110, 158);

    const rows = lastResult.rows;
    const maxAbs = Math.max(...rows.map((row) => Math.abs(row.shap)), 0.000001);
    const startY = 265;
    const rowHeight = 132;
    const labelX = 110;
    const trackLeft = 700;
    const trackRight = 1880;
    const middle = (trackLeft + trackRight) / 2;

    rows.forEach((row, index) => {
      const y = startY + index * rowHeight;
      ctx.fillStyle = "#263247";
      ctx.font = 'bold 31px "Microsoft YaHei", sans-serif';
      ctx.fillText(row.label, labelX, y);
      ctx.fillStyle = "#77859a";
      ctx.font = '25px "Microsoft YaHei", sans-serif';
      ctx.fillText(row.displayValue, labelX, y + 38);

      ctx.strokeStyle = "#94a3b8";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(middle, y - 38);
      ctx.lineTo(middle, y + 42);
      ctx.stroke();

      const width = Math.abs(row.shap) / maxAbs * ((trackRight - trackLeft) / 2 - 50);
      ctx.fillStyle = row.shap >= 0 ? "#d05a4e" : "#1b8a8f";
      ctx.fillRect(row.shap >= 0 ? middle : middle - width, y - 24, width, 48);
      ctx.fillStyle = row.shap >= 0 ? "#b5463d" : "#14777a";
      ctx.font = 'bold 27px "Segoe UI", sans-serif';
      ctx.textAlign = row.shap >= 0 ? "left" : "right";
      ctx.fillText(`${row.shap >= 0 ? "+" : ""}${(100 * row.shap).toFixed(2)}`, row.shap >= 0 ? middle + width + 14 : middle - width - 14, y + 8);
      ctx.textAlign = "left";
    });

    ctx.fillStyle = "#64748b";
    ctx.font = '27px "Microsoft YaHei", sans-serif';
    ctx.fillText(
      `合成参考基线风险 ${(100 * lastResult.baseline).toFixed(1)}% + SHAP贡献 = 个体预测风险 ${(100 * lastResult.probability).toFixed(1)}%`,
      110, 1285
    );
    ctx.fillText("横轴单位：预测概率百分点；SHAP为模型归因，不代表因果效应。", 110, 1330);

    canvas.toBlob((blob) => {
      if (blob) downloadBlob(`RF_SHAP_patient_${timestamp()}.png`, blob);
    }, "image/png");
  }

  function verifyBrowserModel() {
    try {
      const verification = engine.verifyFixtures(1e-12);
      if (!verification.passed) throw new Error(`maximum error ${verification.maximumError}`);
      calculateButton.disabled = false;
    } catch (error) {
      calculateButton.disabled = true;
      showError("浏览器模型与原R模型核对失败，本次未启用预测。")
    }
  }

  calculateButton.addEventListener("click", calculate);
  byId("reset").addEventListener("click", () => {
    if (busy) return;
    setDefaults();
    clearError();
  });
  byId("download-csv").addEventListener("click", downloadCsv);
  byId("download-png").addEventListener("click", downloadPng);

  setDefaults();
  verifyBrowserModel();
})();
