// RoadSafe AI - Frontend Interactive Application Logic
document.addEventListener("DOMContentLoaded", () => {
  // Tab Switching
  const tabButtons = document.querySelectorAll(".tab-btn");
  const tabPanes = document.querySelectorAll(".tab-pane");

  tabButtons.forEach(btn => {
    btn.addEventListener("click", () => {
      tabButtons.forEach(b => b.classList.remove("active"));
      tabPanes.forEach(p => p.classList.remove("active"));

      btn.classList.add("active");
      const target = btn.getAttribute("data-tab");
      const targetPane = document.getElementById(target);
      if (targetPane) targetPane.classList.add("active");
    });
  });

  // Presets definition
  const presets = {
    blizzard: {
      "Temperature(F)": 18,
      "Humidity(%)": 92,
      "Pressure(in)": 29.4,
      "Visibility(mi)": 1.5,
      "Wind_Speed(mph)": 28,
      "Precipitation(in)": 0.45,
      "Weather_Condition": "Snow",
      "Sunrise_Sunset": "Night",
      "State_Clean": "PA",
      "Hour": 22,
      "Month": 1,
      "Start_Lat": 40.4406,
      "Start_Lng": -79.9959,
      "Junction": 1,
      "Traffic_Signal": 0,
      "Crossing": 0,
      "Stop": 0
    },
    rush_hour: {
      "Temperature(F)": 65,
      "Humidity(%)": 55,
      "Pressure(in)": 30.05,
      "Visibility(mi)": 10.0,
      "Wind_Speed(mph)": 8,
      "Precipitation(in)": 0.0,
      "Weather_Condition": "Clear",
      "Sunrise_Sunset": "Day",
      "State_Clean": "CA",
      "Hour": 17,
      "Month": 5,
      "Start_Lat": 34.0522,
      "Start_Lng": -118.2437,
      "Junction": 1,
      "Traffic_Signal": 1,
      "Crossing": 1,
      "Stop": 0
    },
    rain_fog: {
      "Temperature(F)": 48,
      "Humidity(%)": 98,
      "Pressure(in)": 29.8,
      "Visibility(mi)": 2.0,
      "Wind_Speed(mph)": 15,
      "Precipitation(in)": 0.30,
      "Weather_Condition": "Rain",
      "Sunrise_Sunset": "Night",
      "State_Clean": "FL",
      "Hour": 20,
      "Month": 9,
      "Start_Lat": 28.5383,
      "Start_Lng": -81.3792,
      "Junction": 0,
      "Traffic_Signal": 1,
      "Crossing": 0,
      "Stop": 1
    },
    clear_highway: {
      "Temperature(F)": 72,
      "Humidity(%)": 40,
      "Pressure(in)": 30.12,
      "Visibility(mi)": 10.0,
      "Wind_Speed(mph)": 5,
      "Precipitation(in)": 0.0,
      "Weather_Condition": "Clear",
      "Sunrise_Sunset": "Day",
      "State_Clean": "TX",
      "Hour": 11,
      "Month": 4,
      "Start_Lat": 32.7767,
      "Start_Lng": -96.7970,
      "Junction": 0,
      "Traffic_Signal": 0,
      "Crossing": 0,
      "Stop": 0
    }
  };

  // Preset buttons click
  document.querySelectorAll(".preset-chip").forEach(chip => {
    chip.addEventListener("click", () => {
      const pKey = chip.getAttribute("data-preset");
      const pData = presets[pKey];
      if (pData) {
        for (const [key, val] of Object.entries(pData)) {
          const el = document.getElementById(key);
          if (el) {
            if (el.type === "checkbox") {
              el.checked = Boolean(val);
            } else {
              el.value = val;
            }
          }
        }
      }
    });
  });

  // Prediction Form Submission
  const form = document.getElementById("prediction-form");
  if (form) {
    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      const submitBtn = document.getElementById("predict-submit-btn");
      submitBtn.disabled = true;
      submitBtn.innerHTML = "Processing Inference...";

      const formData = new FormData(form);
      const payload = {};
      formData.forEach((value, key) => {
        payload[key] = value;
      });

      // Handle checkboxes explicitly
      const checkboxes = ['Crossing', 'Give_Way', 'Junction', 'No_Exit', 'Railway', 'Roundabout', 'Station', 'Stop', 'Traffic_Calming', 'Traffic_Signal', 'Amenity', 'Bump'];
      checkboxes.forEach(cb => {
        payload[cb] = document.getElementById(cb) && document.getElementById(cb).checked ? 1 : 0;
      });

      try {
        const response = await fetch("/api/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        });

        if (!response.ok) throw new Error("Inference error");
        const data = await response.json();

        // Update UI
        renderPredictionResults(data);

        // Fetch lifecycle trace for this request
        const traceRes = await fetch("/api/lifecycle_trace", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        });
        if (traceRes.ok) {
          const traceData = await traceRes.json();
          renderLifecycleTrace(traceData.lifecycle_stages);
        }

      } catch (err) {
        alert("Prediction failed: " + err.message);
      } finally {
        submitBtn.disabled = false;
        submitBtn.innerHTML = "Predict Severity & Risk Assessment";
      }
    });
  }

  // Load Initial Metrics & Hotspots
  loadMetricsData();

  // Simulate Drift Button
  const driftBtn = document.getElementById("simulate-drift-btn");
  if (driftBtn) {
    driftBtn.addEventListener("click", async () => {
      driftBtn.disabled = true;
      driftBtn.innerHTML = "Testing Distribution Shift...";
      try {
        const res = await fetch("/api/monitoring");
        const data = await res.json();
        renderMonitoringReport(data);
      } catch (e) {
        console.error(e);
      } finally {
        driftBtn.disabled = false;
        driftBtn.innerHTML = "Simulate Extreme Weather Drift";
      }
    });
  }
});

function renderPredictionResults(res) {
  const sevBanner = document.getElementById("result-severity-banner");
  sevBanner.className = `severity-banner severity-${res.predicted_severity}`;
  document.getElementById("res-severity-title").innerText = `Severity ${res.predicted_severity}`;
  document.getElementById("res-severity-label").innerText = res.severity_label;

  // Risk Meter
  document.getElementById("res-risk-index").innerText = `${res.risk_index} / 100`;
  document.getElementById("res-risk-category").innerText = res.risk_category;
  document.getElementById("res-risk-bar").style.width = `${res.risk_index}%`;

  // Duration
  document.getElementById("res-duration").innerText = `${res.predicted_duration_minutes} min`;
  document.getElementById("res-latency").innerText = `${res.latency_ms} ms`;

  // Action
  document.getElementById("res-action").innerText = res.recommended_action;

  // Probability bars
  const probs = res.class_probabilities;
  for (let s = 1; s <= 4; s++) {
    const p = probs[`Severity_${s}`] || 0;
    const pct = (p * 100).toFixed(1);
    const fillEl = document.getElementById(`prob-fill-${s}`);
    const pctEl = document.getElementById(`prob-pct-${s}`);
    if (fillEl) fillEl.style.width = `${pct}%`;
    if (pctEl) pctEl.innerText = `${pct}%`;
  }
}

function renderLifecycleTrace(stages) {
  const container = document.getElementById("lifecycle-timeline");
  if (!container || !stages) return;

  container.innerHTML = "";
  stages.forEach(st => {
    const div = document.createElement("div");
    div.className = "timeline-step";
    div.innerHTML = `
      <div class="step-marker">${st.stage_number}</div>
      <div class="step-content">
        <div class="step-header">
          <div class="step-title">Stage ${st.stage_number}: ${st.stage_name}</div>
          <span class="badge badge-info">Verified</span>
        </div>
        <div class="step-desc">${st.description}</div>
        <div class="step-details"><pre>${JSON.stringify(st.details, null, 2)}</pre></div>
      </div>
    `;
    container.appendChild(div);
  });
}

async function loadMetricsData() {
  try {
    const res = await fetch("/api/metrics");
    if (!res.ok) return;
    const data = await res.json();

    // 1. Render CO2 Linear Table
    const co2Reg = data.co2_linear_models?.regression_comparison || {};
    const co2Tbody = document.getElementById("co2-reg-tbody");
    if (co2Tbody) {
      co2Tbody.innerHTML = "";
      for (const [mName, mMetrics] of Object.entries(co2Reg)) {
        const row = document.createElement("tr");
        row.innerHTML = `
          <td><strong>${mName.replace(/_/g, ' ')}</strong></td>
          <td>${mMetrics.test_r2}</td>
          <td>${mMetrics.test_rmse} min</td>
          <td>${mMetrics.test_mae} min</td>
          <td><span class="badge ${mMetrics.sparsity_percentage > 20 ? 'badge-warning' : 'badge-info'}">${mMetrics.sparsity_percentage}% (${mMetrics.zero_coefficients} zeros)</span></td>
          <td>${mMetrics.mean_abs_weight}</td>
        `;
        co2Tbody.appendChild(row);
      }
    }

    // 2. Render CO3 Tree Models Table
    const co3Data = data.co3_tree_models?.comparison || {};
    const co3Tbody = document.getElementById("co3-tree-tbody");
    if (co3Tbody) {
      co3Tbody.innerHTML = "";
      for (const [mName, mMetrics] of Object.entries(co3Data)) {
        const row = document.createElement("tr");
        row.innerHTML = `
          <td><strong>${mName.replace(/_/g, ' ')}</strong></td>
          <td><span class="badge badge-success">${(mMetrics.accuracy * 100).toFixed(1)}%</span></td>
          <td>${(mMetrics.f1_macro * 100).toFixed(1)}%</td>
          <td>${(mMetrics.precision_macro * 100).toFixed(1)}%</td>
          <td>${(mMetrics.recall_macro * 100).toFixed(1)}%</td>
          <td><strong>${mMetrics.roc_auc_ovr}</strong></td>
        `;
        co3Tbody.appendChild(row);
      }
    }

    // 3. Render Top Features
    const fiData = data.co3_tree_models?.feature_importances || [];
    const fiContainer = document.getElementById("feature-importance-list");
    if (fiContainer && fiData.length > 0) {
      fiContainer.innerHTML = "";
      fiData.forEach(item => {
        const div = document.createElement("div");
        div.className = "prob-item";
        div.innerHTML = `
          <div class="prob-labels">
            <span>${item.feature}</span>
            <span>${item.importance}</span>
          </div>
          <div class="prob-bar">
            <div class="prob-fill" style="width: ${Math.min(100, item.importance * 350)}%; background-color: var(--primary);"></div>
          </div>
        `;
        fiContainer.appendChild(div);
      });
    }

    // 4. Render Hotspot Clusters (CO4)
    const kmClusters = data.co4_unsupervised?.kmeans?.clusters || [];
    const kmTbody = document.getElementById("kmeans-tbody");
    if (kmTbody) {
      kmTbody.innerHTML = "";
      kmClusters.forEach(c => {
        const row = document.createElement("tr");
        row.innerHTML = `
          <td><strong>Region Zone ${c.cluster_id + 1}</strong></td>
          <td>Lat ${c.centroid_lat}, Lng ${c.centroid_lng}</td>
          <td>${c.sample_count.toLocaleString()}</td>
          <td><span class="badge ${c.avg_severity >= 3 ? 'badge-danger' : 'badge-warning'}">Severity ${c.avg_severity}</span></td>
          <td>${c.avg_temperature} °F</td>
        `;
        kmTbody.appendChild(row);
      });
    }

    // 5. Render Initial Monitoring status
    renderMonitoringReport(data.co6_mlops?.drift_monitoring);

  } catch (err) {
    console.error("Error loading metrics:", err);
  }
}

function renderMonitoringReport(rep) {
  if (!rep) return;
  const statusEl = document.getElementById("monitoring-status-badge");
  if (statusEl) {
    statusEl.className = rep.overall_drift_detected ? "badge badge-danger" : "badge badge-success";
    statusEl.innerText = rep.overall_drift_detected ? "DRIFT DETECTED" : "NOMINAL / STABLE";
  }

  const tbody = document.getElementById("drift-tbody");
  if (tbody && rep.features_monitored) {
    tbody.innerHTML = "";
    for (const [feat, metrics] of Object.entries(rep.features_monitored)) {
      const row = document.createElement("tr");
      row.innerHTML = `
        <td><strong>${feat}</strong></td>
        <td>${metrics.ks_statistic}</td>
        <td>${metrics.p_value}</td>
        <td>${metrics.psi}</td>
        <td><span class="badge ${metrics.drift_status === 'STABLE' ? 'badge-success' : 'badge-danger'}">${metrics.drift_status}</span></td>
      `;
      tbody.appendChild(row);
    }
  }

  // Update Latency
  if (rep.latency_metrics_ms) {
    document.getElementById("lat-p50").innerText = `${rep.latency_metrics_ms.p50} ms`;
    document.getElementById("lat-p95").innerText = `${rep.latency_metrics_ms.p95} ms`;
    document.getElementById("lat-p99").innerText = `${rep.latency_metrics_ms.p99} ms`;
  }
}
