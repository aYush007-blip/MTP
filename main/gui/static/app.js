/* ==========================================================================
   Jetson Orin Speech-to-LaTeX Minimal GUI - Frontend Controller
   100% Offline Edge Audio Capture, Device Control & Multi-Format Exporters
   ========================================================================== */

document.addEventListener("DOMContentLoaded", () => {
  // -------------------------------------------------------------------------
  // UI Elements - Hardware Audio Devices
  // -------------------------------------------------------------------------
  const selectMicDevice = document.getElementById("selectMicDevice");
  const selectOutputDevice = document.getElementById("selectOutputDevice");
  const deviceStatusBadge = document.getElementById("deviceStatusBadge");

  // Primary Action Button & Timers
  const btnToggleRecord = document.getElementById("btnToggleRecord");
  const mainRecordIcon = document.getElementById("mainRecordIcon");
  const mainRecordLabel = document.getElementById("mainRecordLabel");
  const keyboardHintPill = document.getElementById("keyboardHintPill");

  const recordingTimerBadge = document.getElementById("recordingTimerBadge");
  const recordingTimerText = document.getElementById("recordingTimerText");

  // Secondary Inputs
  const btnUpload = document.getElementById("btnUpload");
  const fileInput = document.getElementById("fileInput");
  const savedRecordingsSelect = document.getElementById("savedRecordingsSelect");
  const btnRunSaved = document.getElementById("btnRunSaved");
  const directTextInput = document.getElementById("directTextInput");
  const btnSendText = document.getElementById("btnSendText");

  // Audio Visualizer & Player
  const audioPlayback = document.getElementById("audioPlayback");
  const audioPlayerWrapper = document.getElementById("audioPlayerWrapper");
  const btnDownloadAudioPlayer = document.getElementById("btnDownloadAudioPlayer");
  const audioCanvas = document.getElementById("audioCanvas");
  const visualizerDot = document.getElementById("visualizerDot");
  const visualizerLabel = document.getElementById("visualizerLabel");

  // Outputs & LaTeX Display
  const rawTranscriptBox = document.getElementById("rawTranscriptBox");
  const latexRenderContainer = document.getElementById("latexRenderContainer");
  const routeBadge = document.getElementById("routeBadge");
  const latexCodeCard = document.getElementById("latexCodeCard");
  const latexCodeBox = document.getElementById("latexCodeBox");
  const spansCard = document.getElementById("spansCard");
  const spansBox = document.getElementById("spansBox");

  const btnCopyLatex = document.getElementById("btnCopyLatex");
  const btnDownloadOutputAudio = document.getElementById("btnDownloadOutputAudio");
  const btnDownloadOutputJson = document.getElementById("btnDownloadOutputJson");

  // Latency Metrics
  const valAsrTime = document.getElementById("valAsrTime");
  const valRouterTime = document.getElementById("valRouterTime");
  const valT5Time = document.getElementById("valT5Time");
  const valTotalTime = document.getElementById("valTotalTime");
  const lblTranslationEngine = document.getElementById("lblTranslationEngine");

  // Engine Selector
  const btnEngineSlm = document.getElementById("btnEngineSlm");
  const btnEngineT5 = document.getElementById("btnEngineT5");
  let currentEngine = "slm"; // Default: Qwen2.5-1.5B (Edge SLM)

  // Live Tests Session & Exports
  const sessionPathTag = document.getElementById("sessionPathTag");
  const sessionCountBadge = document.getElementById("sessionCountBadge");
  const btnExportSrt = document.getElementById("btnExportSrt");
  const btnExportVtt = document.getElementById("btnExportVtt");
  const btnExportMd = document.getElementById("btnExportMd");
  const btnExportTex = document.getElementById("btnExportTex");
  const btnExportTxt = document.getElementById("btnExportTxt");
  const btnExportZip = document.getElementById("btnExportZip");
  const btnNewSession = document.getElementById("btnNewSession");
  const historyList = document.getElementById("historyList");
  const btnClearHistory = document.getElementById("btnClearHistory");

  // -------------------------------------------------------------------------
  // Application State
  // -------------------------------------------------------------------------
  let isHardwareRecording = false;
  let animFrameId = null;
  let timerIntervalId = null;
  let timerStartTime = 0;

  let currentOutputData = null;
  let currentAudioUrl = null;
  let currentAudioFilename = null;
  let currentLatex = "";
  const sessionHistoryRecords = [];

  // -------------------------------------------------------------------------
  // Engine Toggle
  // -------------------------------------------------------------------------
  function setEngine(engine) {
    currentEngine = engine;
    if (engine === "slm") {
      btnEngineSlm?.classList.add("active");
      btnEngineT5?.classList.remove("active");
      btnEngineSlm?.querySelector(".engine-badge-tag")?.classList.add("active-tag");
      btnEngineT5?.querySelector(".engine-badge-tag")?.classList.remove("active-tag");
      if (lblTranslationEngine) lblTranslationEngine.textContent = "SLM (Qwen)";
    } else {
      btnEngineT5?.classList.add("active");
      btnEngineSlm?.classList.remove("active");
      btnEngineT5?.querySelector(".engine-badge-tag")?.classList.add("active-tag");
      btnEngineSlm?.querySelector(".engine-badge-tag")?.classList.remove("active-tag");
      if (lblTranslationEngine) lblTranslationEngine.textContent = "T5-v2";
    }
  }

  btnEngineSlm?.addEventListener("click", () => setEngine("slm"));
  btnEngineT5?.addEventListener("click", () => setEngine("t5-v2"));

  function updateMainRecordButton(stateClass, badgeText, labelText) {
    if (!btnToggleRecord) return;
    btnToggleRecord.className = `btn-main-record ${stateClass}`;
    if (mainRecordIcon) mainRecordIcon.textContent = badgeText;
    if (mainRecordLabel) mainRecordLabel.textContent = labelText;
  }

  // -------------------------------------------------------------------------
  // Jetson Hardware Audio Device Enumeration & Selection
  // -------------------------------------------------------------------------
  async function fetchAudioDevices() {
    try {
      const resp = await fetch("/api/audio_devices");
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json();

      // Populate Mic Sources
      if (selectMicDevice && Array.isArray(data.sources) && data.sources.length > 0) {
        selectMicDevice.innerHTML = "";
        data.sources.forEach(src => {
          const opt = document.createElement("option");
          opt.value = src.id;
          opt.textContent = `${src.name}${src.is_earphones ? " (Earphones Mic)" : ""}`;
          if (src.is_default || (src.id && src.id.toLowerCase().includes("ab13x"))) {
            opt.selected = true;
          }
          selectMicDevice.appendChild(opt);
        });
      }

      // Populate Earphone/Playback Sinks
      if (selectOutputDevice && Array.isArray(data.sinks) && data.sinks.length > 0) {
        selectOutputDevice.innerHTML = "";
        data.sinks.forEach(sink => {
          const opt = document.createElement("option");
          opt.value = sink.id;
          opt.textContent = `${sink.name}${sink.is_earphones ? " (Earphones Playback)" : ""}`;
          if (sink.is_default || (sink.id && sink.id.toLowerCase().includes("ab13x"))) {
            opt.selected = true;
          }
          selectOutputDevice.appendChild(opt);
        });
      }

      if (deviceStatusBadge) {
        const hasUsb = (data.sources?.some(s => s.id && s.id.toLowerCase().includes("ab13x")) ||
                        data.sinks?.some(s => s.id && s.id.toLowerCase().includes("ab13x")));
        deviceStatusBadge.textContent = hasUsb ? "AB13X Earphones Active" : "Hardware Audio Ready";
      }
    } catch (err) {
      console.warn("Could not reload audio devices (using pre-populated defaults):", err);
      if (deviceStatusBadge) deviceStatusBadge.textContent = "AB13X Earphones Ready";
    }
  }

  async function handleDeviceChange(type, deviceName) {
    if (!deviceName) return;
    try {
      await fetch("/api/audio_devices/select", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ type, name: deviceName }),
      });
      if (deviceStatusBadge) deviceStatusBadge.textContent = "Device Updated";
    } catch (err) {
      console.error("Device select error:", err);
    }
  }

  selectMicDevice?.addEventListener("change", () => {
    handleDeviceChange("source", selectMicDevice.value);
  });
  selectOutputDevice?.addEventListener("change", () => {
    handleDeviceChange("sink", selectOutputDevice.value);
  });

  // Call on load
  fetchAudioDevices();

  // -------------------------------------------------------------------------
  // Live Timer Badge
  // -------------------------------------------------------------------------
  function startLiveTimer(prefix = "REC") {
    timerStartTime = Date.now();
    if (recordingTimerBadge) recordingTimerBadge.style.display = "inline-flex";
    if (recordingTimerText) recordingTimerText.textContent = `${prefix} 00:00`;
    clearInterval(timerIntervalId);
    timerIntervalId = setInterval(() => {
      const elapsedSec = Math.floor((Date.now() - timerStartTime) / 1000);
      const mm = String(Math.floor(elapsedSec / 60)).padStart(2, "0");
      const ss = String(elapsedSec % 60).padStart(2, "0");
      if (recordingTimerText) {
        recordingTimerText.textContent = `${prefix} ${mm}:${ss}`;
      }
    }, 400);
  }

  function stopLiveTimer() {
    clearInterval(timerIntervalId);
    timerIntervalId = null;
    if (recordingTimerBadge) recordingTimerBadge.style.display = "none";
  }

  // -------------------------------------------------------------------------
  // Audio Visualizer Setup (Canvas)
  // -------------------------------------------------------------------------
  const canvasCtx = audioCanvas?.getContext("2d");

  function resizeCanvas() {
    if (audioCanvas && audioCanvas.parentElement) {
      audioCanvas.width = audioCanvas.parentElement.clientWidth;
      audioCanvas.height = 100;
    }
  }
  window.addEventListener("resize", resizeCanvas);
  resizeCanvas();

  function drawIdleWave() {
    if (!canvasCtx || !audioCanvas) return;
    canvasCtx.clearRect(0, 0, audioCanvas.width, audioCanvas.height);
    const width = audioCanvas.width;
    const height = audioCanvas.height;
    const centerY = height / 2;

    canvasCtx.lineWidth = 2;
    canvasCtx.strokeStyle = "rgba(56, 189, 248, 0.25)";
    canvasCtx.beginPath();

    const time = Date.now() * 0.002;
    for (let x = 0; x < width; x++) {
      const y = centerY + Math.sin(x * 0.02 + time) * 5;
      if (x === 0) canvasCtx.moveTo(x, y);
      else canvasCtx.lineTo(x, y);
    }
    canvasCtx.stroke();

    if (!isHardwareRecording) {
      animFrameId = requestAnimationFrame(drawIdleWave);
    }
  }
  drawIdleWave();

  function drawSimulatedActiveWave() {
    if (!canvasCtx || !audioCanvas) return;
    canvasCtx.fillStyle = "rgba(2, 6, 23, 0.35)";
    canvasCtx.fillRect(0, 0, audioCanvas.width, audioCanvas.height);

    const width = audioCanvas.width;
    const height = audioCanvas.height;
    const centerY = height / 2;
    const time = Date.now() * 0.007;

    canvasCtx.lineWidth = 2.5;
    const grad = canvasCtx.createLinearGradient(0, 0, width, 0);
    grad.addColorStop(0, "#10b981");
    grad.addColorStop(0.5, "#38bdf8");
    grad.addColorStop(1, "#8b5cf6");
    canvasCtx.strokeStyle = grad;

    canvasCtx.beginPath();
    for (let x = 0; x < width; x += 3) {
      const envelope = Math.sin((x / width) * Math.PI);
      const amp = (Math.sin(x * 0.05 + time) * 16 + Math.cos(x * 0.08 - time * 1.5) * 10) * envelope;
      const y = centerY + amp;
      if (x === 0) canvasCtx.moveTo(x, y);
      else canvasCtx.lineTo(x, y);
    }
    canvasCtx.stroke();

    if (isHardwareRecording) {
      requestAnimationFrame(drawSimulatedActiveWave);
    }
  }

  // -------------------------------------------------------------------------
  // Unified Hardware Microphone Recording (Press Enter to Start & Stop)
  // -------------------------------------------------------------------------
  async function startHardwareRecording() {
    if (isHardwareRecording) return;

    isHardwareRecording = true;
    updateMainRecordButton("recording", "[STOP]", "Transcribe [Enter]");

    if (visualizerDot) visualizerDot.style.background = "#ef4444";
    if (visualizerLabel) visualizerLabel.textContent = "Earphones Mic Active (Press Enter to transcribe)...";

    startLiveTimer("REC");
    cancelAnimationFrame(animFrameId);
    drawSimulatedActiveWave();

    try {
      const selectedSource = selectMicDevice ? selectMicDevice.value : "";
      const resp = await fetch("/api/hardware_mic/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ device: selectedSource }),
      });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.error || "Failed to start recording");
    } catch (err) {
      console.error(err);
      alert("Hardware Mic Error: " + err.message);
      resetHardwareUi();
    }
  }

  async function stopHardwareRecording() {
    if (!isHardwareRecording) return;
    stopLiveTimer();

    updateMainRecordButton("idle", "[PROC]", "Transcribing...");
    if (visualizerLabel) visualizerLabel.textContent = "Running Whisper ASR & SLM...";

    try {
      const resp = await fetch(`/api/hardware_mic/stop?engine=${currentEngine}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ engine: currentEngine }),
      });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.error || "Hardware mic processing failed");

      setAudioSource(data.audio_url, data.audio_file);
      renderOutput(data);
    } catch (err) {
      console.error(err);
      alert("Hardware Mic Processing Error: " + err.message);
    } finally {
      resetHardwareUi();
      loadRecordingsList();
      fetchLiveTestsSummary();
    }
  }

  function resetHardwareUi() {
    stopLiveTimer();
    isHardwareRecording = false;
    updateMainRecordButton("idle", "[REC]", "Record [Enter]");

    if (visualizerDot) visualizerDot.style.background = "#64748b";
    if (visualizerLabel) visualizerLabel.textContent = "Microphone Idle";
    drawIdleWave();
  }

  // -------------------------------------------------------------------------
  // Main Primary Button Click Handler
  // -------------------------------------------------------------------------
  btnToggleRecord?.addEventListener("click", () => {
    if (!isHardwareRecording) {
      startHardwareRecording();
    } else {
      stopHardwareRecording();
    }
  });

  // -------------------------------------------------------------------------
  // Global Enter Key Handler (Press Enter to Start / Stop)
  // -------------------------------------------------------------------------
  window.addEventListener("keydown", (e) => {
    // If focused on an input or textarea, let normal typing occur
    if (document.activeElement && (document.activeElement.tagName === "INPUT" || document.activeElement.tagName === "TEXTAREA")) {
      return;
    }

    if (e.key === "Enter") {
      e.preventDefault();
      if (!isHardwareRecording) {
        startHardwareRecording();
      } else {
        stopHardwareRecording();
      }
    } else if (e.key === " " || e.code === "Space") {
      e.preventDefault();
      if (!isHardwareRecording) {
        startHardwareRecording();
      } else {
        stopHardwareRecording();
      }
    }
  });

  // -------------------------------------------------------------------------
  // Live Tests Session Summary & Multi-Format Exporters
  // -------------------------------------------------------------------------
  async function fetchLiveTestsSummary() {
    try {
      const resp = await fetch("/api/live_tests/summary");
      if (!resp.ok) return;
      const data = await resp.json();
      if (sessionPathTag) sessionPathTag.textContent = `live_tests/${data.session_id}/`;
      const count = data.total_utterances ?? data.utterance_count ?? 0;
      if (sessionCountBadge) sessionCountBadge.textContent = `${count} utterance${count === 1 ? '' : 's'}`;
    } catch (e) {
      console.warn("Could not fetch live tests summary:", e);
    }
  }

  fetchLiveTestsSummary();

  btnExportSrt?.addEventListener("click", () => {
    window.open("/api/live_tests/export/srt", "_blank");
  });
  btnExportVtt?.addEventListener("click", () => {
    window.open("/api/live_tests/export/vtt", "_blank");
  });
  btnExportMd?.addEventListener("click", () => {
    window.open("/api/live_tests/export/md", "_blank");
  });
  btnExportTex?.addEventListener("click", () => {
    window.open("/api/live_tests/export/tex", "_blank");
  });
  btnExportTxt?.addEventListener("click", () => {
    window.open("/api/live_tests/export/txt", "_blank");
  });
  btnExportZip?.addEventListener("click", () => {
    window.open("/api/live_tests/export/zip", "_blank");
  });

  btnNewSession?.addEventListener("click", async () => {
    if (confirm("Start a new test session in live_tests? Previous outputs remain saved.")) {
      try {
        const resp = await fetch("/api/live_tests/new_session", { method: "POST" });
        const data = await resp.json();
        alert(`New session started: ${data.session_id}`);
        sessionHistoryRecords.length = 0;
        if (historyList) historyList.innerHTML = '<div class="history-empty">New test session active. Press Enter to record.</div>';
        fetchLiveTestsSummary();
      } catch (e) {
        alert("Failed to start new session: " + e.message);
      }
    }
  });

  btnClearHistory?.addEventListener("click", () => {
    sessionHistoryRecords.length = 0;
    if (historyList) historyList.innerHTML = '<div class="history-empty">History view cleared. Data remains safely saved in live_tests/.</div>';
  });

  // -------------------------------------------------------------------------
  // File Upload Handling
  // -------------------------------------------------------------------------
  btnUpload?.addEventListener("click", () => fileInput?.click());

  fileInput?.addEventListener("change", async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const audioUrl = URL.createObjectURL(file);
    setAudioSource(audioUrl, file.name);

    if (visualizerLabel) visualizerLabel.textContent = `Processing ${file.name}...`;
    await uploadAndTranscribe(file);
  });

  async function uploadAndTranscribe(blob) {
    const formData = new FormData();
    formData.append("audio", blob, "speech.wav");

    try {
      const resp = await fetch(`/api/transcribe_audio?engine=${currentEngine}`, {
        method: "POST",
        body: formData,
      });
      const rawText = await resp.text();
      let data = {};
      try { data = JSON.parse(rawText); } catch (_) { throw new Error(rawText || `Server error ${resp.status}`); }
      if (!resp.ok) throw new Error(data.error || "Transcription failed");

      if (data.audio_url) {
        setAudioSource(data.audio_url, data.audio_file);
      }
      renderOutput(data);
    } catch (err) {
      console.error(err);
      alert("Error: " + err.message);
    } finally {
      loadRecordingsList();
      fetchLiveTestsSummary();
    }
  }

  // -------------------------------------------------------------------------
  // Benchmark Audio Selector
  // -------------------------------------------------------------------------
  async function loadRecordingsList() {
    try {
      const res = await fetch("/api/recordings");
      const data = await res.json();
      if (!savedRecordingsSelect) return;
      savedRecordingsSelect.innerHTML = '<option value="">Benchmark WAV...</option>';
      data.recordings.forEach(r => {
        const opt = document.createElement("option");
        opt.value = r.filename;
        opt.textContent = `${r.filename} (${r.size_kb} KB)`;
        savedRecordingsSelect.appendChild(opt);
      });
    } catch (err) {
      console.warn("Could not load recordings list:", err);
    }
  }
  loadRecordingsList();

  btnRunSaved?.addEventListener("click", async () => {
    const filename = savedRecordingsSelect?.value;
    if (!filename) {
      alert("Please select a benchmark recording first.");
      return;
    }

    setAudioSource(`/recordings/${filename}`, filename);
    if (visualizerLabel) visualizerLabel.textContent = `Processing ${filename}...`;

    try {
      const res = await fetch("/api/transcribe_saved", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ filename, engine: currentEngine }),
      });
      const rawText = await res.text();
      let data = {};
      try { data = JSON.parse(rawText); } catch (_) { throw new Error(rawText || `Server error ${res.status}`); }
      if (!res.ok) throw new Error(data.error || "Processing failed");

      renderOutput(data);
    } catch (err) {
      alert("Error running recording: " + err.message);
    } finally {
      fetchLiveTestsSummary();
    }
  });

  // -------------------------------------------------------------------------
  // Direct Text Input Simulation
  // -------------------------------------------------------------------------
  btnSendText?.addEventListener("click", sendDirectText);
  directTextInput?.addEventListener("keydown", (e) => {
    if (e.key === "Enter") sendDirectText();
  });

  async function sendDirectText() {
    const text = directTextInput?.value.trim();
    if (!text) return;

    if (visualizerLabel) visualizerLabel.textContent = "Analyzing input text...";
    try {
      const res = await fetch("/api/transcribe_text", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, engine: currentEngine }),
      });
      const rawText = await res.text();
      let data = {};
      try { data = JSON.parse(rawText); } catch (_) { throw new Error(rawText || `Server error ${res.status}`); }
      if (!res.ok) throw new Error(data.error || "Processing failed");

      clearAudioSource();
      renderOutput(data);
    } catch (err) {
      alert("Error: " + err.message);
    } finally {
      fetchLiveTestsSummary();
    }
  }

  // -------------------------------------------------------------------------
  // Audio Player State & Helpers
  // -------------------------------------------------------------------------
  function setAudioSource(url, filename) {
    currentAudioUrl = url;
    currentAudioFilename = filename;
    if (audioPlayback) audioPlayback.src = url;
    if (audioPlayerWrapper) audioPlayerWrapper.style.display = "block";

    if (btnDownloadAudioPlayer) btnDownloadAudioPlayer.disabled = false;
    if (btnDownloadOutputAudio) btnDownloadOutputAudio.disabled = false;
  }

  function clearAudioSource() {
    currentAudioUrl = null;
    currentAudioFilename = null;
    if (audioPlayerWrapper) audioPlayerWrapper.style.display = "none";
    if (btnDownloadAudioPlayer) btnDownloadAudioPlayer.disabled = true;
    if (btnDownloadOutputAudio) btnDownloadOutputAudio.disabled = true;
  }

  function downloadAudioFile(filename, fallbackUrl) {
    const url = filename ? `/api/download_audio/${filename}` : fallbackUrl;
    if (!url) return;
    const a = document.createElement("a");
    a.href = url;
    a.download = filename || `audio_${Date.now()}.wav`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  }

  btnDownloadAudioPlayer?.addEventListener("click", () => {
    downloadAudioFile(currentAudioFilename, currentAudioUrl);
  });
  btnDownloadOutputAudio?.addEventListener("click", () => {
    downloadAudioFile(currentAudioFilename, currentAudioUrl);
  });

  // -------------------------------------------------------------------------
  // KaTeX Visual LaTeX Renderer
  // -------------------------------------------------------------------------
  function escapeHtml(str) {
    if (!str) return "";
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  function renderLatexToVisualHtml(text, routeType) {
    if (!text) return '<span style="color: var(--text-dim);">(empty output)</span>';

    if (!window.katex) {
      return `<div class="prose-display">${escapeHtml(text)}</div>`;
    }

    if (routeType === "PURE_ENGLISH" && !text.includes("$") && !text.includes("\\")) {
      return `<div class="prose-display">${escapeHtml(text)}</div>`;
    }

    if (routeType === "PURE_MATH") {
      const cleanFormula = text.replace(/^\$\$|\$\$$/g, "").replace(/^\$|\$$/g, "").trim();
      try {
        const rendered = window.katex.renderToString(cleanFormula, {
          displayMode: true,
          throwOnError: false,
        });
        return `<div class="pure-math-display">${rendered}</div>`;
      } catch (e) {
        return `<div class="pure-math-display"><code>${escapeHtml(cleanFormula)}</code></div>`;
      }
    }

    let resultHtml = "";
    let lastIndex = 0;
    const mathRegex = /\$\$([\s\S]+?)\$\$|\$([^\$\n]+?)\$/g;
    let match;

    while ((match = mathRegex.exec(text)) !== null) {
      const prefix = text.substring(lastIndex, match.index);
      if (prefix) {
        resultHtml += `<span class="prose-text">${escapeHtml(prefix)}</span>`;
      }

      const isDisplay = Boolean(match[1]);
      const mathContent = (match[1] || match[2]).trim();

      try {
        const renderedMath = window.katex.renderToString(mathContent, {
          displayMode: isDisplay,
          throwOnError: false,
        });
        resultHtml += `<span class="inline-math-wrapper">${renderedMath}</span>`;
      } catch (e) {
        resultHtml += `<code class="math-error">$${escapeHtml(mathContent)}$</code>`;
      }

      lastIndex = match.index + match[0].length;
    }

    const suffix = text.substring(lastIndex);
    if (suffix) {
      resultHtml += `<span class="prose-text">${escapeHtml(suffix)}</span>`;
    }

    if (lastIndex === 0 && (text.includes("\\") || text.includes("^{") || text.includes("_{"))) {
      try {
        const rendered = window.katex.renderToString(text.trim(), {
          displayMode: true,
          throwOnError: false,
        });
        return `<div class="pure-math-display">${rendered}</div>`;
      } catch (e) {
        return `<div class="prose-display">${escapeHtml(text)}</div>`;
      }
    }

    return `<div class="mixed-utterance-display">${resultHtml}</div>`;
  }

  // -------------------------------------------------------------------------
  // Rendering Pipeline Output & Typography
  // -------------------------------------------------------------------------
  function renderOutput(data) {
    currentOutputData = data;

    // 1. Raw Whisper Transcript
    if (rawTranscriptBox) {
      rawTranscriptBox.textContent = data.raw_transcript || data.raw_text || "(empty)";
    }

    // 2. Route Badge
    const isSlm = data.engine === "qwen2.5-1.5b" || data.route_type === "slm_direct" || currentEngine === "slm";
    if (routeBadge) {
      if (isSlm) {
        routeBadge.className = "route-badge MIXED";
        routeBadge.textContent = "SLM (Qwen2.5-1.5B)";
      } else {
        routeBadge.className = `route-badge ${data.route_type || "PURE_ENGLISH"}`;
        routeBadge.textContent = data.route_type || "IDLE";
      }
    }

    // 3. Latency Metrics
    if (valAsrTime) valAsrTime.textContent = (data.asr_time_ms || 0) + " ms";
    if (valRouterTime) valRouterTime.textContent = (data.router_time_ms || 0) + " ms";
    if (lblTranslationEngine) {
      lblTranslationEngine.textContent = isSlm ? "SLM (Qwen)" : "T5-v2";
    }
    if (valT5Time) {
      valT5Time.textContent = (data.route_type === "PURE_ENGLISH" ? "0 ms" : (data.router_time_ms || 0) + " ms");
    }
    if (valTotalTime) valTotalTime.textContent = (data.total_time_ms || 0) + " ms";

    // 4. Render Visual LaTeX Typography
    const processed = data.processed_text || "";
    currentLatex = processed;
    if (latexRenderContainer) {
      latexRenderContainer.innerHTML = renderLatexToVisualHtml(processed, data.route_type);
    }

    // 5. LaTeX Code Box
    if (latexCodeCard && latexCodeBox) {
      if (data.route_type !== "PURE_ENGLISH" && (processed.includes("$") || processed.includes("\\") || processed.includes("="))) {
        latexCodeCard.style.display = "block";
        latexCodeBox.textContent = processed;
      } else {
        latexCodeCard.style.display = "none";
      }
    }

    // 6. Math Spans
    if (spansCard && spansBox) {
      if (data.spans && data.spans.length > 0) {
        spansCard.style.display = "block";
        spansBox.innerHTML = data.spans.map(s => `• Math Span: "<strong>${s.spoken}</strong>" → <code>${s.latex || s.latex_text || ""}</code>`).join("<br>");
      } else {
        spansCard.style.display = "none";
      }
    }

    // 7. Enable Output Download Actions
    if (btnDownloadOutputJson) btnDownloadOutputJson.disabled = false;
    if (data.audio_file || data.audio_url) {
      currentAudioFilename = data.audio_file;
      currentAudioUrl = data.audio_url;
      if (btnDownloadOutputAudio) btnDownloadOutputAudio.disabled = false;
    }

    // 8. Add to Session Dataset and History
    recordSessionSample(data);
  }

  // Copy LaTeX
  btnCopyLatex?.addEventListener("click", () => {
    if (!currentLatex) return;
    navigator.clipboard.writeText(currentLatex).then(() => {
      const originalText = btnCopyLatex.innerHTML;
      btnCopyLatex.innerHTML = "<span>Copied</span>";
      setTimeout(() => btnCopyLatex.innerHTML = originalText, 1400);
    });
  });

  // -------------------------------------------------------------------------
  // History Record Management
  // -------------------------------------------------------------------------
  function recordSessionSample(data) {
    const isSlm = data.engine === "qwen2.5-1.5b" || data.route_type === "slm_direct" || currentEngine === "slm";
    const record = {
      timestamp: new Date().toISOString(),
      engine: isSlm ? "slm" : "t5-v2",
      audio_file: data.audio_file || currentAudioFilename || null,
      audio_url: data.audio_url || currentAudioUrl || null,
      raw_transcript: data.raw_transcript || data.raw_text || "",
      route_type: data.route_type,
      processed_text: data.processed_text,
      spans: data.spans || [],
      latency_ms: {
        asr: data.asr_time_ms || 0,
        router: data.router_time_ms || 0,
        total: data.total_time_ms || 0,
      },
    };
    sessionHistoryRecords.unshift(record);

    if (historyList && historyList.children.length === 1 && historyList.children[0].classList.contains("history-empty")) {
      historyList.innerHTML = "";
    }

    const item = document.createElement("div");
    item.className = "history-item";
    const timeStr = new Date().toLocaleTimeString();

    item.innerHTML = `
      <div class="history-item-left">
        <div class="history-raw">${escapeHtml(data.raw_transcript || data.raw_text || "")}</div>
        <div class="history-processed">${renderLatexToVisualHtml(data.processed_text || "", data.route_type)}</div>
      </div>
      <div class="history-meta">
        <div class="history-badge-row">
          <span class="route-badge ${data.route_type || 'PURE_ENGLISH'}">${isSlm ? 'SLM' : (data.route_type || 'MIXED')}</span>
          ${data.audio_file ? `<button class="btn-history-dl" data-file="${data.audio_file}">WAV</button>` : ''}
        </div>
        <div class="history-time-meta">${timeStr} &bull; ${data.total_time_ms || 0} ms</div>
      </div>
    `;

    const dlBtn = item.querySelector(".btn-history-dl");
    if (dlBtn) {
      dlBtn.addEventListener("click", () => {
        downloadAudioFile(dlBtn.getAttribute("data-file"), null);
      });
    }

    if (historyList) historyList.prepend(item);
  }

  // -------------------------------------------------------------------------
  // WebSocket Connection (Real-time Telemetry)
  // -------------------------------------------------------------------------
  let ws = null;
  let wsReconnectTimer = null;

  function initWebSocket() {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${window.location.host}/ws`;
    try {
      ws = new WebSocket(wsUrl);

      ws.onopen = () => {
        console.log("[WS] Connected to Jetson Speech-to-LaTeX server");
        clearTimeout(wsReconnectTimer);
      };

      ws.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data);

          if (msg.type === "telemetry" && isHardwareRecording) {
            if (visualizerDot) visualizerDot.style.background = "#ef4444";
            if (visualizerLabel) visualizerLabel.textContent = `Recording ${msg.elapsed_sec.toFixed(1)}s (Press Enter to transcribe)...`;
            if (recordingTimerText) {
              const mm = String(Math.floor(msg.elapsed_sec / 60)).padStart(2, "0");
              const ss = String(Math.floor(msg.elapsed_sec % 60)).padStart(2, "0");
              recordingTimerText.textContent = `REC ${mm}:${ss}`;
            }
          } else if (msg.type === "transcription_result") {
            setAudioSource(msg.audio_url, msg.audio_file);
            renderOutput(msg);
            resetHardwareUi();
            loadRecordingsList();
            fetchLiveTestsSummary();
          }
        } catch (e) {
          console.error("[WS Message Error]", e);
        }
      };

      ws.onclose = () => {
        ws = null;
        clearTimeout(wsReconnectTimer);
        wsReconnectTimer = setTimeout(initWebSocket, 2500);
      };

      ws.onerror = (err) => {
        console.warn("[WS Error]", err);
      };
    } catch (e) {
      console.warn("[WS Init Warning]", e);
    }
  }

  initWebSocket();
});
