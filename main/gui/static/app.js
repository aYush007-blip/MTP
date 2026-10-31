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
  let isLectureRunning = false;
  let animFrameId = null;
  let timerIntervalId = null;
  let timerStartTime = 0;
  let lastRenderedChunkIndex = 0;
  let lecturePollInterval = null;
  let lectureAccumulatedRaw = "";
  let lectureAccumulatedLatex = "";

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
  // -------------------------------------------------------------------------
  // Jetson Hardware Audio Device Enumeration & Selection
  // -------------------------------------------------------------------------
  // -------------------------------------------------------------------------
  // Jetson Hardware Audio Device Enumeration & Selection
  // -------------------------------------------------------------------------
  let isScanningAudio = false;

  async function fetchAudioDevices(silent = false) {
    if (isScanningAudio && silent) return;
    isScanningAudio = true;

    const btnRefresh = document.getElementById("btnRefreshAudioDevices");
    const lblScanAudio = document.getElementById("lblScanAudio");

    if (btnRefresh && !silent) {
      btnRefresh.classList.add("spinning");
      btnRefresh.disabled = true;
      if (lblScanAudio) lblScanAudio.textContent = "Scanning...";
    }
    if (deviceStatusBadge && !silent) {
      deviceStatusBadge.textContent = "Scanning Hardware Audio...";
      deviceStatusBadge.style.color = "var(--text-muted)";
      deviceStatusBadge.style.borderColor = "var(--bg-card-border)";
    }

    try {
      const url = silent ? "/api/audio_devices" : "/api/audio_devices?rescan=true";
      const resp = await fetch(url);
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json();

      const usbSource = data.sources?.find(s => s.is_earphones || (s.id && (s.id.toLowerCase().includes("ab13x") || s.id.toLowerCase().includes("usb"))));
      const usbSink = data.sinks?.find(s => s.is_earphones || (s.id && (s.id.toLowerCase().includes("ab13x") || s.id.toLowerCase().includes("usb"))));

      // Populate Mic Sources (only rebuild if list changed or manual scan to prevent DOM jitter)
      if (selectMicDevice && Array.isArray(data.sources) && data.sources.length > 0) {
        const curVal = selectMicDevice.value;
        const currentIds = Array.from(selectMicDevice.options).map(o => o.value).join(",");
        const newIds = data.sources.map(s => s.id).join(",");
        if (currentIds !== newIds || !silent) {
          selectMicDevice.innerHTML = "";
          let selectedSet = false;

          data.sources.forEach(src => {
            const opt = document.createElement("option");
            opt.value = src.id;
            opt.textContent = `${src.name}${src.is_earphones ? " (Earphones Mic)" : ""}`;

            if (!silent && usbSource && src.id === usbSource.id) {
              opt.selected = true;
              selectedSet = true;
            } else if (curVal && curVal === src.id && !selectedSet) {
              opt.selected = true;
              selectedSet = true;
            } else if (!selectedSet && (src.is_default || (src.id && src.id.toLowerCase().includes("ab13x")))) {
              opt.selected = true;
              selectedSet = true;
            }
            selectMicDevice.appendChild(opt);
          });

          if (!silent && usbSource) {
            selectMicDevice.value = usbSource.id;
            handleDeviceChange("source", usbSource.id);
          }
        }
      }

      // Populate Earphone/Playback Sinks (only rebuild if list changed or manual scan)
      if (selectOutputDevice && Array.isArray(data.sinks) && data.sinks.length > 0) {
        const curVal = selectOutputDevice.value;
        const currentIds = Array.from(selectOutputDevice.options).map(o => o.value).join(",");
        const newIds = data.sinks.map(s => s.id).join(",");
        if (currentIds !== newIds || !silent) {
          selectOutputDevice.innerHTML = "";
          let selectedSet = false;

          data.sinks.forEach(sink => {
            const opt = document.createElement("option");
            opt.value = sink.id;
            opt.textContent = `${sink.name}${sink.is_earphones ? " (Earphones Playback)" : ""}`;

            if (!silent && usbSink && sink.id === usbSink.id) {
              opt.selected = true;
              selectedSet = true;
            } else if (curVal && curVal === sink.id && !selectedSet) {
              opt.selected = true;
              selectedSet = true;
            } else if (!selectedSet && (sink.is_default || (sink.id && sink.id.toLowerCase().includes("ab13x")))) {
              opt.selected = true;
              selectedSet = true;
            }
            selectOutputDevice.appendChild(opt);
          });

          if (!silent && usbSink) {
            selectOutputDevice.value = usbSink.id;
            handleDeviceChange("sink", usbSink.id);
          }
        }
      }

      // Visual Status Badge Feedback
      if (deviceStatusBadge) {
        const hasUsb = (usbSource != null || usbSink != null);
        if (hasUsb) {
          deviceStatusBadge.textContent = "✓ AB13X Earphones Active";
          deviceStatusBadge.style.color = "var(--accent-emerald)";
          deviceStatusBadge.style.borderColor = "rgba(16, 185, 129, 0.4)";
          deviceStatusBadge.style.background = "rgba(16, 185, 129, 0.15)";
        } else {
          deviceStatusBadge.textContent = silent ? "Built-in Audio Ready" : "Built-in Audio (No Earphones Detected)";
          deviceStatusBadge.style.color = "var(--text-muted)";
          deviceStatusBadge.style.borderColor = "var(--bg-card-border)";
          deviceStatusBadge.style.background = "rgba(255, 255, 255, 0.04)";
        }
      }
    } catch (err) {
      console.warn("Could not reload audio devices:", err);
      if (deviceStatusBadge && !silent) {
        deviceStatusBadge.textContent = "Scan Error - Retrying...";
      }
    } finally {
      isScanningAudio = false;
      if (btnRefresh) {
        setTimeout(() => {
          btnRefresh.classList.remove("spinning");
          btnRefresh.disabled = false;
          if (lblScanAudio) lblScanAudio.textContent = "Scan Audio";
        }, 600);
      }
    }
  }

  // Export to window for global access
  window.fetchAudioDevices = fetchAudioDevices;

  async function handleDeviceChange(type, deviceName) {
    if (!deviceName) return;
    try {
      await fetch("/api/audio_devices/select", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ type, name: deviceName }),
      });
      if (deviceStatusBadge) {
        const prevText = deviceStatusBadge.textContent;
        deviceStatusBadge.textContent = "Device Updated";
        setTimeout(() => {
          if (deviceStatusBadge.textContent === "Device Updated") {
            deviceStatusBadge.textContent = prevText;
          }
        }, 1200);
      }
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

  // Hotplug refresh triggers: manual button, dropdown focus/click, and periodic scan
  document.getElementById("btnRefreshAudioDevices")?.addEventListener("click", () => fetchAudioDevices(false));
  selectMicDevice?.addEventListener("focus", () => fetchAudioDevices(true));
  selectMicDevice?.addEventListener("mousedown", () => fetchAudioDevices(true));
  selectOutputDevice?.addEventListener("focus", () => fetchAudioDevices(true));
  selectOutputDevice?.addEventListener("mousedown", () => fetchAudioDevices(true));

  // Periodic polling every 3s when idle to catch hotplug events automatically
  setInterval(() => {
    if (!isHardwareRecording && !isLectureRunning) {
      fetchAudioDevices(true);
    }
  }, 3000);

  // Call on load
  fetchAudioDevices(false);

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

    if (!isHardwareRecording && !isLectureRunning) {
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

    if (isHardwareRecording || isLectureRunning) {
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
  window.toggleHardwareRecord = () => {
    if (isLectureRunning) {
      stopContinuousLecture();
      return;
    }
    if (!isHardwareRecording) {
      startHardwareRecording();
    } else {
      stopHardwareRecording();
    }
  };

  btnToggleRecord?.addEventListener("click", window.toggleHardwareRecord);

  // -------------------------------------------------------------------------
  // Hands-free Continuous Live Lecture Mode (Auto-VAD streaming)
  // -------------------------------------------------------------------------
  const btnStartLecture = document.getElementById("btnStartLecture");
  const btnStopLecture = document.getElementById("btnStopLecture");
  const lectureRecordIcon = document.getElementById("lectureRecordIcon");
  const lectureRecordLabel = document.getElementById("lectureRecordLabel");

  async function pollRecentLectureChunks() {
    try {
      const resp = await fetch(`/api/lecture/chunks?since_index=${lastRenderedChunkIndex}`);
      if (!resp.ok) return;
      const data = await resp.json();
      if (data.chunks && data.chunks.length > 0) {
        for (const ch of data.chunks) {
          if (ch.index > lastRenderedChunkIndex) {
            lastRenderedChunkIndex = ch.index;
            const chunkData = {
              raw_transcript: ch.raw_transcript,
              processed_text: ch.processed_text,
              asr_time_ms: ch.asr_time_ms,
              router_time_ms: ch.router_time_ms,
              total_time_ms: Math.round((ch.asr_time_ms || 0) + (ch.router_time_ms || 0)),
              route_type: (ch.processed_text || "").includes("$") ? "MIXED" : "PURE_ENGLISH",
              engine: "qwen2.5-1.5b",
              audio_url: ch.audio_url,
              audio_file: `chunk_${ch.index}.wav`,
              start_sec: ch.start_sec,
              end_sec: ch.end_sec,
              start_time: ch.start_time,
              end_time: ch.end_time,
              is_continuous_chunk: true,
            };
            renderOutput(chunkData);
          }
        }
        fetchLiveTestsSummary();
      }
    } catch (e) {
      console.warn("[Lecture Polling]", e);
    }
  }

  async function startContinuousLecture() {
    if (isLectureRunning) return;
    console.log("[Continuous Lecture] Starting...");
    if (isHardwareRecording) {
      await stopHardwareRecording();
    }
    isLectureRunning = true;
    lastRenderedChunkIndex = 0;
    lectureAccumulatedRaw = "";
    lectureAccumulatedLatex = "";

    if (rawTranscriptBox) {
      rawTranscriptBox.textContent = "Listening to live lecture... Speak freely.";
    }
    if (latexRenderContainer) {
      latexRenderContainer.style.justifyContent = "flex-start";
      latexRenderContainer.style.textAlign = "left";
      latexRenderContainer.innerHTML = '<span style="color: var(--text-dim); font-style: italic;">Continuous lecture text and equations will stream here in real time...</span>';
    }

    // Instant Visual State Change
    if (btnStartLecture) {
      btnStartLecture.className = "btn-main-record lecture-active";
      btnStartLecture.style.background = "linear-gradient(135deg, #d97706, #b45309)";
      btnStartLecture.disabled = true;
      btnStartLecture.style.opacity = "0.75";
      btnStartLecture.style.cursor = "default";
      if (lectureRecordIcon) lectureRecordIcon.textContent = "[STREAMING]";
      if (lectureRecordLabel) lectureRecordLabel.textContent = "Live Streaming...";
    }
    if (btnStopLecture) {
      btnStopLecture.disabled = false;
      btnStopLecture.className = "btn-main-record recording";
      btnStopLecture.style.background = "linear-gradient(135deg, #dc2626, #b91c1c)";
      btnStopLecture.style.color = "#ffffff";
      btnStopLecture.style.border = "1px solid #ef4444";
      btnStopLecture.style.opacity = "1";
      btnStopLecture.style.cursor = "pointer";
    }

    if (visualizerDot) visualizerDot.style.background = "#f59e0b";
    if (visualizerLabel) visualizerLabel.textContent = "Continuous Mode: Speak freely (auto-detects pauses & streams live)...";
    startLiveTimer("LIVE");
    cancelAnimationFrame(animFrameId);
    drawSimulatedActiveWave();

    // Start robust backup polling every 900ms to guarantee live chunk updates even if WebSocket has jitter
    if (lecturePollInterval) clearInterval(lecturePollInterval);
    lecturePollInterval = setInterval(() => {
      if (isLectureRunning) {
        pollRecentLectureChunks();
      }
    }, 900);

    try {
      const resp = await fetch("/api/lecture/start", { method: "POST" });
      const res = await resp.json();
      if (!resp.ok && res.status !== "already_running") {
        throw new Error(res.error || "Failed to start");
      }
    } catch (e) {
      console.error(e);
      alert("Could not start continuous lecture: " + e.message);
      stopContinuousLecture();
    }
  }

  async function stopContinuousLecture() {
    if (!isLectureRunning) return;
    console.log("[Continuous Lecture] Stopping...");
    isLectureRunning = false;
    if (lecturePollInterval) {
      clearInterval(lecturePollInterval);
      lecturePollInterval = null;
    }

    // Reset Buttons
    if (btnStartLecture) {
      btnStartLecture.disabled = false;
      btnStartLecture.className = "btn-main-record idle";
      btnStartLecture.style.background = "linear-gradient(135deg, #4338ca, #6366f1)";
      btnStartLecture.style.opacity = "1";
      btnStartLecture.style.cursor = "pointer";
      if (lectureRecordIcon) lectureRecordIcon.textContent = "[LIVE]";
      if (lectureRecordLabel) lectureRecordLabel.textContent = "Start Live Lecture";
    }
    if (btnStopLecture) {
      btnStopLecture.disabled = true;
      btnStopLecture.className = "btn-main-record idle";
      btnStopLecture.style.background = "#1e293b";
      btnStopLecture.style.color = "#64748b";
      btnStopLecture.style.border = "1px solid #334155";
      btnStopLecture.style.opacity = "0.45";
      btnStopLecture.style.cursor = "not-allowed";
    }

    if (visualizerDot) visualizerDot.style.background = "#64748b";
    if (visualizerLabel) visualizerLabel.textContent = "Microphone Idle";
    stopLiveTimer();
    drawIdleWave();

    try {
      const resp = await fetch("/api/lecture/stop", { method: "POST" });
      await resp.json();
    } catch (e) {
      console.warn(e);
    } finally {
      // Final poll to capture any in-flight chunk
      setTimeout(() => {
        pollRecentLectureChunks();
        fetchLiveTestsSummary();
        loadRecordingsList();
      }, 500);
    }
  }

  window.startContinuousLecture = startContinuousLecture;
  window.stopContinuousLecture = stopContinuousLecture;

  btnStartLecture?.addEventListener("click", startContinuousLecture);
  btnStopLecture?.addEventListener("click", stopContinuousLecture);


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
    const isContinuous = isLectureRunning || Boolean(data.is_continuous_chunk);

    if (isContinuous) {
      const newRaw = (data.raw_transcript || data.raw_text || "").trim();
      const newProcessed = (data.processed_text || "").trim();

      if (newRaw) {
        lectureAccumulatedRaw = lectureAccumulatedRaw ? `${lectureAccumulatedRaw} ${newRaw}` : newRaw;
      }
      if (newProcessed) {
        lectureAccumulatedLatex = lectureAccumulatedLatex ? `${lectureAccumulatedLatex} ${newProcessed}` : newProcessed;
      }

      // 1. Continuous Raw Whisper Transcript
      if (rawTranscriptBox) {
        rawTranscriptBox.textContent = lectureAccumulatedRaw || "(listening...)";
        rawTranscriptBox.scrollTop = rawTranscriptBox.scrollHeight;
      }

      // 2. Route Badge
      if (routeBadge) {
        routeBadge.className = "route-badge MIXED";
        routeBadge.textContent = "LIVE LECTURE STREAMING";
      }

      // 3. Latency Metrics (displays latest chunk processing latency)
      if (valAsrTime) valAsrTime.textContent = (data.asr_time_ms || 0) + " ms";
      if (valRouterTime) valRouterTime.textContent = (data.router_time_ms || 0) + " ms";
      if (lblTranslationEngine) lblTranslationEngine.textContent = "SLM (Qwen)";
      if (valT5Time) valT5Time.textContent = (data.router_time_ms || 0) + " ms";
      if (valTotalTime) valTotalTime.textContent = (data.total_time_ms || 0) + " ms";

      // 4. Render Visual LaTeX Typography (Continuous flowing lecture text & formulas)
      currentLatex = lectureAccumulatedLatex;
      if (latexRenderContainer) {
        latexRenderContainer.style.justifyContent = "flex-start";
        latexRenderContainer.style.textAlign = "left";
        latexRenderContainer.innerHTML = renderLatexToVisualHtml(lectureAccumulatedLatex, "MIXED");
        latexRenderContainer.scrollTop = latexRenderContainer.scrollHeight;
      }

      // 5. LaTeX Code Box (Continuous)
      if (latexCodeCard && latexCodeBox) {
        if (lectureAccumulatedLatex.includes("$") || lectureAccumulatedLatex.includes("\\") || lectureAccumulatedLatex.includes("=")) {
          latexCodeCard.style.display = "block";
          latexCodeBox.textContent = lectureAccumulatedLatex;
          latexCodeBox.scrollTop = latexCodeBox.scrollHeight;
        } else {
          latexCodeCard.style.display = "none";
        }
      }
    } else {
      // Single Utterance Mode
      if (rawTranscriptBox) {
        rawTranscriptBox.textContent = data.raw_transcript || data.raw_text || "(empty)";
      }

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

      if (valAsrTime) valAsrTime.textContent = (data.asr_time_ms || 0) + " ms";
      if (valRouterTime) valRouterTime.textContent = (data.router_time_ms || 0) + " ms";
      if (lblTranslationEngine) lblTranslationEngine.textContent = isSlm ? "SLM (Qwen)" : "T5-v2";
      if (valT5Time) {
        valT5Time.textContent = (data.route_type === "PURE_ENGLISH" ? "0 ms" : (data.router_time_ms || 0) + " ms");
      }
      if (valTotalTime) valTotalTime.textContent = (data.total_time_ms || 0) + " ms";

      const processed = data.processed_text || "";
      currentLatex = processed;
      if (latexRenderContainer) {
        latexRenderContainer.style.justifyContent = "center";
        latexRenderContainer.style.textAlign = "center";
        latexRenderContainer.innerHTML = renderLatexToVisualHtml(processed, data.route_type);
      }

      if (latexCodeCard && latexCodeBox) {
        if (data.route_type !== "PURE_ENGLISH" && (processed.includes("$") || processed.includes("\\") || processed.includes("="))) {
          latexCodeCard.style.display = "block";
          latexCodeBox.textContent = processed;
        } else {
          latexCodeCard.style.display = "none";
        }
      }

      if (spansCard && spansBox) {
        if (data.spans && data.spans.length > 0) {
          spansCard.style.display = "block";
          spansBox.innerHTML = data.spans.map(s => `• Math Span: "<strong>${s.spoken}</strong>" → <code>${s.latex || s.latex_text || ""}</code>`).join("<br>");
        } else {
          spansCard.style.display = "none";
        }
      }
    }

    // Enable Output Download Actions
    if (btnDownloadOutputJson) btnDownloadOutputJson.disabled = false;
    if (data.audio_file || data.audio_url) {
      currentAudioFilename = data.audio_file;
      currentAudioUrl = data.audio_url;
      if (btnDownloadOutputAudio) btnDownloadOutputAudio.disabled = false;
    }

    // Add to Session Dataset and History
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

  // Clear Continuous Live Text
  const btnClearLiveText = document.getElementById("btnClearLiveText");
  btnClearLiveText?.addEventListener("click", () => {
    lectureAccumulatedRaw = "";
    lectureAccumulatedLatex = "";
    currentLatex = "";
    if (rawTranscriptBox) rawTranscriptBox.textContent = "(cleared)";
    if (latexRenderContainer) {
      latexRenderContainer.style.justifyContent = "center";
      latexRenderContainer.style.textAlign = "center";
      latexRenderContainer.innerHTML = '<span style="color: var(--text-dim);">Formula or typography will render here</span>';
    }
    if (latexCodeBox) latexCodeBox.textContent = "";
    if (latexCodeCard) latexCodeCard.style.display = "none";
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

          if (msg.type === "telemetry" || msg.type === "lecture_telemetry") {
            const isRec = isHardwareRecording;
            const isLec = isLectureRunning || msg.is_lecture_running;
            if (isRec || isLec) {
              if (visualizerDot) visualizerDot.style.background = isLec ? "#f59e0b" : "#ef4444";
              if (visualizerLabel) {
                const elap = msg.elapsed_sec ? msg.elapsed_sec.toFixed(1) : "0.0";
                const utts = msg.utterance_count ? ` (${msg.utterance_count} chunks captured)` : "";
                visualizerLabel.textContent = isLec
                  ? `Live Lecture Streaming ${elap}s${utts}...`
                  : `Recording ${elap}s (Press Enter to transcribe)...`;
              }
              if (recordingTimerText && msg.elapsed_sec !== undefined) {
                const mm = String(Math.floor(msg.elapsed_sec / 60)).padStart(2, "0");
                const ss = String(Math.floor(msg.elapsed_sec % 60)).padStart(2, "0");
                recordingTimerText.textContent = `${isLec ? "LIVE" : "REC"} ${mm}:${ss}`;
              }
            }
          } else if (msg.type === "lecture_chunk") {
            // Live streamed chunk from continuous lecture worker
            if (msg.index && msg.index <= lastRenderedChunkIndex) {
              return; // Already rendered by backup poll
            }
            if (msg.index) {
              lastRenderedChunkIndex = Math.max(lastRenderedChunkIndex, msg.index);
            }
            const chunkData = {
              raw_transcript: msg.raw_transcript,
              processed_text: msg.processed_text,
              asr_time_ms: msg.asr_time_ms,
              router_time_ms: msg.router_time_ms,
              total_time_ms: Math.round((msg.asr_time_ms || 0) + (msg.router_time_ms || 0)),
              route_type: (msg.processed_text || "").includes("$") ? "MIXED" : "PURE_ENGLISH",
              engine: "qwen2.5-1.5b",
              audio_url: msg.audio_url,
              audio_file: `chunk_${msg.index}.wav`,
              start_sec: msg.start_sec,
              end_sec: msg.end_sec,
              start_time: msg.start_time,
              end_time: msg.end_time,
              is_continuous_chunk: true,
            };
            renderOutput(chunkData);
            fetchLiveTestsSummary();
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
