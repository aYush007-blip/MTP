/* ==========================================================================
   Jetson Orin Speech-to-LaTeX Minimal GUI - Frontend Controller
   Continuous Audio Recording + LaTeX Rendering + Dataset / Learning Export
   ========================================================================== */

document.addEventListener("DOMContentLoaded", () => {
  // UI Elements - Inputs & Capture
  const btnRecord = document.getElementById("btnRecord");
  const btnStopBrowser = document.getElementById("btnStopBrowser");
  const recordLabel = document.getElementById("recordLabel");
  const recordIcon = document.getElementById("recordIcon");

  const btnRecordHw = document.getElementById("btnRecordHw");
  const btnStopHw = document.getElementById("btnStopHw");
  const hwRecordLabel = document.getElementById("hwRecordLabel");
  const hwRecordIcon = document.getElementById("hwRecordIcon");

  const recordingTimerBadge = document.getElementById("recordingTimerBadge");
  const recordingTimerText = document.getElementById("recordingTimerText");

  const btnUpload = document.getElementById("btnUpload");
  const fileInput = document.getElementById("fileInput");
  const savedRecordingsSelect = document.getElementById("savedRecordingsSelect");
  const btnRunSaved = document.getElementById("btnRunSaved");
  const directTextInput = document.getElementById("directTextInput");
  const btnSendText = document.getElementById("btnSendText");

  // UI Elements - Audio Player & Downloads
  const audioPlayback = document.getElementById("audioPlayback");
  const audioPlayerWrapper = document.getElementById("audioPlayerWrapper");
  const btnDownloadAudioPlayer = document.getElementById("btnDownloadAudioPlayer");
  const audioCanvas = document.getElementById("audioCanvas");
  const visualizerDot = document.getElementById("visualizerDot");
  const visualizerLabel = document.getElementById("visualizerLabel");

  // UI Elements - Outputs & Actions
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

  const valAsrTime = document.getElementById("valAsrTime");
  const valRouterTime = document.getElementById("valRouterTime");
  const valT5Time = document.getElementById("valT5Time");
  const valTotalTime = document.getElementById("valTotalTime");
  const lblTranslationEngine = document.getElementById("lblTranslationEngine");

  // UI Elements - Engine Selector
  const btnEngineSlm = document.getElementById("btnEngineSlm");
  const btnEngineT5 = document.getElementById("btnEngineT5");
  let currentEngine = "slm"; // Default: Qwen2.5-1.5B (SLM)

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

  // UI Elements - History
  const historyList = document.getElementById("historyList");
  const btnClearHistory = document.getElementById("btnClearHistory");
  const btnExportSessionJson = document.getElementById("btnExportSessionJson");

  // State
  let isBrowserRecording = false;
  let isHardwareRecording = false;
  let audioContext = null;
  let mediaStream = null;
  let scriptProcessor = null;
  let recordedPcmChunks = [];
  let animFrameId = null;
  let timerIntervalId = null;
  let timerStartTime = 0;
  let hwCountdownTimerId = null;

  // Active Session & Export State
  let currentOutputData = null;
  let currentAudioUrl = null;
  let currentAudioFilename = null;
  let currentLatex = "";
  const sessionHistoryRecords = [];

  // -------------------------------------------------------------------------
  // Live Recording Elapsed Timer Badge
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
  // Audio Visualizer Setup
  // -------------------------------------------------------------------------
  const canvasCtx = audioCanvas.getContext("2d");

  function resizeCanvas() {
    if (audioCanvas && audioCanvas.parentElement) {
      audioCanvas.width = audioCanvas.parentElement.clientWidth;
      audioCanvas.height = 100;
    }
  }
  window.addEventListener("resize", resizeCanvas);
  resizeCanvas();

  function drawIdleWave() {
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

    if (!isBrowserRecording && !isHardwareRecording) {
      animFrameId = requestAnimationFrame(drawIdleWave);
    }
  }
  drawIdleWave();

  function drawSimulatedActiveWave() {
    canvasCtx.fillStyle = "rgba(2, 6, 23, 0.35)";
    canvasCtx.fillRect(0, 0, audioCanvas.width, audioCanvas.height);

    const width = audioCanvas.width;
    const height = audioCanvas.height;
    const centerY = height / 2;
    const time = Date.now() * 0.006;

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
  // 16kHz 16-bit Mono PCM WAV Encoder (Browser Audio API)
  // -------------------------------------------------------------------------
  function encodePcmToWavBlob(chunks, inputSampleRate, targetSampleRate = 16000) {
    let totalLength = 0;
    for (const c of chunks) totalLength += c.length;
    const merged = new Float32Array(totalLength);
    let offset = 0;
    for (const c of chunks) {
      merged.set(c, offset);
      offset += c.length;
    }

    let resampled = merged;
    if (inputSampleRate !== targetSampleRate && merged.length > 0) {
      const ratio = inputSampleRate / targetSampleRate;
      const newLength = Math.round(merged.length / ratio);
      resampled = new Float32Array(newLength);
      for (let i = 0; i < newLength; i++) {
        const origIndex = i * ratio;
        const indexFloor = Math.floor(origIndex);
        const indexCeil = Math.min(merged.length - 1, Math.ceil(origIndex));
        const fraction = origIndex - indexFloor;
        resampled[i] = merged[indexFloor] * (1 - fraction) + merged[indexCeil] * fraction;
      }
    }

    const pcm16 = new Int16Array(resampled.length);
    for (let i = 0; i < resampled.length; i++) {
      const s = Math.max(-1, Math.min(1, resampled[i]));
      pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
    }

    const numChannels = 1;
    const bitsPerSample = 16;
    const byteRate = targetSampleRate * numChannels * (bitsPerSample / 8);
    const blockAlign = numChannels * (bitsPerSample / 8);
    const dataSize = pcm16.length * 2;
    const buffer = new ArrayBuffer(44 + dataSize);
    const view = new DataView(buffer);

    function writeString(v, off, str) {
      for (let i = 0; i < str.length; i++) {
        v.setUint8(off + i, str.charCodeAt(i));
      }
    }

    writeString(view, 0, "RIFF");
    view.setUint32(4, 36 + dataSize, true);
    writeString(view, 8, "WAVE");

    writeString(view, 12, "fmt ");
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, numChannels, true);
    view.setUint32(24, targetSampleRate, true);
    view.setUint32(28, byteRate, true);
    view.setUint16(32, blockAlign, true);
    view.setUint16(34, bitsPerSample, true);

    writeString(view, 36, "data");
    view.setUint32(40, dataSize, true);

    const pcmUint8 = new Uint8Array(pcm16.buffer);
    new Uint8Array(buffer, 44).set(pcmUint8);

    return new Blob([buffer], { type: "audio/wav" });
  }

  // -------------------------------------------------------------------------
  // Mode 1: Jetson Onboard Hardware USB Mic (Continuous / Unlimited)
  // -------------------------------------------------------------------------
  async function startHardwareRecording() {
    if (isHardwareRecording || isBrowserRecording) return;

    isHardwareRecording = true;

    btnRecordHw.classList.add("recording");
    btnRecordHw.disabled = true;
    hwRecordIcon.textContent = "[REC]";
    hwRecordLabel.textContent = "Recording...";
    btnStopHw.disabled = false;
    btnStopHw.classList.add("btn-stop-active");

    btnRecord.disabled = true;
    if (btnStopBrowser) btnStopBrowser.disabled = true;
    btnUpload.disabled = true;

    visualizerDot.style.background = "#ef4444";
    visualizerLabel.textContent = "Hardware mic active (Press Enter to transcribe)...";

    startLiveTimer("REC");
    cancelAnimationFrame(animFrameId);
    drawSimulatedActiveWave();

    try {
      const resp = await fetch("/api/hardware_mic/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}),
      });
      const rawText = await resp.text();
      let data = {};
      try { data = JSON.parse(rawText); } catch (_) { throw new Error(rawText || `Server error ${resp.status}`); }
      if (!resp.ok) throw new Error(data.error || "Recording failed to start");
    } catch (err) {
      console.error(err);
      alert("Hardware Mic Error: " + err.message);
      resetHardwareUi();
    }
  }

  async function stopHardwareRecording() {
    if (!isHardwareRecording) return;
    clearTimeout(hwCountdownTimerId);
    stopLiveTimer();

    btnStopHw.disabled = true;
    btnStopHw.classList.remove("btn-stop-active");
    hwRecordLabel.textContent = "Processing...";
    visualizerLabel.textContent = "Running Whisper ASR & T5...";

    try {
      const resp = await fetch(`/api/hardware_mic/stop?engine=${currentEngine}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ engine: currentEngine }),
      });
      const rawText = await resp.text();
      let data = {};
      try { data = JSON.parse(rawText); } catch (_) { throw new Error(rawText || `Server error ${resp.status}`); }
      if (!resp.ok) throw new Error(data.error || "Hardware mic processing failed");

      setAudioSource(data.audio_url, data.audio_file);
      renderOutput(data);
    } catch (err) {
      console.error(err);
      alert("Hardware Mic Processing Error: " + err.message);
    } finally {
      resetHardwareUi();
      loadRecordingsList();
    }
  }

  function resetHardwareUi() {
    clearTimeout(hwCountdownTimerId);
    stopLiveTimer();
    isHardwareRecording = false;
    btnRecordHw.classList.remove("recording");
    btnRecordHw.disabled = false;
    hwRecordIcon.textContent = "[HW]";
    hwRecordLabel.textContent = "Hardware Mic";
    btnStopHw.disabled = true;
    btnStopHw.classList.remove("btn-stop-active");

    btnRecord.disabled = false;
    if (btnStopBrowser) btnStopBrowser.disabled = true;
    btnUpload.disabled = false;

    visualizerDot.style.background = "#64748b";
    visualizerLabel.textContent = "Microphone Idle";
    drawIdleWave();
  }

  btnRecordHw.addEventListener("click", startHardwareRecording);
  btnStopHw.addEventListener("click", stopHardwareRecording);

  // -------------------------------------------------------------------------
  // Mode 2: Browser Microphone Recording (Web Audio 16kHz PCM WAV)
  // -------------------------------------------------------------------------
  btnRecord.addEventListener("click", async () => {
    if (isHardwareRecording) return;
    if (!isBrowserRecording) {
      await startBrowserRecording();
    } else {
      stopBrowserRecording();
    }
  });

  if (btnStopBrowser) {
    btnStopBrowser.addEventListener("click", () => {
      if (isBrowserRecording) stopBrowserRecording();
    });
  }

  async function startBrowserRecording() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      alert("Browser microphone not available. Use Hardware Mic to capture audio directly.");
      return;
    }

    try {
      mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });

      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      audioContext = new AudioCtx();
      const source = audioContext.createMediaStreamSource(mediaStream);

      const analyser = audioContext.createAnalyser();
      analyser.fftSize = 256;
      source.connect(analyser);

      const bufferSize = 4096;
      scriptProcessor = audioContext.createScriptProcessor(bufferSize, 1, 1);
      recordedPcmChunks = [];

      scriptProcessor.onaudioprocess = (e) => {
        if (!isBrowserRecording) return;
        const channelData = e.inputBuffer.getChannelData(0);
        recordedPcmChunks.push(new Float32Array(channelData));
      };

      source.connect(scriptProcessor);
      scriptProcessor.connect(audioContext.destination);

      isBrowserRecording = true;
      btnRecord.classList.add("recording");
      btnRecord.disabled = true;
      recordLabel.textContent = "Recording...";
      recordIcon.textContent = "[REC]";

      if (btnStopBrowser) {
        btnStopBrowser.disabled = false;
        btnStopBrowser.classList.add("btn-stop-active");
      }

      btnRecordHw.disabled = true;
      btnStopHw.disabled = true;
      btnUpload.disabled = true;

      visualizerDot.style.background = "#ef4444";
      visualizerLabel.textContent = "Browser mic active (Press Enter to transcribe)...";
      startLiveTimer("REC");

      cancelAnimationFrame(animFrameId);
      const bufferLength = analyser.frequencyBinCount;
      const dataArray = new Uint8Array(bufferLength);

      function renderLiveSpectrum() {
        if (!isBrowserRecording) return;
        analyser.getByteFrequencyData(dataArray);

        canvasCtx.fillStyle = "rgba(2, 6, 23, 0.4)";
        canvasCtx.fillRect(0, 0, audioCanvas.width, audioCanvas.height);

        const barWidth = (audioCanvas.width / bufferLength) * 2.2;
        let x = 0;

        for (let i = 0; i < bufferLength; i++) {
          const barHeight = (dataArray[i] / 255) * audioCanvas.height * 0.85;
          const grad = canvasCtx.createLinearGradient(0, audioCanvas.height, 0, 0);
          grad.addColorStop(0, "#06b6d4");
          grad.addColorStop(1, "#8b5cf6");
          canvasCtx.fillStyle = grad;
          canvasCtx.fillRect(x, audioCanvas.height - barHeight, barWidth, barHeight);
          x += barWidth + 2;
        }

        requestAnimationFrame(renderLiveSpectrum);
      }
      renderLiveSpectrum();

    } catch (err) {
      console.error("Microphone error:", err);
      alert("Microphone error: " + err.message);
      resetBrowserUi();
    }
  }

  function stopBrowserRecording() {
    if (!isBrowserRecording) return;
    isBrowserRecording = false;
    stopLiveTimer();

    if (scriptProcessor) {
      scriptProcessor.disconnect();
      scriptProcessor = null;
    }
    if (mediaStream) {
      mediaStream.getTracks().forEach(track => track.stop());
      mediaStream = null;
    }

    resetBrowserUi();
    visualizerLabel.textContent = "Transcribing audio...";
    drawIdleWave();

    const inputSampleRate = audioContext ? audioContext.sampleRate : 44100;
    const wavBlob = encodePcmToWavBlob(recordedPcmChunks, inputSampleRate, 16000);

    const clientAudioUrl = URL.createObjectURL(wavBlob);
    setAudioSource(clientAudioUrl, `client_mic_${Date.now()}.wav`);

    uploadAndTranscribe(wavBlob);
  }

  function resetBrowserUi() {
    isBrowserRecording = false;
    btnRecord.classList.remove("recording");
    btnRecord.disabled = false;
    recordLabel.textContent = "Browser Mic";
    recordIcon.textContent = "[WEB]";

    if (btnStopBrowser) {
      btnStopBrowser.disabled = true;
      btnStopBrowser.classList.remove("btn-stop-active");
    }

    btnRecordHw.disabled = false;
    btnStopHw.disabled = true;
    btnUpload.disabled = false;

    visualizerDot.style.background = "#64748b";
    visualizerLabel.textContent = "Microphone Idle";
    drawIdleWave();
  }

  // -------------------------------------------------------------------------
  // File Upload Handling
  // -------------------------------------------------------------------------
  btnUpload.addEventListener("click", () => fileInput.click());

  fileInput.addEventListener("change", async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const audioUrl = URL.createObjectURL(file);
    setAudioSource(audioUrl, file.name);

    visualizerLabel.textContent = `Processing ${file.name}...`;
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
    }
  }

  // -------------------------------------------------------------------------
  // Pre-recorded Benchmark Selector
  // -------------------------------------------------------------------------
  async function loadRecordingsList() {
    try {
      const res = await fetch("/api/recordings");
      const data = await res.json();
      savedRecordingsSelect.innerHTML = '<option value="">Select benchmark audio...</option>';
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

  btnRunSaved.addEventListener("click", async () => {
    const filename = savedRecordingsSelect.value;
    if (!filename) {
      alert("Please select a benchmark recording first.");
      return;
    }

    setAudioSource(`/recordings/${filename}`, filename);
    visualizerLabel.textContent = `Processing ${filename}...`;

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
    }
  });

  // -------------------------------------------------------------------------
  // Direct Text Input
  // -------------------------------------------------------------------------
  btnSendText.addEventListener("click", sendDirectText);
  directTextInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") sendDirectText();
  });

  async function sendDirectText() {
    const text = directTextInput.value.trim();
    if (!text) return;

    visualizerLabel.textContent = "Analyzing input text...";
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

      // Clear audio player for pure text run
      clearAudioSource();
      renderOutput(data);
    } catch (err) {
      alert("Error: " + err.message);
    }
  }

  // -------------------------------------------------------------------------
  // Audio Player State & Download Helpers
  // -------------------------------------------------------------------------
  function setAudioSource(url, filename) {
    currentAudioUrl = url;
    currentAudioFilename = filename;
    audioPlayback.src = url;
    audioPlayerWrapper.style.display = "block";

    if (btnDownloadAudioPlayer) btnDownloadAudioPlayer.disabled = false;
    if (btnDownloadOutputAudio) btnDownloadOutputAudio.disabled = false;
  }

  function clearAudioSource() {
    currentAudioUrl = null;
    currentAudioFilename = null;
    audioPlayerWrapper.style.display = "none";
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

  if (btnDownloadAudioPlayer) {
    btnDownloadAudioPlayer.addEventListener("click", () => {
      downloadAudioFile(currentAudioFilename, currentAudioUrl);
    });
  }

  if (btnDownloadOutputAudio) {
    btnDownloadOutputAudio.addEventListener("click", () => {
      downloadAudioFile(currentAudioFilename, currentAudioUrl);
    });
  }

  // -------------------------------------------------------------------------
  // Download Model Output JSON (For Model Learning / Fine-Tuning / Dataset)
  // -------------------------------------------------------------------------
  function downloadLearningSampleJson() {
    if (!currentOutputData) {
      alert("No output data available yet to download.");
      return;
    }

    const payload = {
      timestamp: new Date().toISOString(),
      audio_file: currentOutputData.audio_file || currentAudioFilename || null,
      audio_url: currentOutputData.audio_url || currentAudioUrl || null,
      raw_transcript: currentOutputData.raw_transcript || currentOutputData.raw_text || "",
      route_type: currentOutputData.route_type || "",
      processed_text: currentOutputData.processed_text || "",
      spans: currentOutputData.spans || [],
      latency_ms: {
        asr: currentOutputData.asr_time_ms || 0,
        router: currentOutputData.router_time_ms || 0,
        t5: currentOutputData.route_type === "PURE_ENGLISH" ? 0 : (currentOutputData.router_time_ms || 0),
        total: currentOutputData.total_time_ms || 0,
      },
      device: "NVIDIA Jetson Orin Nano (8GB)",
    };

    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    const ts = new Date().toISOString().replace(/[:.]/g, "-");
    a.download = `learning_sample_${ts}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  if (btnDownloadOutputJson) {
    btnDownloadOutputJson.addEventListener("click", downloadLearningSampleJson);
  }

  // -------------------------------------------------------------------------
  // Export Full Session Dataset (JSONL / JSON)
  // -------------------------------------------------------------------------
  function exportFullSessionDataset() {
    if (sessionHistoryRecords.length === 0) {
      alert("No utterances recorded in this session yet.");
      return;
    }

    const exportBundle = {
      export_timestamp: new Date().toISOString(),
      total_utterances: sessionHistoryRecords.length,
      device: "NVIDIA Jetson Orin Nano (8GB)",
      samples: sessionHistoryRecords,
    };

    const blob = new Blob([JSON.stringify(exportBundle, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    const ts = new Date().toISOString().replace(/[:.]/g, "-");
    a.download = `speech_latex_dataset_${ts}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  if (btnExportSessionJson) {
    btnExportSessionJson.addEventListener("click", exportFullSessionDataset);
  }

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
    rawTranscriptBox.textContent = data.raw_transcript || data.raw_text || "(empty)";

    // 2. Route Badge
    const isSlm = data.engine === "qwen2.5-1.5b" || data.route_type === "slm_direct";
    if (isSlm) {
      routeBadge.className = "route-badge MIXED";
      routeBadge.textContent = "SLM (Qwen2.5-1.5B)";
    } else {
      routeBadge.className = `route-badge ${data.route_type || "PURE_ENGLISH"}`;
      routeBadge.textContent = data.route_type || "IDLE";
    }

    // 3. Latency Metrics
    valAsrTime.textContent = (data.asr_time_ms || 0) + " ms";
    valRouterTime.textContent = (data.router_time_ms || 0) + " ms";
    if (lblTranslationEngine) {
      lblTranslationEngine.textContent = isSlm ? "SLM (Qwen)" : "T5-v2";
    }
    valT5Time.textContent = (data.route_type === "PURE_ENGLISH" ? "0 ms" : (data.router_time_ms || 0) + " ms");
    valTotalTime.textContent = (data.total_time_ms || 0) + " ms";

    // 4. Render Visual LaTeX Typography
    const processed = data.processed_text || "";
    currentLatex = processed;
    latexRenderContainer.innerHTML = renderLatexToVisualHtml(processed, data.route_type);

    // 5. Underlying LaTeX Code Box
    if (latexCodeCard && latexCodeBox) {
      if (data.route_type !== "PURE_ENGLISH" && (processed.includes("$") || processed.includes("\\") || processed.includes("="))) {
        latexCodeCard.style.display = "block";
        latexCodeBox.textContent = processed;
      } else {
        latexCodeCard.style.display = "none";
      }
    }

    // 6. Math Spans Breakdown
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

  // -------------------------------------------------------------------------
  // Copy LaTeX
  // -------------------------------------------------------------------------
  btnCopyLatex.addEventListener("click", () => {
    if (!currentLatex) return;
    navigator.clipboard.writeText(currentLatex).then(() => {
      const originalText = btnCopyLatex.innerHTML;
      btnCopyLatex.innerHTML = "<span>✓</span><span>Copied</span>";
      setTimeout(() => btnCopyLatex.innerHTML = originalText, 1400);
    });
  });

  // -------------------------------------------------------------------------
  // History Log Management
  // -------------------------------------------------------------------------
  function recordSessionSample(data) {
    const isSlm = data.engine === "qwen2.5-1.5b" || data.route_type === "slm_direct" || currentEngine === "slm";
    const engineLabel = isSlm ? "Qwen-1.5B (SLM)" : "T5-v2";
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

    if (historyList.children.length === 1 && historyList.children[0].classList.contains("history-empty")) {
      historyList.innerHTML = "";
    }

    const item = document.createElement("div");
    item.className = "history-item";
    const timeStr = new Date().toLocaleTimeString();

    const audioAction = record.audio_file
      ? `<button class="btn-history-dl" onclick="window.downloadAudioItem('${record.audio_file}')" title="Download audio">WAV</button>`
      : "";

    item.innerHTML = `
      <div class="history-item-left">
        <div class="history-raw">"${escapeHtml(record.raw_transcript)}"</div>
        <div class="history-processed">${escapeHtml(record.processed_text)}</div>
      </div>
      <div class="history-meta">
        <div class="history-badge-row">
          <span class="route-badge ${record.route_type}">${engineLabel}</span>
          ${audioAction}
        </div>
        <div class="history-time-meta">${record.latency_ms.total} ms • ${timeStr}</div>
      </div>
    `;

    historyList.prepend(item);
  }

  // Window helper for inline history audio download
  window.downloadAudioItem = (filename) => {
    downloadAudioFile(filename, `/recordings/${filename}`);
  };

  // -------------------------------------------------------------------------
  // WebSocket Client: Live Telemetry & Instant Transcribe
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
            visualizerDot.style.background = "#ef4444";
            visualizerLabel.textContent = `Recording ${msg.elapsed_sec.toFixed(1)}s (Press Enter to transcribe)...`;
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

  // -------------------------------------------------------------------------
  // Global Keyboard Shortcuts: Enter (Transcribe) & Space (Toggle Mic)
  // -------------------------------------------------------------------------
  window.addEventListener("keydown", (e) => {
    // If focused on an input or textarea, let normal typing occur
    if (document.activeElement && (document.activeElement.tagName === "INPUT" || document.activeElement.tagName === "TEXTAREA")) {
      return;
    }

    if (e.key === "Enter") {
      e.preventDefault();
      if (isHardwareRecording) {
        stopHardwareRecording();
      } else if (isBrowserRecording) {
        stopBrowserRecording();
      }
    } else if (e.key === " " || e.code === "Space") {
      e.preventDefault();
      if (isHardwareRecording) {
        stopHardwareRecording();
      } else if (!isBrowserRecording) {
        startHardwareRecording();
      }
    }
  });

  btnClearHistory.addEventListener("click", () => {
    sessionHistoryRecords.length = 0;
    historyList.innerHTML = '<div class="history-empty">Session history cleared.</div>';
  });
});
