/**
 * OmniSign AI — Frontend Application Logic
 * Integrates WebRTC camera, speech recognition, real-time inference streaming,
 * and the Common Communication Layer.
 */

(function () {
  "use strict";

  // Elements
  const videoElem = document.getElementById("webcam-video");
  const canvasElem = document.getElementById("detection-canvas");
  const cameraStandby = document.getElementById("camera-standby");
  const btnStartCamera = document.getElementById("btn-start-camera");
  const btnStopCamera = document.getElementById("btn-stop-camera");
  const btnTestFrame = document.getElementById("btn-test-frame");

  const selectLanguage = document.getElementById("select-language");
  const selectMode = document.getElementById("select-mode");

  const displaySignText = document.getElementById("display-sign-text");
  const confidencePercentage = document.getElementById("confidence-percentage");
  const confidenceFill = document.getElementById("confidence-fill");
  const thresholdHint = document.getElementById("threshold-hint");
  const statusBanner = document.getElementById("detection-status-banner");
  const statusIcon = document.getElementById("status-icon");
  const statusText = document.getElementById("detection-status-text");

  const hudFps = document.getElementById("hud-fps");
  const hudLatency = document.getElementById("hud-latency");
  const hudLang = document.getElementById("hud-lang");
  const systemStatusBadge = document.getElementById("system-status-badge");
  const systemStatusText = document.getElementById("system-status-text");

  const btnStartSpeech = document.getElementById("btn-start-speech");
  const btnStopSpeech = document.getElementById("btn-stop-speech");
  const speechStatusIndicator = document.getElementById("speech-status-indicator");
  const speechStatusText = document.getElementById("speech-status-text");
  const inputSpeechText = document.getElementById("input-speech-text");
  const btnSendSpeech = document.getElementById("btn-send-speech");

  const transcriptFeed = document.getElementById("transcript-feed");
  const feedEmptyState = document.getElementById("feed-empty-state");
  const fullTranscriptText = document.getElementById("full-transcript-text");
  const btnSpeakTranscript = document.getElementById("btn-speak-transcript");
  const btnClearTranscript = document.getElementById("btn-clear-transcript");
  const btnCopyTranscript = document.getElementById("btn-copy-transcript");

  // Settings elements
  const btnSettingsToggle = document.getElementById("btn-settings-toggle");
  const settingsModal = document.getElementById("settings-modal");
  const btnCloseSettings = document.getElementById("btn-close-settings");
  const btnSaveSettings = document.getElementById("btn-save-settings");
  const islSlider = document.getElementById("isl-threshold-slider");
  const islVal = document.getElementById("isl-thresh-val");
  const aslSlider = document.getElementById("asl-threshold-slider");
  const aslVal = document.getElementById("asl-thresh-val");
  const ttsToggle = document.getElementById("tts-toggle");

  // State
  let cameraStream = null;
  let isCameraActive = false;
  let frameIntervalId = null;
  let lastFrameTime = performance.now();
  let currentFps = 0;
  let isProcessingFrame = false;
  let speechRecognition = null;
  let isSpeechActive = false;

  // Initialize
  async function init() {
    setupEventListeners();
    await checkSystemStatus();
    initSpeechRecognition();
    setInterval(checkSystemStatus, 8000);
  }

  // Check Backend Status
  async function checkSystemStatus() {
    try {
      const res = await fetch("/api/status");
      if (res.ok) {
        const data = await res.json();
        systemStatusBadge.classList.remove("error");
        systemStatusText.textContent = "Online & Connected";
        if (data.thresholds) {
          thresholdHint.textContent = `Threshold: ${Math.round((data.active_language === "ISL" ? data.thresholds.isl : data.thresholds.asl) * 100)}%`;
        }
      } else {
        systemStatusText.textContent = "API Error";
      }
    } catch (e) {
      systemStatusText.textContent = "Offline / Standby";
    }
  }

  // Setup Event Listeners
  function setupEventListeners() {
    btnStartCamera.addEventListener("click", startCamera);
    btnStopCamera.addEventListener("click", stopCamera);
    btnTestFrame.addEventListener("click", testSnapshot);

    selectLanguage.addEventListener("change", onConfigChanged);
    selectMode.addEventListener("change", onConfigChanged);

    btnStartSpeech.addEventListener("click", startSpeech);
    btnStopSpeech.addEventListener("click", stopSpeech);
    btnSendSpeech.addEventListener("click", sendSimulatedSpeech);
    inputSpeechText.addEventListener("keydown", (e) => {
      if (e.key === "Enter") sendSimulatedSpeech();
    });

    btnSpeakTranscript.addEventListener("click", speakLatestTranscript);
    btnClearTranscript.addEventListener("click", clearTranscript);
    btnCopyTranscript.addEventListener("click", copyTranscript);

    // Modal
    btnSettingsToggle.addEventListener("click", () => settingsModal.style.display = "flex");
    btnCloseSettings.addEventListener("click", () => settingsModal.style.display = "none");
    settingsModal.addEventListener("click", (e) => {
      if (e.target === settingsModal) settingsModal.style.display = "none";
    });

    islSlider.addEventListener("input", (e) => islVal.textContent = `${e.target.value}%`);
    aslSlider.addEventListener("input", (e) => aslVal.textContent = `${e.target.value}%`);
    btnSaveSettings.addEventListener("click", saveSettings);
  }

  // Config Update
  async function onConfigChanged() {
    const lang = selectLanguage.value;
    const mode = selectMode.value;
    hudLang.textContent = `${lang} • ${mode}`;

    try {
      await fetch("/api/config", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ language: lang, mode: mode })
      });
    } catch (e) {
      console.warn("Failed to update config on server:", e);
    }
  }

  async function saveSettings() {
    const islThresh = parseFloat(islSlider.value) / 100.0;
    const aslThresh = parseFloat(aslSlider.value) / 100.0;
    const ttsEnabled = ttsToggle.checked;

    try {
      await fetch("/api/config", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          isl_threshold: islThresh,
          asl_threshold: aslThresh,
          tts_enabled: ttsEnabled
        })
      });
      thresholdHint.textContent = `Threshold: ${selectLanguage.value === "ISL" ? islSlider.value : aslSlider.value}%`;
      settingsModal.style.display = "none";
    } catch (e) {
      alert("Error saving settings: " + e.message);
    }
  }

  // Camera Management
  async function startCamera() {
    try {
      cameraStream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 } }
      });
      videoElem.srcObject = cameraStream;
      videoElem.play();
      cameraStandby.style.display = "none";
      isCameraActive = true;
      btnStartCamera.disabled = true;
      btnStopCamera.disabled = false;

      // Start recognition frame sampling loop (8 FPS)
      frameIntervalId = setInterval(captureAndRecognizeFrame, 125);
    } catch (err) {
      console.error("Camera access error:", err);
      statusText.textContent = "Camera permission denied or device not found.";
      statusBanner.classList.add("uncertain");
    }
  }

  function stopCamera() {
    if (cameraStream) {
      cameraStream.getTracks().forEach((t) => t.stop());
      cameraStream = null;
    }
    if (frameIntervalId) {
      clearInterval(frameIntervalId);
      frameIntervalId = null;
    }
    videoElem.srcObject = null;
    cameraStandby.style.display = "flex";
    isCameraActive = false;
    btnStartCamera.disabled = false;
    btnStopCamera.disabled = true;
    hudFps.textContent = "0 FPS";
  }

  // Frame Capture & API Inference
  async function captureAndRecognizeFrame() {
    if (!isCameraActive || isProcessingFrame || !videoElem.videoWidth) return;
    isProcessingFrame = true;

    const now = performance.now();
    const dt = (now - lastFrameTime) / 1000.0;
    lastFrameTime = now;
    if (dt > 0) {
      currentFps = Math.round(1.0 / dt);
      hudFps.textContent = `${currentFps} FPS`;
    }

    // Capture frame to offscreen canvas
    canvasElem.width = videoElem.videoWidth;
    canvasElem.height = videoElem.videoHeight;
    const ctx = canvasElem.getContext("2d");
    ctx.drawImage(videoElem, 0, 0, canvasElem.width, canvasElem.height);
    const frameBase64 = canvasElem.toDataURL("image/jpeg", 0.7);

    try {
      const t0 = performance.now();
      const res = await fetch("/api/recognize/frame", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ frame: frameBase64 })
      });
      const data = await res.json();
      const latency = Math.round(performance.now() - t0);
      hudLatency.textContent = `${latency} ms`;

      updateRecognitionDisplay(data);
    } catch (err) {
      console.warn("Inference error:", err);
    } finally {
      isProcessingFrame = false;
    }
  }

  // Update HUD & Display from Model Output
  function updateRecognitionDisplay(data) {
    if (!data) return;

    const confPct = Math.round((data.confidence || 0) * 100);
    confidencePercentage.textContent = `${confPct}%`;
    confidenceFill.style.width = `${Math.min(confPct, 100)}%`;

    if (data.raw_label === "NO_HAND" || data.raw_label === "NONE") {
      displaySignText.textContent = "--";
      displaySignText.style.color = "var(--text-muted)";
      statusBanner.classList.remove("uncertain");
      statusIcon.textContent = "✋";
      statusText.textContent = "Show your hand clearly to the camera.";
    } else if (data.raw_label && data.raw_label.startsWith("BUFFERING")) {
      displaySignText.textContent = data.raw_label;
      displaySignText.style.color = "var(--accent-cyan)";
      statusBanner.classList.remove("uncertain");
      statusIcon.textContent = "⏳";
      statusText.textContent = `Recording gesture sequence (${data.raw_label})...`;
    } else if (data.is_uncertain) {
      displaySignText.textContent = "UNCERTAIN";
      displaySignText.style.color = "var(--accent-amber)";
      statusBanner.classList.add("uncertain");
      statusIcon.textContent = "⚠️";
      statusText.textContent = "Sign not recognized. Please hold steady or try again.";
    } else {
      const label = data.smoothed_label || data.raw_label || "--";
      displaySignText.textContent = label;
      displaySignText.style.color = "var(--accent-emerald)";
      statusBanner.classList.remove("uncertain");
      statusIcon.textContent = "✨";
      statusText.textContent = `Sign confirmed: [${label}] with ${confPct}% confidence.`;
    }

    // Refresh common text transcript if updated
    if (data.transcript_summary) {
      fullTranscriptText.textContent = data.transcript_summary || "Waiting for input...";
      fetchCommonText();
    }
  }

  // Test Snapshot (Single frame or simulation)
  async function testSnapshot() {
    statusText.textContent = "Testing snapshot inference...";
    if (isCameraActive) {
      await captureAndRecognizeFrame();
    } else {
      // Simulate snapshot test
      try {
        const res = await fetch("/api/recognize/frame", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ frame: "data:image/jpeg;base64," })
        });
        const data = await res.json();
        updateRecognitionDisplay(data);
      } catch (e) {
        console.error(e);
      }
    }
  }

  // Speech Recognition (Web Speech API + Backend Whisper Integration)
  function initSpeechRecognition() {
    const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (SpeechRec) {
      speechRecognition = new SpeechRec();
      speechRecognition.continuous = true;
      speechRecognition.interimResults = false;
      speechRecognition.lang = "en-US";

      speechRecognition.onresult = (event) => {
        const lastIdx = event.results.length - 1;
        const transcript = event.results[lastIdx][0].transcript;
        if (transcript) {
          sendSpeechToBackend(transcript.trim());
        }
      };

      speechRecognition.onerror = (event) => {
        console.warn("Speech recognition error:", event.error);
        speechStatusText.textContent = "Speech recognition: " + event.error;
      };

      speechRecognition.onend = () => {
        if (isSpeechActive) speechRecognition.start();
      };
    }
  }

  function startSpeech() {
    if (speechRecognition) {
      try {
        speechRecognition.start();
      } catch (e) {}
    }
    isSpeechActive = true;
    btnStartSpeech.disabled = true;
    btnStopSpeech.disabled = false;
    speechStatusIndicator.classList.add("listening");
    speechStatusText.textContent = "Listening to microphone...";
  }

  function stopSpeech() {
    if (speechRecognition) {
      try {
        speechRecognition.stop();
      } catch (e) {}
    }
    isSpeechActive = false;
    btnStartSpeech.disabled = false;
    btnStopSpeech.disabled = true;
    speechStatusIndicator.classList.remove("listening");
    speechStatusText.textContent = "Microphone Ready";
  }

  async function sendSpeechToBackend(text) {
    if (!text) return;
    try {
      const res = await fetch("/api/speech/transcribe", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: text })
      });
      const data = await res.json();
      if (data.transcript_summary) {
        fullTranscriptText.textContent = data.transcript_summary;
      }
      fetchCommonText();
    } catch (e) {
      console.error("Speech send error:", e);
    }
  }

  function sendSimulatedSpeech() {
    const text = inputSpeechText.value.trim();
    if (!text) return;
    sendSpeechToBackend(text);
    inputSpeechText.value = "";
  }

  // Common Text Layer Management
  async function fetchCommonText() {
    try {
      const res = await fetch("/api/common-text");
      if (res.ok) {
        const data = await res.json();
        renderTranscriptItems(data.entries || []);
        if (data.full_transcript) {
          fullTranscriptText.textContent = data.full_transcript;
        }
      }
    } catch (e) {
      console.warn("Error fetching common text:", e);
    }
  }

  function renderTranscriptItems(entries) {
    if (!entries || entries.length === 0) {
      feedEmptyState.style.display = "flex";
      return;
    }
    feedEmptyState.style.display = "none";

    // Clear existing
    transcriptFeed.innerHTML = "";
    entries.forEach((entry) => {
      const item = document.createElement("div");
      item.className = "transcript-item";

      const badge = document.createElement("span");
      badge.className = `item-badge badge-${entry.source}`;
      badge.textContent = entry.source;

      const content = document.createElement("div");
      content.className = "item-content";

      const text = document.createElement("div");
      text.className = "item-text";
      text.textContent = entry.text;

      const meta = document.createElement("div");
      meta.className = "item-meta";
      const timeStr = new Date(entry.timestamp * 1000).toLocaleTimeString();
      meta.textContent = `${entry.mode} • Confidence: ${Math.round(entry.confidence * 100)}% • ${timeStr}`;

      content.appendChild(text);
      content.appendChild(meta);
      item.appendChild(badge);
      item.appendChild(content);

      transcriptFeed.appendChild(item);
    });

    transcriptFeed.scrollTop = transcriptFeed.scrollHeight;
  }

  async function clearTranscript() {
    try {
      await fetch("/api/common-text/clear", { method: "POST" });
      transcriptFeed.innerHTML = "";
      feedEmptyState.style.display = "flex";
      fullTranscriptText.textContent = "Waiting for input...";
    } catch (e) {
      console.error("Clear error:", e);
    }
  }

  async function speakLatestTranscript() {
    const text = fullTranscriptText.textContent;
    if (!text || text === "Waiting for input...") return;
    try {
      await fetch("/api/tts/speak", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: text })
      });
    } catch (e) {
      console.error("TTS error:", e);
    }
  }

  function copyTranscript() {
    const text = fullTranscriptText.textContent;
    if (text) {
      navigator.clipboard.writeText(text);
      btnCopyTranscript.textContent = "✓ Copied!";
      setTimeout(() => btnCopyTranscript.textContent = "📋 Copy", 2000);
    }
  }

  // Run on page load
  window.addEventListener("DOMContentLoaded", init);
})();
