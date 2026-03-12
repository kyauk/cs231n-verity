/* ================================================================
   LIBERO+ Perturbation Pipeline — Frontend Logic
   ================================================================ */

(function () {
  "use strict";

  // ---- State ----
  let selectedPath = null;
  let selectedName = "";
  let selectedLang = "";
  let currentJobId = null;
  let allTasks = [];
  let activeFilter = "all";

  // ---- DOM refs ----
  const $ = (sel) => document.querySelector(sel);
  const $$ = (sel) => [...document.querySelectorAll(sel)];

  // ---- Step navigation ----
  function goToStep(n) {
    $$(".step-panel").forEach((p) => p.classList.remove("active"));
    $(`#step-${n}`).classList.add("active");

    $$(".step-item").forEach((s) => {
      const sn = parseInt(s.dataset.step);
      s.classList.remove("active", "completed");
      if (sn === n) s.classList.add("active");
      else if (sn < n) s.classList.add("completed");
    });
  }

  // ---- Tab switching ----
  $$(".tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      $$(".tab").forEach((t) => t.classList.remove("active"));
      tab.classList.add("active");
      $$(".tab-panel").forEach((p) => p.classList.remove("active"));
      $(`#tab-${tab.dataset.tab}`).classList.add("active");
    });
  });

  // ---- Load examples ----
  async function loadExamples() {
    try {
      const res = await fetch("/api/examples");
      const data = await res.json();
      renderExamples(data.groups || []);
    } catch (e) {
      $("#examples-list").innerHTML =
        '<p class="loading-spinner">Failed to load examples.</p>';
    }
  }

  function renderExamples(groups) {
    const container = $("#examples-list");
    container.innerHTML = "";

    groups.forEach((g) => {
      const label = document.createElement("div");
      label.className = "scene-group-label";
      label.textContent = g.label;
      container.appendChild(label);

      g.scenes.forEach((s) => {
        const item = document.createElement("div");
        item.className = "scene-item";
        item.innerHTML = `
          <div class="scene-name">${s.name}</div>
          <div class="scene-lang">${s.language}</div>`;
        item.addEventListener("click", () => {
          $$(".scene-item").forEach((el) => el.classList.remove("selected"));
          item.classList.add("selected");
          selectScene(s.path, s.name, s.language);
        });
        container.appendChild(item);
      });
    });
  }

  // ---- Search filter ----
  $("#scene-search").addEventListener("input", (e) => {
    const q = e.target.value.toLowerCase();
    $$(".scene-item").forEach((item) => {
      const text = item.textContent.toLowerCase();
      item.style.display = text.includes(q) ? "" : "none";
    });
    $$(".scene-group-label").forEach((label) => {
      let next = label.nextElementSibling;
      let anyVisible = false;
      while (next && !next.classList.contains("scene-group-label")) {
        if (next.style.display !== "none") anyVisible = true;
        next = next.nextElementSibling;
      }
      label.style.display = anyVisible ? "" : "none";
    });
  });

  // ---- File upload ----
  const uploadZone = $("#upload-zone");
  const fileInput = $("#file-input");

  uploadZone.addEventListener("click", () => fileInput.click());
  uploadZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    uploadZone.classList.add("dragover");
  });
  uploadZone.addEventListener("dragleave", () =>
    uploadZone.classList.remove("dragover")
  );
  uploadZone.addEventListener("drop", (e) => {
    e.preventDefault();
    uploadZone.classList.remove("dragover");
    if (e.dataTransfer.files.length) uploadFile(e.dataTransfer.files[0]);
  });
  fileInput.addEventListener("change", () => {
    if (fileInput.files.length) uploadFile(fileInput.files[0]);
  });

  async function uploadFile(file) {
    const form = new FormData();
    form.append("file", file);
    const status = $("#upload-status");
    status.hidden = false;
    status.textContent = "Uploading...";

    try {
      const res = await fetch("/api/upload", { method: "POST", body: form });
      const data = await res.json();
      status.textContent = `Uploaded: ${data.filename}`;
      selectScene(data.path, data.filename, data.language);
    } catch (e) {
      status.textContent = "Upload failed.";
    }
  }

  // ---- Select scene ----
  function selectScene(path, name, lang) {
    selectedPath = path;
    selectedName = name;
    selectedLang = lang;
    $("#selected-name").textContent = name;
    $("#selected-lang").textContent = lang || "(no language instruction)";
    $("#selected-info").hidden = false;
  }

  // Step 1 -> 2
  $("#btn-next-1").addEventListener("click", () => {
    if (!selectedPath) return;
    goToStep(2);
  });

  // Step 2 -> 1
  $("#btn-back-2").addEventListener("click", () => goToStep(1));

  // ---- Toggle cards ----
  $$(".pert-card").forEach((card) => {
    const cb = card.querySelector('input[type="checkbox"]');
    const update = () =>
      card.classList.toggle("disabled", !cb.checked);
    cb.addEventListener("change", update);
    update();
  });

  // ---- Generate ----
  $("#btn-generate").addEventListener("click", startGeneration);

  function gatherConfig() {
    const perturbations = {};
    let severity_object_layout = 3;
    let severity_robot_init = 3;
    let noise_type = null;

    $$(".pert-card").forEach((card) => {
      const dim = card.dataset.dim;
      const enabled = card.querySelector('input[type="checkbox"]').checked;
      if (!enabled) return;
      const count = parseInt(
        card.querySelector(".variant-count")?.value || "3"
      );
      perturbations[dim] = count;

      const sev = card.querySelector(".severity-select");
      if (sev) {
        if (dim === "object_layout") severity_object_layout = parseInt(sev.value);
        if (dim === "robot_init") severity_robot_init = parseInt(sev.value);
      }
      const ns = card.querySelector(".noise-select");
      if (ns && ns.value) noise_type = ns.value;
    });

    return {
      input_path: selectedPath,
      perturbations,
      severity_object_layout,
      severity_robot_init,
      noise_type,
      seed: 42,
    };
  }

  async function startGeneration() {
    const config = gatherConfig();
    if (!config.input_path || Object.keys(config.perturbations).length === 0) {
      return;
    }

    goToStep(3);
    resetProgress(config.perturbations);

    try {
      const response = await fetch("/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(config),
      });

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        const lines = buffer.split("\n\n");
        buffer = lines.pop();

        for (const chunk of lines) {
          const match = chunk.match(/^data:\s*(.*)/);
          if (!match) continue;
          try {
            handleSSE(JSON.parse(match[1]));
          } catch (e) {
            /* skip malformed */
          }
        }
      }
    } catch (e) {
      appendLog("Connection error: " + e.message);
    }
  }

  // ---- Progress UI ----
  function resetProgress(perturbations) {
    const grid = $("#dim-status-grid");
    grid.innerHTML = "";
    Object.keys(perturbations).forEach((dim) => {
      const badge = document.createElement("span");
      badge.className = "dim-badge";
      badge.id = `badge-${dim}`;
      badge.textContent = dim.replace("_", " ");
      grid.appendChild(badge);
    });
    // Add render badge
    const rb = document.createElement("span");
    rb.className = "dim-badge";
    rb.id = "badge-render";
    rb.textContent = "rendering";
    grid.appendChild(rb);

    $("#progress-fill").style.width = "0%";
    $("#progress-text").textContent = "Starting...";
    $("#log-box").innerHTML = "";
  }

  function handleSSE(evt) {
    switch (evt.type) {
      case "job_start":
        currentJobId = evt.job_id;
        appendLog(`Job started: ${evt.job_id}`);
        break;

      case "dim_start": {
        const badge = $(`#badge-${evt.dimension}`);
        if (badge) badge.classList.add("running");
        appendLog(`Generating ${evt.dimension} variants...`);
        updateProgress(evt.index, evt.total + 1);
        $("#progress-text").textContent = `Generating ${evt.dimension}...`;
        break;
      }

      case "dim_done": {
        const badge = $(`#badge-${evt.dimension}`);
        if (badge) {
          badge.classList.remove("running");
          badge.classList.add("done");
          badge.textContent = `${evt.dimension.replace("_", " ")} (${evt.count})`;
        }
        appendLog(
          `  ${evt.dimension}: ${evt.count} variants generated`
        );
        updateProgress(evt.index + 1, evt.total + 1);
        break;
      }

      case "render_start":
        $(`#badge-render`).classList.add("running");
        appendLog(`Rendering ${evt.total} preview images...`);
        $("#progress-text").textContent = "Rendering previews...";
        break;

      case "render_progress":
        appendLog(
          `  Rendered ${evt.current}/${evt.total}: ${evt.perturbation}`
        );
        const renderPct = Math.round((evt.current / evt.total) * 100);
        $("#progress-text").textContent = `Rendering ${evt.current}/${evt.total} (${renderPct}%)`;
        break;

      case "render_error":
        appendLog(`Render warning: ${evt.message}`);
        break;

      case "complete":
        $(`#badge-render`).classList.remove("running");
        $(`#badge-render`).classList.add("done");
        $(`#badge-render`).textContent = `rendered (${evt.rendered})`;
        updateProgress(1, 1);
        $("#progress-text").textContent = "Complete!";
        appendLog(
          `Done! ${evt.total_tasks} tasks, ${evt.rendered} previews.`
        );
        allTasks = evt.tasks || [];

        setTimeout(() => showResults(), 600);
        break;
    }
  }

  function updateProgress(current, total) {
    const pct = Math.min(100, Math.round((current / total) * 100));
    $("#progress-fill").style.width = pct + "%";
  }

  function appendLog(msg) {
    const box = $("#log-box");
    const line = document.createElement("div");
    line.textContent = msg;
    box.appendChild(line);
    box.scrollTop = box.scrollHeight;
  }

  // ---- Results ----
  function showResults() {
    goToStep(4);

    const rendered = allTasks.filter((t) => t.preview);
    $("#results-summary").textContent = `${allTasks.length} total variants generated, ${rendered.length} preview images rendered.`;

    // Build filter buttons
    const types = [...new Set(allTasks.map((t) => t.perturbation))];
    const filtersEl = $("#gallery-filters");
    filtersEl.innerHTML = "";

    const allBtn = document.createElement("button");
    allBtn.className = "filter-btn active";
    allBtn.textContent = `All (${allTasks.length})`;
    allBtn.addEventListener("click", () => setFilter("all"));
    filtersEl.appendChild(allBtn);

    types.forEach((type) => {
      const count = allTasks.filter((t) => t.perturbation === type).length;
      const btn = document.createElement("button");
      btn.className = "filter-btn";
      btn.dataset.type = type;
      btn.textContent = `${type.replace("_", " ")} (${count})`;
      btn.addEventListener("click", () => setFilter(type));
      filtersEl.appendChild(btn);
    });

    renderGallery();
  }

  function setFilter(type) {
    activeFilter = type;
    $$(".filter-btn").forEach((b) => {
      if (type === "all") {
        b.classList.toggle("active", !b.dataset.type);
      } else {
        b.classList.toggle("active", b.dataset.type === type);
      }
    });
    renderGallery();
  }

  function renderGallery() {
    const gallery = $("#gallery");
    gallery.innerHTML = "";

    const tasks =
      activeFilter === "all"
        ? allTasks
        : allTasks.filter((t) => t.perturbation === activeFilter);

    tasks.forEach((task) => {
      const card = document.createElement("div");
      card.className = "gallery-card";

      const imgSrc = task.preview
        ? `/api/results/${currentJobId}/images/${task.preview}`
        : "";

      const paramStr = buildParamString(task);

      if (task.preview) {
        card.innerHTML = `
          <img src="${imgSrc}" alt="${task.name}" loading="lazy" />
          <div class="gallery-card-info">
            <span class="pert-type">${task.perturbation.replace("_", " ")}</span>
            <div class="variant-label">Variant ${task.variant}${paramStr ? " — " + paramStr : ""}</div>
          </div>`;
        card.addEventListener("click", () => openLightbox(task, imgSrc));
      } else {
        card.innerHTML = `
          <div style="aspect-ratio:1;display:flex;align-items:center;justify-content:center;background:var(--bg-secondary);color:var(--text-muted);font-size:0.8rem;">No preview</div>
          <div class="gallery-card-info">
            <span class="pert-type">${task.perturbation.replace("_", " ")}</span>
            <div class="variant-label">Variant ${task.variant}${paramStr ? " — " + paramStr : ""}</div>
          </div>`;
      }
      gallery.appendChild(card);
    });
  }

  function buildParamString(task) {
    const parts = [];
    if (task.horizon_view !== undefined) parts.push(`cam ${task.horizon_view}°`);
    if (task.init_state_id !== undefined) parts.push(`init #${task.init_state_id}`);
    if (task.noise_type && task.noise_type !== "unknown")
      parts.push(task.noise_type.replace("_", " "));
    if (task.problem_class) {
      const short = task.problem_class.split("_").slice(-3).join("_");
      parts.push(short);
    }
    if (task.original_language) parts.push(`"${task.language}"`);
    if (task.severity) parts.push(`sev ${task.severity}`);
    return parts.join(", ");
  }

  // ---- Lightbox ----
  function openLightbox(task, imgSrc) {
    const lb = $("#lightbox");
    $("#lightbox-img").src = imgSrc;
    let info = `<strong>${task.perturbation.replace("_", " ")}</strong> — Variant ${task.variant}<br>`;
    info += `<strong>Name:</strong> ${task.name}<br>`;
    if (task.language) info += `<strong>Language:</strong> ${task.language}<br>`;
    const ps = buildParamString(task);
    if (ps) info += `<strong>Params:</strong> ${ps}`;
    $("#lightbox-info").innerHTML = info;
    lb.style.display = "";
  }

  $(".lightbox-backdrop").addEventListener("click", () => {
    $("#lightbox").style.display = "none";
  });
  $(".lightbox-close").addEventListener("click", () => {
    $("#lightbox").style.display = "none";
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") $("#lightbox").style.display = "none";
  });

  // ---- Download ----
  $("#btn-download").addEventListener("click", () => {
    if (!currentJobId) return;
    window.location.href = `/api/download/${currentJobId}`;
  });

  // ---- Restart ----
  $("#btn-restart").addEventListener("click", () => {
    selectedPath = null;
    selectedName = "";
    selectedLang = "";
    currentJobId = null;
    allTasks = [];
    activeFilter = "all";
    $("#selected-info").hidden = true;
    $$(".scene-item").forEach((el) => el.classList.remove("selected"));
    goToStep(1);
  });

  // ---- Boot ----
  loadExamples();
})();
