const PALETTE = [
  "#000000",
  "#0074d9",
  "#ff4136",
  "#2ecc40",
  "#ffdc00",
  "#aaaaaa",
  "#f012be",
  "#ff851b",
  "#7fdbff",
  "#870c25",
];

const PAGE_SIZE = 12;
const MAX_PAGE_CACHE = 24;

const elements = {
  datasetSelect: document.querySelector("#dataset-select"),
  taskSelect: document.querySelector("#task-select"),
  taskPosition: document.querySelector("#task-position"),
  pairPosition: document.querySelector("#pair-position"),
  splitBadge: document.querySelector("#split-badge"),
  pairGrid: document.querySelector("#pair-grid"),
  previousTask: document.querySelector("#previous-task"),
  nextTask: document.querySelector("#next-task"),
  previousPage: document.querySelector("#previous-page"),
  nextPage: document.querySelector("#next-page"),
  pageCurrent: document.querySelector("#page-current"),
  pageTotal: document.querySelector("#page-total"),
  workspace: document.querySelector("#workspace"),
  loading: document.querySelector("#loading"),
  error: document.querySelector("#error"),
  errorMessage: document.querySelector("#error-message"),
  retry: document.querySelector("#retry"),
  sourceLabel: document.querySelector("#source-label"),
  detailDialog: document.querySelector("#detail-dialog"),
  detailTitle: document.querySelector("#detail-title"),
  detailGrids: document.querySelector("#detail-grids"),
  closeDetail: document.querySelector("#close-detail"),
};

const state = {
  datasets: [],
  datasetIndex: 0,
  taskIndex: 0,
  pairPage: 0,
  current: null,
  request: null,
  requestSerial: 0,
  cache: new Map(),
  wheelLockedUntil: 0,
  touchStart: null,
};

function currentDataset() {
  return state.datasets[state.datasetIndex];
}

function currentTask() {
  return currentDataset()?.tasks[state.taskIndex];
}

function pageCacheKey(datasetId, taskKey, page) {
  return `${datasetId}\u0000${taskKey}\u0000${page}`;
}

function cacheGet(key) {
  const value = state.cache.get(key);
  if (!value) return null;
  state.cache.delete(key);
  state.cache.set(key, value);
  return value;
}

function cacheSet(key, value) {
  state.cache.delete(key);
  state.cache.set(key, value);
  while (state.cache.size > MAX_PAGE_CACHE) {
    state.cache.delete(state.cache.keys().next().value);
  }
}

function parseLocation() {
  const params = new URLSearchParams(location.hash.slice(1));
  return {
    dataset: params.get("dataset"),
    task: params.get("task"),
    page: Number.parseInt(params.get("page") || "0", 10),
  };
}

function updateLocation() {
  const dataset = currentDataset();
  const task = currentTask();
  if (!dataset || !task) return;
  const params = new URLSearchParams({
    dataset: dataset.id,
    task: task.key,
    page: String(state.pairPage),
  });
  history.replaceState(null, "", `#${params.toString()}`);
}

function option(value, label) {
  const node = document.createElement("option");
  node.value = value;
  node.textContent = label;
  return node;
}

function populateDatasetSelect() {
  const fragment = document.createDocumentFragment();
  state.datasets.forEach((dataset, index) => {
    fragment.append(option(String(index), `${dataset.name} · ${dataset.taskCount.toLocaleString()} tasks`));
  });
  elements.datasetSelect.replaceChildren(fragment);
  elements.datasetSelect.value = String(state.datasetIndex);
}

function populateTaskSelect() {
  const dataset = currentDataset();
  const fragment = document.createDocumentFragment();
  dataset.tasks.forEach((task, index) => {
    const prefix = task.group ? `${task.group} / ` : "";
    fragment.append(option(String(index), `${prefix}${task.name}`));
  });
  elements.taskSelect.replaceChildren(fragment);
  elements.taskSelect.value = String(state.taskIndex);
}

function restoreLocation() {
  const requested = parseLocation();
  const datasetIndex = state.datasets.findIndex((dataset) => dataset.id === requested.dataset);
  state.datasetIndex = datasetIndex >= 0 ? datasetIndex : 0;
  const dataset = currentDataset();
  const taskIndex = dataset.tasks.findIndex((task) => task.key === requested.task);
  state.taskIndex = taskIndex >= 0 ? taskIndex : 0;
  state.pairPage = Number.isFinite(requested.page) ? Math.max(0, requested.page) : 0;
}

async function fetchJson(url, signal) {
  const response = await fetch(url, { signal });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || `Request failed (${response.status})`);
  return payload;
}

async function loadCurrentPage({ keepGrid = false } = {}) {
  const dataset = currentDataset();
  const task = currentTask();
  if (!dataset || !task) return;

  state.request?.abort();
  state.request = null;
  const serial = ++state.requestSerial;
  elements.error.hidden = true;
  updateControls();
  updateLocation();

  const cacheKey = pageCacheKey(dataset.id, task.key, state.pairPage);
  const cached = cacheGet(cacheKey);
  if (cached) {
    state.current = cached;
    state.pairPage = cached.page;
    elements.loading.hidden = true;
    renderCurrent();
    return;
  }

  state.request = new AbortController();
  if (!keepGrid) renderSkeleton();
  elements.loading.hidden = false;

  const query = new URLSearchParams({
    dataset: dataset.id,
    task: task.key,
    page: String(state.pairPage),
  });

  try {
    const payload = await fetchJson(`/api/task?${query}`, state.request.signal);
    if (serial !== state.requestSerial) return;
    state.current = payload;
    state.pairPage = payload.page;
    cacheSet(pageCacheKey(dataset.id, task.key, payload.page), payload);
    renderCurrent();
  } catch (error) {
    if (error.name === "AbortError") return;
    showError(error.message);
  } finally {
    if (serial === state.requestSerial) elements.loading.hidden = true;
  }
}

function renderSkeleton() {
  stopObservingCanvases(elements.pairGrid);
  const fragment = document.createDocumentFragment();
  for (let index = 0; index < PAGE_SIZE; index += 1) {
    const card = document.createElement("div");
    card.className = "pair-placeholder";
    fragment.append(card);
  }
  elements.pairGrid.replaceChildren(fragment);
}

function renderCurrent() {
  const payload = state.current;
  if (!payload) return;
  stopObservingCanvases(elements.pairGrid);
  const fragment = document.createDocumentFragment();
  payload.pairs.forEach((pair, index) => {
    fragment.append(createPairCard(pair, payload.start + index));
  });
  for (let index = payload.pairs.length; index < PAGE_SIZE; index += 1) {
    const placeholder = document.createElement("div");
    placeholder.className = "pair-placeholder";
    fragment.append(placeholder);
  }
  elements.pairGrid.replaceChildren(fragment);

  const group = payload.task.group;
  elements.splitBadge.hidden = !group;
  elements.splitBadge.textContent = group;
  elements.pairPosition.textContent = `${payload.pairCount.toLocaleString()} pair${payload.pairCount === 1 ? "" : "s"}`;
  elements.pageCurrent.textContent = String(payload.page + 1);
  elements.pageTotal.textContent = String(payload.pageCount);
  updateControls();
  updateLocation();
  document.title = `${payload.task.name} · ${payload.dataset} · mdlARC`;
}

function createPairCard(pair, absoluteIndex) {
  const card = document.createElement("article");
  card.className = "pair-card";
  card.style.animationDelay = `${Math.min(absoluteIndex % PAGE_SIZE, 6) * 12}ms`;

  const header = document.createElement("header");
  header.className = "pair-header";
  const indexLabel = document.createElement("span");
  indexLabel.className = "pair-index";
  indexLabel.textContent = `PAIR ${String(absoluteIndex + 1).padStart(3, "0")}`;
  const split = document.createElement("span");
  split.className = "pair-split";
  split.dataset.split = pair.split;
  split.textContent = pair.split;
  header.append(indexLabel, split);

  const content = document.createElement("div");
  content.className = "pair-content";
  content.append(
    createGridView(pair.input, "INPUT", () => showDetail(pair, absoluteIndex)),
    arrowNode(),
    createGridView(pair.output, "OUTPUT", () => showDetail(pair, absoluteIndex)),
  );

  card.append(header, content);
  return card;
}

function arrowNode() {
  const arrow = document.createElement("div");
  arrow.className = "transform-arrow";
  arrow.textContent = "→";
  arrow.setAttribute("aria-hidden", "true");
  return arrow;
}

function createGridView(grid, label, onClick) {
  const wrap = document.createElement("div");
  wrap.className = "canvas-wrap";
  wrap.dataset.label = gridDimensions(grid, label);
  if (!grid) {
    const unknown = document.createElement("div");
    unknown.className = "unknown-output";
    unknown.textContent = "?";
    wrap.append(unknown);
    return wrap;
  }

  const canvas = document.createElement("canvas");
  canvas.setAttribute("role", "img");
  canvas.setAttribute("aria-label", `${label.toLowerCase()} grid, ${grid.length} rows by ${grid[0]?.length || 0} columns`);
  wrap.append(canvas);
  wrap.addEventListener("click", onClick);
  observeCanvas(canvas, grid);
  return wrap;
}

function gridDimensions(grid, label) {
  if (!grid) return `${label} · UNKNOWN`;
  return `${label} · ${grid.length}×${grid[0]?.length || 0}`;
}

const resizeObserver = new ResizeObserver((entries) => {
  for (const entry of entries) {
    const canvas = entry.target;
    drawGrid(canvas, canvas._grid);
  }
});

function stopObservingCanvases(root) {
  root.querySelectorAll("canvas").forEach((canvas) => {
    resizeObserver.unobserve(canvas);
    canvas._grid = null;
  });
}

function observeCanvas(canvas, grid) {
  canvas._grid = grid;
  resizeObserver.observe(canvas);
}

function drawGrid(canvas, grid) {
  if (!grid?.length || !grid[0]?.length) return;
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  if (!width || !height) return;

  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const pixelWidth = Math.max(1, Math.round(width * dpr));
  const pixelHeight = Math.max(1, Math.round(height * dpr));
  if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
    canvas.width = pixelWidth;
    canvas.height = pixelHeight;
  }

  const context = canvas.getContext("2d", { alpha: false });
  context.setTransform(dpr, 0, 0, dpr, 0, 0);
  context.imageSmoothingEnabled = false;
  context.fillStyle = "#0b0e12";
  context.fillRect(0, 0, width, height);

  const rows = grid.length;
  const columns = grid[0].length;
  const cell = Math.max(1, Math.min(width / columns, height / rows));
  const gridWidth = cell * columns;
  const gridHeight = cell * rows;
  const left = (width - gridWidth) / 2;
  const top = (height - gridHeight) / 2;
  const gap = cell >= 8 ? Math.min(0.8, cell * 0.045) : 0;

  for (let row = 0; row < rows; row += 1) {
    for (let column = 0; column < columns; column += 1) {
      const value = Number(grid[row][column]);
      context.fillStyle = PALETTE[value] || "#ffffff";
      context.fillRect(
        left + column * cell + gap / 2,
        top + row * cell + gap / 2,
        Math.max(0.5, cell - gap),
        Math.max(0.5, cell - gap),
      );
    }
  }

  context.strokeStyle = "rgba(255, 255, 255, 0.22)";
  context.lineWidth = Math.max(0.5, 1 / dpr);
  context.strokeRect(left, top, gridWidth, gridHeight);
}

function showDetail(pair, absoluteIndex) {
  elements.detailTitle.textContent = `${currentDataset().name} / ${currentTask().name} / pair ${absoluteIndex + 1} / ${pair.split}`;
  stopObservingCanvases(elements.detailGrids);
  elements.detailGrids.replaceChildren(
    createGridView(pair.input, "INPUT", () => {}),
    arrowNode(),
    createGridView(pair.output, "OUTPUT", () => {}),
  );
  elements.detailDialog.showModal();
}

function updateControls() {
  const dataset = currentDataset();
  const pageCount = state.current?.task.key === currentTask()?.key ? state.current.pageCount : 1;
  elements.taskPosition.textContent = dataset
    ? `${(state.taskIndex + 1).toLocaleString()} / ${dataset.taskCount.toLocaleString()}`
    : "— / —";
  elements.previousTask.disabled = !dataset || state.taskIndex <= 0;
  elements.nextTask.disabled = !dataset || state.taskIndex >= dataset.tasks.length - 1;
  elements.previousPage.disabled = state.pairPage <= 0;
  elements.nextPage.disabled = state.pairPage >= pageCount - 1;
  elements.taskSelect.value = String(state.taskIndex);
  elements.datasetSelect.value = String(state.datasetIndex);
}

function switchDataset(index) {
  if (!state.datasets[index]) return;
  state.datasetIndex = index;
  state.taskIndex = 0;
  state.pairPage = 0;
  state.current = null;
  populateTaskSelect();
  loadCurrentPage();
}

function switchTask(delta) {
  const dataset = currentDataset();
  const nextIndex = state.taskIndex + delta;
  if (!dataset || nextIndex < 0 || nextIndex >= dataset.tasks.length) return;
  state.taskIndex = nextIndex;
  state.pairPage = 0;
  state.current = null;
  loadCurrentPage();
}

function goToTask(index) {
  if (!currentDataset()?.tasks[index]) return;
  state.taskIndex = index;
  state.pairPage = 0;
  state.current = null;
  loadCurrentPage();
}

function switchPairPage(delta) {
  const pageCount = state.current?.pageCount || 1;
  const nextPage = state.pairPage + delta;
  if (nextPage < 0 || nextPage >= pageCount) return;
  state.pairPage = nextPage;
  loadCurrentPage({ keepGrid: true });
}

function showError(message) {
  elements.loading.hidden = true;
  elements.errorMessage.textContent = message;
  elements.error.hidden = false;
}

function installInteractions() {
  elements.datasetSelect.addEventListener("change", () => switchDataset(Number(elements.datasetSelect.value)));
  elements.taskSelect.addEventListener("change", () => goToTask(Number(elements.taskSelect.value)));
  elements.previousTask.addEventListener("click", () => switchTask(-1));
  elements.nextTask.addEventListener("click", () => switchTask(1));
  elements.previousPage.addEventListener("click", () => switchPairPage(-1));
  elements.nextPage.addEventListener("click", () => switchPairPage(1));
  elements.retry.addEventListener("click", () => loadCurrentPage());
  elements.closeDetail.addEventListener("click", () => elements.detailDialog.close());
  elements.detailDialog.addEventListener("close", () => stopObservingCanvases(elements.detailGrids));
  elements.detailDialog.addEventListener("click", (event) => {
    if (event.target === elements.detailDialog) elements.detailDialog.close();
  });

  document.addEventListener("keydown", (event) => {
    if (elements.detailDialog.open || ["SELECT", "INPUT"].includes(event.target.tagName)) return;
    const actions = {
      ArrowLeft: () => switchTask(-1),
      ArrowRight: () => switchTask(1),
      ArrowUp: () => switchPairPage(-1),
      ArrowDown: () => switchPairPage(1),
    };
    if (actions[event.key]) {
      event.preventDefault();
      actions[event.key]();
    }
  });

  elements.workspace.addEventListener("wheel", (event) => {
    event.preventDefault();
    const now = performance.now();
    if (now < state.wheelLockedUntil) return;
    const horizontal = Math.abs(event.deltaX) > Math.abs(event.deltaY);
    const delta = horizontal ? event.deltaX : event.deltaY;
    if (Math.abs(delta) < 8) return;
    state.wheelLockedUntil = now + 280;
    if (horizontal) switchTask(Math.sign(delta));
    else switchPairPage(Math.sign(delta));
  }, { passive: false });

  elements.workspace.addEventListener("pointerdown", (event) => {
    if (event.pointerType === "mouse") return;
    state.touchStart = { x: event.clientX, y: event.clientY };
  });
  elements.workspace.addEventListener("pointerup", (event) => {
    if (!state.touchStart) return;
    const x = event.clientX - state.touchStart.x;
    const y = event.clientY - state.touchStart.y;
    state.touchStart = null;
    if (Math.max(Math.abs(x), Math.abs(y)) < 36) return;
    if (Math.abs(x) > Math.abs(y)) switchTask(x < 0 ? 1 : -1);
    else switchPairPage(y < 0 ? 1 : -1);
  });
}

async function boot() {
  renderSkeleton();
  installInteractions();
  elements.loading.hidden = false;
  try {
    const payload = await fetchJson("/api/index");
    state.datasets = payload.datasets;
    if (!state.datasets.length) throw new Error("No JSON tasks were found in assets_tmp.");
    restoreLocation();
    populateDatasetSelect();
    populateTaskSelect();
    const sourceParts = payload.source.split(/[\\/]/).filter(Boolean);
    elements.sourceLabel.textContent = sourceParts.at(-1) || payload.source;
    await loadCurrentPage();
  } catch (error) {
    showError(error.message);
  } finally {
    elements.loading.hidden = true;
  }
}

boot();
