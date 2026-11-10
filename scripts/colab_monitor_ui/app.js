"use strict";
const token = document.querySelector('meta[name="monitor-token"]').content;
const cards = new Map();
const $ = id => document.getElementById(id);
const node = (tag, text, cls) => {
  const item = document.createElement(tag);
  if (text != null) item.textContent = text;
  if (cls) item.className = cls;
  return item;
};
const number = (v, digits = 3) => typeof v === "number" && Number.isFinite(v) ? v.toFixed(digits) : "—";
const pct = v => typeof v === "number" ? `${number(v, 2)}%` : "—";
const fractionPct = v => v == null ? "—" : pct(100 * v);
const stamp = v => v ? new Date(v).toLocaleTimeString() : "";
async function api(path, body) {
  const response = await fetch(path, {method:"POST", headers:{"Content-Type":"application/json", "X-Monitor-Token":token}, body:JSON.stringify(body)});
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || "Check failed.");
  return result;
}
function error(target, value) { target.replaceChildren(node("p", value.message || String(value), "error")); }
function speedText(speed) {
  if (!speed || speed.per_second == null) return speed?.note || "Speed unavailable.";
  if (speed.per_second === 0) return `No logged progress during ${number(speed.elapsed_seconds, 0)}s. This alone does not prove a stall.`;
  return `~${number(speed.per_second, 4)} ${speed.unit}/s · ${number(speed.seconds_per_unit, 1)} s/${speed.unit === "epochs" ? "epoch" : "batch"} (${number(speed.elapsed_seconds, 0)}s window)`;
}
function renderGPU(target, data) {
  target.replaceChildren();
  for (const gpu of data.gpus || []) {
    const used = gpu.memoryUsedBytes, total = gpu.memoryTotalBytes;
    const memory = typeof used === "number" && total > 0 ? `${number(used / 2**30, 1)} / ${number(total / 2**30, 1)} GiB (${number(100 * used / total, 1)}%)` : "unavailable";
    target.append(node("p", `${gpu.name || "GPU"} · RAM ${memory} · utilization ${gpu.gpuUtilization == null ? "—" : pct(100 * gpu.gpuUtilization)}`, "numbers"));
  }
  if (!data.gpus?.length) target.append(node("p", "No GPU reported."));
  target.append(node("small", `GPU snapshot ${stamp(data.observed_at)} · current use, not peak`));
  if (data.note) target.append(node("p", data.note));
  if (data.speed) target.append(node("p", speedText(data.speed), "numbers"));
}
function metricLine(metrics) {
  return `loss ${number(metrics.loss, 4)} · MTP ${number(metrics.mtp_loss, 4)} · total ${number(metrics.total_loss, 4)} · LR ${metrics.lr == null ? "—" : metrics.lr.toExponential(2)}`;
}
function renderMetrics(target, data) {
  target.replaceChildren();
  const p = data.progress || {}, epoch = p.epoch_metrics || {};
  const current = p.epoch == null ? "No epoch reported yet" : `Epoch ${p.epoch}${p.epoch_total ? ` / ${p.epoch_total}` : ""}${p.step != null ? ` · step ${p.step}${p.steps_per_epoch ? ` / ~${p.steps_per_epoch}` : ""}` : " completed"}`;
  target.append(node("p", `${data.run_name || "Run"} · ${p.stage || p.kind || "status"} · ${current}`, "numbers"));
  if (p.error_observed) target.append(node("p", "An error was observed in the run console. Inspect its logs and completion status.", "error"));
  if (p.evaluation) target.append(node("p", `Decode ${p.evaluation.split}: batch ${p.evaluation.batch} / ${p.evaluation.total_batches}`));
  if (Object.keys(p.step_metrics || {}).length) target.append(node("p", `Latest step: ${metricLine(p.step_metrics)}`, "numbers"));
  if (epoch.epoch != null) {
    target.append(node("p", `Completed epoch ${epoch.epoch}: ${metricLine(epoch)}`, "numbers"));
    target.append(node("p", `Training accuracy: color ${pct(epoch.color_accuracy)} · special tokens ${pct(epoch.special_token_accuracy)} · exact output ${pct(epoch.exact_output_accuracy)}`, "numbers"));
  }
  if (data.scores?.length) {
    for (const score of data.scores) target.append(node("p", `Held-out score (${score.eval}): ${number(score.score, 2)} / ${number(score.max_score, 0)} · ${pct(score.percentage)}`, "numbers"));
  } else target.append(node("p", data.evaluation_configured === false ? "Held-out evaluation: not configured for this run." : "Held-out score: pending / no scored result yet."));
  for (const report of data.evaluation_accuracy || []) {
    for (const [split, value] of Object.entries(report.splits || {})) {
      target.append(node("p", `Teacher-forced evaluation (${report.eval}, ${split}): color ${fractionPct(value.color_accuracy)} · exact output ${fractionPct(value.exact_output_accuracy)}`));
    }
  }
  target.append(node("small", `Read ${stamp(data.observed_at)} · training accuracy and held-out score are different metrics`));
  if (data.completion?.finalized) target.append(node("p", `${data.completion.ok ? "Run succeeded" : "Run finished; inspect its outcome"} · backups ${data.completion.backups_complete ? "complete" : "incomplete / unknown"}`));
  const refs = data.comparisons;
  if (refs?.rows?.length) {
    const details = node("details");
    details.open = true;
    details.append(node("summary", `Compare completed epoch ${refs.epoch ?? "—"} with saved references`));
    const table = node("table"), head = node("tr");
    for (const title of ["Run", "Loss", "Total", "Color", "Exact output", "Final score"]) head.append(node("th", title));
    const thead = node("thead"); thead.append(head); table.append(thead);
    const tbody = node("tbody");
    for (const ref of [{label:"Current", metrics:epoch, score:data.scores?.[0] || {}}, ...refs.rows]) {
      const row = node("tr"), m = ref.metrics || {};
      for (const value of [ref.label, number(m.loss, 4), number(m.total_loss, 4), pct(m.color_accuracy), pct(m.exact_output_accuracy), pct(ref.score?.percentage)]) row.append(node("td", value));
      tbody.append(row);
    }
    table.append(tbody); details.append(table);
    const notes = node("details"); notes.append(node("summary", "Comparison differences"));
    for (const caveat of refs.caveats || []) notes.append(node("p", caveat, "muted"));
    details.append(notes); target.append(details);
  }
}
function card(run) {
  const key = `${run.account}/${run.endpoint}`;
  if (cards.has(key)) return cards.get(key);
  const element = node("article", null, "vm"), title = node("h3", run.session), description = node("small", `${run.gpu} · ${run.shape}${run.branch ? ` · ${run.branch}` : ""}`);
  const actions = node("div", null, "actions"), performance = node("button", "Speed / GPU"), metrics = node("button", "Training metrics");
  const gpu = node("div", null, "output"), training = node("div", null, "output");
  gpu.setAttribute("aria-live", "polite"); training.setAttribute("aria-live", "polite");
  actions.append(performance, metrics); element.append(title, description, actions, gpu, training);
  metrics.onclick = async () => {
    metrics.disabled = true; metrics.textContent = "Reading…";
    try { renderMetrics(training, await api("/api/check", {account:run.account, endpoint:run.endpoint, kind:"metrics"})); }
    catch (e) { error(training, e); }
    finally { metrics.disabled = false; metrics.textContent = "Training metrics"; }
  };
  performance.onclick = async () => {
    performance.disabled = true; performance.textContent = "Reading GPU…";
    try {
      const result = await api("/api/check", {account:run.account, endpoint:run.endpoint, kind:"performance"});
      renderGPU(gpu, result);
      if (result.sample_id) {
        performance.textContent = "Measuring speed · 30s…";
        const line = node("p", "Sampling logged progress for 30 seconds…"); gpu.append(line);
        await new Promise(resolve => setTimeout(resolve, result.sample_seconds * 1000));
        const last = await api("/api/speed", {sample_id:result.sample_id});
        line.textContent = speedText(last.speed);
      }
    } catch (e) { gpu.append(node("p", e.message, "error")); }
    finally { performance.disabled = false; performance.textContent = "Speed / GPU"; }
  };
  if (run.cached?.performance) renderGPU(gpu, run.cached.performance);
  if (run.cached?.metrics) renderMetrics(training, run.cached.metrics);
  cards.set(key, element);
  return element;
}
async function refresh(force = true) {
  $("refresh").disabled = true; $("status").textContent = "Reading both accounts…";
  try {
    const data = await api("/api/runs", {refresh:force});
    const fragment = document.createDocumentFragment(); const visible = new Set(); let count = 0;
    for (const account of data.accounts) {
      const section = node("section"); section.append(node("h2", `Account ${account.account} · ${account.email}`));
      if (account.error) section.append(node("p", account.error, "error"));
      else if (!account.runs.length) section.append(node("p", "No active VMs.", "muted"));
      for (const run of account.runs) { section.append(card(run)); visible.add(`${run.account}/${run.endpoint}`); count++; }
      fragment.append(section);
    }
    $("accounts").replaceChildren(fragment);
    for (const key of cards.keys()) if (!visible.has(key)) cards.delete(key);
    $("status").textContent = `${count} VMs listed · ${stamp(data.observed_at)}`;
  } catch (e) { error($("status"), e); }
  finally { $("refresh").disabled = false; }
}
$("refresh").onclick = () => refresh();
refresh(false);
