/* Offline rendering: all numerical fields and metrics come from evidence.js. */
"use strict";

const slides = [...document.querySelectorAll(".slide")];
const data = window.DEFENSE_EVIDENCE;
const stage = document.getElementById("stage");
const notesDialog = document.getElementById("notes-dialog");
const overviewDialog = document.getElementById("overview-dialog");
const chapterLabels = {
  overview: "目录",
  context: "01 / 背景与问题",
  method: "02 / 建模与算法",
  evidence: "03 / 数值证据",
  hardware: "04 / 实物闭环",
  closing: "05 / 结论",
};
let current = 0;
let controlsTimer;
let touchStart = null;

function fit() {
  const scale = Math.min(window.innerWidth / 1600, window.innerHeight / 900);
  document.documentElement.style.setProperty("--scale", scale);
}

function showControls() {
  document.body.classList.add("controls-visible");
  clearTimeout(controlsTimer);
  controlsTimer = setTimeout(() => document.body.classList.remove("controls-visible"), 2200);
}

function show(index, updateHash = true) {
  current = Number.isFinite(index) ? Math.max(0, Math.min(slides.length - 1, Math.trunc(index))) : 0;
  slides.forEach((slide, i) => {
    slide.hidden = i !== current;
    slide.classList.toggle("active", i === current);
  });
  document.getElementById("slide-count").textContent = `${String(current + 1).padStart(2, "0")} / ${String(slides.length).padStart(2, "0")}`;
  document.getElementById("previous").disabled = current === 0;
  document.getElementById("next").disabled = current === slides.length - 1;
  document.title = `${current + 1}/${slides.length} · ${slides[current].dataset.title} · 超声相位优化`;
  if (updateHash) {
    try { history.replaceState(null, "", `#${current + 1}`); }
    catch (_) { /* Some local-file viewers do not support replaceState. */ }
  }
}

function toast(message) {
  const status = document.getElementById("status");
  status.textContent = message;
  status.classList.add("visible");
  setTimeout(() => status.classList.remove("visible"), 4000);
}

function showNotes() {
  document.getElementById("notes-title").textContent = `${current + 1}. ${slides[current].dataset.title}`;
  document.getElementById("notes-content").replaceChildren(
    slides[current].querySelector(".speaker-note").content.cloneNode(true),
  );
  const before = slides.slice(0, current).reduce((sum, slide) => sum + Number(slide.dataset.duration), 0);
  const end = before + Number(slides[current].dataset.duration);
  const format = value => `${Math.floor(value / 60)}:${String(value % 60).padStart(2, "0")}`;
  const total = slides.reduce((sum, slide) => sum + Number(slide.dataset.duration), 0);
  document.getElementById("notes-duration").textContent =
    `建议 ${format(before)}–${format(end)} · 本页 ${slides[current].dataset.duration} 秒 · 总讲述 ${format(total)}，预留 ${300 - total} 秒`;
  notesDialog.showModal();
}

async function fullscreen() {
  try {
    if (document.fullscreenElement) await document.exitFullscreen();
    else if (document.documentElement.requestFullscreen) await document.documentElement.requestFullscreen();
    else toast("此浏览器不支持全屏，可使用浏览器自身的全屏功能。");
  } catch (_) { toast("全屏请求未获允许，请使用浏览器自身的全屏功能。"); }
}

function metricChart(id, records, threshold, names, limit, ticks) {
  const root = document.getElementById(id);
  root.innerHTML = '<div class="metric-header"><span>方法</span><span>首次达标时间</span><span>秒</span><span>评估次数</span></div>';
  for (const [key, name] of names) {
    const record = records.find(row => row.id === key && row.threshold === threshold);
    if (!record) throw new Error(`Missing evidence: ${key} at ${threshold}`);
    const row = document.createElement("div");
    row.className = `metric-row${["adjoint", "adam"].includes(key) ? " featured" : ""}`;
    const label = document.createElement("span");
    label.className = "metric-name"; label.textContent = name;
    const track = document.createElement("span"); track.className = "track";
    const seconds = document.createElement("span"); seconds.className = "metric-number";
    const count = document.createElement("span"); count.className = "metric-number";
    if (record.reached) {
      track.style.setProperty("--value", `${record.seconds / limit * 100}%`);
      track.innerHTML = '<span class="track-line"></span><span class="track-dot"></span>';
      seconds.textContent = record.seconds.toFixed(2);
      count.textContent = record.evaluations.toLocaleString("en-US");
    } else {
      track.classList.add("empty");
      seconds.textContent = "未达到"; seconds.classList.add("not-reached");
      count.textContent = "—";
    }
    row.append(label, track, seconds, count);
    root.append(row);
  }
  const axis = document.createElement("div"); axis.className = "metric-axis";
  axis.innerHTML = `<span></span><div class="axis-ticks">${ticks.map(tick => `<span>${tick}</span>`).join("")}</div><span></span><span></span>`;
  root.append(axis);
}

function renderEvidence() {
  if (!data) throw new Error("assets/evidence.js did not load");
  metricChart("algorithm-chart", data.algorithms, -1593.498,
    [["adjoint", "Adam"], ["cmaes", "CMA-ES"], ["lshade", "L-SHADE"]], 50, [0, 25, 50]);
  metricChart("ablation-chart", data.ablation, -1860,
    [["adam", "Adam"], ["adamw", "AdamW"], ["lion", "Lion"], ["nonlinear_cg", "Nonlinear CG"], ["lbfgsb", "L-BFGS-B"]], 25, [0, 12.5, 25]);
  const adam = data.algorithms.find(row => row.id === "adjoint" && row.threshold === -1593.498);
  const lshade = data.algorithms.find(row => row.id === "lshade" && row.threshold === -1593.498);
  document.getElementById("speed-ratio").innerHTML = `${(lshade.seconds / adam.seconds).toFixed(1)}<span>×</span>`;
  const comparison = data.field.comparison;
  document.getElementById("geometric-target").textContent =
    Math.round(comparison.geometric_target_pa).toLocaleString("en-US");
  document.getElementById("optimized-target").textContent =
    Math.round(comparison.optimized_target_pa).toLocaleString("en-US");
  document.getElementById("target-gain").textContent =
    `${(comparison.optimized_target_pa / comparison.geometric_target_pa).toFixed(1)}×`;
  document.getElementById("geometric-contrast").textContent =
    comparison.geometric_contrast.toFixed(2);
  document.getElementById("optimized-contrast").textContent =
    comparison.optimized_contrast.toFixed(2);
  document.getElementById("contrast-gain").textContent =
    `${(comparison.optimized_contrast / comparison.geometric_contrast).toFixed(1)}×`;
}

slides.forEach((slide, index) => {
  const number = String(index + 1).padStart(2, "0");
  slide.querySelector(".page-number").textContent =
    `${number} / ${String(slides.length).padStart(2, "0")}`;
  slide.setAttribute("aria-label", `第 ${index + 1} 页：${slide.dataset.title}`);
  const sectionNumber = slide.querySelector(".section-label b");
  if (sectionNumber) sectionNumber.textContent = number;
  const runningTitle = slide.querySelector(".running-title");
  if (runningTitle && chapterLabels[slide.dataset.chapter]) {
    runningTitle.textContent = chapterLabels[slide.dataset.chapter];
  }
  const li = document.createElement("li"), button = document.createElement("button");
  button.innerHTML = `<span>${number}</span>${slide.dataset.title}`;
  button.addEventListener("click", () => { overviewDialog.close(); show(index); });
  li.append(button); document.getElementById("overview-list").append(li);
});
document.getElementById("previous").addEventListener("click", () => show(current - 1));
document.getElementById("next").addEventListener("click", () => show(current + 1));
document.getElementById("overview-button").addEventListener("click", () => overviewDialog.showModal());
document.getElementById("notes-button").addEventListener("click", showNotes);
document.getElementById("fullscreen-button").addEventListener("click", fullscreen);
document.getElementById("print-button").addEventListener("click", () => window.print());
document.querySelectorAll(".close-dialog").forEach(button => button.addEventListener("click", () => button.closest("dialog").close()));
document.addEventListener("keydown", event => {
  if (notesDialog.open || overviewDialog.open || event.ctrlKey || event.altKey || event.metaKey) return;
  if (["ArrowRight", "ArrowDown", "PageDown", " "].includes(event.key)) {
    if (event.key === " " && event.target.closest("button")) return;
    event.preventDefault(); show(current + 1);
  }
  if (["ArrowLeft", "ArrowUp", "PageUp"].includes(event.key)) { event.preventDefault(); show(current - 1); }
  if (event.key === "Home") { event.preventDefault(); show(0); }
  if (event.key === "End") { event.preventDefault(); show(slides.length - 1); }
  if (/^[1-9]$/.test(event.key) && Number(event.key) <= slides.length) show(Number(event.key) - 1);
  if (event.key.toLowerCase() === "n") showNotes();
  if (event.key.toLowerCase() === "o") overviewDialog.showModal();
  if (event.key.toLowerCase() === "f") fullscreen();
});
stage.addEventListener("touchstart", event => {
  if (event.touches.length === 1) touchStart = {x: event.touches[0].clientX, y: event.touches[0].clientY};
}, {passive: true});
stage.addEventListener("touchend", event => {
  if (!touchStart || !event.changedTouches.length) return;
  const dx = event.changedTouches[0].clientX - touchStart.x;
  const dy = event.changedTouches[0].clientY - touchStart.y;
  if (Math.abs(dx) > 45 && Math.abs(dx) > Math.abs(dy)) show(current + (dx < 0 ? 1 : -1));
  touchStart = null;
}, {passive: true});
window.addEventListener("resize", fit);
window.addEventListener("hashchange", () => show(Number(location.hash.slice(1)) - 1 || 0, false));
document.addEventListener("mousemove", showControls, {passive: true});
document.addEventListener("touchstart", showControls, {passive: true});
window.addEventListener("beforeprint", renderEvidence);
window.lucide?.createIcons();
fit();
const requestedSlide = Number(location.hash.slice(1));
show(
  Number.isInteger(requestedSlide)
    && requestedSlide >= 1
    && requestedSlide <= slides.length
    ? requestedSlide - 1
    : 0,
);
try { renderEvidence(); } catch (error) { console.error(error); toast("数值素材未能加载，请保持 assets 文件夹与 HTML 同目录。"); }
document.fonts.ready.then(renderEvidence).catch(console.error);
showControls();
