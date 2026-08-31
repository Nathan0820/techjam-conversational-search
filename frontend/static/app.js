const state = { sessionId: null, turn: 1 };

const messages = document.querySelector("#messages");
const form = document.querySelector("#chat-form");
const input = document.querySelector("#message-input");
const sendButton = document.querySelector("#send-button");
const statusText = document.querySelector("#status");
const turnLabel = document.querySelector("#turn-label");
const stateView = document.querySelector("#state-view");
const recommendationList = document.querySelector("#recommendations");
const newSessionButton = document.querySelector("#new-session");

function metric(label, value) {
  return `<div class="metric"><strong>${value}</strong><span>${label}</span></div>`;
}

function percentage(value) {
  return value == null ? "—" : `${(value * 100).toFixed(1)}%`;
}

function renderMetrics(data) {
  const grid = document.querySelector("#metric-grid");
  if (!data.available) {
    grid.innerHTML = metric("Evaluation", "Not run");
    document.querySelector("#sample-count").textContent = "No results.json";
    return;
  }
  grid.innerHTML = [
    metric("Technical score", Number(data.technical_score).toFixed(3)),
    metric("Hit Rate@10", percentage(data.hit_rate_at_10)),
    metric("MRR", Number(data.mrr).toFixed(3)),
    metric("Mean turns", Number(data.mttc).toFixed(2)),
    metric("Efficiency", percentage((11 - data.mttc) / 10)),
  ].join("");
  document.querySelector("#sample-count").textContent = `${data.sample_count} sessions`;
}

function showEvaluationRunning() {
  document.querySelector("#metric-grid").innerHTML = metric("Evaluation", "Running…");
  document.querySelector("#sample-count").textContent = "200 sessions · about 1 minute";
}

async function loadMetrics() {
  const response = await fetch("/api/metrics", { cache: "no-store" });
  if (!response.ok) throw new Error("Could not refresh evaluation metrics");
  renderMetrics(await response.json());
}

async function evaluateCurrentAgent() {
  showEvaluationRunning();
  statusText.textContent = "Evaluating the current agent on the public set…";
  const response = await fetch("/api/evaluate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: "{}",
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "Could not evaluate the current agent");
  renderMetrics(data);
}

function addMessage(role, text) {
  const element = document.createElement("div");
  element.className = `message ${role}`;
  element.textContent = text;
  messages.appendChild(element);
  messages.scrollTop = messages.scrollHeight;
}

function renderState(agentState) {
  const rows = [
    ["Intent", [agentState.intent || "unknown"]],
    ...Object.entries(agentState.slots),
  ];
  if (rows.length === 1 && !agentState.intent) {
    stateView.innerHTML = '<div class="empty">No preferences extracted yet.</div>';
    return;
  }
  stateView.innerHTML = rows.map(([label, values]) => `
    <div class="state-row">
      <span>${label.replaceAll("_", " ")}</span>
      <div class="chips">${values.map(value => `<span class="chip">${escapeHtml(value)}</span>`).join("")}</div>
    </div>
  `).join("");
}

function renderRecommendations(items) {
  if (!items.length) {
    recommendationList.innerHTML = '<li class="empty">No matching products returned.</li>';
    return;
  }
  recommendationList.innerHTML = items.map(item => `
    <li class="product-card">
      <div class="recommendation-head">
        <span class="product-title">${escapeHtml(item.title)}</span>
        <span class="score">${item.score == null ? "" : Number(item.score).toFixed(3)}</span>
      </div>
      <div class="product-meta">
        <span>${escapeHtml(item.store)}</span>
        <span>${escapeHtml(item.category)}</span>
        ${item.price == null ? "" : `<span>$${Number(item.price).toFixed(2)}</span>`}
      </div>
      <span class="asin">${escapeHtml(item.parent_asin)}</span>
    </li>
  `).join("");
}

function escapeHtml(value) {
  const div = document.createElement("div");
  div.textContent = String(value);
  return div.innerHTML;
}

async function newSession({ reevaluate = false } = {}) {
  input.disabled = true;
  sendButton.disabled = true;
  newSessionButton.disabled = true;
  statusText.textContent = "Starting a session…";
  const response = await fetch("/api/reset", { method: "POST", body: "{}" });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "Could not start a new session");
  state.sessionId = data.session_id;
  state.turn = 1;
  turnLabel.textContent = "Turn 1 / 10";
  if (reevaluate) {
    await evaluateCurrentAgent();
  } else {
    await loadMetrics();
  }
  statusText.textContent = "Ready";
  input.disabled = false;
  sendButton.disabled = false;
  newSessionButton.disabled = false;
  input.focus();
}

form.addEventListener("submit", async event => {
  event.preventDefault();
  const message = input.value.trim();
  if (!message || !state.sessionId || state.turn > 10) return;
  addMessage("user", message);
  input.value = "";
  input.disabled = true;
  sendButton.disabled = true;
  statusText.textContent = "Searching the product catalog…";
  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: state.sessionId, message, turn: state.turn }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Request failed");
    addMessage("assistant", data.message);
    renderState(data.state);
    renderRecommendations(data.recommendations);
    state.turn += 1;
    turnLabel.textContent = `Turn ${Math.min(state.turn, 10)} / 10`;
    statusText.textContent = state.turn > 10 ? "Session complete" : "Ready";
  } catch (error) {
    addMessage("error", error.message);
    statusText.textContent = "Something went wrong";
  } finally {
    input.disabled = state.turn > 10;
    sendButton.disabled = state.turn > 10;
    if (!input.disabled) input.focus();
  }
});

newSessionButton.addEventListener("click", async () => {
  messages.innerHTML = '<div class="message assistant">New session started. What are you shopping for?</div>';
  stateView.innerHTML = '<div class="empty">No preferences extracted yet.</div>';
  recommendationList.innerHTML = '<li class="empty">Send a message to rank products.</li>';
  try {
    await newSession({ reevaluate: true });
  } catch (error) {
    statusText.textContent = error.message;
    input.disabled = !state.sessionId;
    sendButton.disabled = !state.sessionId;
    newSessionButton.disabled = false;
  }
});

newSession().catch(error => {
  statusText.textContent = error.message;
  newSessionButton.disabled = false;
});
