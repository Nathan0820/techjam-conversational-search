const state = { sessionId: null, turn: 1 };
const BASELINE_SCORE = 0.1067;

const messages = document.querySelector("#messages");
const form = document.querySelector("#chat-form");
const input = document.querySelector("#message-input");
const sendButton = document.querySelector("#send-button");
const statusText = document.querySelector("#status");
const statusCopy = document.querySelector(".status-copy");
const turnLabel = document.querySelector("#turn-label");
const turnProgressBar = document.querySelector("#turn-progress-bar");
const stateView = document.querySelector("#state-view");
const signalCount = document.querySelector("#signal-count");
const recommendationList = document.querySelector("#recommendations");
const resultCount = document.querySelector("#result-count");
const newSessionButton = document.querySelector("#new-session");
const newSessionLabel = newSessionButton.querySelector(".button-label");
const promptButtons = [...document.querySelectorAll(".prompt-chip")];

function clamp(value, minimum = 0, maximum = 1) {
  return Math.min(maximum, Math.max(minimum, Number(value) || 0));
}

function formattedNumber(value, digits) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(digits) : "—";
}

function metric(label, value, ratio, note = "", primary = false) {
  const width = `${(clamp(ratio) * 100).toFixed(1)}%`;
  return `
    <article class="metric${primary ? " primary" : ""}">
      <div class="metric-topline">
        <span class="metric-label">${label}</span>
        ${note ? `<span class="metric-note">${note}</span>` : ""}
      </div>
      <strong>${value}</strong>
      <div class="metric-track" aria-hidden="true"><span style="width: ${width}"></span></div>
    </article>
  `;
}

function percentage(value) {
  const number = Number(value);
  return Number.isFinite(number) ? `${(number * 100).toFixed(1)}%` : "—";
}

function setStatus(message, mode = "ready") {
  statusCopy.textContent = message;
  statusText.dataset.status = mode;
}

function setInteractionDisabled(disabled) {
  input.disabled = disabled;
  sendButton.disabled = disabled;
  promptButtons.forEach(button => { button.disabled = disabled; });
}

function updateTurn(turn) {
  const visibleTurn = Math.min(Math.max(turn, 1), 10);
  turnLabel.textContent = `Turn ${visibleTurn} / 10`;
  turnProgressBar.style.width = `${visibleTurn * 10}%`;
}

function renderMetrics(data) {
  const grid = document.querySelector("#metric-grid");
  if (!data.available) {
    grid.innerHTML = metric("Evaluation", "Not run", 0, "Run a session", true);
    document.querySelector("#sample-count").textContent = "No results.json";
    return;
  }

  const score = Number(data.technical_score);
  const meanTurns = Number(data.mttc);
  const efficiency = Number.isFinite(meanTurns) ? (11 - meanTurns) / 10 : null;
  const baselineGain = Number.isFinite(score) ? `${(score / BASELINE_SCORE).toFixed(1)}× baseline` : "";
  grid.innerHTML = [
    metric("Technical score", formattedNumber(score, 3), score, baselineGain, true),
    metric("Hit Rate@10", percentage(data.hit_rate_at_10), data.hit_rate_at_10),
    metric("Mean reciprocal rank", formattedNumber(data.mrr, 3), data.mrr),
    metric("Mean turns", formattedNumber(meanTurns, 2), efficiency, "lower is better"),
    metric("Efficiency", percentage(efficiency), efficiency),
  ].join("");
  document.querySelector("#sample-count").textContent = `${data.sample_count} sessions · verified`;
}

function showEvaluationRunning() {
  document.querySelector("#metric-grid").innerHTML = metric(
    "Evaluation",
    "Running…",
    0.35,
    "200 sessions",
    true,
  );
  document.querySelector("#sample-count").textContent = "About 1 minute";
}

async function loadMetrics() {
  const response = await fetch("/api/metrics", { cache: "no-store" });
  if (!response.ok) throw new Error("Could not refresh evaluation metrics");
  renderMetrics(await response.json());
}

async function evaluateCurrentAgent() {
  showEvaluationRunning();
  setStatus("Replaying 200 public sessions with the current agent…", "working");
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
  if (role === "assistant") {
    const author = document.createElement("span");
    author.className = "message-author";
    author.textContent = "Findly";
    element.appendChild(author);
  }
  element.appendChild(document.createTextNode(text));
  messages.appendChild(element);
  messages.scrollTop = messages.scrollHeight;
}

function addProductHighlights(items) {
  const products = items.slice(0, 3);
  if (!products.length) return;

  const block = document.createElement("section");
  block.className = "chat-recommendations";
  block.setAttribute("aria-label", "Top three product recommendations");
  block.innerHTML = `
    <div class="chat-picks-heading">
      <div>
        <span class="chat-picks-kicker">Findly shortlist</span>
        <strong>Top ${products.length} right now</strong>
      </div>
      <span class="chat-picks-note">Updated this turn</span>
    </div>
    <ol class="chat-pick-list">
      ${products.map((item, index) => `
        <li class="chat-pick-card">
          <span class="chat-pick-rank">${String(index + 1).padStart(2, "0")}</span>
          <div class="chat-pick-content">
            <div class="chat-pick-title-row">
              <strong>${escapeHtml(item.title)}</strong>
              <span class="chat-pick-score" title="Reranker score">
                ${item.score == null ? "Unscored" : formattedNumber(item.score, 3)}
              </span>
            </div>
            <div class="chat-pick-details">
              <span><small>Store</small>${escapeHtml(item.store)}</span>
              <span><small>Category</small>${escapeHtml(item.category)}</span>
              <span><small>Price</small>${item.price == null ? "Not listed" : `$${formattedNumber(item.price, 2)}`}</span>
            </div>
            ${item.highlights?.length ? `
              <ul class="chat-pick-highlights" aria-label="Product details">
                ${item.highlights.map(highlight => `<li>${escapeHtml(highlight)}</li>`).join("")}
              </ul>
            ` : ""}
            <span class="chat-pick-asin">ASIN · ${escapeHtml(item.parent_asin)}</span>
          </div>
        </li>
      `).join("")}
    </ol>
  `;
  messages.appendChild(block);
  messages.scrollTop = messages.scrollHeight;
}

function renderState(agentState) {
  const slotEntries = Object.entries(agentState.slots || {});
  const count = slotEntries.reduce((total, [, values]) => total + values.length, 0);
  signalCount.textContent = `${count} signal${count === 1 ? "" : "s"}`;

  const hard = agentState.hard_constraints || [];
  const soft = agentState.soft_preferences || [];
  const summary = `
    <div class="state-summary">
      <div class="constraint-group intent">
        <span>Detected intent</span>
        <strong>${escapeHtml(agentState.intent || "Still learning")}</strong>
      </div>
      <div class="constraint-group hard">
        <span>Must have</span>
        <strong>${hard.length ? hard.map(formatLabel).join(", ") : "None yet"}</strong>
      </div>
      <div class="constraint-group soft">
        <span>Nice to have</span>
        <strong>${soft.length ? soft.map(formatLabel).join(", ") : "None yet"}</strong>
      </div>
    </div>
  `;

  const rows = slotEntries.map(([label, values]) => `
    <div class="state-row">
      <span class="state-label">${escapeHtml(formatLabel(label))}</span>
      <div class="chips">
        ${values.map(value => `<span class="chip" title="${escapeHtml(value)}">${escapeHtml(value)}</span>`).join("")}
      </div>
    </div>
  `).join("");

  stateView.className = "state-content";
  stateView.innerHTML = summary + rows;
}

function renderRecommendations(items) {
  resultCount.textContent = items.length ? `${items.length} matches` : "No matches";
  if (!items.length) {
    recommendationList.innerHTML = `
      <li class="empty-state compact">
        <strong>No matching products</strong>
        <span>Try relaxing one preference or describing the use case differently.</span>
      </li>
    `;
    return;
  }

  recommendationList.innerHTML = items.map((item, index) => `
    <li class="product-card">
      <span class="rank">${String(index + 1).padStart(2, "0")}</span>
      <div class="recommendation-head">
        <span class="product-title">${escapeHtml(item.title)}</span>
        <span class="score" title="Reranker score">${item.score == null ? "—" : formattedNumber(item.score, 3)}</span>
      </div>
      <div class="product-meta">
        <span>${escapeHtml(item.store)}</span>
        <span>${escapeHtml(item.category)}</span>
        ${item.price == null ? "" : `<span class="price">$${formattedNumber(item.price, 2)}</span>`}
      </div>
      <span class="asin">${escapeHtml(item.parent_asin)}</span>
    </li>
  `).join("");
}

function formatLabel(value) {
  return String(value).replaceAll("_", " ");
}

function escapeHtml(value) {
  const div = document.createElement("div");
  div.textContent = String(value);
  return div.innerHTML;
}

function resetConversationView() {
  messages.innerHTML = "";
  addMessage("assistant", "Fresh start. What are you shopping for?");
  stateView.className = "empty-state";
  stateView.innerHTML = "<strong>Listening for clues</strong><span>Your intent and preferences will appear here as you chat.</span>";
  recommendationList.innerHTML = "<li class=\"empty-state compact\"><strong>No search yet</strong><span>Recommendations will respond to every new clue.</span></li>";
  signalCount.textContent = "0 signals";
  resultCount.textContent = "Waiting";
  updateTurn(1);
}

async function newSession({ reevaluate = false } = {}) {
  setInteractionDisabled(true);
  newSessionButton.disabled = true;
  setStatus("Starting a clean session…", "working");
  const response = await fetch("/api/reset", { method: "POST", body: "{}" });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "Could not start a new session");
  state.sessionId = data.session_id;
  state.turn = 1;
  updateTurn(state.turn);
  if (reevaluate) {
    await evaluateCurrentAgent();
  } else {
    await loadMetrics();
  }
  setStatus(reevaluate ? "Fresh evaluation loaded. Ready to search." : "Ready for your first clue.");
  setInteractionDisabled(false);
  newSessionButton.disabled = false;
  input.focus();
}

form.addEventListener("submit", async event => {
  event.preventDefault();
  const message = input.value.trim();
  if (!message || !state.sessionId || state.turn > 10) return;
  addMessage("user", message);
  input.value = "";
  setInteractionDisabled(true);
  newSessionButton.disabled = true;
  form.setAttribute("aria-busy", "true");
  setStatus("Searching 50,000 products and updating your preference map…", "working");
  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: state.sessionId, message, turn: state.turn }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Request failed");
    addMessage("assistant", data.message);
    addProductHighlights(data.recommendations);
    renderState(data.state);
    renderRecommendations(data.recommendations);
    state.turn += 1;
    updateTurn(state.turn);
    setStatus(state.turn > 10 ? "Session complete. Start a new session to continue." : "Preference map updated.");
  } catch (error) {
    addMessage("error", error.message);
    setStatus("Something went wrong. Your session is still here.", "error");
  } finally {
    const sessionComplete = state.turn > 10;
    setInteractionDisabled(sessionComplete);
    newSessionButton.disabled = false;
    form.removeAttribute("aria-busy");
    if (!sessionComplete) input.focus();
  }
});

promptButtons.forEach(button => {
  button.addEventListener("click", () => {
    input.value = button.dataset.prompt || "";
    input.focus();
    setStatus("Prompt added — edit it or send when ready.");
  });
});

newSessionButton.addEventListener("click", async () => {
  resetConversationView();
  newSessionLabel.textContent = "Starting…";
  try {
    await newSession({ reevaluate: false });
  } catch (error) {
    setStatus(error.message, "error");
    setInteractionDisabled(!state.sessionId);
    newSessionButton.disabled = false;
  } finally {
    newSessionLabel.textContent = "New session";
  }
});

newSession({ reevaluate: true }).catch(error => {
  setStatus(error.message, "error");
  newSessionButton.disabled = false;
});
