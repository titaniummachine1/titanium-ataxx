import init, { Game } from "./pkg/titanium.js";

const SIZE = 7;
const CANVAS = 500;
const PAD = 24;
const CELL = (CANVAS - PAD * 2) / SIZE;
const COLS = "abcdefg";

const canvas = document.getElementById("board");
const ctx = canvas.getContext("2d");
const statusEl = document.getElementById("status");
const countB = document.getElementById("count-black");
const countW = document.getElementById("count-white");
const infoEl = document.getElementById("search-info");
const listEl = document.getElementById("move-list");
const newBtn = document.getElementById("new-game");
const undoBtn = document.getElementById("undo");
const diffSel = document.getElementById("difficulty");
const sideSel = document.getElementById("side");

let game = null;
let human = 0;        // 0 black, 1 white
let busy = true;      // engine thinking / animating
let selected = -1;
let hover = -1;
let legal = [];       // packed moves (from << 8) | to for side to move
let moves = [];       // packed moves as played (0xffff = pass)
let session = 0;      // invalidates stale async callbacks
let lastPacked = 0;   // for last-move highlight + pop animation

const sqName = (sq) => COLS[sq % 7] + (7 - Math.floor(sq / 7));
const fmtNodes = (n) =>
  n >= 1e6 ? (n / 1e6).toFixed(1) + "M" : n >= 1e3 ? (n / 1e3).toFixed(0) + "k" : String(n);

function cellRect(sq) {
  const col = sq % SIZE;
  const row = Math.floor(sq / SIZE);
  return { x: PAD + col * CELL, y: PAD + row * CELL };
}

function sqAt(px, py) {
  const col = Math.floor((px - PAD) / CELL);
  const row = Math.floor((py - PAD) / CELL);
  if (col < 0 || col >= SIZE || row < 0 || row >= SIZE) return -1;
  return row * SIZE + col;
}

// ---------- game flow ----------

function applyDifficulty() {
  const [t, d] = diffSel.value.split(",").map(Number);
  game.set_difficulty(t, d);
}

function newGame() {
  session++;
  game = new Game();
  human = Number(sideSel.value);
  applyDifficulty();
  selected = -1;
  hover = -1;
  moves = [];
  lastPacked = 0;
  listEl.replaceChildren();
  infoEl.textContent = "";
  update();
}

function pushMove(packed) {
  moves.push(packed);
  const li = document.createElement("li");
  if (packed === 0xffff) {
    li.textContent = `${moves.length}. pass`;
    li.className = "pass";
  } else {
    li.textContent = `${moves.length}. ${sqName(packed >> 8)}-${sqName(packed & 0xff)}`;
  }
  listEl.appendChild(li);
  listEl.scrollTop = listEl.scrollHeight;
  lastPacked = packed;
}

function undo() {
  if (busy || game.history_len() <= 1) return;
  let popped = 0;
  do {
    if (!game.undo()) break;
    popped++;
  } while (game.history_len() > 1 && game.turn() !== human);
  moves = moves.slice(0, moves.length - popped);
  lastPacked = moves.length ? moves[moves.length - 1] : 0;
  listEl.querySelectorAll("li").forEach((li, i) => {
    if (i >= moves.length) li.remove();
  });
  selected = -1;
  update();
}

function update() {
  const mySession = session;
  legal = game.legal_moves();
  const [b, w] = game.counts();
  countB.textContent = b;
  countW.textContent = w;
  undoBtn.disabled = busy || game.history_len() <= 1;

  if (game.over()) {
    busy = false;
    const [bl, wh] = game.counts();
    const win = game.winner();
    statusEl.textContent =
      win === 2 ? `Draw ${bl}\u2013${wh}`
      : win === 0 ? `Game over \u2014 Black wins ${bl}\u2013${wh}`
      : `Game over \u2014 White wins ${wh}\u2013${bl}`;
    render();
    return;
  }

  if (game.turn() === human) {
    if (!game.has_moves()) {
      busy = true;
      statusEl.textContent = "You have no moves \u2014 passing\u2026";
      const s = session;
      setTimeout(() => {
        if (s !== session) return;
        game.pass();
        pushMove(0xffff);
        update();
      }, 600);
    } else {
      busy = false;
      statusEl.textContent = human === 0 ? "Your move \u2014 Black" : "Your move \u2014 White";
    }
  } else {
    busy = true;
    statusEl.textContent = "Engine thinking\u2026";
    const s = session;
    setTimeout(() => {
      if (s !== session) return;
      const packed = game.ai_move();
      pushMove(packed === null || packed === undefined ? 0xffff : packed);
      const [d, n, ms] = game.search_info();
      infoEl.textContent = d
        ? `depth ${d} \u00b7 ${fmtNodes(n)} nodes \u00b7 ${(ms / 1000).toFixed(1)}s`
        : "";
      update();
    }, 30);
  }
  render();
}

function animatePop() {
  const start = performance.now();
  const s = session;
  const step = (t) => {
    if (s !== session) return;
    const k = Math.min(1, (t - start) / 150);
    render(0.6 + 0.4 * k);
    if (k < 1) requestAnimationFrame(step);
    else render(1);
  };
  requestAnimationFrame(step);
}

// ---------- rendering ----------

function render(scaleP = 1) {
  ctx.clearRect(0, 0, CANVAS, CANVAS);

  for (let sq = 0; sq < SIZE * SIZE; sq++) {
    const { x, y } = cellRect(sq);
    ctx.fillStyle = ((sq + Math.floor(sq / SIZE)) % 2 === 0) ? "#2c3a4f" : "#243144";
    ctx.fillRect(x, y, CELL, CELL);

    if (sq === selected) {
      ctx.fillStyle = "rgba(55, 200, 195, 0.16)";
      ctx.fillRect(x, y, CELL, CELL);
    }
    if (sq === hover && game.piece(sq) === human + 1 && !busy && game.turn() === human) {
      ctx.fillStyle = "rgba(255, 255, 255, 0.06)";
      ctx.fillRect(x, y, CELL, CELL);
    }
    if (lastPacked !== 0 && lastPacked !== 0xffff &&
        (sq === (lastPacked >> 8) || sq === (lastPacked & 0xff))) {
      ctx.strokeStyle = "#f0c04a";
      ctx.lineWidth = 3;
      ctx.strokeRect(x + 2, y + 2, CELL - 4, CELL - 4);
      ctx.lineWidth = 1;
    }
    ctx.strokeStyle = "rgba(0,0,0,0.25)";
    ctx.strokeRect(x + 0.5, y + 0.5, CELL - 1, CELL - 1);
  }

  // legal targets for the selected piece
  if (!busy && selected >= 0) {
    for (const packed of legal) {
      if ((packed >> 8) !== selected) continue;
      const to = packed & 0xff;
      const dr = Math.abs(Math.floor(to / 7) - Math.floor(selected / 7));
      const df = Math.abs((to % 7) - (selected % 7));
      drawTarget(to, Math.max(dr, df) <= 1);
    }
  }

  // pieces
  for (let sq = 0; sq < SIZE * SIZE; sq++) {
    const p = game.piece(sq);
    if (p === 0 || p === 3) continue;
    const isLanding = lastPacked !== 0 && lastPacked !== 0xffff && sq === (lastPacked & 0xff);
    const scale = isLanding ? scaleP : 1;
    drawPiece(sq, p === 1 ? "black" : "white", scale);
  }

  // coordinates
  ctx.fillStyle = "#7f8ea6";
  ctx.font = "11px Consolas, monospace";
  for (let i = 0; i < SIZE; i++) {
    ctx.textAlign = "center";
    ctx.fillText(COLS[i], PAD + i * CELL + CELL / 2, CANVAS - 7);
    ctx.textAlign = "left";
    ctx.fillText(String(7 - i), 7, PAD + i * CELL + CELL / 2 + 4);
  }
}

function drawTarget(sq, clone) {
  const { x, y } = cellRect(sq);
  const cx = x + CELL / 2, cy = y + CELL / 2;
  ctx.fillStyle = "rgba(55, 200, 195, 0.18)";
  ctx.fillRect(x + 1, y + 1, CELL - 2, CELL - 2);
  if (clone) {
    ctx.beginPath();
    ctx.arc(cx, cy, CELL * 0.14, 0, Math.PI * 2);
    ctx.fillStyle = "rgba(55, 200, 195, 0.9)";
    ctx.fill();
  } else {
    ctx.beginPath();
    ctx.arc(cx, cy, CELL * 0.30, 0, Math.PI * 2);
    ctx.strokeStyle = "rgba(55, 200, 195, 0.85)";
    ctx.lineWidth = 3;
    ctx.stroke();
    ctx.lineWidth = 1;
  }
}

function drawPiece(sq, color, scale = 1) {
  const { x, y } = cellRect(sq);
  const cx = x + CELL / 2, cy = y + CELL / 2, r = CELL * 0.36 * scale;
  if (r <= 0.5) return;
  const g = ctx.createRadialGradient(cx - r * 0.35, cy - r * 0.35, r * 0.15, cx, cy, r);
  if (color === "black") {
    g.addColorStop(0, "#5a6b85");
    g.addColorStop(0.7, "#0c0f14");
  } else {
    g.addColorStop(0, "#ffffff");
    g.addColorStop(0.7, "#b9c3d4");
  }
  ctx.beginPath();
  ctx.arc(cx, cy, r, 0, Math.PI * 2);
  ctx.fillStyle = g;
  ctx.shadowColor = "rgba(0, 0, 0, 0.4)";
  ctx.shadowBlur = 6;
  ctx.shadowOffsetY = 2;
  ctx.fill();
  ctx.shadowColor = "transparent";
  ctx.shadowBlur = 0;
  ctx.shadowOffsetY = 0;
  ctx.strokeStyle = color === "black" ? "#000" : "#7f8ea6";
  ctx.stroke();
}

// ---------- input ----------

canvas.addEventListener("mousemove", (e) => {
  if (busy || game.turn() !== human || game.over()) { hover = -1; return; }
  const rect = canvas.getBoundingClientRect();
  const scale = CANVAS / rect.width;
  hover = sqAt((e.clientX - rect.left) * scale, (e.clientY - rect.top) * scale);
  canvas.style.cursor = game.piece(hover) === human + 1 ? "pointer" : "default";
  render();
});

canvas.addEventListener("mouseleave", () => { hover = -1; });

canvas.addEventListener("click", (e) => {
  if (busy || game.over() || game.turn() !== human) return;
  const rect = canvas.getBoundingClientRect();
  const scale = CANVAS / rect.width;
  const sq = sqAt((e.clientX - rect.left) * scale, (e.clientY - rect.top) * scale);
  if (sq < 0) { selected = -1; render(); return; }

  if (selected >= 0) {
    const target = legal.find((m) => (m >> 8) === selected && (m & 0xff) === sq);
    if (target !== undefined) {
      game.make(selected, sq);
      pushMove(target);
      selected = -1;
      update();
      return;
    }
  }
  selected = game.piece(sq) === human + 1 ? sq : -1;
  render();
});

newBtn.addEventListener("click", newGame);
undoBtn.addEventListener("click", undo);
diffSel.addEventListener("change", applyDifficulty);

// ---------- boot ----------

(async () => {
  try {
    await init();
    newGame();
  } catch (err) {
    statusEl.textContent = "Failed to load engine: " + err;
    console.error(err);
  }
})();
