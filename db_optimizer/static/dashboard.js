const POLL_MS = 2000;
let polling = null;

// ── Hyperparameter defaults & wiring ────────────────
const defaults = {
    n_islands: 32,
    batch_size: 8192,
    max_pending: 200000,
    elite_k: 20,
    tournament_size: 4,
    migration_prob: 0.08,
    p_mutate_coeff: 0.55,
    p_mutate_gaussian: 0.25,
    p_crossover: 0.12,
    p_shadow_reinject: 0.08,
    tier1_threshold: 0.20,
    tier2_threshold: 0.15,
    max_gpu_batch: 100000,
    hit_threshold: 1e-10,
    minimax_sweeps: 5,
    minimax_fine_range: 0.003,
    max_generations: 100000,
    cpu_workers: Math.max(1, (navigator.hardwareConcurrency || 4) - 2),
    refine_batch: 44,
    gpu_minimax_sweeps: 3,
    gpu_minimax_batch: 40000,
    gpu_minimax_n_trials: 64,
    gpu_minimax_fine_range: 0.005,
};

const fieldMap = {
    n_islands: "cfg-n-islands",
    batch_size: "cfg-batch-size",
    max_pending: "cfg-max-pending",
    elite_k: "cfg-elite-k",
    tournament_size: "cfg-tournament-size",
    migration_prob: "cfg-migration-prob",
    p_mutate_coeff: "cfg-p-coeff",
    p_mutate_gaussian: "cfg-p-gaussian",
    p_crossover: "cfg-p-crossover",
    p_shadow_reinject: "cfg-p-shadow",
    tier1_threshold: "cfg-tier1-thresh",
    tier2_threshold: "cfg-tier2-thresh",
    max_gpu_batch: "cfg-max-gpu-batch",
    hit_threshold: "cfg-hit-thresh",
    minimax_sweeps: "cfg-minimax-sweeps",
    minimax_fine_range: "cfg-minimax-fine",
    max_generations: "cfg-max-gens",
    cpu_workers: "cfg-cpu-workers",
    refine_batch: "cfg-refine-batch",
    gpu_minimax_sweeps: "cfg-gpu-mm-sweeps",
    gpu_minimax_batch: "cfg-gpu-mm-batch",
    gpu_minimax_n_trials: "cfg-gpu-mm-trials",
    gpu_minimax_fine_range: "cfg-gpu-mm-fine",
};

const settingHelp = {
    n_islands: "Number of active islands sampled from the full role×size lattice. Higher values cover more role/size combinations; lower values keep the search tighter.",
    batch_size: "Children generated per generation across all islands. Larger batches exploit GPU parallelism better but increase memory. Reasonable range: 256–8192.",
    max_pending: "Maximum candidates waiting for GPU screening. Acts as backpressure — generation pauses when this is full. Reasonable range: 10000–200000.",
    elite_k: "Top K candidates per island used as parents for the next generation. Higher values increase diversity; lower values increase selection pressure. Reasonable range: 3–30.",
    tournament_size: "Tournament size for parent selection. Higher = stronger selection pressure. Reasonable range: 2–10.",
    migration_prob: "Probability of assigning a child to a random island instead of its parent's island. Enables cross-island gene flow. Reasonable range: 0.01–0.20.",
    p_mutate_coeff: "Probability of algebraic coefficient mutation (snap to nearest algebraic value). The dominant mutation operator. Reasonable range: 0.30–0.80.",
    p_mutate_gaussian: "Probability of Gaussian coefficient perturbation. Continuous local exploration. Reasonable range: 0.10–0.40.",
    p_crossover: "Probability of uniform term crossover between two parents. Recombines structural blocks. Reasonable range: 0.05–0.30.",
    p_shadow_reinject: "Probability of reinjecting a candidate from the shadow archive. Recovers lost diversity. Reasonable range: 0.01–0.15.",
    tier1_threshold: "Max-abs fitness threshold for passing GPU screening (Tier 1). Candidates worse than this are discarded. Reasonable range: 0.10–0.30.",
    tier2_threshold: "Max-abs fitness threshold for CPU minimax refinement (Tier 2→3). Only candidates better than this get expensive CPU polish. Reasonable range: 0.08–0.15.",
    max_gpu_batch: "Maximum candidates per GPU evaluation batch. Capped by VRAM. On RTX 3060 12GB, 50000 is safe. Reasonable range: 5000–100000.",
    hit_threshold: "Residual threshold for declaring an exact solution found. Should be extremely small. Reasonable range: 1e-12 to 1e-6.",
    minimax_sweeps: "Number of greedy minimax coordinate descent sweeps during CPU refinement. Each sweep iterates all 513 coefficients. More sweeps = better local search, slower per candidate. Reasonable range: 1–10.",
    minimax_fine_range: "Half-width of fine grid around each coefficient during minimax sweeps. Smaller = tighter local moves. Reasonable range: 0.001–0.01.",
    max_generations: "Maximum generations before the optimizer stops. Use a large value for open-ended search. Reasonable range: 1000–10000000.",
    cpu_workers: "Number of parallel CPU processes for minimax coordinate descent refinement. Default is (CPU cores − 2). Higher values increase CPU utilization but also memory usage. Reasonable range: 4–24.",
    refine_batch: "Number of candidates sent to the CPU refine pool per cycle. Should be ≥ cpu_workers to keep all workers busy. Reasonable range: 10–100.",
    gpu_minimax_sweeps: "Number of GPU minimax coordinate descent sweeps (Tier 2). Each sweep iterates all 513 coefficients on GPU. More sweeps = better refinement, higher GPU utilization. Reasonable range: 1–10.",
    gpu_minimax_batch: "Maximum candidates per GPU minimax batch. Larger batches fill more GPU SMs. Capped by VRAM. Reasonable range: 5000–60000.",
    gpu_minimax_n_trials: "Trial perturbations per coefficient during GPU minimax. More trials = more parallel GPU work per step. Reasonable range: 16–128.",
    gpu_minimax_fine_range: "Half-width of perturbation grid during GPU minimax. Wider = more exploration per sweep. Reasonable range: 0.001–0.01.",
};

function resetSettings() {
    localStorage.removeItem("dbopt_settings");
    for (const [key, elId] of Object.entries(fieldMap)) {
        const el = $(elId);
        if (el) el.value = defaults[key];
    }
    markDirty();
}

function loadSettings() {
    const saved = localStorage.getItem("dbopt_settings");
    const state = saved ? { ...defaults, ...JSON.parse(saved) } : { ...defaults };
    for (const [key, elId] of Object.entries(fieldMap)) {
        const el = $(elId);
        if (el) el.value = state[key] ?? defaults[key];
    }
}

function saveSettings() {
    const state = {};
    for (const [key, elId] of Object.entries(fieldMap)) {
        const el = $(elId);
        if (el) state[key] = parseFloat(el.value);
    }
    localStorage.setItem("dbopt_settings", JSON.stringify(state));
    return state;
}

function getSettings() {
    const state = {};
    for (const [key, elId] of Object.entries(fieldMap)) {
        const el = $(elId);
        if (el) state[key] = parseFloat(el.value);
    }
    return state;
}

function setLocked(locked) {
    for (const elId of Object.values(fieldMap)) {
        const el = $(elId);
        if (el) el.disabled = locked;
    }
}

function initTooltips() {
    for (const [key, elId] of Object.entries(fieldMap)) {
        const label = document.querySelector(`label[for="${elId}"]`);
        if (label && settingHelp[key]) {
            label.title = settingHelp[key];
        }
    }
}

function markDirty() {
    for (const [key, elId] of Object.entries(fieldMap)) {
        const el = $(elId);
        if (!el) continue;
        const cur = parseFloat(el.value);
        const def = defaults[key];
        el.classList.toggle("cfg-dirty", cur !== def);
    }
}

const PLOTLY_LAYOUT = {
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(255,251,245,0.92)",
    margin: { l: 72, r: 28, t: 24, b: 56 },
    font: { family: 'Aptos, "Segoe UI Variable Text", "Segoe UI", sans-serif', color: "#1f2430", size: 12 },
    hovermode: "closest",
    legend: { orientation: "h", y: -0.22, x: 0, bgcolor: "rgba(0,0,0,0)" },
    xaxis: {
        title: "Generation",
        gridcolor: "rgba(31, 36, 48, 0.08)",
        linecolor: "rgba(31, 36, 48, 0.12)",
        zeroline: false,
        automargin: true,
    },
    yaxis: {
        title: "max |residual|",
        gridcolor: "rgba(31, 36, 48, 0.08)",
        linecolor: "rgba(31, 36, 48, 0.12)",
        zeroline: false,
        automargin: true,
    },
};
const PLOTLY_CONFIG = { displayModeBar: false, responsive: true, scrollZoom: true };

// ── API helpers ─────────────────────────────────────
async function api(endpoint, method = "GET") {
    const resp = await fetch(endpoint, { method });
    if (!resp.ok) throw new Error(`${resp.status}`);
    return resp.json();
}

// ── DOM helpers ─────────────────────────────────────
function $(id) { return document.getElementById(id); }
function setText(id, text) {
    const el = $(id);
    if (el) el.textContent = text;
}

function clearPlot(divId) {
    const plot = $(divId);
    if (plot) {
        Plotly.purge(plot);
    }
}

function clearDashboardForFreshRun() {
    chartIslandsInit = false;
    chartEnvelopeInit = false;
    clearPlot("chart-islands");
    clearPlot("chart-envelope");
    updateIslands([]);
    updateLeaderboard([]);
    setText("generation", "0");
    setText("best-fitness", "—");
    setText("total-candidates", "0");
    setText("pending-count", "0");
    setText("screened-count", "0");
    setText("minimax-count", "0");
    setText("refined-count", "0");
    setText("shadow-count", "0");
    setText("queue-count", "0");
}

// ── Poll loop ───────────────────────────────────────
async function pollOnce() {
    try {
        const [status, stats, activity] = await Promise.all([
            api("/api/status"),
            api("/api/stats"),
            api("/api/activity"),
        ]);

        // Hero metrics
        setText("status-label", status.running ? "Running" : (status.error ? `Error: ${status.error}` : "Stopped"));
        setText("generation", status.generation != null ? status.generation.toLocaleString() : "—");
        setText("best-fitness", stats.best != null ? stats.best.toFixed(10) : "—");
        setText("total-candidates", stats.total != null ? stats.total.toLocaleString() : "—");

        // Buttons + lockout
        $("btn-start").disabled = status.running;
        $("btn-stop").disabled = !status.running;
        setLocked(status.running);

        // Pipeline counts
        setText("pending-count", stats.pending != null ? stats.pending.toLocaleString() : "—");
        setText("screened-count", stats.screened != null ? stats.screened.toLocaleString() : "—");
        setText("minimax-count", stats.minimax != null ? stats.minimax.toLocaleString() : "—");
        setText("refined-count", stats.refined != null ? stats.refined.toLocaleString() : "—");
        setText("shadow-count", stats.shadow != null ? stats.shadow.toLocaleString() : "—");
        setText("queue-count", stats.gpu_out_queue != null ? stats.gpu_out_queue.toLocaleString() : "—");

        // Islands
        updateIslands(stats.islands || []);

        // Charts
        const hist = stats.history || [];
        updateIslandChart(hist);
        updateEnvelopeChart(hist);

        // Leaderboard
        updateLeaderboard(stats.top || []);

        // Pipeline activity boxes
        updateActivity(activity || {});

    } catch (err) {
        setText("status-label", "Disconnected");
    }
}

const ACTIVITY_STAGES = ["generator", "gpu_screen", "gpu_minimax", "cpu_refine", "db_io", "monitor"];

function updateActivity(activity) {
    for (const stage of ACTIVITY_STAGES) {
        const box = $(`act-${stage}`);
        if (!box) continue;
        const data = activity[stage];
        if (!data) {
            box.classList.remove("active", "error");
            continue;
        }
        const isActive = data.active;
        const isError = data.state === "error";
        box.classList.toggle("active", isActive);
        box.classList.toggle("error", isError);

        const detailEl = $(`act-${stage}-detail`);
        if (detailEl) {
            detailEl.textContent = data.detail || data.state || "idle";
        }
        const rateEl = $(`act-${stage}-rate`);
        if (rateEl) {
            if (data.rate > 0 && data.cycles > 0) {
                rateEl.textContent = `${data.rate.toLocaleString()} items/s · ${data.cycles} cycles`;
            } else if (data.cycles > 0) {
                rateEl.textContent = `${data.cycles} cycles`;
            } else {
                rateEl.textContent = "";
            }
        }
    }
}

function updateIslands(islands) {
    const tbody = $("island-tbody");
    if (!tbody) return;
    tbody.innerHTML = "";
    for (const s of islands) {
        if (s.count === 0) continue;
        const tr = document.createElement("tr");
        tr.innerHTML = `
            <td>${s.island}</td>
            <td>${s.role || "—"}</td>
            <td>2^${s.size_exp ?? "—"}</td>
            <td>${s.cap != null ? s.cap.toLocaleString() : "—"}</td>
            <td>${s.count}</td>
            <td>${s.best != null ? s.best.toFixed(6) : "—"}</td>
            <td>${s.mean != null ? s.mean.toFixed(6) : "—"}</td>
        `;
        tbody.appendChild(tr);
    }
}

let chartIslandsInit = false;
let chartEnvelopeInit = false;

const ISLAND_COLORS = [
    "#bf5c36", "#2b7a78", "#5f4bb6", "#d4943a", "#3a86ff", "#e05780",
    "#43aa8b", "#9b5de5", "#f15bb5", "#00bbf9", "#6a994e", "#ff6d00",
];

const ENVELOPE_LAYOUT = {
    ...PLOTLY_LAYOUT,
    yaxis: { ...PLOTLY_LAYOUT.yaxis, title: "max |residual| (best across islands)" },
};

function updateIslandChart(history) {
    if (!history.length) {
        chartIslandsInit = false;
        clearPlot("chart-islands");
        return;
    }

    // Collect all island IDs that appear
    const islandSet = new Set();
    for (const h of history) {
        if (h.island_bests) {
            for (const k of Object.keys(h.island_bests)) islandSet.add(k);
        }
    }
    const islands = [...islandSet].sort((a, b) => +a - +b);
    if (!islands.length) {
        chartIslandsInit = false;
        clearPlot("chart-islands");
        return;
    }

    const traces = islands.map((isl, idx) => {
        // Carry forward last known value to fill gaps
        const xs = [], ys = [];
        let last = null;
        for (const h of history) {
            const v = h.island_bests && h.island_bests[isl] != null ? h.island_bests[isl] : null;
            if (v != null) last = v;
            if (last != null) {
                xs.push(h.generation);
                ys.push(last);
            }
        }
        return {
            x: xs,
            y: ys,
            mode: "lines",
            name: `Island ${isl}`,
            line: { color: ISLAND_COLORS[idx % ISLAND_COLORS.length], width: 1.5 },
        };
    });

    if (!chartIslandsInit) {
        Plotly.newPlot("chart-islands", traces, PLOTLY_LAYOUT, PLOTLY_CONFIG);
        chartIslandsInit = true;
    } else {
        Plotly.react("chart-islands", traces, PLOTLY_LAYOUT, PLOTLY_CONFIG);
    }
}

function updateEnvelopeChart(history) {
    if (!history.length) {
        chartEnvelopeInit = false;
        clearPlot("chart-envelope");
        return;
    }

    const gens = [];
    const lower = [];  // best (min) across all islands
    const upper = [];  // worst-of-best (max) across all islands

    for (const h of history) {
        if (!h.island_bests) continue;
        const vals = Object.values(h.island_bests).filter(v => v != null);
        if (!vals.length) continue;
        gens.push(h.generation);
        lower.push(Math.min(...vals));
        upper.push(Math.max(...vals));
    }
    if (!gens.length) {
        chartEnvelopeInit = false;
        clearPlot("chart-envelope");
        return;
    }

    const traces = [
        {
            x: gens,
            y: upper,
            mode: "lines",
            name: "Worst island best",
            line: { color: "rgba(191,92,54,0.3)", width: 0 },
            showlegend: false,
        },
        {
            x: gens,
            y: lower,
            mode: "lines",
            name: "Best island best",
            line: { color: "#bf5c36", width: 2 },
            fill: "tonexty",
            fillcolor: "rgba(191,92,54,0.12)",
        },
        {
            x: gens,
            y: upper,
            mode: "lines",
            name: "Worst island best",
            line: { color: "#9b5de5", width: 1.5, dash: "dot" },
        },
        {
            x: history.map(h => h.generation),
            y: history.map(h => h.best_fitness),
            mode: "lines",
            name: "Global best",
            line: { color: "#2b7a78", width: 2 },
        },
    ];

    if (!chartEnvelopeInit) {
        Plotly.newPlot("chart-envelope", traces, ENVELOPE_LAYOUT, PLOTLY_CONFIG);
        chartEnvelopeInit = true;
    } else {
        Plotly.react("chart-envelope", traces, ENVELOPE_LAYOUT, PLOTLY_CONFIG);
    }
}

function updateLeaderboard(top) {
    const tbody = $("top-tbody");
    if (!tbody) return;
    tbody.innerHTML = "";
    for (const c of top) {
        const tr = document.createElement("tr");
        tr.innerHTML = `
            <td>${c.id}</td>
            <td>${c.fitness != null ? c.fitness.toFixed(10) : "—"}</td>
            <td>${c.fro != null ? c.fro.toFixed(6) : "—"}</td>
            <td>${c.origin || "—"}</td>
            <td>${c.island}</td>
            <td>${c.gen}</td>
            <td>T${c.tier}</td>
        `;
        tbody.appendChild(tr);
    }
}

// ── Button handlers ─────────────────────────────────
$("btn-start").addEventListener("click", async () => {
    try {
        const cfg = saveSettings();
        const modeEl = $("cfg-seed-mode");
        const pathEl = $("cfg-seed-path");
        const mode = modeEl ? modeEl.value : "seed";
        cfg.mode = mode;
        if (pathEl && pathEl.value) cfg.seed_path = pathEl.value;

        if (mode === "fresh" || mode === "seed") {
            const label = mode === "seed" ? "Seed from JSON" : "Fresh random";
            const confirmed = window.confirm(
                `${label} will permanently clear the current optimizer database before starting. Continue?`
            );
            if (!confirmed) {
                return;
            }
            cfg.backup_db = window.confirm(
                "Create a backup copy of the current database before clearing it?"
            );
        }

        const resp = await fetch("/api/start", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(cfg),
        });

        const data = await resp.json();
        if (!resp.ok) {
            throw new Error(data.error || `${resp.status}`);
        }

        if (mode === "fresh" || mode === "seed") {
            clearDashboardForFreshRun();
            if (data.backup_path) {
                alert(`Database backed up to:\n${data.backup_path}`);
            }
        }

        await pollOnce();
    } catch (err) {
        alert("Failed to start: " + err.message);
    }
});

$("btn-stop").addEventListener("click", async () => {
    try {
        await api("/api/stop", "POST");
    } catch (err) {
        alert("Failed to stop: " + err.message);
    }
});

// ── Init ────────────────────────────────────────────
loadSettings();
initTooltips();
markDirty();

// Track changes to highlight dirty fields
for (const elId of Object.values(fieldMap)) {
    const el = $(elId);
    if (el) el.addEventListener("input", markDirty);
}

pollOnce();
polling = setInterval(pollOnce, POLL_MS);
