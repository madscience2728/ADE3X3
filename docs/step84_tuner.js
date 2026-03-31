const STORAGE_KEY = "ade3x3-step84-tuner";
const STORAGE_SCHEMA_VERSION = 2;
const POLL_INTERVAL_MS = 2000;
const PLOTLY_LAYOUT_BASE = {
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
        automargin: true
    },
    yaxis: {
        gridcolor: "rgba(31, 36, 48, 0.08)",
        linecolor: "rgba(31, 36, 48, 0.12)",
        zeroline: false,
        automargin: true
    }
};
const PLOTLY_CONFIG = {
    displayModeBar: false,
    responsive: true,
    scrollZoom: true
};
const COPY_COLORS = [
    "#bf5c36",
    "#1e5e63",
    "#8b3418",
    "#597a33",
    "#92598f",
    "#2b6cb0",
    "#c05621",
    "#00695c"
];
let pollHandle = null;
let fitnessScaleMode = "log";
let shadowScaleMode = "linear";
let activeSettingsEnvelope = null;
let serverState = null;
let freshDraftUnlocked = false;
let lockedBatchState = null;
const TAB_ID = `tab-${Math.random().toString(36).slice(2, 10)}`;

const defaults = {
    topologyMode: "explicit",
    islandPopulations: "32,24,16,12,8,8,6,6",
    islandMinExp: 2,
    islandMaxExp: 8,
    islandCopies: 2,
    workers: 24,
    batchCopies: 1,
    timeoutSeconds: 120,
    generations: 1000000,
    seed: 8401001,
    offspringMultiplier: 1,
    warmSeeds: 12,
    migrationInterval: 25,
    migrationSize: 1,
    checkpointInterval: 100,
    tournamentSize: 3,
    executorKind: "thread",
    resumeCheckpoint: "",
    signature333Fraction: 0.15,
    shadowSurvivorFraction: 0.30,
    shadowParentRate: 0.40,
    shadowVariableBucket: 16,
    shadowMetricsInterval: 20,
    hitThreshold: 1e-8,
    crossoverRate: 0.70,
    factorCrossoverRate: 0.20,
    mutationSupportRate: 0.30,
    mutationCoeffRate: 0.50,
    mutationReplaceRate: 0.05,
    alsSweeps: 1,
    newtonThreshold: 0.25,
    polishMaxNfev: 20,
    polishVariableCap: 260,
    activeValueFloor: 1e-4,
    supportMin: 2,
    supportMax: 7,
    initSupportMin: 2,
    initSupportMax: 6,
    supportAddProb: 0.50,
    supportDropProb: 0.50,
    supportAddLow: 0.01,
    supportAddHigh: 0.10,
    coeffSigmaMin: 0.01,
    coeffSigmaMax: 0.35,
    coeffInitLow: 1.0,
    coeffInitHigh: 2.0,
    bestExportIncludeDense: false
};

const presetMap = {
    balanced: {
        topologyMode: "explicit",
        islandPopulations: "32,24,16,12,8,8,6,6",
        offspringMultiplier: 1,
        workers: 24,
        batchCopies: 1,
        timeoutSeconds: 120,
        executorKind: "thread"
    },
    moderate: {
        topologyMode: "power",
        islandMinExp: 2,
        islandMaxExp: 8,
        islandCopies: 2,
        offspringMultiplier: 1,
        workers: 24,
        batchCopies: 1,
        timeoutSeconds: 120,
        executorKind: "thread"
    },
    wide: {
        topologyMode: "power",
        islandMinExp: 2,
        islandMaxExp: 9,
        islandCopies: 2,
        offspringMultiplier: 1,
        workers: 24,
        batchCopies: 1,
        timeoutSeconds: 120,
        executorKind: "thread"
    }
};

const fieldMap = {
    islandPopulations: "island-populations",
    islandMinExp: "island-min-exp",
    islandMaxExp: "island-max-exp",
    islandCopies: "island-copies",
    workers: "workers",
    batchCopies: "batch-copies",
    timeoutSeconds: "timeout-seconds",
    generations: "generations",
    seed: "seed",
    offspringMultiplier: "offspring-multiplier",
    warmSeeds: "warm-seeds",
    migrationInterval: "migration-interval",
    migrationSize: "migration-size",
    checkpointInterval: "checkpoint-interval",
    tournamentSize: "tournament-size",
    executorKind: "executor-kind",
    resumeCheckpoint: "resume-checkpoint",
    signature333Fraction: "signature333-fraction",
    shadowSurvivorFraction: "shadow-survivor-fraction",
    shadowParentRate: "shadow-parent-rate",
    shadowVariableBucket: "shadow-variable-bucket",
    shadowMetricsInterval: "shadow-metrics-interval",
    hitThreshold: "hit-threshold",
    crossoverRate: "crossover-rate",
    factorCrossoverRate: "factor-crossover-rate",
    mutationSupportRate: "mutation-support-rate",
    mutationCoeffRate: "mutation-coeff-rate",
    mutationReplaceRate: "mutation-replace-rate",
    alsSweeps: "als-sweeps",
    newtonThreshold: "newton-threshold",
    polishMaxNfev: "polish-max-nfev",
    polishVariableCap: "polish-variable-cap",
    activeValueFloor: "active-value-floor",
    supportMin: "support-min",
    supportMax: "support-max",
    initSupportMin: "init-support-min",
    initSupportMax: "init-support-max",
    supportAddProb: "support-add-prob",
    supportDropProb: "support-drop-prob",
    supportAddLow: "support-add-low",
    supportAddHigh: "support-add-high",
    coeffSigmaMin: "coeff-sigma-min",
    coeffSigmaMax: "coeff-sigma-max",
    coeffInitLow: "coeff-init-low",
    coeffInitHigh: "coeff-init-high",
    bestExportIncludeDense: "best-export-include-dense"
};

const settingHelp = {
    topologyMode: "Choose how island sizes are specified. Explicit mode is better when you already know the exact schedule you want. Power mode is better for broad scaling experiments. Reasonable usage: explicit for careful hand tuning, power for exploratory sweeps like 2^2..2^8 or 2^2..2^9.",
    islandPopulations: "Island population sizes used directly when topology mode is explicit. Commas, semicolons, spaces, and newlines are all accepted. This controls both diversity and RAM pressure. Reasonable values are usually between 4 and 64 per island. Small islands 4..12 keep niche churn high; medium 16..32 are balanced; large 48..96 are expensive and should be used sparingly.",
    islandMinExp: "Smallest exponent included when generating a power-of-two island schedule. Generated sizes are 2^max down to 2^min. Reasonable range: 2..5, meaning island sizes from 4 to 32 at the low end. Keep this less than or equal to the max exponent.",
    islandMaxExp: "Largest exponent included when generating a power-of-two island schedule. Higher values create large islands quickly and can dominate RAM and evaluation time. Reasonable range: 6..9, meaning largest islands from 64 to 512. On this repo, 8 or 9 is already heavy.",
    islandCopies: "Number of copies to create for each power-of-two island size. More copies increase niche replication and total population linearly. Reasonable range: 1..4 for manual tuning, 6..8 only if you intentionally want a huge ensemble and can afford the memory cost.",
    workers: "Maximum concurrent evaluation workers used by Step 84. On Windows with thread mode, reasonable values are usually 8..32. Match this to actual useful parallelism, not just core count. Too high can increase contention without improving evaluations per second.",
    batchCopies: "Number of independent Step 84 runs launched in parallel by the tuner backend. Each copy gets its own seed and export directory. Reasonable range: 1..8 for normal use. Fifty is a brute-force last resort and usually only makes sense if you also lower workers and keep dense exports off.",
    timeoutSeconds: "Wall-clock time budget before the run exits with time_limit. For tuning, 60..180 seconds is a good range. For real search, use 300+ seconds once you know the schedule is stable.",
    generations: "Upper bound on generations if the time limit does not stop the run first. In practice the timeout usually fires first. Reasonable values: 10000 for short experiments, 1000000 as a safe effectively-unbounded cap.",
    seed: "Random seed for initialization and stochastic operators. Change this when you want a new stochastic trajectory under the same hyperparameters. Any integer is fine; keeping a log of seeds is more important than the magnitude.",
    offspringMultiplier: "Children produced per island relative to island population size. Larger values increase exploration and evaluation load per generation. Reasonable range: 1..3. Use 1 for steady-state efficiency, 2 for stronger search pressure, 3 only if CPU is underused and RAM is stable.",
    warmSeeds: "Number of warm-start individuals loaded from Step 83b style seeds. More warm seeds bias the run harder toward prior structure. Reasonable range: 4..16. Too low wastes prior information; too high can reduce diversity.",
    migrationInterval: "How often cross-island migration runs, measured in generations. Lower means more frequent mixing. Reasonable range: 10..50. Use smaller values when islands diverge too hard, larger values when you want islands to stay independent longer.",
    migrationSize: "How many migrants are exchanged during each migration event. Reasonable range: 0..3. Zero disables migration pressure; one is conservative; two or three are enough for most experiments.",
    checkpointInterval: "How many generations between checkpoint writes. Lower values are safer but create more disk traffic. Reasonable range: 50..250. If artifacts are getting large, move this upward.",
    tournamentSize: "Tournament size used for parent and survivor selection. Higher values increase selection pressure. Reasonable range: 2..5. Two is gentle, three is balanced, four or five can collapse diversity faster.",
    executorKind: "Execution backend. Thread is usually the right choice on Windows because it avoids heavy process spawn and duplicated memory. Process mode is only worth trying if thread mode clearly fails to saturate useful work.",
    resumeCheckpoint: "Optional explicit checkpoint path. If left blank, the tuner backend will automatically continue from each copy's latest checkpoint when you restart after a finished batch in the same session. Use the Start fresh run button to ignore both auto-continue and any explicit resume path.",
    signature333Fraction: "Reserved survivor fraction kept for exact (3,3,3) signatures. This is a guardrail for the Step 83b prior. Reasonable range: 0.10..0.25. Below 0.05 the reservation may become too weak; above 0.30 it can crowd out broader exploration.",
    shadowSurvivorFraction: "Fraction of survivor slots allocated through selection shadow niches. Higher values preserve more niche families. Reasonable range: 0.20..0.45. Too low recreates collapse into easy basins; too high can slow improvement.",
    shadowParentRate: "Probability of selecting parents from the shadow pool instead of the main elite pool. Reasonable range: 0.25..0.60. Lower values emphasize exploitation; higher values increase structural diversity.",
    shadowVariableBucket: "Bucket size used when grouping variable counts inside the shadow key. Smaller buckets preserve finer distinctions. Reasonable range: 8..32. Sixteen is a balanced default.",
    shadowMetricsInterval: "How often shadow occupancy metrics are recomputed and logged. Lower values give more visibility but add overhead. Reasonable range: 10..50. Use larger values if you only care about throughput.",
    hitThreshold: "Residual threshold treated as an exact hit. This should stay very small. Reasonable range: 1e-10 to 1e-6 depending on how strict you want the success criterion. The default 1e-8 is already strict.",
    crossoverRate: "Probability of term crossover during offspring generation. Reasonable range: 0.50..0.85. Lower values reduce recombination; higher values make search more dependent on inherited structure.",
    factorCrossoverRate: "Probability of factor-level crossover instead of term-level crossover. Reasonable range: 0.10..0.35. Keep this lower than the main crossover rate so term-level exchange stays dominant.",
    mutationSupportRate: "Probability of mutating factor supports. This is a major structural exploration knob. Reasonable range: 0.20..0.45. Higher values broaden search but can erase useful patterns quickly.",
    mutationCoeffRate: "Probability of perturbing coefficient values. Reasonable range: 0.30..0.70. This can usually be higher than support mutation because coefficient perturbations are cheaper and less disruptive structurally.",
    mutationReplaceRate: "Probability of replacing a whole term with a fresh random term. Reasonable range: 0.01..0.10. Keep this low; it is a high-disruption operator.",
    alsSweeps: "Number of alternating least-squares sweeps run during local improvement. More sweeps improve local polish but increase evaluation cost. Reasonable range: 0..3. One is the usual tuning default; two or three only if local refinement is clearly paying off.",
    newtonThreshold: "Residual threshold below which support-fixed nonlinear polish is attempted. Larger values trigger polish more often. Reasonable range: 0.05..0.30. If polish is expensive or unstable, lower this. If promising individuals are being missed, raise it modestly.",
    polishMaxNfev: "Maximum function evaluations allowed for nonlinear polish. Reasonable range: 10..50. Lower values cap runaway polish cost; higher values are only useful if polish is consistently productive.",
    polishVariableCap: "Skip nonlinear polish when the reduced variable count exceeds this cap. This is a runtime safety knob. Reasonable range: 180..300. Lower caps are safer; higher caps allow more ambitious but heavier polish attempts.",
    activeValueFloor: "Absolute value floor used to decide whether a coefficient is considered active. Reasonable range: 1e-5 to 1e-3. Lower values keep more nearly-zero terms alive; higher values prune aggressively.",
    supportMin: "Minimum nonzeros allowed per factor support after mutation. This should stay less than or equal to support max. Reasonable range: 2..3. Keeping this low preserves sparse structure.",
    supportMax: "Maximum nonzeros allowed per factor support after mutation. This should stay greater than or equal to support min. Reasonable range: 5..7. Higher values can reach denser regimes but drift away from the sparse Step 83b prior.",
    initSupportMin: "Minimum nonzeros used when creating random initial supports. This should stay less than or equal to init support max. Reasonable range: 2..3.",
    initSupportMax: "Maximum nonzeros used when creating random initial supports. This should stay greater than or equal to init support min. Reasonable range: 4..6. Usually keep this a bit tighter than the post-mutation support max so initialization starts in a cleaner sparse basin.",
    supportAddProb: "Probability that a support mutation attempts an add move. Together with drop probability this shapes sparsity drift. Reasonable range: 0.30..0.70. If supports are shrinking too much, raise this.",
    supportDropProb: "Probability that a support mutation attempts a drop move. Reasonable range: 0.30..0.70. If supports are getting too dense, raise this. The balance with add probability matters more than either value alone.",
    supportAddLow: "Minimum coefficient magnitude used when adding a new active support entry. This should stay less than or equal to support add high. Reasonable range: 0.001..0.05. Lower values seed gentler additions.",
    supportAddHigh: "Maximum coefficient magnitude used when adding a new active support entry. This should stay greater than or equal to support add low. Reasonable range: 0.05..0.20. If new entries are too disruptive, lower this upper bound.",
    coeffSigmaMin: "Lower bound for Gaussian coefficient perturbation strength. This should stay less than or equal to coeff sigma max. Reasonable range: 0.005..0.05. Small values support fine local adjustment.",
    coeffSigmaMax: "Upper bound for Gaussian coefficient perturbation strength. This should stay greater than or equal to coeff sigma min. Reasonable range: 0.15..0.50. Larger values increase jumpiness and can help escape flat basins.",
    coeffInitLow: "Lower bound for random initial coefficient magnitudes. This should stay less than or equal to coeff init high. Reasonable range: 0.5..1.5.",
    coeffInitHigh: "Upper bound for random initial coefficient magnitudes. This should stay greater than or equal to coeff init low. Reasonable range: 1.5..3.0. Wider spreads increase diversity but can make ALS stabilization harder.",
    bestExportIncludeDense: "Include dense matrices and arrays in the best-individual export. This is mainly for forensic debugging. Leave this off for normal runs because artifact size can balloon quickly. Recommended setting: 0 except for short diagnostic runs."
};

const numericFields = new Set([
    "islandMinExp", "islandMaxExp", "islandCopies", "workers", "batchCopies", "timeoutSeconds", "generations", "seed",
    "offspringMultiplier", "warmSeeds", "migrationInterval", "migrationSize", "checkpointInterval",
    "tournamentSize", "signature333Fraction", "shadowSurvivorFraction", "shadowParentRate",
    "shadowVariableBucket", "shadowMetricsInterval", "hitThreshold", "crossoverRate", "factorCrossoverRate",
    "mutationSupportRate", "mutationCoeffRate", "mutationReplaceRate", "alsSweeps", "newtonThreshold",
    "polishMaxNfev", "polishVariableCap", "activeValueFloor", "supportMin", "supportMax", "initSupportMin",
    "initSupportMax", "supportAddProb", "supportDropProb", "supportAddLow", "supportAddHigh", "coeffSigmaMin",
    "coeffSigmaMax", "coeffInitLow", "coeffInitHigh"
]);

const checkboxFields = new Set(["bestExportIncludeDense"]);
const RUN_LOCKED_SETTING_KEYS = new Set(["topologyMode", ...Object.keys(fieldMap)]);
const ENV_TO_STATE_KEY = {
    STEP84_WORKERS: "workers",
    STEP84_TIMEOUT_SECONDS: "timeoutSeconds",
    STEP84_GENERATIONS: "generations",
    STEP84_SEED: "seed",
    STEP84_OFFSPRING_MULTIPLIER: "offspringMultiplier",
    STEP84_WARM_SEEDS: "warmSeeds",
    STEP84_MIGRATION_INTERVAL: "migrationInterval",
    STEP84_MIGRATION_SIZE: "migrationSize",
    STEP84_CHECKPOINT_INTERVAL: "checkpointInterval",
    STEP84_TOURNAMENT_SIZE: "tournamentSize",
    STEP84_EXECUTOR_KIND: "executorKind",
    STEP84_RESUME_CHECKPOINT: "resumeCheckpoint",
    STEP84_SIGNATURE333_SURVIVOR_FRACTION: "signature333Fraction",
    STEP84_SHADOW_SURVIVOR_FRACTION: "shadowSurvivorFraction",
    STEP84_SHADOW_PARENT_RATE: "shadowParentRate",
    STEP84_SHADOW_VARIABLE_BUCKET: "shadowVariableBucket",
    STEP84_SHADOW_METRICS_INTERVAL: "shadowMetricsInterval",
    STEP84_HIT_THRESHOLD: "hitThreshold",
    STEP84_CROSSOVER_RATE: "crossoverRate",
    STEP84_FACTOR_CROSSOVER_RATE: "factorCrossoverRate",
    STEP84_MUTATION_SUPPORT_RATE: "mutationSupportRate",
    STEP84_MUTATION_COEFF_RATE: "mutationCoeffRate",
    STEP84_MUTATION_REPLACE_RATE: "mutationReplaceRate",
    STEP84_ALS_SWEEPS: "alsSweeps",
    STEP84_NEWTON_THRESHOLD: "newtonThreshold",
    STEP84_POLISH_MAX_NFEV: "polishMaxNfev",
    STEP84_POLISH_VARIABLE_CAP: "polishVariableCap",
    STEP84_ACTIVE_VALUE_FLOOR: "activeValueFloor",
    STEP84_SUPPORT_MIN: "supportMin",
    STEP84_SUPPORT_MAX: "supportMax",
    STEP84_INIT_SUPPORT_MIN: "initSupportMin",
    STEP84_INIT_SUPPORT_MAX: "initSupportMax",
    STEP84_SUPPORT_ADD_PROB: "supportAddProb",
    STEP84_SUPPORT_DROP_PROB: "supportDropProb",
    STEP84_SUPPORT_ADD_LOW: "supportAddLow",
    STEP84_SUPPORT_ADD_HIGH: "supportAddHigh",
    STEP84_COEFF_SIGMA_MIN: "coeffSigmaMin",
    STEP84_COEFF_SIGMA_MAX: "coeffSigmaMax",
    STEP84_COEFF_INIT_LOW: "coeffInitLow",
    STEP84_COEFF_INIT_HIGH: "coeffInitHigh",
    STEP84_BEST_EXPORT_INCLUDE_DENSE: "bestExportIncludeDense"
};

function cloneDefaults() {
    return JSON.parse(JSON.stringify(defaults));
}

function sanitizeState(candidate = {}) {
    const state = cloneDefaults();
    if (candidate.topologyMode === "explicit" || candidate.topologyMode === "power") {
        state.topologyMode = candidate.topologyMode;
    }

    Object.keys(fieldMap).forEach((key) => {
        if (!(key in candidate)) {
            return;
        }
        if (checkboxFields.has(key)) {
            state[key] = Boolean(candidate[key]);
            return;
        }
        if (numericFields.has(key)) {
            const value = Number(candidate[key]);
            if (Number.isFinite(value)) {
                state[key] = value;
            }
            return;
        }
        if (candidate[key] !== null && candidate[key] !== undefined) {
            state[key] = String(candidate[key]);
        }
    });

    return state;
}

function statesEqual(left, right) {
    const leftState = sanitizeState(left);
    const rightState = sanitizeState(right);
    if (leftState.topologyMode !== rightState.topologyMode) {
        return false;
    }

    return Object.keys(fieldMap).every((key) => normalizeValue(key, leftState[key]) === normalizeValue(key, rightState[key]));
}

function normalizeEnvelope(candidate) {
    if (!candidate || typeof candidate !== "object" || Array.isArray(candidate)) {
        return null;
    }

    const hasNestedSettings = candidate.settings && typeof candidate.settings === "object" && !Array.isArray(candidate.settings);
    const savedAtMsRaw = Number(candidate.savedAtMs);
    const parsedSavedAtMs = Number.isFinite(savedAtMsRaw)
        ? savedAtMsRaw
        : Date.parse(typeof candidate.savedAt === "string" ? candidate.savedAt : "");
    const savedAtMs = Number.isFinite(parsedSavedAtMs) ? parsedSavedAtMs : 0;

    return {
        schemaVersion: hasNestedSettings ? Number(candidate.schemaVersion) || 1 : 1,
        savedAtMs,
        savedAt: savedAtMs ? new Date(savedAtMs).toISOString() : "",
        sourceTabId: hasNestedSettings && candidate.sourceTabId ? String(candidate.sourceTabId) : "",
        settings: sanitizeState(hasNestedSettings ? candidate.settings : candidate)
    };
}

function createSettingsEnvelope(state, overrides = {}) {
    const savedAtMs = overrides.savedAtMs ?? Date.now();
    return {
        schemaVersion: STORAGE_SCHEMA_VERSION,
        savedAtMs,
        savedAt: new Date(savedAtMs).toISOString(),
        sourceTabId: overrides.sourceTabId || TAB_ID,
        settings: sanitizeState(state)
    };
}

function readStoredEnvelope() {
    try {
        const raw = window.localStorage.getItem(STORAGE_KEY);
        if (!raw) {
            return null;
        }
        return normalizeEnvelope(JSON.parse(raw));
    } catch (_error) {
        return null;
    }
}

function formatSavedTime(savedAtMs) {
    if (!Number.isFinite(savedAtMs) || savedAtMs <= 0) {
        return "unknown time";
    }
    return new Date(savedAtMs).toLocaleTimeString();
}

function setSavedStatus(message) {
    const saved = document.getElementById("saved-status");
    if (saved) {
        saved.textContent = message;
    }
}

function setLockStatus(message) {
    const lock = document.getElementById("lock-status");
    if (lock) {
        lock.textContent = message;
    }
}

function applyEnvelope(envelope, options = {}) {
    if (!envelope) {
        return;
    }
    activeSettingsEnvelope = envelope;
    applyState(envelope.settings, { persist: false });
    if (options.message) {
        setSavedStatus(options.message);
    }
}

function getHelpText(key) {
    const description = settingHelp[key] || "";
    if (!(key in defaults)) {
        return description;
    }
    return `${description} Default: ${formatValue(defaults[key])}.`;
}

function getSavedState() {
    const envelope = readStoredEnvelope();
    activeSettingsEnvelope = envelope;
    return envelope ? envelope.settings : cloneDefaults();
}

function buildStateFromEnv(env = {}) {
    const state = cloneDefaults();
    if (typeof env.STEP84_ISLAND_POPULATIONS === "string" && env.STEP84_ISLAND_POPULATIONS.trim()) {
        state.topologyMode = "explicit";
        state.islandPopulations = env.STEP84_ISLAND_POPULATIONS;
    } else if (
        env.STEP84_ISLAND_MIN_EXP !== undefined ||
        env.STEP84_ISLAND_MAX_EXP !== undefined ||
        env.STEP84_ISLAND_COPIES !== undefined
    ) {
        state.topologyMode = "power";
        if (env.STEP84_ISLAND_MIN_EXP !== undefined) {
            state.islandMinExp = Number(env.STEP84_ISLAND_MIN_EXP);
        }
        if (env.STEP84_ISLAND_MAX_EXP !== undefined) {
            state.islandMaxExp = Number(env.STEP84_ISLAND_MAX_EXP);
        }
        if (env.STEP84_ISLAND_COPIES !== undefined) {
            state.islandCopies = Number(env.STEP84_ISLAND_COPIES);
        }
    }

    Object.entries(ENV_TO_STATE_KEY).forEach(([envKey, stateKey]) => {
        if (!(envKey in env)) {
            return;
        }
        if (checkboxFields.has(stateKey)) {
            const raw = String(env[envKey]).trim().toLowerCase();
            state[stateKey] = ["1", "true", "yes", "on"].includes(raw);
            return;
        }
        if (numericFields.has(stateKey)) {
            const value = Number(env[envKey]);
            if (Number.isFinite(value)) {
                state[stateKey] = value;
            }
            return;
        }
        state[stateKey] = String(env[envKey]);
    });

    return sanitizeState(state);
}

function parsePopulationList(text) {
    return text
    .split(/[;,\s]+/)
        .map((item) => item.trim())
        .filter(Boolean)
        .map((item) => Number.parseInt(item, 10))
        .filter((value) => Number.isFinite(value) && value > 0);
}

function buildPowerSchedule(minExp, maxExp, copies) {
    const values = [];
    for (let exponent = maxExp; exponent >= minExp; exponent -= 1) {
        for (let count = 0; count < copies; count += 1) {
            values.push(2 ** exponent);
        }
    }
    return values;
}

function getState() {
    const state = cloneDefaults();
    state.topologyMode = document.querySelector(".segmented__item.is-active")?.dataset.mode || "explicit";

    Object.entries(fieldMap).forEach(([key, id]) => {
        const element = document.getElementById(id);
        if (!element) {
            return;
        }
        if (checkboxFields.has(key)) {
            state[key] = element.checked;
            return;
        }
        if (numericFields.has(key)) {
            state[key] = Number(element.value);
            return;
        }
        state[key] = element.value;
    });

    return state;
}

function applyState(state, options = {}) {
    Object.entries(fieldMap).forEach(([key, id]) => {
        const element = document.getElementById(id);
        if (!element || !(key in state)) {
            return;
        }
        if (checkboxFields.has(key)) {
            element.checked = Boolean(state[key]);
            return;
        }
        element.value = state[key];
    });

    setTopologyMode(state.topologyMode || "explicit");
    render({
        persist: options.persist ?? true,
        forcePersist: options.forcePersist ?? false,
        saveMessage: options.saveMessage
    });
}

function setElementValue(key, value) {
    if (key === "topologyMode") {
        setTopologyMode(value);
        return;
    }
    const element = document.getElementById(fieldMap[key]);
    if (!element) {
        return;
    }
    if (checkboxFields.has(key)) {
        element.checked = Boolean(value);
        return;
    }
    element.value = value;
}

function setTopologyMode(mode) {
    document.querySelectorAll(".segmented__item").forEach((button) => {
        button.classList.toggle("is-active", button.dataset.mode === mode);
    });
    document.getElementById("explicit-fields").classList.toggle("is-hidden", mode !== "explicit");
    document.getElementById("power-fields").classList.toggle("is-hidden", mode !== "power");
}

function getDerived(state) {
    const populations = state.topologyMode === "power"
        ? buildPowerSchedule(state.islandMinExp, state.islandMaxExp, state.islandCopies)
        : parsePopulationList(state.islandPopulations);
    const totalPopulation = populations.reduce((sum, value) => sum + value, 0);
    const offspring = populations.map((value) => value * state.offspringMultiplier);
    const totalOffspring = offspring.reduce((sum, value) => sum + value, 0);
    return {
        populations,
        totalPopulation,
        totalOffspring,
        islandCount: populations.length
    };
}

function formatValue(value) {
    if (typeof value === "boolean") {
        return value ? "1" : "0";
    }
    if (typeof value === "number") {
        return Number.isInteger(value) ? String(value) : String(value);
    }
    return String(value || "");
}

function normalizeValue(key, value) {
    if (checkboxFields.has(key)) {
        return Boolean(value);
    }
    if (numericFields.has(key)) {
        return Number(value);
    }
    return String(value ?? "");
}

function hasContinuationContext(payload) {
    return Boolean(payload && Number(payload.copy_count) > 0);
}

function shouldLockRunSettings() {
    if (serverState?.running) {
        return true;
    }
    return hasContinuationContext(serverState) && !freshDraftUnlocked;
}

function isDefaultValue(key, state) {
    return normalizeValue(key, state[key]) === normalizeValue(key, defaults[key]);
}

function resetSetting(key) {
    const state = getState();
    state[key] = defaults[key];
    applyState(state, { persist: true });
}

function buildEnvEntries(state) {
    const entries = [
        ["STEP84_GENERATIONS", state.generations],
        ["STEP84_WORKERS", state.workers],
        ["STEP84_TIMEOUT_SECONDS", state.timeoutSeconds],
        ["STEP84_CHECKPOINT_INTERVAL", state.checkpointInterval],
        ["STEP84_ALS_SWEEPS", state.alsSweeps],
        ["STEP84_SIGNATURE333_SURVIVOR_FRACTION", state.signature333Fraction],
        ["STEP84_SHADOW_SURVIVOR_FRACTION", state.shadowSurvivorFraction],
        ["STEP84_SHADOW_PARENT_RATE", state.shadowParentRate],
        ["STEP84_SHADOW_VARIABLE_BUCKET", state.shadowVariableBucket],
        ["STEP84_SHADOW_METRICS_INTERVAL", state.shadowMetricsInterval],
        ["STEP84_EXECUTOR_KIND", state.executorKind],
        ["STEP84_OFFSPRING_MULTIPLIER", state.offspringMultiplier],
        ["STEP84_WARM_SEEDS", state.warmSeeds],
        ["STEP84_MIGRATION_INTERVAL", state.migrationInterval],
        ["STEP84_MIGRATION_SIZE", state.migrationSize],
        ["STEP84_SEED", state.seed],
        ["STEP84_TOURNAMENT_SIZE", state.tournamentSize],
        ["STEP84_CROSSOVER_RATE", state.crossoverRate],
        ["STEP84_FACTOR_CROSSOVER_RATE", state.factorCrossoverRate],
        ["STEP84_MUTATION_SUPPORT_RATE", state.mutationSupportRate],
        ["STEP84_MUTATION_COEFF_RATE", state.mutationCoeffRate],
        ["STEP84_MUTATION_REPLACE_RATE", state.mutationReplaceRate],
        ["STEP84_SUPPORT_MIN", state.supportMin],
        ["STEP84_SUPPORT_MAX", state.supportMax],
        ["STEP84_INIT_SUPPORT_MIN", state.initSupportMin],
        ["STEP84_INIT_SUPPORT_MAX", state.initSupportMax],
        ["STEP84_SUPPORT_ADD_PROB", state.supportAddProb],
        ["STEP84_SUPPORT_DROP_PROB", state.supportDropProb],
        ["STEP84_NEWTON_THRESHOLD", state.newtonThreshold],
        ["STEP84_POLISH_MAX_NFEV", state.polishMaxNfev],
        ["STEP84_POLISH_VARIABLE_CAP", state.polishVariableCap],
        ["STEP84_HIT_THRESHOLD", state.hitThreshold],
        ["STEP84_ACTIVE_VALUE_FLOOR", state.activeValueFloor],
        ["STEP84_BEST_EXPORT_INCLUDE_DENSE", state.bestExportIncludeDense],
        ["STEP84_COEFF_SIGMA_MIN", state.coeffSigmaMin],
        ["STEP84_COEFF_SIGMA_MAX", state.coeffSigmaMax],
        ["STEP84_COEFF_INIT_LOW", state.coeffInitLow],
        ["STEP84_COEFF_INIT_HIGH", state.coeffInitHigh],
        ["STEP84_SUPPORT_ADD_LOW", state.supportAddLow],
        ["STEP84_SUPPORT_ADD_HIGH", state.supportAddHigh]
    ];

    if (state.topologyMode === "explicit") {
        entries.push(["STEP84_ISLAND_POPULATIONS", state.islandPopulations]);
    } else {
        entries.push(["STEP84_ISLAND_MIN_EXP", state.islandMinExp]);
        entries.push(["STEP84_ISLAND_MAX_EXP", state.islandMaxExp]);
        entries.push(["STEP84_ISLAND_COPIES", state.islandCopies]);
    }

    if (String(state.resumeCheckpoint || "").trim()) {
        entries.push(["STEP84_RESUME_CHECKPOINT", state.resumeCheckpoint.trim()]);
    }

    return entries;
}

function buildRunCommand(state) {
    const envLines = buildEnvEntries(state)
        .map(([key, value]) => `$env:${key}='${formatValue(value).replace(/'/g, "''")}'`);
    return [
        "Set-Location \"C:\\Users\\madsc\\Desktop\\Github\\ADE3X3\"",
        "$env:OMP_NUM_THREADS='1'",
        "$env:OPENBLAS_NUM_THREADS='1'",
        "$env:MKL_NUM_THREADS='1'",
        "$env:NUMEXPR_NUM_THREADS='1'",
        ...envLines,
        ".\\.venv\\Scripts\\python.exe .\\src\\ade3x3\\steps\\ade3x3_step84_metaheuristic_rank19_search.py"
    ].join("; \n");
}

function buildEnvObject(state) {
    return Object.fromEntries(buildEnvEntries(state).map(([key, value]) => [key, formatValue(value)]));
}

function buildProfileCommand(state) {
    const envLines = buildEnvEntries(state)
        .map(([key, value]) => `$env:${key}='${formatValue(value).replace(/'/g, "''")}'`)
        .join("; ");
    return [
        "$profileName = 'manual_profile'",
        "$outDir = '.\\outputs\\exports\\step84_resource_profiles\\' + $profileName",
        "New-Item -ItemType Directory -Force -Path $outDir | Out-Null",
        "$env:OMP_NUM_THREADS='1'; $env:OPENBLAS_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'; $env:NUMEXPR_NUM_THREADS='1'",
        envLines,
        "$shim = Start-Process -FilePath '.\\.venv\\Scripts\\python.exe' -ArgumentList '.\\src\\ade3x3\\steps\\ade3x3_step84_metaheuristic_rank19_search.py' -WorkingDirectory (Get-Location).Path -RedirectStandardOutput \"$outDir\\stdout.log\" -RedirectStandardError \"$outDir\\stderr.log\" -PassThru",
        "$sw = [System.Diagnostics.Stopwatch]::StartNew()",
        "$real = $null",
        "while (($null -eq $real) -and (-not $shim.HasExited)) { Start-Sleep -Milliseconds 250; $real = Get-CimInstance Win32_Process | Where-Object { $_.ParentProcessId -eq $shim.Id -and $_.Name -like 'python*' } | Select-Object -First 1 }",
        "if ($null -eq $real) { throw 'Failed to locate child python process for Step 84.' }",
        "$cores = [int](Get-CimInstance Win32_ComputerSystem).NumberOfLogicalProcessors",
        "$rows = New-Object System.Collections.Generic.List[object]",
        "$prevCpu = $null; $prevTime = Get-Date",
        "while (-not $shim.HasExited) { Start-Sleep -Seconds 1; $proc = Get-Process -Id $real.ProcessId -ErrorAction SilentlyContinue; if ($null -eq $proc) { break }; $now = Get-Date; $cpuPct = $null; if ($null -ne $prevCpu) { $dt = ($now - $prevTime).TotalSeconds; if ($dt -gt 0) { $cpuPct = 100.0 * (($proc.CPU - $prevCpu) / ($dt * $cores)) } }; $rows.Add([pscustomobject]@{ second = [math]::Round($sw.Elapsed.TotalSeconds, 3); cpu_percent = if ($null -eq $cpuPct) { $null } else { [math]::Round($cpuPct, 3) }; working_set_mb = [math]::Round($proc.WorkingSet64 / 1MB, 3); private_memory_mb = [math]::Round($proc.PrivateMemorySize64 / 1MB, 3); threads = $proc.Threads.Count; handles = $proc.Handles }) | Out-Null; $prevCpu = $proc.CPU; $prevTime = $now }",
        "$shim.WaitForExit()",
        "$rows | Export-Csv -NoTypeInformation -Encoding UTF8 -Path \"$outDir\\samples.csv\"",
        "$rows | Select-Object -Last 8"
    ].join("; \n");
}

function downloadJson(state) {
    const envelope = createSettingsEnvelope(state);
    const blob = new Blob([JSON.stringify(envelope, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "step84_tuner_config.json";
    link.click();
    URL.revokeObjectURL(url);
}

async function copyText(text) {
    try {
        await navigator.clipboard.writeText(text);
    } catch (_error) {
        window.alert("Clipboard access failed. Select and copy the command manually.");
    }
}

function saveState(state, options = {}) {
    const currentState = sanitizeState(state);
    const currentStoredEnvelope = readStoredEnvelope();

    if (!options.force && currentStoredEnvelope && activeSettingsEnvelope && currentStoredEnvelope.savedAtMs > activeSettingsEnvelope.savedAtMs) {
        if (!statesEqual(currentStoredEnvelope.settings, currentState)) {
            applyEnvelope(currentStoredEnvelope, {
                message: `A newer tab saved settings at ${formatSavedTime(currentStoredEnvelope.savedAtMs)}. Reloaded the latest snapshot.`
            });
            return { saved: false, conflict: true, envelope: currentStoredEnvelope };
        }
        activeSettingsEnvelope = currentStoredEnvelope;
        if (options.message) {
            setSavedStatus(options.message);
        }
        return { saved: false, conflict: false, envelope: currentStoredEnvelope };
    }

    if (currentStoredEnvelope && statesEqual(currentStoredEnvelope.settings, currentState)) {
        activeSettingsEnvelope = currentStoredEnvelope;
        if (options.message) {
            setSavedStatus(options.message);
        }
        return { saved: false, conflict: false, envelope: currentStoredEnvelope };
    }

    const envelope = createSettingsEnvelope(currentState);
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(envelope));
    activeSettingsEnvelope = envelope;
    setSavedStatus(options.message || `Autosaved locally at ${formatSavedTime(envelope.savedAtMs)}.`);
    return { saved: true, conflict: false, envelope };
}

function createTooltipNode(key) {
    const tooltip = document.createElement("span");
    tooltip.className = "setting-tooltip";
    tooltip.textContent = "?";
    tooltip.title = getHelpText(key);
    tooltip.setAttribute("aria-label", getHelpText(key));
    return tooltip;
}

function createResetNode(key) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "setting-reset";
    button.textContent = "Reset";
    button.dataset.settingReset = key;
    button.addEventListener("click", () => resetSetting(key));
    return button;
}

function decorateStandardSetting(key, element) {
    const wrapper = element.parentElement;
    if (!wrapper || wrapper.querySelector("[data-setting-reset]")) {
        return;
    }
    const label = wrapper.querySelector(`label[for="${element.id}"]`);
    if (!label) {
        return;
    }
    wrapper.classList.add("setting-control");
    const toolbar = document.createElement("div");
    toolbar.className = "setting-toolbar";
    const labelBox = document.createElement("div");
    labelBox.className = "setting-toolbar__label";
    const actions = document.createElement("div");
    actions.className = "setting-toolbar__actions";
    label.parentNode.insertBefore(toolbar, label);
    toolbar.appendChild(labelBox);
    toolbar.appendChild(actions);
    labelBox.appendChild(label);
    label.title = getHelpText(key);
    actions.appendChild(createTooltipNode(key));
    actions.appendChild(createResetNode(key));
}

function decorateToggleSetting(key, element) {
    const toggleLabel = element.closest(".toggle-card");
    if (!toggleLabel || toggleLabel.parentElement?.classList.contains("setting-control--toggle")) {
        return;
    }
    const outer = document.createElement("div");
    outer.className = "setting-control setting-control--toggle";
    const toolbar = document.createElement("div");
    toolbar.className = "setting-toolbar";
    const labelBox = document.createElement("div");
    labelBox.className = "setting-toolbar__label";
    const name = document.createElement("span");
    name.className = "setting-name";
    name.textContent = "STEP84_BEST_EXPORT_INCLUDE_DENSE";
    name.title = getHelpText(key);
    const actions = document.createElement("div");
    actions.className = "setting-toolbar__actions";
    labelBox.appendChild(name);
    actions.appendChild(createTooltipNode(key));
    actions.appendChild(createResetNode(key));
    toolbar.appendChild(labelBox);
    toolbar.appendChild(actions);
    toggleLabel.parentElement.insertBefore(outer, toggleLabel);
    outer.appendChild(toolbar);
    outer.appendChild(toggleLabel);
    const textSpan = toggleLabel.querySelector("span");
    if (textSpan) {
        textSpan.title = getHelpText(key);
    }
}

function decorateTopologyMode() {
    const segmented = document.querySelector(".segmented");
    if (!segmented || document.getElementById("topology-mode-toolbar")) {
        return;
    }
    const toolbar = document.createElement("div");
    toolbar.className = "setting-toolbar setting-toolbar--standalone";
    toolbar.id = "topology-mode-toolbar";
    const labelBox = document.createElement("div");
    labelBox.className = "setting-toolbar__label";
    const name = document.createElement("span");
    name.className = "setting-name";
    name.textContent = "Topology mode";
    name.title = getHelpText("topologyMode");
    const actions = document.createElement("div");
    actions.className = "setting-toolbar__actions";
    labelBox.appendChild(name);
    actions.appendChild(createTooltipNode("topologyMode"));
    actions.appendChild(createResetNode("topologyMode"));
    toolbar.appendChild(labelBox);
    toolbar.appendChild(actions);
    segmented.parentNode.insertBefore(toolbar, segmented);
    document.querySelectorAll(".segmented__item").forEach((button) => {
        const mode = button.dataset.mode;
        button.title = mode === "explicit"
            ? "Explicit populations: you type the exact island sizes yourself, for example 32,24,16,12,8,8,6,6. Best for deliberate hand tuning."
            : "Power schedule: generate island sizes from min exponent, max exponent, and copies, for example 2^2..2^8 with 2 copies each. Best for broad sweeps.";
    });
}

function decorateControls() {
    decorateTopologyMode();
    Object.entries(fieldMap).forEach(([key, id]) => {
        const element = document.getElementById(id);
        if (!element) {
            return;
        }
        if (checkboxFields.has(key)) {
            decorateToggleSetting(key, element);
        } else {
            decorateStandardSetting(key, element);
        }
    });
}

function updateResetButtons(state) {
    const locked = shouldLockRunSettings();
    document.querySelectorAll("[data-setting-reset]").forEach((button) => {
        const key = button.dataset.settingReset;
        const atDefault = isDefaultValue(key, state);
        const lockedSetting = locked && RUN_LOCKED_SETTING_KEYS.has(key);
        button.disabled = atDefault || lockedSetting;
        button.title = lockedSetting
            ? "Locked to the current batch. Start a fresh run to change this setting."
            : atDefault
                ? "Already at default."
                : `Reset to default (${formatValue(defaults[key])}).`;
    });
}

function updateRunLockedControls() {
    const locked = shouldLockRunSettings();
    document.querySelectorAll(".segmented__item").forEach((button) => {
        button.disabled = locked;
    });
    document.querySelectorAll("[data-preset]").forEach((button) => {
        button.disabled = locked;
    });

    Object.entries(fieldMap).forEach(([key, id]) => {
        const element = document.getElementById(id);
        if (!element) {
            return;
        }
        element.disabled = locked && RUN_LOCKED_SETTING_KEYS.has(key);
    });

    const saveButton = document.getElementById("save-config");
    if (saveButton) {
        saveButton.disabled = locked;
    }

    const importInput = document.getElementById("import-config");
    if (importInput) {
        importInput.disabled = locked;
        importInput.parentElement?.classList.toggle("is-disabled", locked);
        importInput.parentElement?.setAttribute(
            "title",
            locked ? "Import is locked to the current batch. Start a fresh run to change settings." : ""
        );
    }
}

function updateRunActionControls() {
    const startRunButton = document.getElementById("start-run");
    const startFreshButton = document.getElementById("start-fresh-run");
    const stopButton = document.getElementById("stop-run");
    if (!startRunButton || !startFreshButton || !stopButton) {
        return;
    }

    const running = Boolean(serverState?.running);
    const hasBatch = hasContinuationContext(serverState);

    if (running) {
        freshDraftUnlocked = false;
        startRunButton.textContent = "Run active";
        startRunButton.disabled = true;
        startFreshButton.textContent = "Fresh run unavailable";
        startFreshButton.disabled = true;
        stopButton.disabled = false;
        setLockStatus("Run-defining settings are locked while the current batch is running.");
    } else if (hasBatch && freshDraftUnlocked) {
        startRunButton.textContent = "Return to current batch";
        startRunButton.disabled = false;
        startFreshButton.textContent = "Launch fresh run";
        startFreshButton.disabled = false;
        stopButton.disabled = true;
        setLockStatus("Fresh draft unlocked. Edit settings now, then launch a fresh run to replace the current batch lineage.");
    } else if (hasBatch) {
        startRunButton.textContent = "Continue current batch";
        startRunButton.disabled = false;
        startFreshButton.textContent = "New fresh draft";
        startFreshButton.disabled = false;
        stopButton.disabled = true;
        setLockStatus("Run-defining settings are locked to the current batch. Use New fresh draft to change them before starting a fresh run.");
    } else {
        freshDraftUnlocked = false;
        lockedBatchState = null;
        startRunButton.textContent = "Start run";
        startRunButton.disabled = false;
        startFreshButton.textContent = "Start fresh run";
        startFreshButton.disabled = false;
        stopButton.disabled = true;
        setLockStatus("No resumable batch is active. All settings are editable.");
    }

    updateRunLockedControls();
}

function enterFreshDraftMode() {
    if (serverState?.running || !hasContinuationContext(serverState)) {
        return false;
    }
    lockedBatchState = sanitizeState(getState());
    freshDraftUnlocked = true;
    updateRunActionControls();
    setSavedStatus("Fresh draft unlocked. Edit settings, then use Launch fresh run.");
    return true;
}

function restoreLockedBatchState() {
    freshDraftUnlocked = false;
    if (lockedBatchState) {
        applyState(lockedBatchState, {
            persist: true,
            forcePersist: true,
            saveMessage: "Restored the locked current-batch settings."
        });
        return;
    }
    updateRunActionControls();
}

async function apiRequest(path, options = {}) {
    const response = await fetch(path, {
        headers: {
            "Content-Type": "application/json"
        },
        ...options
    });
    if (!response.ok) {
        const data = await response.json().catch(() => ({}));
        throw new Error(data.error || `Request failed: ${response.status}`);
    }
    return response.json();
}

function setRunStatus(message) {
    document.getElementById("run-status").textContent = message;
}

function formatMetric(value, digits = 6) {
    if (value === null || value === undefined || Number.isNaN(value)) {
        return "-";
    }
    return Number(value).toFixed(digits);
}

function formatCompactNumber(value, digits = 6) {
    if (value === null || value === undefined || Number.isNaN(value)) {
        return "-";
    }
    const numeric = Number(value);
    if (numeric === 0) {
        return "0";
    }
    if (Math.abs(numeric) >= 1000 || Math.abs(numeric) < 0.001) {
        return numeric.toExponential(3);
    }
    return numeric.toFixed(digits).replace(/0+$/, "").replace(/\.$/, "");
}

function getCopyBestResidual(copy) {
    if (copy?.best_fitness_ever !== null && copy?.best_fitness_ever !== undefined && !Number.isNaN(Number(copy.best_fitness_ever))) {
        return Number(copy.best_fitness_ever);
    }
    if (copy?.latest?.best_fitness !== null && copy?.latest?.best_fitness !== undefined && !Number.isNaN(Number(copy.latest.best_fitness))) {
        return Number(copy.latest.best_fitness);
    }
    return null;
}

function getBestCopy(payload) {
    const copies = (payload.copies || []).filter((copy) => getCopyBestResidual(copy) !== null);
    if (!copies.length) {
        return null;
    }
    return copies.reduce((best, copy) => (getCopyBestResidual(copy) < getCopyBestResidual(best) ? copy : best));
}

function formatAxisValue(value, mode) {
    if (mode === "log") {
        return value.toExponential(2);
    }
    if (Math.abs(value) >= 1000 || (Math.abs(value) > 0 && Math.abs(value) < 0.01)) {
        return value.toExponential(2);
    }
    return value.toFixed(2);
}

function updateLiveMetrics(payload) {
    const bestCopy = getBestCopy(payload);
    const bestLatest = bestCopy?.latest || null;
    document.getElementById("live-copies").textContent = payload.copy_count !== undefined ? String(payload.copy_count) : "-";
    document.getElementById("live-running").textContent = payload.running_count !== undefined ? String(payload.running_count) : "-";
    document.getElementById("live-best-copy").textContent = bestCopy ? `Copy ${bestCopy.index}` : "-";
    document.getElementById("live-best").textContent = getCopyBestResidual(bestCopy) !== null
        ? formatCompactNumber(getCopyBestResidual(bestCopy), 9)
        : payload.summary?.best_fitness_ever
            ? formatCompactNumber(payload.summary.best_fitness_ever, 9)
            : "-";
    document.getElementById("live-generation").textContent = bestLatest && bestLatest.generation !== undefined
        ? String(bestLatest.generation)
        : bestCopy?.total_generations !== undefined && bestCopy?.total_generations !== null
            ? String(bestCopy.total_generations)
            : "-";
    document.getElementById("live-wall").textContent = bestLatest && bestLatest.wall_seconds_cumulative !== undefined
        ? formatCompactNumber(bestLatest.wall_seconds_cumulative, 2)
        : bestCopy?.total_wall_seconds !== undefined && bestCopy?.total_wall_seconds !== null
            ? formatCompactNumber(bestCopy.total_wall_seconds, 2)
            : "-";
}

function renderCopyMetrics(payload) {
    const container = document.getElementById("copy-metrics");
    container.innerHTML = "";
    const copies = payload.copies || [];
    if (!copies.length) {
        container.innerHTML = '<div class="copy-card"><div class="copy-card__header"><div class="copy-card__title">No copy data</div></div><div class="copy-card__metric"><span>Status</span><strong>Start a run to populate per-copy metrics.</strong></div></div>';
        return;
    }

    copies.forEach((copy) => {
        const latest = copy.latest || {};
        const card = document.createElement("div");
        card.className = "copy-card";
        card.innerHTML = `
            <div class="copy-card__header">
                <div class="copy-card__title">Copy ${copy.index}</div>
                <div class="copy-card__status${copy.running ? " is-running" : ""}">${copy.running ? "running" : (copy.exit_code === 0 ? "done" : (copy.exit_code === null ? "idle" : `exit ${copy.exit_code}`))}</div>
            </div>
            <div class="copy-card__grid">
                <div class="copy-card__metric"><span>Seed</span><strong>${copy.seed ?? "-"}</strong></div>
                <div class="copy-card__metric"><span>Generation</span><strong>${latest.generation ?? copy.total_generations ?? "-"}</strong></div>
                <div class="copy-card__metric"><span>Best residual</span><strong>${formatCompactNumber(getCopyBestResidual(copy), 9)}</strong></div>
                <div class="copy-card__metric"><span>Wall seconds</span><strong>${formatCompactNumber(latest.wall_seconds_cumulative ?? copy.total_wall_seconds, 2)}</strong></div>
                <div class="copy-card__metric"><span>Signature (3,3,3)</span><strong>${latest.signature_333_count ?? "-"}</strong></div>
                <div class="copy-card__metric"><span>Shadow pool</span><strong>${latest.shadow_pool_size ?? "-"}</strong></div>
            </div>
        `;
        container.appendChild(card);
    });
}

function buildCopyTrace(copy, valueKey, color, name, yAxis = "y", visible = true) {
    const history = (copy.history || []).filter((row) => row[valueKey] !== null && row[valueKey] !== undefined && (valueKey !== "best_fitness" || row[valueKey] > 0));
    return {
        type: "scatter",
        mode: "lines+markers",
        name,
        x: history.map((row) => row.generation),
        y: history.map((row) => row[valueKey]),
        line: { color, width: 2, shape: "linear", simplify: false },
        marker: { color, size: 4, symbol: "circle" },
        hovertemplate: `Copy ${copy.index}<br>Generation %{x}<br>${name}: %{y:.6g}<extra></extra>`,
        connectgaps: false,
        yaxis: yAxis,
        visible: visible ? true : "legendonly"
    };
}

const SQRT27 = Math.sqrt(27);

function buildRelativeCopyTrace(copy, color, name) {
    const history = (copy.history || []).filter(
        (row) => row.best_fitness !== null && row.best_fitness !== undefined && row.best_fitness > 0
    );
    return {
        type: "scatter",
        mode: "lines+markers",
        name,
        x: history.map((row) => row.generation),
        y: history.map((row) => row.best_fitness / SQRT27),
        line: { color, width: 1.5, shape: "linear", simplify: false, dash: "dot" },
        marker: { color, size: 3, symbol: "diamond" },
        hovertemplate: `Copy ${copy.index}<br>Generation %{x}<br>${name}: %{y:.6g}<extra></extra>`,
        connectgaps: false,
        yaxis: "y2",
        visible: true
    };
}

function renderPlot(targetId, traces, layout) {
    const target = document.getElementById(targetId);
    if (!window.Plotly) {
        target.textContent = "Plotly failed to load. Refresh the page after reconnecting network access.";
        return;
    }
    const nonEmpty = traces.filter((trace) => Array.isArray(trace.x) && trace.x.length);
    if (!nonEmpty.length) {
        window.Plotly.react(target, [], {
            ...PLOTLY_LAYOUT_BASE,
            ...layout,
            annotations: [{
                text: "No data yet.",
                x: 0.5,
                y: 0.5,
                xref: "paper",
                yref: "paper",
                showarrow: false,
                font: { size: 16, color: "#5f655f" }
            }]
        }, PLOTLY_CONFIG);
        return;
    }
    window.Plotly.react(target, nonEmpty, { ...PLOTLY_LAYOUT_BASE, ...layout }, PLOTLY_CONFIG);
}

function updateScaleButtons() {
    const fitnessButton = document.getElementById("fitness-scale-toggle");
    const shadowButton = document.getElementById("shadow-scale-toggle");
    if (fitnessButton) {
        fitnessButton.textContent = fitnessScaleMode === "log" ? "Log Y" : "Linear Y";
        fitnessButton.title = fitnessScaleMode === "log"
            ? "Residual chart is using a log-scaled Y axis. Click to switch to linear."
            : "Residual chart is using a linear Y axis. Click to switch to log.";
    }
    if (shadowButton) {
        shadowButton.textContent = shadowScaleMode === "log" ? "Log Y" : "Linear Y";
        shadowButton.title = shadowScaleMode === "log"
            ? "Shadow chart is using a log-scaled Y axis. Click to switch to linear."
            : "Shadow chart is using a linear Y axis. Click to switch to log.";
    }
}

function updateCharts(payload) {
    const copies = payload.copies || [];
    const fitnessTraces = [];
    const shadowTraces = [];

    copies.forEach((copy, index) => {
        const color = COPY_COLORS[index % COPY_COLORS.length];
        fitnessTraces.push(buildCopyTrace(copy, "best_fitness", color, `copy ${copy.index} best`));
        fitnessTraces.push(buildCopyTrace(copy, "mean_fitness", color, `copy ${copy.index} mean`, "y", false));
        fitnessTraces.push(buildRelativeCopyTrace(copy, color, `copy ${copy.index} rel (÷√27)`));

        shadowTraces.push(buildCopyTrace(copy, "shadow_pool_size", color, `copy ${copy.index} shadow`, "y"));
        shadowTraces.push(buildCopyTrace(copy, "signature_333_count", color, `copy ${copy.index} (3,3,3)`, "y2", false));
    });

    renderPlot("fitness-chart", fitnessTraces, {
        title: { text: "", font: { size: 14 } },
        yaxis: {
            ...PLOTLY_LAYOUT_BASE.yaxis,
            title: "Absolute residual",
            type: fitnessScaleMode === "log" ? "log" : "linear"
        },
        yaxis2: {
            title: "Relative residual (÷√27)",
            overlaying: "y",
            side: "right",
            type: fitnessScaleMode === "log" ? "log" : "linear",
            gridcolor: "rgba(0,0,0,0)",
            linecolor: "rgba(31, 36, 48, 0.12)",
            zeroline: false,
            automargin: true,
            tickformat: ".3g"
        },
        uirevision: `fitness-${fitnessScaleMode}`
    });

    renderPlot("shadow-chart", shadowTraces, {
        title: { text: "", font: { size: 14 } },
        yaxis: {
            ...PLOTLY_LAYOUT_BASE.yaxis,
            title: "Shadow pool size",
            type: shadowScaleMode === "log" ? "log" : "linear"
        },
        yaxis2: {
            title: "(3,3,3) count",
            overlaying: "y",
            side: "right",
            gridcolor: "rgba(0,0,0,0)",
            linecolor: "rgba(31, 36, 48, 0.12)",
            zeroline: false,
            automargin: true
        },
        uirevision: `shadow-${shadowScaleMode}`
    });
    updateScaleButtons();
}

function updateLogs(payload) {
    document.getElementById("stdout-tail").value = (payload.stdout_tail || []).join("\n");
    document.getElementById("stderr-tail").value = (payload.stderr_tail || []).join("\n");
}

function applyServerState(payload) {
    serverState = payload;
    if (hasContinuationContext(payload) && !freshDraftUnlocked) {
        lockedBatchState = buildStateFromEnv(payload.env || {});
        lockedBatchState.batchCopies = Number(payload.copy_count) || defaults.batchCopies;
        if (!statesEqual(getState(), lockedBatchState)) {
            applyState(lockedBatchState, { persist: false });
        }
    }
    updateLiveMetrics(payload);
    renderCopyMetrics(payload);
    updateCharts(payload);
    updateLogs(payload);
    if (payload.running) {
        if (payload.continued_from_previous) {
            setRunStatus(`Continuing ${payload.running_count}/${payload.copy_count} copies from the current population. Resumed copies: ${payload.resumed_copy_count}. Batch output: ${payload.batch_dir || "-"}.`);
        } else {
            setRunStatus(`Running ${payload.running_count}/${payload.copy_count} Step 84 copies. Batch output: ${payload.batch_dir || "-"}.`);
        }
    } else if (payload.copy_count) {
        if (payload.continued_from_previous) {
            setRunStatus(`Last continuation finished. Copies: ${payload.copy_count}. Resumed copies: ${payload.resumed_copy_count}. Batch output: ${payload.batch_dir || "-"}.`);
        } else {
            setRunStatus(`Last batch finished. Copies: ${payload.copy_count}. Batch output: ${payload.batch_dir || "-"}.`);
        }
    } else {
        setRunStatus("Backend reachable. No active Step 84 run.");
    }
    updateRunActionControls();
}

async function refreshServerState() {
    try {
        const payload = await apiRequest("/api/state", { method: "GET", headers: {} });
        applyServerState(payload);
    } catch (_error) {
        setRunStatus("Backend not reachable. Start the local server to enable run control and live graphs.");
        updateRunActionControls();
    }
}

async function startRun(freshStart = false) {
    if (freshStart && hasContinuationContext(serverState) && !serverState?.running && !freshDraftUnlocked) {
        enterFreshDraftMode();
        return;
    }

    if (!freshStart && freshDraftUnlocked && hasContinuationContext(serverState) && !serverState?.running) {
        restoreLockedBatchState();
        return;
    }

    try {
        const state = !freshStart && hasContinuationContext(serverState) && lockedBatchState
            ? sanitizeState(lockedBatchState)
            : getState();
        const saveResult = saveState(state);
        if (saveResult.conflict) {
            setRunStatus("A newer settings snapshot was applied from another tab. Review it before starting a run.");
            return;
        }
        const payload = await apiRequest("/api/run/start", {
            method: "POST",
            body: JSON.stringify({ env: buildEnvObject(state), copies: state.batchCopies, fresh_start: freshStart })
        });
        freshDraftUnlocked = false;
        lockedBatchState = sanitizeState(state);
        applyServerState(payload.state);
    } catch (error) {
        setRunStatus(error.message);
    }
}

async function stopRun() {
    try {
        const payload = await apiRequest("/api/run/stop", { method: "POST", body: "{}" });
        applyServerState(payload.state);
    } catch (error) {
        setRunStatus(error.message);
    }
}

function render(options = {}) {
    const state = getState();
    const derived = getDerived(state);

    document.getElementById("topology-label").textContent = state.topologyMode === "explicit" ? "Explicit list" : "Power schedule";
    document.getElementById("population-total").textContent = derived.totalPopulation.toLocaleString();
    document.getElementById("offspring-total").textContent = derived.totalOffspring.toLocaleString();
    document.getElementById("island-count").textContent = derived.islandCount.toLocaleString();
    document.getElementById("derived-summary").textContent = derived.populations.length
        ? `Schedule: [${derived.populations.join(", ")}]. Total population ${derived.totalPopulation}, total offspring ${derived.totalOffspring}.`
        : "Schedule is empty. Add at least one positive island population.";
    document.getElementById("step84-command").value = buildRunCommand(state);
    document.getElementById("profile-command").value = buildProfileCommand(state);
    updateResetButtons(state);
    updateRunActionControls();
    if (options.persist ?? true) {
        saveState(state, {
            force: options.forcePersist ?? false,
            message: options.saveMessage
        });
    }
}

function wirePresets() {
    document.querySelectorAll("[data-preset]").forEach((button) => {
        button.addEventListener("click", () => {
            const current = getState();
            applyState({ ...current, ...presetMap[button.dataset.preset] }, { persist: true });
        });
    });
}

function wireModeToggle() {
    document.querySelectorAll(".segmented__item").forEach((button) => {
        button.addEventListener("click", () => {
            setTopologyMode(button.dataset.mode);
            render();
        });
    });
}

function wireFieldUpdates() {
    Object.values(fieldMap).forEach((id) => {
        const element = document.getElementById(id);
        if (!element) {
            return;
        }
        element.addEventListener("input", render);
        element.addEventListener("change", render);
    });
}

function wireActions() {
    document.getElementById("save-config").addEventListener("click", () => saveState(getState()));
    document.getElementById("copy-run-command").addEventListener("click", () => copyText(document.getElementById("step84-command").value));
    document.getElementById("copy-profile-command").addEventListener("click", () => copyText(document.getElementById("profile-command").value));
    document.getElementById("download-config").addEventListener("click", () => downloadJson(getState()));
    document.getElementById("start-run").addEventListener("click", () => startRun(false));
    document.getElementById("start-fresh-run").addEventListener("click", () => startRun(true));
    document.getElementById("stop-run").addEventListener("click", stopRun);
    document.getElementById("refresh-state").addEventListener("click", refreshServerState);
    document.getElementById("fitness-scale-toggle").addEventListener("click", () => {
        fitnessScaleMode = fitnessScaleMode === "log" ? "linear" : "log";
        refreshServerState();
    });
    document.getElementById("shadow-scale-toggle").addEventListener("click", () => {
        shadowScaleMode = shadowScaleMode === "log" ? "linear" : "log";
        refreshServerState();
    });
    document.getElementById("import-config").addEventListener("change", async (event) => {
        const [file] = event.target.files || [];
        if (!file) {
            return;
        }
        const text = await file.text();
        try {
            const parsed = normalizeEnvelope(JSON.parse(text));
            if (!parsed) {
                throw new Error("Invalid JSON config.");
            }
            applyState(parsed.settings, {
                persist: true,
                forcePersist: true,
                saveMessage: "Imported JSON and promoted it to the latest local snapshot."
            });
        } catch (_error) {
            window.alert("Invalid JSON config.");
        }
        event.target.value = "";
    });
    window.addEventListener("beforeunload", () => saveState(getState()));
    window.addEventListener("storage", (event) => {
        if (event.key !== STORAGE_KEY || !event.newValue) {
            return;
        }
        let envelope = null;
        try {
            envelope = normalizeEnvelope(JSON.parse(event.newValue));
        } catch (_error) {
            return;
        }
        if (!envelope) {
            return;
        }
        if (!activeSettingsEnvelope || envelope.savedAtMs > activeSettingsEnvelope.savedAtMs) {
            applyEnvelope(envelope, {
                message: `Loaded newer settings from another tab saved at ${formatSavedTime(envelope.savedAtMs)}.`
            });
        }
    });
}

function init() {
    decorateControls();
    wirePresets();
    wireModeToggle();
    wireFieldUpdates();
    wireActions();
    applyState(getSavedState(), { persist: false });
    updateRunActionControls();
    refreshServerState();
    pollHandle = window.setInterval(refreshServerState, POLL_INTERVAL_MS);
}

init();