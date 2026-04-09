## Next-Phase Handoff

Date: 2026-04-08

This file records the immediately usable state of the ED-degree search without waiting for all active basins or target-tracking calls to finish.

### Hard Checkpoints Already On Disk

Rank 19
- Status: `tracking_started`
- Orbit config: `[1, 6, 12]`
- Effective parameters: `24`
- Monodromy-complete ED degree: `1`
- Timings:
  - build system: `0.497s`
  - start pair: `1.462s`
  - monodromy: `597.488s`
- Source file: `ed_R19.json`

Rank 22
- Status: `monodromy_complete`
- Orbit config: `[1, 6, 8, 12]`
- Effective parameters: `30`
- Monodromy-complete ED degree: `1`
- Timings:
  - build system: `0.567s`
  - start pair: `1.273s`
  - monodromy: `143.535s`
- Source file: `ed_R22.json`

### Live Session Evidence From The Active Run

Session file: `session_20260408_164657.jsonl`
Launcher configuration: `24` Julia threads per rank

Latest observed monodromy counts from JSONL:
- Rank 13: `301` observed solutions at `132.127s`
- Rank 20: `7` observed solutions at `129.842s`

These are live monodromy counts parsed from stdout, not final checkpoint files.

### What You Can Safely Use Right Now

1. R22 is already a confirmed one-solution monodromy case on the `[1, 6, 8, 12]` chart.
2. R19 is also already a confirmed one-solution monodromy case on the `[1, 6, 12]` chart, and target tracking had started in a prior completed checkpoint.
3. The current active run shows R13 is highly productive under monodromy on the `[1, 12]` chart, with at least `301` observed solutions already.
4. The current active run shows R20 is also productive on the `[8, 12]` chart, with at least `7` observed solutions already.

### What Is Not Available Yet

1. No monodromy endpoint-coordinate dump files have been written yet by the current live run.
2. No tracked real-endpoint dump files have been written yet.
3. No basin-geometry or clustering information exists yet.

### Recommended Immediate Next-Phase Starting Point

If the next phase is about prioritization rather than exact endpoint geometry:

- Treat `R13` as the main high-yield monodromy regime.
- Treat `R20` as a secondary productive regime worth watching.
- Treat `R19` and `R22` as already-established one-solution cases that can anchor comparison.

If the next phase requires actual parameter vectors, wait specifically for the first `monodromy checkpoint written` event from the active run; that is the first moment when endpoint coordinates should become extractable under the patched solver.