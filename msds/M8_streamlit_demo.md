# MSD — M8: Streamlit Demo

## Purpose
Interactive side-by-side showcase of the trained ATAR policy vs. the ReAct
baseline, for demoing the project live (interviews, portfolio, lab review).

## Dependency
Needs M5 (environment) and M7 (trained policy + ReAct baseline).

## Layout
- **Left column**: ReAct baseline running on the current query — show its
  chain-of-thought text and each tool call as it happens.
- **Right column**: ATAR policy running on the same query — show the
  policy's action probabilities (as a bar chart) at each step, and running
  cumulative reward.
- **Bottom panel**: live metrics — accuracy so far this session, average
  tool calls, latency, estimated token cost — updated after each query.
- **Controls**: a dropdown of pre-loaded example tasks (a few from each
  tier), a free-text query box, a max-turns slider, and a "step through"
  mode that pauses after each action for the viewer to click "next."

## Live vs simulated toggle (added scope — see project ARCHITECTURE_DECISIONS)
Add a toggle: "Simulated tools" (default, uses M1) vs. "Live tools" (routes
`search_web` to a real search API call instead of the BM25 simulator, while
leaving calculator/SQL/Python as-is). Clearly label which mode is active in
the UI — don't let a viewer mistake a live-mode run for the trained
distribution's actual training conditions.

## Definition of Done
- [ ] `streamlit run demo/app.py` launches without error and completes a
  full query end-to-end in simulated mode.
- [ ] Side-by-side comparison actually runs both ReAct and ATAR on the same
  query and displays both, not a mocked/precomputed pair.
- [ ] Step-through mode correctly pauses between actions.
- [ ] Live-tools toggle actually swaps the search backend (verified by a
  test query where simulated and live modes return visibly different
  results).
- [ ] Basic error handling: a malformed free-text query doesn't crash the
  app (show an error message in the UI instead).
- [ ] `pytest tests/demo/ -v` for any testable logic (metrics computation,
  not the Streamlit UI itself) passes, report actual count.

---

## Amendment 2026-10-14 (approved by the architect)

**Why.** The policy's input is the training model's hidden states, so the
demo can only run live on a GPU host that loads that same 7B model; hosted
text-only APIs (e.g. NVIDIA NIM) cannot run it. The architect also wants
viewers to see the policy *improve over training*. See ADR-19 and ADR-21.

**Added scope.**
- **Policy versions:** a version selector over training checkpoints
  (step 0 = untrained … final), each evaluated on the held-out set.
- **Training progress view:** held-out success vs. training step with 95%
  bootstrap CIs and the baselines as reference lines; the action mix per
  version; a table view. Improvement claims come only from held-out
  evaluation, never from training reward.
- **Recorded replays (default mode):** episodes recorded on the GPU with
  `scripts/record_replays.py` and replayed in the app, labelled as recorded.
  Live mode remains for GPU hosts.
- **Sample bundle:** a bundle built with the 0.5B stand-in and toy
  checkpoints, always shown with a "SAMPLE DATA — not ATAR's results" banner.

**Changed DoD items.**
- "Side-by-side comparison actually runs both ReAct and ATAR on the same
  query … not a mocked/precomputed pair" now applies to **live mode**;
  replay mode shows *recorded real runs* of both on the same task.
- `streamlit run atar/demo/app.py` (the app lives in the `atar.demo`
  package; the repo-root `demo.py` is the original vertical-slice script).
- The live-tools search backend is Wikipedia's public API (the simulated
  search is BM25 over Wikipedia abstracts); verified by a manual query, and
  in tests with a fake HTTP opener (tests never touch the network).
