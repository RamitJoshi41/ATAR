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
