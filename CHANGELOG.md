# Public preparation decisions

This is a development snapshot, not a published release. Detailed private
evidence stays in the research repository. Stable decision IDs support review.

| ID | Change | Reason / evidence |
|---|---|---|
| PUB-01 | Separate self-contained source tree, curated one-way export. | Private development remains authoritative; no private Git history or result archive is exported. Exported-copy tests check independence. |
| PUB-02 | Explicit `Problem.reach/avoid/reach_avoid`, negative-inside input sets. | Avoids manually coordinating input signs and control optimization. Tests check opposite controller directions and a forced-translation failure barrier. |
| PUB-03 | Apply immediate hybrid projection at horizon zero; remove final source-goal clamps from reach projection. | Mandatory resets precede source stopping. Tests use different source/destination goals, including zero horizon. Archived private BRT conventions remain unchanged. |
| PUB-04 | Avoid clearance caps before/after every reset stage. | Failure at a source or intermediate reset cannot be erased. Independent projection, HLRF and simulator tests exercise a failing intermediate mode. |
| PUB-05 | Active out-of-domain resets error by default; clipping requires an edge setting. | Prevents silently altering a new model. Inactive reset outputs need not lie inside the grid. Periodic PDE resets still wrap. |
| PUB-06 | Shared local interpolation, results, avoid HLRF and event simulation. | Policies read post-reset gradients, finite-horizon queries cannot extrapolate, and local stencils avoid full-grid gradient preparation. Numerical/physical time and reset traces are explicit. |
| PUB-07 | Online candidate paths reverse listed Bellman stage composition. | `T_B(T_A(V))` describes physical B then A. A cooldown-like forced→optional test compares online candidates to the projected field. |
| PUB-08 | Aircraft optimal-speed ties use admissible endpoints; dissipation bounds use the full input-box corners. | Corrects documented AIR-05 issues in the new copy. Hamiltonian corner tests verify max/min choices; private reference implementation is retained. |
| PUB-09 | Avoid-only HLRF; deterministic-forced continuous feedback for reach/RA. | No reach/RA LRF is advertised. Controlled discrete reach timing and forced online nondeterminism require explicit additional machinery. |
| PUB-10 | Portable NumPy/JSON storage, selectable checkpoints, XLA execution-budget check. | Avoids private fixtures and pickling; retains all coupling updates. Sparse/full history equivalence and budget rejection are tested. Streaming was subsequently added in PUB-13. |
| PUB-11 | Explicit identity/asinh coordinate adapters and physical-policy wrapper. | Physical ODEs, resets and guards use the appropriate chart; gradient queries apply the full diagonal chain rule. Roundtrip, finite-difference gradient and transformed-flow tests pass. |
| PUB-12 | Optional hold-lookahead for avoid HLRF and explicit hold-based hybrid goal feedback. | A projected stay value includes immediate switching. Satellite idle candidates instead commit to one feedback interval at decreasing time-to-go. Out-of-domain predictions are ineligible, never clipped. Tests expose dummy-stay ties for both objectives. |
| PUB-13 | Streamed HJVI continuation with dynamic times/initial fields and cached chunk executables. | Checkpoints go to a host memmap. Terminal/reset projection runs once, not at chunk boundaries. Three-objective sparse/full equivalence, empty-output chunks, projection chains and failed-run loading are tested. |
| PUB-14 | Guard events can explicitly own a computational domain face; held coordinates omit identically zero exit events. | Satellite phase=1 triggers a required reset, not a false domain exit. Physical tests check 60 s thrust, 300 s cooldown and immediate cooldown→idle→thrust. |
| PUB-15 | Saved grid axes accept up to eight floating-point epsilons of backend representation difference. | GPU and CPU linspace roundoff previously prevented loading a GPU solve for CPU feedback. Larger coordinate changes still error; interpolation uses the saved axes. No physical trajectories are clipped. |
| PUB-16 | Shared physical rollouts store pre-reset segment endpoints and reset traces; public independent replay/geometry audit. | Satellite replay uses inertial Kepler dynamics, jumper replay uses exact constant-acceleration flow. Input availability, clocks, time-to-go and conditional clearance are checked separately from feedback. |
| PUB-17 | Canonicalize an already-located guard/domain-face root within eight float64 epsilons, recording every nonzero correction. | A phase root returned 1.0000000000000002 and caused a false domain exit at 728 s. The diagnostic prefix is retained privately. Only the declared event face is normalized; larger errors raise, and arbitrary states/predictions are never clipped. |
| PUB-18 | Public demo profiles use moderate grids and finite feedback deadlines. Satellite avoid shares the two listed reach-avoid starts. | The public 1200 s avoid trajectories are new validation cases, not the archived 2800 s stationary-field experiments or exact paper-figure initial conditions. Reference profiles remain explicit further-refinement options. |
| PUB-19 | Standalone `AGENTS.md` with environment/import checks, CPU/GPU commands, mathematical invariants and export boundaries. | Prevents loading the private package with the same name and documents the validated workflow. Curated exports and source distributions include the guide; the export test verifies its presence and hash. |
| PUB-20 | Organize public documentation around concepts, a first CPU tutorial and paper experiments; move setup/development details into dedicated guides. Include a generated tutorial preview, script/preset map, expected outputs and explicit reproduction status. | Readers previously had to infer the workflow from implementation notes. Local documentation links, executable Python walkthroughs, tutorial CLI, Dog1D recipe and IJRR smoke/replay are checked from a fresh source export. Full RSS plotting workflows and exact paper-figure reproduction remain incomplete. |
| PUB-21 | Fix tutorial summary construction in `tutorials/jumper.py`; retain trajectory `status` and save solver completion as `solver_status`. Example commands print concise outcomes and output locations. | Streamed-solver metadata added a `status` key, causing duplicate-key `TypeError` before the first tutorial figure was written. The complete CPU tutorial now finishes reach/avoid/reach-avoid and writes all summaries and plots. Numerical solver, model and feedback behavior are unchanged. |

Remaining work before a complete public release:

- Revalidate representative fine-grid outputs under the public semantics.
- Extend full rollout adapters to the remaining RSS Dog1D/aircraft/three-mode jumper recipes.
- Select a repository license and review upstream attribution before publication.

No private archive, MATLAB environment or fine-grid benchmark is needed to run
the tutorial or public tests. Fine-grid accuracy and global invariance remain
separate validation questions.
