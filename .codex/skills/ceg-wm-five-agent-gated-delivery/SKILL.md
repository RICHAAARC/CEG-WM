---
name: ceg-wm-five-agent-gated-delivery
description: Run or review an authorized CEG-WM implementation, Colab handoff, artifact audit, or release using the Agent1–5 roles, exact-identity binding, and evidence ceiling.
allowed-tools: Read, Grep, Glob, Bash
---

# CEG-WM five-agent gated delivery

## When to use

Use for an authorized CEG-WM method/runtime/runner change, controlled handoff or push, real-artifact audit, or a request to assess a gate. Do not use to infer missing user authorization, perform formal GPU/Colab work, or turn an audit/readiness result into a scientific decision.

## Inputs to gather

1. Current goal and authorization boundary: branch/base/HEAD, strict allowlist, action, and explicit stop conditions.
2. Bound identities: method/protocol/run/roster/key/artifact digests and whether an earlier failure is frozen evidence.
3. Minimum relevant tests, current diff, and the current controller/user approval for the exact action.

## Procedure

1. Classify the action. If it touches method semantics (carrier/detector/joint statistic/band/allocation/budget), gates/thresholds/ties, roster/denominator/failure/retry, blind/key/PRG/secret, formal identity/conclusion, GPU/Drive/provenance, irreversible external state, cross-route integration, or a second same-class real failure, stop for the user’s current exact approval. Agent verdicts and historic approval cannot substitute. [ad-hoc note]
2. Agent1 works only in the allowlist. Prefer a real production path over proxy logic; use a transparent non-amend forward commit. Do not self-approve.
3. Run only the minimum targeted tests plus `git diff --check`; reserve full/profile validation for the cumulative final exact.
4. Have Agent2 and Agent3 independently review the same final exact in parallel. Agent2 checks method relevance/minimal governance; Agent3 traces the real production path and independently checks formal artifacts rather than trusting self-reports.
5. If either returns `REQUEST CHANGES`, preserve the exact blocker and stop later gates. One first same-allowlist (or narrower), non-semantic mechanical correction may be a transparent forward correction followed by a fresh Agent2 review; do not add files or weaken assertions. A second request, real API incompatibility, semantic/runtime-authority change, GPU, push, or any external state change returns to the user. [ad-hoc note]
6. Start Agent4 only after fresh Agent2 and Agent3 approvals. Agent4 may declare completeness/readiness to request an already authorized controlled push; it cannot authorize GPU, science, merge, or promotion.
7. Keep Agent5 inactive until a real non-roster canary, formal artifact, or final route milestone. It can return readiness to present a personal audit/adjudication, never make that adjudication.
8. State the evidence ceiling explicitly: local/fake/CPU = engineering; non-roster canary = operational with denominator 0; formal RC0 = evaluable, not automatically gate-passing. Only user personal approval can record `SCIENTIFIC_NEGATIVE` or promotion.

## Efficiency plan

- Start from a compact active-state summary: goal; branch/base/HEAD; allowlist; identities; tests; verdicts; blocked actions; next authorization node.
- Send status only when branch/exact, allowlist, tests, review, push, or blocker changes.
- Use the compact review return order: identity → diff/allowlist → method delta → tests → Agent2/3/4/5 verdict → evidence ceiling → next action.
- Run Agent2/3 read-only work concurrently; never run multiple writers. Do not repeat unchanged tests or copy frozen logs/artifacts into follow-up context.
- Treat dependency/version/GPU observations as record-only unless a repeated real failure establishes a narrow public-contract need; do not turn them into generic pre-commit gates. [ad-hoc note]

## Pitfalls and fixes

- Reproducibility or package versions become a generic gate → keep them record-only unless repeated real failure supports a narrow method/API requirement.
- Notebook reads ZIP bytes or hashes artifacts → runner creates artifacts; a thin notebook checks only sidecar/filename binding; independent audit owns byte hashing.
- Artifact integrity or false all-gates conjunction is treated as science → preserve `scientific_status=not_adjudicated` and request the user’s exact bound adjudication.
- Old authorization or another agent’s verdict is treated as permission → request current approval bound to exact/action/scope.

## Verification checklist

- Exact parent, final exact, allowlist, and identities are recorded.
- Targeted tests and diff checks are tied to that exact.
- Agent2 and Agent3 reviewed the same exact; Agent4 ran only after both approved.
- Any push/external action had explicit current authorization and pre/postflight evidence.
- Result names its evidence ceiling and does not claim science/promotion beyond it.
