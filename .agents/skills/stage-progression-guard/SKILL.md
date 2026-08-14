---
name: stage-progression-guard
description: Evaluate and govern CEG-WM semantic stage changes. Use when changing project_stage, research-definition readiness, component implementation readiness, runtime verification, experiment admission, or evidence claims.
---

# Stage Progression Guard

## Workflow

1. Read the current stage and allowed work from `.codex/project_contract.md`.
2. Confirm the current and requested stage names are registered in `governance/policies/method_readiness_rules.yaml` `stage_order`, then list the evidence required by the requested next stage.
3. Separate structural readiness, implemented capability, and formal experimental evidence.
4. For `research_defined` and later stages, verify `.codex/research_state/research_definition.yaml`, all required design roles and every frozen method invariant.
5. To enter `method_construction_authorized`, verify candidate specification closure, independent review, explicit user authorization and an auditable repository revision, then fill `method_construction_admission.yaml` from its template.
6. Make the transition to `method_construction_authorized` in a revision that contains no `main/` change; use the research-definition audit to verify that transition from the admission base revision, and begin implementation only in a later authorized revision.
7. For `method_implemented` and later stages, read the required responsibility count and
   bindings from `governance/policies/method_readiness_rules.yaml`, the actual
   readiness-bound implementation identities from
   `.codex/research_state/method_readiness.yaml`, and the design registry identities
   from `docs/design/candidate_specifications.md`; verify the readiness binds every
   policy-required component to its fixed architecture path, exact candidate IDs,
   unique implementation symbol and declared responsibility without conflating these
   three authority planes. Treat the readiness candidate digest as the reviewed
   implementation snapshot and verify it from the candidate-specification Git blob at
   the recorded review revision; treat the current candidate specification as the live
   design authority, which must retain all policy/readiness-bound identities but may
   additionally describe candidates that remain pending implementation admission.
8. Verify every required candidate-specific behavior node binds the policy-required component symbols, calls them, and makes data-dependent, non-isomorphic assertions.
9. Treat the AST audit only as a necessary structure/wiring gate. Require a separate
   independent semantic review `approve` bound to the same reviewed candidate snapshot
   digest and repository revision. After review, keep implementation paths and
   registered behavior-test paths stale-protected; do not require later live-design
   additions to rewrite or re-sign the historical readiness snapshot.
10. Verify required runtime evidence, protocols, records, reports and harness gates for later stages.
11. Distinguish full CEG-WM evidence from `research_question_closed_negative` and a separately named, separately authorized reduced-scope method.
12. Change the stage only when all declared gates are satisfied.

## Blocking Rules

- Use semantic stage names, not numbered, `new`, `final`, or weak version labels.
- Do not advance a stage using placeholder fields, dry runs, empty directories, or structural audits as experimental evidence.
- Do not weaken a gate to make the transition pass.
- Do not use an unregistered stage name to bypass an active gate.
- Behavior-check nodes must be in the default suite and must directly import and call the registered implementation with non-constant assertions.
- Do not treat method-readiness metadata itself as scientific effectiveness evidence.
- Do not make research code or runtime configuration depend on the outer method-readiness metadata.
- Do not enter `method_implemented` with any policy-required responsibility component
  missing, aliased or folded together, including content embedder, LF detector and
  geometry reliability.
- Do not advance beyond `research_defined` merely because design documents exist.
- Do not enter `method_construction_authorized` without explicit user authorization and an auditable repository revision.
- Do not combine the stage transition into `method_construction_authorized` with substantive method implementation.
- Do not treat `method_construction_authorized` as implemented capability or scientific evidence.
- Do not let one implementation symbol satisfy multiple required method components.
- Do not let one centralized proxy module satisfy components assigned to distinct architecture paths.
- Do not accept generic arithmetic checks in place of the candidate-specific key
  schedule/golden vectors, HF carrier/direct-score, LF, routing, Q/K, reliability,
  rectification and joint-decision nodes.
- Do not treat repeated or structurally isomorphic behavior tests as independent readiness evidence.
- Do not treat the readiness AST audit as proof of method semantics or non-proxy implementation; an independent revision-bound semantic review is mandatory.
- Do not let HF-only plus geometry pass the full CEG-WM success gate after LF or routing fails promotion.

## Required Validation

- Run every gate declared for the target stage, the default test suite, governance self-tests and all harness audits.
