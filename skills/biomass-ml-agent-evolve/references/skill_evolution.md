# Skill Evolution

## Purpose
Define a bounded, opt-in mode in which the skill reflects on completed experiments and proposes incremental, conservative updates to its own `references/`, `assets/`, and `scripts/` files. This mode never edits skill files automatically; it produces proposals that a human reviewer must approve. The goal is steady, evidence-driven refinement without architectural drift.

## Triggering conditions
Run this mode ONLY when all of the following are true:

- The current experiment has reached a stop condition: target met, benchmark met, max iterations reached, compute/runtime budget exhausted, or user escalation closed.
- No in-flight Actor or Critic stage is open.
- The Orchestrator has recorded the experiment as `COMPLETE` or `ESCALATED_CLOSED` in `IMPLEMENTATION_PLAN.md`.

Never trigger:

- Mid-pipeline, between Actor stages.
- While any rejected artefact is still being revised.
- During an active rerun of a pipeline suffix.

The mode must never edit in-flight artefacts under `experiments/{experiment_id}/`. It reads them, it does not modify them.

## Required inputs
The reflection actor must read all of:

- `experiments/{experiment_id}/IMPLEMENTATION_PLAN.md` (latest)
- `experiments/{experiment_id}/evaluation/evaluation_matrix.yaml`
- `experiments/{experiment_id}/error_analysis/error_analysis.md`
- `experiments/{experiment_id}/reports/improvement_plan.md`
- Current contents of `skills/biomass-ml-agent-evolve/references/` and `skills/biomass-ml-agent-evolve/assets/`
- Existing ledger if present. The file at `skills/biomass-ml-agent-evolve/assets/skill_evolution_ledger.template.md` is the **template only** — copy it to a project-level live ledger at `skill_evolution_ledger.md` (one per skill instance) on the first evolution pass, and append to that live copy thereafter. Do not append to the template itself.

If any required input is missing, write a blocker section in the proposal artefact and stop.

## Procedure
1. **Reflection diff.** Run a reflection actor that compares the accepted artefacts of the experiment against the acceptance gates in each relevant reference file. Produce a short list of drift findings: each finding names the reference file, the gate, the artefact, and the observed deviation or unmet need.
2. **Targeted proposals.** Draft at most 5 proposals. Each proposal contains:
   - one-line summary
   - exact target file under `skills/biomass-ml-agent-evolve/`
   - before-and-after snippet showing the precise edit
   - rationale tied to one or more drift findings
   - supporting evidence file path inside `experiments/{experiment_id}/`
3. **Evidence rule.** Every proposal must cite at least one experiment artefact as evidence. Proposals without evidence are dropped before critic review.
4. **Write the proposal artefact.** Write `experiments/{experiment_id}/reports/skill_evolution_proposal.md`. Append a row to the project-level live ledger `skill_evolution_ledger.md` (initialise it from `assets/skill_evolution_ledger.template.md` if it does not yet exist) recording experiment_id, date, number of proposals, and ledger row status `PENDING_REVIEW`. Never write to the template file itself.
5. **Skill Evolution Critic.** A dedicated Critic reviews the proposal artefact and returns `ACCEPT` only if every proposal satisfies all of:
   - (a) cites at least one experiment artefact as evidence
   - (b) is minimal (no architectural rewrites, no whole-file replacements)
   - (c) does not weaken any existing acceptance gate
   - (d) does not introduce new external dependencies without explicit justification
   - (e) preserves the Orchestrator/Actor/Critic contract
6. **Human-in-the-loop gate.** The Orchestrator presents accepted proposals to the user and requests explicit approval BEFORE editing any file under `skills/biomass-ml-agent-evolve/`. No silent edits. Approved edits are applied one proposal at a time; the ledger row is updated to `APPLIED` or `REJECTED_BY_USER`.

## Required output
`experiments/{experiment_id}/reports/skill_evolution_proposal.md` with sections:

- **Summary** — one paragraph stating what this evolution pass proposes and why.
- **Evidence Snapshot** — bulleted list of the experiment artefacts cited.
- **Proposals** — one row per proposal with the five fields from step 2.
- **Risks & Counterarguments** — explicit risks of adopting each proposal.
- **Recommendation** — overall stance: adopt all, adopt some, defer, or reject.

## Caps
- Maximum 5 proposals per evolution pass.
- Maximum 1 evolution pass per experiment.
- Maximum 3 evolution passes per calendar month per skill instance.

If a cap is reached, the mode writes a short note in the ledger and exits without producing a new proposal.

## Forbidden changes
The reflection actor and Critic must reject proposals that:

- delete any acceptance gate in any reference file
- relax leakage controls in `references/database_preprocessing.md`
- reduce the per-stage retry limit below 3
- remove or weaken Critic review for any stage
- introduce new heavy ML dependencies without an explicit user-approved RFC

## Critic addendum
REJECT the proposal artefact if any single proposal:

- lacks cited evidence
- would weaken any existing acceptance gate
- is non-minimal (for example, rewrites a whole reference file or restructures a directory)
- adds an external dependency without justification
- alters the Orchestrator/Actor/Critic contract

A single failing proposal causes the whole pass to be REJECTED; the actor must trim or revise and resubmit, subject to the per-experiment cap.

## Failure modes
- **Drift accumulation** if evolution always defers and no proposals are ever applied.
- **Over-tuning** to a single experiment whose conditions do not generalise.
- **Gate erosion** through repeated small relaxations that individually look harmless.
- **Scope creep** into architectural rewrites disguised as small edits.
- **Ledger neglect** where proposals are produced but never recorded, breaking auditability.
