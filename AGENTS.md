# ENN-PyTorch Agent Entry Point

Read `docs/CONTEXT.md` first. It is the canonical source of truth for AI-facing repository context and instructions.

Do not change production Python code, dependencies, lockfiles, secrets, or generated artifacts unless the user explicitly asks for that change.

AI-facing documentation must be kept current. Before finishing every task, review whether the change affects AI-facing documentation. If it affects repository structure, architecture, public APIs, package boundaries, configuration, dependencies, test commands, compatibility contracts, runtime safety rules, artifact handling, documented workflows, or current-state classification, update the affected AI-facing documents in the same PR. Do not defer required updates to a follow-up task, and do not edit unrelated AI-facing documents merely to create churn.

Use these focused references instead of duplicating details here:

- Repository context: `docs/CONTEXT.md`
- Current state: `docs/CURRENT_STATE.md`
- Testing guidance: `docs/TESTING.md`
- Runtime and artifact safety: `docs/RUNTIME_SAFETY.md`
- Change checklist: `docs/CHANGE_CHECKLIST.md`

Final reports must include exactly one AI documentation impact result:

```text
AI docs updated:
- <documents updated>
```

or

```text
AI docs impact: none
Reason: <concrete reason>
```
