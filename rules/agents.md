# Agent Orchestration

Rules for when and how to use specialized agents.

## Available Agents

| Agent | Purpose | When to Use |
|-------|---------|-------------|
| planner | Experiment planning | Complex features, multi-step implementations |
| code-reviewer | Code quality review | After writing/modifying code |
| architect | ML architecture design | Architectural decisions, system design |
| tdd-guide | Test-driven development | New features, bug fixes |
| build-error-resolver | Environment issues | Import errors, dependency conflicts |
| ml-trainer | Training workflows | Training setup, optimization |
| cheminformatics-reviewer | Domain review | Molecular data processing, RDKit code |
| refactor-cleaner | Code cleanup | Dead code, duplication |
| doc-updater | Documentation | README, docstrings, API docs |

## Immediate Agent Usage

Use agents PROACTIVELY without explicit user request:

1. **Complex experiment request** → Use **planner** agent
2. **Code just written/modified** → Use **code-reviewer** agent
3. **New feature implementation** → Use **tdd-guide** agent
4. **Training setup** → Use **ml-trainer** agent
5. **Molecular data code** → Use **cheminformatics-reviewer** agent
6. **Architecture decision** → Use **architect** agent

## Parallel Task Execution

ALWAYS use parallel execution for independent operations:

```markdown
# GOOD: Parallel execution
Launch 3 agents in parallel:
1. Agent 1: Review src/models/vae.py
2. Agent 2: Review src/data/dataset.py
3. Agent 3: Review src/training/trainer.py

# BAD: Sequential when unnecessary
First agent 1, wait, then agent 2, wait, then agent 3
```

## Agent Selection Matrix

| Task | Primary Agent | Secondary |
|------|---------------|-----------|
| New model implementation | planner | tdd-guide |
| Bug fix | tdd-guide | code-reviewer |
| Code review | code-reviewer | cheminformatics-reviewer |
| Training issues | ml-trainer | build-error-resolver |
| SMILES processing | cheminformatics-reviewer | code-reviewer |
| Import errors | build-error-resolver | - |
| Documentation | doc-updater | - |
| Code cleanup | refactor-cleaner | code-reviewer |

## Multi-Agent Workflows

### New Feature
1. **planner** → Create implementation plan
2. **tdd-guide** → Write tests, implement
3. **code-reviewer** → Review implementation
4. **doc-updater** → Update documentation

### Code Review
1. **code-reviewer** → General review
2. **cheminformatics-reviewer** → Domain-specific review (if molecular code)

### Training Setup
1. **planner** → Plan experiment
2. **ml-trainer** → Configure training
3. **code-reviewer** → Review configuration
