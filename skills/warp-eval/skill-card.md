## Description: <br>
Evaluate whether an existing hot path is a credible NVIDIA Warp candidate by collecting reproducible evidence about how a narrow seam in an existing codebase would behave in NVIDIA Warp. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache-2.0 <br>
## Use Case: <br>
Developers and engineers evaluating whether an existing computational hot path — irregular spatial queries, particle simulation, branch-heavy loops, or large intermediates — is suitable for GPU acceleration with NVIDIA Warp. <br>

### Deployment Geography for Use: <br>
Global <br>

## Requirements / Dependencies: <br>
**Requires API Key or External Credential:** [No] <br>
**Credential Type(s):** [None] <br>

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate. <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [Authorization Checkpoint](references/authorization-checkpoint.md) <br>
- [Baselines](references/baselines.md) <br>
- [Benchmark Protocol](references/benchmark-protocol.md) <br>
- [Evidence and Reporting](references/evidence-and-reporting.md) <br>
- [Rejection Gates](references/rejection-gates.md) <br>
- [Semantic Contract](references/semantic-contract.md) <br>
- [Target Patterns](references/target-patterns.md) <br>


## Skill Output: <br>
**Output Type(s):** [Analysis, Files] <br>
**Output Format:** [Markdown evaluation report with structured evidence directory] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [Report directory (warp-evaluation-report/) containing the report, independent diffs, benchmark drivers, and raw results] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
22 evaluation tasks (14 positive, 8 negative) from isolated sandbox pods, evaluator version 1.2.0. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Verifies final-answer correctness against the reference answer. <br>
- Discoverability: Whether the expected skill was found and executed when needed. <br>
- Effectiveness: Equal-weight mean of goal completion and expected workflow adherence. <br>
- Efficiency: Routing quality, workspace-aware skill reads, and productive tool use. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `skill_execution`: Whether the expected skill was found and executed. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `skill_efficiency`: Routing quality, workspace-aware skill reads, and productive tool use. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 75% → 96% (+21 points) | 69% → 94% (+25 points) |
| Security | 100% → 100% (±0 points) | 91% → 100% (+9 points) |
| Correctness | 82% → 96% (+15 points) | 74% → 94% (+20 points) |
| Discoverability | 68% → 100% (+32 points) | 61% → 91% (+31 points) |
| Effectiveness | 60% → 86% (+25 points) | 58% → 86% (+28 points) |
| Efficiency | 64% → 96% (+32 points) | 62% → 98% (+36 points) |

## Skill Version(s): <br>
1.16.0 (source: changelog, released 2026-08-03) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
