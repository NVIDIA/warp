## Description: <br>
Diagnoses and fixes incorrect gradients in differentiable NVIDIA Warp programs by measuring autodiff against finite differences, localizing the failure, and applying minimal verified corrections. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers debugging gradient correctness in differentiable simulation, optimization, calibration, or training workflows built on NVIDIA Warp kernels and wp.Tape. <br>

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
- [Quick-Checks Checklist](references/quick-checks.md) <br>
- [Verification Tooling](references/verification.md) <br>
- [Custom Gradients](references/custom-gradients.md) <br>
- [Case Studies](references/case-studies.md) <br>
- [NVIDIA Warp Documentation](https://nvidia.github.io/warp/stable/) <br>
- [NVIDIA Warp GitHub](https://github.com/NVIDIA/warp) <br>


## Skill Output: <br>
**Output Type(s):** [Analysis, Code] <br>
**Output Format:** [Markdown with inline code blocks and diagnostic scripts] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
Evaluated against 6 gradient-debugging tasks (6 positive) in isolated sandbox pods. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Checks final-answer correctness against the reference answer. <br>
- Discoverability: Checks whether the expected skill was found and executed when needed. <br>
- Effectiveness: Checks whether the skill helped complete the user's goal and followed expected workflow behavior. <br>
- Efficiency: Checks routing quality, workspace-aware skill reads, and productive tool use. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was found and executed. <br>
- `skill_efficiency`: Routing quality, workspace-aware skill reads, and productive tool use. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 79% → 93% (+14 points) | 80% → 90% (+11 points) |
| Security | 100% → 100% (±0 points) | 100% → 83% (-17 points) |
| Correctness | 100% → 100% (±0 points) | 100% → 100% (±0 points) |
| Discoverability | 58% → 84% (+26 points) | 60% → 84% (+24 points) |
| Effectiveness | 80% → 94% (+15 points) | 78% → 90% (+13 points) |
| Efficiency | 55% → 86% (+31 points) | 61% → 93% (+33 points) |

## Skill Version(s): <br>
0.1.0 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
