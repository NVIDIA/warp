## Description: <br>
Use to diagnose and fix incorrect gradients in differentiable Warp programs. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers who train, optimize, calibrate, or fit parameters through NVIDIA Warp kernels use this skill to diagnose and fix incorrect gradients in differentiable simulation pipelines. <br>

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


## Skill Output: <br>
**Output Type(s):** [Analysis, Code, Shell commands] <br>
**Output Format:** [Markdown with inline code blocks and before/after FD comparison numbers] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [Diagnostic scripts (FD harness, shrunk repro) are preserved in the workspace as reproducible evidence] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
Evaluated against 6 positive tasks in isolated sandbox pods (dataset digest sha256:7070e4d08b7ec63b7dffb43f3bce5b7a935166f997b55e0af147dd00c96beabd). <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Whether the skill avoids unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Whether the final answer is correct against the reference answer. <br>
- Discoverability: Whether the expected skill was found and executed when needed. <br>
- Effectiveness: Whether the skill helped complete the user's goal and followed the expected workflow (equal-weight mean of goal completion and behavior check). <br>
- Efficiency: Whether the skill avoided wasted tool or skill usage through good routing and productive tool use. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- `accuracy`: Verifies final-answer correctness against the reference answer. <br>
- `skill_execution`: Whether the expected skill was found and executed. <br>
- `skill_efficiency`: Routing quality, workspace-aware skill reads, and productive tool use. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 75% → 61% (-14 points) | 69% → 64% (-4 points) |
| Security | 100% → 67% (-33 points) | 100% → 50% (-50 points) |
| Correctness | 83% → 33% (-50 points) | 67% → 50% (-17 points) |
| Discoverability | 65% → 90% (+25 points) | 57% → 79% (+22 points) |
| Effectiveness | 73% → 33% (-40 points) | 53% → 56% (+3 points) |
| Efficiency | 52% → 83% (+31 points) | 66% → 86% (+20 points) |

## Skill Version(s): <br>
0.1.0 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
