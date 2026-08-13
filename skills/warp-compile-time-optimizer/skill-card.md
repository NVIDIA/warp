## Description: <br>
Use when compile time or startup time is the problem in code that uses Warp: diagnoses and reduces JIT compilation overhead, module identity churn, and cold-start latency for Warp kernels. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers using NVIDIA Warp who need to diagnose and reduce JIT compilation time, cold-start latency, or module rebuild overhead in their GPU-accelerated Python applications. <br>

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
- [Measurement Protocol](references/measurement.md) <br>
- [Compile-Time Mechanisms Reference](references/mechanisms.md) <br>
- [NVIDIA Warp Documentation](https://nvidia.github.io/warp/stable/) <br>


## Skill Output: <br>
**Output Type(s):** [Analysis, Code, Shell commands] <br>
**Output Format:** [Markdown with inline code blocks and measurement tables] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
18 evaluation tasks (18 positive) executed in isolated sandbox pods, covering safety, correctness, discoverability, effectiveness, and efficiency. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Verifies final-answer correctness against reference answers. <br>
- Discoverability: Whether the expected skill was found and executed when needed. <br>
- Effectiveness: Whether the skill helped complete the user's goal and expected workflow (equal-weight mean of goal completion and behavior adherence). <br>
- Efficiency: Routing quality, workspace-aware skill reads, and productive tool use without waste. <br>

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
| Overall | Not available | 88% → 95% (+7 points) |
| Security | Not available | 83% → 94% (+11 points) |
| Correctness | Not available | 94% → 100% (+6 points) |
| Discoverability | Not available | 84% → 89% (+5 points) |
| Effectiveness | Not available | 90% → 95% (+5 points) |
| Efficiency | Not available | 89% → 96% (+8 points) |

## Skill Version(s): <br>
0.1.0 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
