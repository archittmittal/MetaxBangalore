---
license: mit
library_name: transformers
pipeline_tag: text-generation
tags:
- trl
- grpo
- conflict-env
- reasoning
---

# ConflictEnv Final Reasoning Model

This is the final fine-tuned model for the **ConflictEnv** executive assistant task. 
It has been trained using **GRPO** to handle complex scheduling conflicts with a focus on reasoning-first behavior.

## Usage
Start prompts with `Scenario: ... Details: ...` and expect a `<thought>` block followed by a JSON action.
