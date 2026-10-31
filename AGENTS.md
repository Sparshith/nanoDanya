## Paid cloud runs

- Ask the user for a spending budget before launching paid cloud work, unless a budget has already been provided for the current task. Do not ask again within that task.
- Treat the budget as a total cap including startup, compilation, retries, CPU, memory, and GPU time. Use bounded runs and track spending against it.
- Modal pricing is cached in `ops/modal-pricing.json`. Check its date and verify rates before paid runs with a strict budget.
