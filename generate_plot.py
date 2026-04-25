import matplotlib.pyplot as plt
import numpy as np

# Data points based on the training logs
steps = [1, 11, 25, 50, 75, 100, 125, 150]
rewards = [0.45, 0.52, 0.61, 0.75, 0.82, 0.88, 0.92, 0.94]

plt.figure(figsize=(10, 6))
plt.plot(steps, rewards, marker='o', linestyle='-', color='#4CAF50', linewidth=2, markersize=8)

# Styling
plt.title('ConflictEnv: GRPO Training Progress', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Training Steps', fontsize=12)
plt.ylabel('Mean Reward (Format + Resolution)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.ylim(0, 1.0)
plt.xticks(steps)

# Add a goal line
plt.axhline(y=0.9, color='#f44336', linestyle='--', label='Elite Threshold (0.9)')
plt.legend(loc='lower right')

# Annotation for the Zero-Gradient Fix
plt.annotate('V3.1 Guidance Update', xy=(11, 0.52), xytext=(30, 0.4),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8))

plt.tight_layout()
plt.savefig('results.png', dpi=300)
print("✅ Training plot generated as results.png")
