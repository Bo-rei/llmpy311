# Architecture figure prompt

Create a publication-ready vector figure for an ACL-style paper on open intent detection. Use the visual language of the method figure in `s2c/paper/oos_intent论文.pdf`: a clean horizontal flow, restrained colors, thin arrows, compact labels, and a white background. The figure must be readable in a two-column paper at approximately 7.2 inches wide.

## Main content

Show one left-to-right inference pipeline:

```text
Input utterance x
        ↓
MiniLM Gate
        ├── OOS → Reject
        └── Known → Router → Expert → Known intent
```

Use the following exact English labels:

- `Input utterance`
- `MiniLM Gate`
- `Known / OOS`
- `Reject as OOS`
- `Domain Router`
- `Intent Expert`
- `Known intent`

Inside or immediately below the Gate box, add only these compact annotations:

- `all-MiniLM-L6-v2`
- `384-d normalized embedding`
- `K = 1`
- `Diagonal Mahalanobis`
- `score ≤ 1: Known`
- `score > 1: OOS`

Use a small side panel above or below the Gate, connected by a light dashed arrow, to show the training contract:

```text
Known training utterances
        ↓
Trainable last 2 Transformer blocks
        + residual projection (384 → 256 → 384)
        ↓
Known calibration checkpoint selection
```

The side panel must explicitly say `Known-only training`; do not draw OOS samples entering the training path. Add a small note `Router / Expert: LoRA` under the corresponding downstream boxes. Do not show internal loss formulas, optimizer details, or parameter counts in the main figure.

## Style and color

- Use a restrained academic palette: dark navy for the Gate and main flow, muted blue for Known, muted rose/red for OOS rejection, and neutral gray for training annotations.
- Use rounded rectangles with consistent corner radius and equal padding.
- Use solid arrows for inference and dashed arrows only for training or explanatory links.
- Use one simple circle or dot for the Known branch and one simple diamond or tag for the OOS branch. Do not use decorative icons, 3D effects, gradients, shadows, or cartoon characters.
- Make the branch point visually unambiguous: the OOS path terminates at `Reject as OOS`; only the Known path continues to Router and Expert.
- Use a single sans-serif font, sentence case, and no Chinese text.
- Keep the diagram flat and balanced. Avoid dense legends and avoid putting labels on top of arrows.

## Caption draft

`Overview of the proposed cascade. A Known-only trainable MiniLM Gate performs binary OOS rejection with a K=1 distance boundary. Only inputs accepted as Known are passed to the LoRA-based Domain Router and Intent Expert.`

## Negative instructions

Do not depict multi-centroid inference as the main method. Do not imply that Router or Expert performs OOS detection. Do not show OOS data in the training branch. Do not write `multi-cluster is always better`, `SOTA`, `strictly separable`, or any unsupported performance claim. Do not add extra agents or a multi-agent LLM loop.
