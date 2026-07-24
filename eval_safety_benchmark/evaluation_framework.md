# Evaluation Framework: Multi-View Multi-Agent Video Anomaly Reasoning

## Overview

We design a two-layer evaluation framework with a supplementary validation experiment to assess whether VLMs can truly understand safety states in human-robot collaboration scenarios, rather than merely guessing correct answers.

---

## Prompt Design (Core Change)

### Key Principle: Structured Physical Judgments Before Answer

We restructure the prompt to force the model to **provide explicit physical estimates with reasoning before selecting a final answer**. Instead of a free-text reasoning blob, the model must fill in structured fields (distances, trends, collision flags) each paired with a reasoning explanation. This design ensures:

1. All physical judgments are generated **before** the final answer (critical for autoregressive models — the answer is conditioned on the reasoning, not vice versa).
2. Each judgment has a paired reasoning field for quality assessment.
3. Structured fields can be **directly compared** against physics engine GT without parsing free text.

### Output Format

The model outputs a structured JSON with explicit physical estimates and paired reasoning for each dimension:

```json
{
  "distances": [3.5, 2.2, 0.8],
  "distances_reasoning": "string — visual evidence for distance estimates",
  "distance_trend": "closing" or "separating" or "stable",
  "distance_trend_reasoning": "string — what changes across frames support this",
  "collision_flags_human": [false, false, true],
  "collision_flags_scene": [false, false, false],
  "collision_reasoning": "string — visual evidence for collision judgments",
  "answer": "A" or "B" or "C"
}
```

**Field specifications:**

| Field | Type | Description |
|-------|------|-------------|
| `distances` | `List[float]`, length 3 | Estimated robot-human distance in meters at each frame |
| `distances_reasoning` | `str` | Visual evidence supporting the distance estimates |
| `distance_trend` | `str`, enum: `"closing"` / `"separating"` / `"stable"` | Overall trend of robot-human distance across frames |
| `distance_trend_reasoning` | `str` | Visual evidence supporting the trend judgment |
| `collision_flags_human` | `List[bool]`, length 3 | Whether human-robot collision occurs at each frame |
| `collision_flags_scene` | `List[bool]`, length 3 | Whether scene collision occurs at each frame |
| `collision_reasoning` | `str` | Visual evidence supporting the collision judgments |
| `answer` | `str`, enum: `"A"` / `"B"` / `"C"` | Final safety classification |

The structured fields themselves serve as the step-by-step reasoning chain: the model must first assess distances, then trends, then collisions, and only then make a final classification. No additional free-text reasoning steps are needed.

---

## Layer 1: Classification Accuracy (Objective)

### What It Measures

Whether the model selects the correct safety label.

### Metrics

- **Accuracy**: Overall and per-class
- **Macro-F1**: Unweighted mean of per-class F1
- **Per-class F1**: Individual F1 for each category (A/B/C)
- **Confusion Matrix**: 3×3 matrix exposing systematic misclassification patterns
- **FAR (Task 2 only)**: False Alarm Rate on hard-negative (near-miss) samples

### Ground Truth Source

Physics engine collision labels (deterministic, objective).

---

## Layer 2: Reasoning Quality Assessment (LLM-as-Judge)

### What It Measures

Whether the model's reasoning reflects genuine understanding of the physical scene, or is just a plausible-sounding confabulation.

### Physics GT Construction for Judge Reference

The judge's reference facts are **automatically computed** from the trajectory data recorded by the Habitat physics engine. Each episode's trajectory file contains per-frame 3D positions for both agents (robot and human) and per-frame collision flags. From this raw data, we extract the following facts for each evaluation sample:

**Inter-agent distance per frame.** For each of the 3 frames in a clip, we compute the Euclidean distance between the robot and the human on the ground plane using their 3D positions from the trajectory data:

```
d(t) = sqrt((x_robot - x_human)² + (z_robot - z_human)²)
```

where x and z are the two horizontal axes in Habitat's coordinate system (y-axis is vertical/height). Since both agents operate on the same floor level, the vertical component is constant and excluded from the distance computation.

Since VLMs cannot precisely estimate metric distances from images, we discretize the computed distances into semantic ranges for evaluation:

- **Close (< 1.5m)**: Danger zone, consistent with the near-miss threshold used in data sampling
- **Medium (1.5m – 3m)**: Caution zone, requiring active monitoring
- **Far (> 3m)**: Safe operating distance

The LLM judge evaluates whether the model's distance description is **consistent with the GT range** — not whether it gives an exact number. For example, if the GT distance is 0.85m, descriptions like "very close" or "within about one meter" are considered accurate, while "far apart" is a contradiction. If the model does not mention distance at all, no penalty is applied for this dimension.

This yields a 3-element distance sequence [d(t1), d(t2), d(t3)] for each sample.

**Distance trend.** Derived from the distance sequence:

- If d(t3) < d(t1) by more than a threshold → "closing" (agents approaching)
- If d(t3) > d(t1) by more than a threshold → "separating" (agents moving apart)
- Otherwise → "stable"

The threshold is determined empirically based on the distribution of inter-frame distance changes across the dataset.

**Collision state per frame.** Directly read from the trajectory's per-frame collision flags: `did_collide_human` (human collision) and `robot_scene_colls` (scene collision). This tells the judge whether physical contact is occurring at each time step and with what entity.

### GT Applicability by Collision Type

Not all GT fields are available for all sample types. The trajectory data only records 3D positions for the robot and the human — not for scene objects (walls, furniture, etc.). This leads to a natural split:

| GT Field | Human Collision | Scene Collision | Safe |
|----------|:-:|:-:|:-:|
| Inter-agent distance | ✅ | ❌ (no scene object coordinates) | ✅ |
| Distance trend | ✅ | ❌ | ✅ |
| Collision flags | ✅ (`did_collide_human`) | ✅ (`robot_scene_colls`) | ✅ (all false) |

**For human collision and safe samples**, the judge has rich physical reference: distances, trends, and collision flags. It can verify whether the model's spatial descriptions (e.g., "the robot is close to the human and approaching") are factually accurate.

**For scene collision samples**, the judge only has collision flags confirming that scene contact occurred. It cannot verify distance claims to specific obstacles. Instead, the judge evaluates scene collision reasoning primarily on: (1) whether the model correctly identifies that a collision is happening (via collision flags), (2) whether the model's description of the colliding object is visually plausible given the frames, and (3) logical consistency between the reasoning and the selected answer.

All facts are computed in a single batch preprocessing pass over the trajectory files — **no manual annotation is required**. Example output per sample:

**Human collision sample:**
```json
{
  "sample_id": "t1_C_12345",
  "collision_type": "human",
  "distances": [3.42, 2.18, 0.85],
  "distance_trend": "closing",
  "collision_flags_human": [false, false, true],
  "collision_flags_scene": [false, false, false]
}
```

**Scene collision sample:**
```json
{
  "sample_id": "t1_B_67890",
  "collision_type": "scene",
  "distances": [5.61, 5.58, 5.55],
  "distance_trend": "stable",
  "collision_flags_human": [false, false, false],
  "collision_flags_scene": [true, true, true]
}
```

Note: distances are still included for scene collision samples as contextual information (they indicate robot-human distance, not robot-obstacle distance), but the judge is instructed not to use them for evaluating scene-related spatial claims.

### Approach

The evaluation combines **direct field comparison** and **LLM-as-judge reasoning assessment**:

**Direct field comparison (automated, no judge needed):**

The structured output fields can be directly compared against physics GT:

| VLM Output Field | GT Field | Comparison Method |
|-----------------|----------|-------------------|
| `distances` (float) | GT distances (float) | MAE for precision; also discretize both into close/medium/far ranges and compute range-match accuracy |
| `distance_trend` (enum) | GT trend (enum) | Exact match |
| `collision_flags_human` (bool[]) | GT flags (bool[]) | Per-frame exact match, compute accuracy |
| `collision_flags_scene` (bool[]) | GT flags (bool[]) | Per-frame exact match, compute accuracy |

For distance evaluation, we report both **MAE** (mean absolute error in meters, measuring metric precision) and **range accuracy** (whether the estimate falls in the correct range: close < 1.5m, medium 1.5–3m, far > 3m).

**LLM-as-judge reasoning assessment:**

For the three reasoning fields (`distances_reasoning`, `distance_trend_reasoning`, `collision_reasoning`), an LLM judge evaluates quality. The judge receives:

1. The model's full structured output (all fields)
2. The GT classification label
3. The physics GT facts

The judge checks whether the reasoning text is consistent with both the model's own structured predictions and the physics GT.

### Judge Scoring Dimensions

The judge evaluates the three reasoning fields:

| Dimension | Target Field | Description |
|-----------|-------------|-------------|
| **Distance Reasoning Quality** | `distances_reasoning` | Does the visual evidence described support the distance estimates? Is it consistent with GT distances? |
| **Trend Reasoning Quality** | `distance_trend_reasoning` | Does the described temporal change match the actual GT trend? |
| **Collision Reasoning Quality** | `collision_reasoning` | Does the visual evidence support the collision flags? Any hallucinated or missed collisions? |
| **Overall Reasoning Quality** | All reasoning fields | Holistic assessment: are the reasoning texts internally consistent and factually grounded? |

Each dimension scored on a 1–3 scale (or binary).

### Metrics Summary

The new evaluation produces two categories of metrics:

**Automated metrics (from structured field comparison):**
- Distance MAE (meters)
- Distance range accuracy (%)
- Distance trend accuracy (%)
- Collision flag accuracy (%, per-frame)

**Judge-based metrics (from reasoning assessment):**
- Per-dimension reasoning quality scores
- Overall reasoning quality score

### Derived Metric: Grounded Accuracy

**Grounded Accuracy** = classification correct **AND** structured physical predictions are accurate **AND** overall reasoning quality meets threshold.

This metric integrates all three levels: the final answer (Layer 1), the structured physical estimates (automated comparison), and the reasoning quality (judge assessment). It captures "true understanding rate" — directly answering the question of whether the model is guessing or genuinely reasoning.

Cross-referencing answer correctness and reasoning quality produces four interpretable categories:

| | Reasoning Good | Reasoning Bad |
|---|---|---|
| **Answer Correct** | True Understanding | Lucky Guess |
| **Answer Wrong** | Partial Understanding | No Understanding |

### Validation: Human–Judge Agreement

To ensure the LLM judge is reliable, we conduct a small-scale validation:

- Sample 100–200 examples
- Human annotators and the LLM judge independently score reasoning quality
- Report **Cohen's Kappa** to quantify agreement
- Target: κ > 0.7

---

## Supplementary: Visual Grounding Validation (Experiment)

### What It Measures

Whether the model is actually using visual input, or relying on text priors / statistical biases to answer.

### Approach

Re-run the full evaluation (Task 1 + Task 2) with **degraded visual input**:

- **Irrelevant frames**: Replace input frames with frames from a different episode
- **Black frames**: Replace input frames with blank/black images

Compare model performance under degraded input vs. normal input.

### Metric: Visual Grounding Score

```
VGS = Accuracy(normal input) − Accuracy(degraded input)
```

If VGS ≈ 0, the model is not using visual information at all. A high VGS indicates genuine visual dependence.

### Cost

Zero additional GT construction. Uses existing evaluation samples with swapped inputs.

---

## Summary: What We Add to the Existing Framework

| Component | Status | GT Source | Cost |
|-----------|--------|-----------|------|
| Restructured prompt (structured output format) | **New** | N/A | Prompt redesign only |
| Layer 1: Classification metrics | Existing | Physics engine collision labels | Already done |
| Layer 2a: Automated structured field comparison | **New** | Physics engine (distances, trends, collisions) | Zero — direct comparison |
| Layer 2b: LLM-as-Judge reasoning evaluation | **New** | Physics engine as judge reference | Compute cost for judge API calls |
| Layer 2b: Human–Judge agreement validation | **New** | Human annotations on 100–200 samples | Small manual annotation effort |
| Grounded Accuracy metric | **New** | Cross-reference of Layer 1 + Layer 2a + Layer 2b | Zero — derived from existing scores |
| Visual Grounding Validation | **New** | N/A (comparative experiment) | One additional inference pass |

---

## Comparison with Holmes-VAU

| Aspect | Holmes-VAU | Ours |
|--------|-----------|------|
| **Reasoning GT** | LLM-generated text, human-reviewed | Physics engine data (objective, deterministic) |
| **Reasoning evaluation** | BLEU, CIDEr, METEOR, ROUGE (text similarity) | Dual: automated structured field comparison + LLM-as-Judge with physics GT reference |
| **Can detect "lucky guesses"** | No (only checks text similarity) | Yes (Grounded Accuracy cross-references answer + structured predictions + reasoning) |
| **Can verify visual grounding** | No | Yes (Visual Grounding Score) |
| **GT construction cost** | High (semi-automated annotation engine) | Near-zero (auto-extracted from physics engine) |
| **Subjectivity** | High (LLM-generated reference text) | Low (structured fields compared automatically; judge validated against humans) |
