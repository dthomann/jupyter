# How Neuromodulators (NE, DA) and Memory Work in the Brain

## Quick Summary

**NE (Norepinephrine)** = Surprise signal
- **Measures**: World model prediction error magnitude
- **Does**: Boosts learning rate and exploration when surprised
- **Formula**: `LR_factor = 1.0 + 2.0*NE + 1.0*ACh` (up to 5x boost!)

**DA (Dopamine)** = Reward prediction error
- **Measures**: `TD_error = reward + gamma*next_value - current_value`
- **Does**: Modulates exploration (high DA → exploit, low DA → explore)
- **Also**: Determines memory importance (`salience = |DA| + NE`)

**Memory** = Selective consolidation
- **STM**: Last ~10 transitions (working memory)
- **Episodic**: Important experiences (salience > 0.5) stored immediately
- **Consolidation**: All STM → Episodic at episode end
- **Usage**: Similarity-based retrieval (currently available but not actively used)

---

## Overview

The brain uses neuromodulators to **dynamically regulate learning** instead of fixed hyperparameters. Memories are used for **consolidation** and **context retrieval**.

---

## 🧪 Neuromodulators: NE and DA

### **Norepinephrine (NE) - "Surprise Signal"**

**What it measures:**
- NE = magnitude of world model prediction error
- High NE = "I didn't expect this!" (surprise)
- Low NE = "This matches my predictions" (familiar)

**What it does:**

1. **Boosts Learning Rate** (`get_learning_rate_factor()`):
   ```python
   lr_factor = 1.0 + 2.0 * NE + 1.0 * ACh
   ```
   - When surprised (high NE), learning rate increases (up to 5x)
   - When familiar (low NE), learning rate decreases (down to 0.1x)
   - **Example**: If NE=0.5 (moderate surprise), LR is boosted by 2.0x

2. **Boosts Exploration** (`get_exploration_entropy()`):
   ```python
   entropy_factor = 1.0 + 1.0 * NE - 0.5 * (tonic_DA - 0.5)
   ```
   - High surprise → more random exploration
   - **Example**: If NE=0.3, exploration entropy increases by 30%

3. **Gates World Model Plasticity**:
   ```python
   neuromod_factor = 0.5 * NE + 0.5 * ACh
   world_model.learn(x, neuromod_factor, lr_model)
   ```
   - High NE → world model learns more from this experience
   - Low NE → world model barely updates

**In the algorithm:**
- Updated every step: `NE = prediction_error_magnitude`
- Used immediately to modulate learning rates and exploration
- Tracks both phasic (instantaneous) and tonic (long-term average) levels

---

### **Dopamine (DA) - "Reward Prediction Error"**

**What it measures:**
- DA = TD error = `reward + gamma*next_value - current_value`
- Positive DA = "Better than expected!" (pleasant surprise)
- Negative DA = "Worse than expected" (disappointment)
- Zero DA = "Exactly as predicted"

**What it does:**

1. **Modulates Exploration** (`get_exploration_entropy()`):
   ```python
   entropy_factor = 1.0 + 1.0 * NE - 0.5 * (tonic_DA - 0.5)
   ```
   - High tonic DA (doing well) → less exploration (exploit)
   - Low tonic DA (struggling) → more exploration (try new things)
   - **Example**: If tonic_DA=0.8 (doing well), exploration decreases

2. **Used in Memory Salience**:
   ```python
   salience = abs(DA) + NE
   ```
   - High |DA| (big reward or big disappointment) → important memory
   - Combined with NE to determine what gets stored long-term

**In the algorithm:**
- Updated every step based on TD error
- Tracks both phasic (instantaneous RPE) and tonic (long-term mood)
- Tonic DA slowly integrates: `tonic_DA = 0.99 * tonic_DA + 0.01 * (0.5 + tanh(RPE))`

---

## 🧠 Memory System: STM → Episodic Consolidation

### **Short-Term Memory (STM)**

**What it is:**
- Buffer holding last ~10 transitions
- Like "working memory" - what you're thinking about right now
- Capacity: 10 items (configurable)

**What happens:**
- Every experience goes into STM immediately
- STM is cleared at end of episode

**Code:**
```python
self.hippocampus.stm.add(transition)  # Every step
```

---

### **Episodic Memory (Long-Term)**

**What it is:**
- Permanent storage of important experiences
- Capacity: 100,000 transitions
- Stores: `(obs, state, action, reward, next_obs, next_state, done)`

**How experiences get stored:**

1. **Immediate Consolidation** (high salience):
   ```python
   salience = abs(DA) + NE
   if salience > 0.5:
       episodic.store(transition, importance=salience)
   ```
   - Big reward (high |DA|) → stored immediately
   - High surprise (high NE) → stored immediately
   - **Example**: Winning move (DA=+1.0) → salience=1.0 → stored immediately

2. **End-of-Episode Consolidation**:
   ```python
   hippocampus.consolidate_stm()  # At episode end
   ```
   - Everything in STM moves to episodic memory
   - Default importance: 0.5
   - STM is cleared

**How memories are used:**

1. **Similarity-Based Retrieval** (context):
   ```python
   similar_memories = hippocampus.recall(current_state, k=1)
   ```
   - Finds k most similar past states (Euclidean distance)
   - Could be used for "one-shot" adaptation (currently commented out)
   - **Future use**: Bias action selection based on past similar situations

2. **Replay Sampling** (offline learning):
   ```python
   batch = hippocampus.sample_replay(batch_size=32)
   ```
   - Uniform sampling from episodic memory
   - Used for offline replay (currently placeholder)
   - **Future use**: "Dreaming" - replay past experiences during rest

---

## 🔄 Complete Learning Loop Flow

### Timing Diagram:

```
Step N-1                    Step N                      Step N+1
─────────────────────────────────────────────────────────────────
NE[N-1] = 0.3              NE[N] = 0.5                 NE[N+1] = ?
  │                          │                            │
  │                          │                            │
  └─→ Gates Step N ──────────┼─→ Gates Step N+1 ─────────┘
     learning                 │     learning
                              │
                              ↓
                    World Model learns
                    (using NE[N-1])
                              │
                              ↓
                    pred_error_norm = 0.5
                              │
                              ↓
                    NE[N] = 0.5 (updated!)
                              │
                              └─→ Used for Step N+1
```

**Key Point**: NE from step N-1 gates learning in step N, then NE[N] is computed from step N's prediction error and used for step N+1.

### Step-by-Step (per transition):

1. **Observe** → Encode state (x, z)

2. **Act** → Policy selects action (exploration modulated by previous NE+DA)

3. **Receive reward** → External reward from environment

4. **World Model learns** (gated by PREVIOUS NE+ACh):
   ```python
   neuromod_factor = 0.5 * previous_NE + 0.5 * previous_ACh
   pred_error_norm = world_model.learn(x, neuromod_factor, lr_model)
   ```
   - Uses NE from last step to gate plasticity
   - Computes NEW prediction error
   - **NEW NE** = `pred_error_norm` (surprise from this step)

5. **Intrinsic Motivation**:
   - Curiosity = NEW_NE * scale
   - Competence = improvement in prediction error

6. **Drives update**:
   - Energy depletes with action
   - Restored by reward
   - Drive multiplier scales external reward

7. **Total reward** = `external_reward * drive_multiplier + intrinsic_reward`

8. **Neuromodulators update**:
   ```python
   DA = TD_error = reward + gamma*next_value - current_value
   NE = pred_error_norm  # From step 4
   ACh = decay * previous_ACh + (1-decay) * NE  # Running average
   ```
   - DA = Reward Prediction Error (RPE)
   - NE = Current prediction error magnitude
   - ACh = Running average of NE (expected uncertainty)
   - Serotonin = Reacts to punishment
   - Cortisol = Accumulates stress

9. **Policy updates** (using NEW neuromodulator values):
   ```python
   lr_factor = 1.0 + 2.0 * NEW_NE + 1.0 * NEW_ACh
   actual_lr = base_lr * lr_factor
   
   entropy_factor = 1.0 + 1.0 * NEW_NE - 0.5 * (tonic_DA - 0.5)
   actual_entropy = base_entropy * entropy_factor
   
   gamma = base_gamma + 0.09 * (tonic_serotonin * 2 - 1)
   ```
   - Learning rate boosted by surprise (NE) and uncertainty (ACh)
   - Exploration boosted by surprise (NE), reduced by success (DA)
   - Discount factor increased by patience (Serotonin)

10. **Memory processing**:
    ```python
    transition = (x, z, action, total_reward, x_next, z_next, done)
    salience = abs(NEW_DA) + NEW_NE
    if salience > 0.5:
        episodic.store(transition, importance=salience)  # Immediate storage
    stm.add(transition)  # Always goes to STM
    ```
    - Transition added to STM (always)
    - If salience (|DA| + NE) > 0.5 → immediate episodic storage
    - At episode end → consolidate all STM to episodic

---

## 📊 Concrete Example: Tic-Tac-Toe

**Scenario**: Agent makes a winning move

1. **Before move**: Value estimate = 0.3 (moderate position)
2. **After move**: Reward = +1.0 (win!), next_value = 0.0 (terminal)
3. **TD Error**: `1.0 + 0.99*0.0 - 0.3 = 0.7`
4. **DA spikes**: `DA = 0.7` (big positive RPE!)
5. **NE**: Moderate (prediction error from world model)
6. **Salience**: `|0.7| + NE ≈ 0.8` → **Stored immediately in episodic memory**
7. **Learning rate**: Boosted by NE (surprise) → learns faster from this win
8. **Exploration**: Reduced (high DA → exploit this strategy)

**Scenario**: Agent makes a losing move

1. **TD Error**: `-1.0 + 0.99*0.0 - 0.2 = -1.2`
2. **DA spikes negative**: `DA = -1.2` (big disappointment!)
3. **Salience**: `|-1.2| + NE ≈ 1.3` → **Stored immediately** (learn from mistakes!)
4. **Learning rate**: Boosted (high NE from surprise)
5. **Exploration**: Increased (low DA → try different strategies)

---

## 🎯 Key Insights

1. **NE (Surprise) = Learning Rate Modulator**
   - High surprise → learn faster
   - Low surprise → learn slower (or not at all)

2. **DA (RPE) = Exploration Modulator**
   - High DA → exploit (do more of this)
   - Low DA → explore (try something else)

3. **Memory = Selective Consolidation**
   - Only important experiences (high salience) stored long-term
   - Salience = |DA| + NE (reward OR surprise = important)
   - Everything else stays in STM until episode end

4. **No Fixed Hyperparameters**
   - Learning rate adapts to surprise (NE)
   - Exploration adapts to performance (DA)
   - The brain self-regulates!

---

## 🔬 Current Limitations & Future Enhancements

**Current:**
- Memory retrieval (`recall()`) exists but isn't actively used for action selection
- Offline replay (`sample_replay()`) is placeholder
- Similarity search is naive (Euclidean distance) - slow for large buffers

**Future possibilities:**
- Use retrieved memories to bias action selection (episodic control)
- Implement offline replay during "rest" periods
- Use priority-based sampling (TD error priorities)
- Add semantic memory (generalized knowledge vs. specific episodes)

