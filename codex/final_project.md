# Final Project: Reinforcement Learning for Bomberman

> **Source:** [`final_project.pdf`](final_project.pdf), Machine Learning Essentials, Summer Semester 2026. This is a Markdown transcription of the project brief; formatting was normalized for readability.

**Deadline for agent code:** Monday, 21 September 2026, 21:00  
**Deadline for report:** Monday, 28 September 2026, 21:00  
**Questions:** Discord `#final-project-questions`

In this final project, you will use reinforcement-learning techniques to train an agent to play the classic arcade game Bomberman.

The game is played simultaneously by four agents in discrete time steps. Your agent can move, drop bombs, or stand still. Well-placed bombs clear crates and sometimes reveal coins, which can be collected for points. The deciding factor for the final score, however, is blowing up opponents while avoiding being blown up yourself. Special items and power-ups are not available.

After the deadline, trained agents will compete in a tournament with real prizes. Tournament performance contributes to the final grade, but report quality and a systematic, scientific approach to agent design, optimization, and testing carry substantially more weight (see [Report and code repository](#9-report-and-code-repository)).

Develop **two or more different models** and submit the best-performing one to the tournament. Describe all models in the report. Do not split the work so that each team member independently implements a separate model: real teamwork is important.

This is an open project. You may use any combination of methods learned in the semester, and may extend this knowledge with more advanced methods. At least one model should focus on lecture techniques. The one strict rule is that the solution **must involve machine learning**; otherwise it will be rejected.

## 1. Regulations

### Use of AI

You may consult AI to brainstorm ideas, help write code, and draft the report. It must nevertheless be clear in the code and report that the main work was done by the team. Submitting AI drafts without refining them in your own style results in failing the examination.

### Submission

The first submission, the agent code, must contain:

- A directory containing the code and all trained parameters of the best-performing model. This is the subdirectory of `agent_code` in which you develop the agent.

Zip all files into one archive named `final-project-agent-code.zip` and upload it to MaMPF before the deadline.

Set the MaMPF display name and tutorial-group name to your real name, identical to your Muesli name. Join your team's submission with the invitation code before the deadline. See <https://mampf.blog/handing-in-homework-assignments> for instructions.

The second submission, the report, must contain:

- A PDF report describing the approach, of about 4,000 words per team member (not much more). For legal reasons, state the responsible author after headings for each subsection.
- The URL of a public repository containing the complete code base, including every model developed. Mention the URL in the report. Do **not** upload the report to that repository.

You may change groups between homework assignments and the final project. Organize teams, for example via `#homework-team-finding`, and announce them—including a tournament team name—at <https://tinyurl.com/fml-final-project-teams>.

Cheating, including resubmitting a solution from an earlier edition even with small revisions, is forbidden and results in a failed grade.

### Development

To share resources fairly, the final agent must not use multiprocessing. Multiprocessing or any other technique is allowed during training.

Neural networks are permitted. They run on the CPU in official games, although GPUs may be used for training. Account for training time: an unconverged model by the deadline will not be competitive.

You may modify the framework to facilitate training, for example by changing the board. The final agent code will be plugged into the original framework for official games, so changes outside the agent directory will not be present.

Discussion with other teams is encouraged. You may share trained agents (without training code) on `#final-project-beat-my-agent`, and download agents from other teams for testing. Free libraries such as PyTorch are allowed when installable from an official source such as pip or conda. You may not copy an existing solution in whole or in part; plagiarism results in a failed grade.

## 2. Setup

The framework is available at:

```bash
git clone https://github.com/ukoethe/bomberman_rl
```

It includes:

- the game environment and its reinforcement-learning APIs;
- rule-based agents of varying strength and a weak random agent (none can learn);
- a template for a custom agent.

Direct framework questions to `#final-project-questions`. Consider filing a GitHub issue or pull request for framework bugs.

The GUI uses `pygame`; without it the game still runs, but without a GUI. `tqdm` is also recommended to show simulation speed. In the exercise conda environment:

```bash
conda activate ml_homework
pip install pygame tqdm
```

Otherwise a standard Python 3 installation with NumPy, SciPy, and scikit-learn is assumed. Clearly state additional required libraries in the repository and at the beginning of the report.

Watch the strong rule-based agent with:

```bash
cd bomberman_rl
python main.py play
```

## 3. General setting and rules of the game

One to four colored robot agents play in discrete steps. Multiple episodes are played because the game has random elements; the winner is determined by total score.

At each step, a robot can move one tile horizontally or vertically, drop a bomb, or wait. It can move only onto empty black tiles; stone walls, crates, bombs, and other agents block movement. Coins are collectible by moving onto them. Stone-wall placement is fixed each round, but crate and hidden-coin positions vary. Agents start in randomly assigned board corners.

A placed bomb detonates after four steps. Its explosion reaches three tiles up, down, left, and right, stops at stone walls, and does not turn corners. It destroys crates and agents. The explosion remains dangerous for one additional round, then becomes harmless smoke. An agent can place a new bomb after its own prior bomb's explosion is gone.

A fixed number of coins is hidden randomly each episode. Destroying a crate hiding a coin reveals it. A collected coin is worth one point; blowing up an opponent is worth five.

Each episode ends after 400 steps. An agent has **0.5 seconds per step** to decide. If it exceeds this limit, its chosen action is not executed and it waits; the following step's time budget is reduced by the prior excess. Official games provide one thread of an AMD Ryzen 5 2600 and up to 8 GB RAM.

Exact values are in `settings.py`. They may change until seven days before the deadline if major problems are reported; changes will be announced.

## 4. Tasks your agents will have to solve

The most important grading factor is systematic, scientific design, optimization, and testing. Break the overall task into manageable subgoals. At every stage, run meaningful experiments comparing agent variants with one another and supplied agents. Demonstrate improvements resulting from changes motivated by previous tests, define suitable performance metrics, and document the process well.

The following preliminary tasks build on each other; an agent capable of Task 4 should also handle Tasks 1–3.

1. **Coin navigation:** With no crates or opponents, collect revealed coins as quickly as possible. Bombing is unnecessary; learn efficient navigation.
2. **Crates and survival:** With random crates and no opponents, find all hidden coins within the step limit. Learn to destroy crates with bombs, escape bombs without self-destruction, and retain efficient navigation. Bomb escape is crucial.
3. **Hunting:** With crates, hunt and blow up the supplied `peaceful_agent` (easy) and `coin_collector_agent` (hard). The former moves randomly and never drops bombs; the latter drops bombs only to collect coins.
4. **Competition:** With crates, compete for top score against one or more opponents, such as `rule_based_agent` and variants of your own design. In practice, beating `rule_based_agent` is necessary to have a tournament chance.

You may adjust board size and other parameters early in experimentation. Reward shaping may guide desirable states, but auxiliary rewards do not exist in official games, so avoid overfitting to them.

## 5. Framework structure and interface for your agent

### Environment

`environment.py` defines the world and game logic in `BombeRLeWorld`. It tracks the board and objects, runs steps, starts rounds, and optionally references the GUI. It manages agents using the `Agent` class in `agents.py`; each `Agent` loads code from an agent directory.

This framework is fixed during the tournament. Training-time modifications, including board changes, are allowed.

### Agent

Agent code must be in a subdirectory of `agent_code`.

#### Always loaded: `callbacks.py`

Before the first game, the evaluation code in `callbacks.py` is imported. It must provide:

```python
def setup(self): ...
def act(self, game_state: dict): ...
```

`setup(self)` runs once before the first round and initializes persistent state. `self` is passed to all future callbacks, so attributes can be stored on it:

```python
def setup(self):
    self.model = MyModel()

def act(self, game_state: dict):
    return self.model.propose_action(game_state)
```

Pre-set fields include:

| Field | Meaning |
| --- | --- |
| `logger: logging.Logger` | Logger for debugging. |
| `train: bool` | `True` when the model should train; for example, create a fresh model then and load a trained model otherwise. |

`act(self, game_state)` is called once each step. Return one of `'UP'`, `'DOWN'`, `'LEFT'`, `'RIGHT'`, `'BOMB'`, or `'WAIT'`. `WAIT` is the default when the decision exceeds the time limit.

`game_state` contains:

| Key | Type | Meaning |
| --- | --- | --- |
| `round` | `int` | Rounds since environment launch, starting at 1. |
| `step` | `int` | Steps in the current episode, starting at 1. |
| `field` | `np.array(width, height)` | Board: `1` crate, `-1` stone wall, `0` free tile. Coordinates are image coordinates `(x, y)`, so printing it appears transposed relative to the GUI. |
| `bombs` | `[((int, int), int)]` | `((x, y), t)` coordinates and countdown for active bombs; `0` means about to explode. |
| `explosion_map` | `np.array(width, height)` | Number of further steps each tile remains explosive; `0` means no explosion. |
| `coins` | `[(x, y)]` | Collectable coin positions. |
| `self` | `(str, int, bool, (int, int))` | `(name, score, bomb_available, (x, y))` for the agent. A bomb is available when no own bomb is ticking. |
| `others` | `[(str, int, bool, (int, int))]` | Equivalent tuples for living opponents. |
| `user_input` | `str \| None` | GUI input; see practical hints. |

#### Loaded only during training: `train.py`

When training, `train.py` is also imported from the same directory. It must provide:

- `setup_training(self)`, called after `callbacks.setup`, for training-only initialization;
- `game_events_occurred(self, old_game_state, self_action, new_game_state, events)`, called after every non-final step, for learning data and/or experience-buffer updates;
- `end_of_round(self, last_game_state, last_action, events)`, similar to the previous callback but called once after the final step.

Learning may occur in either of the last two callbacks; neither has a time limit.

`events` lists outcomes caused by agent and opponent actions. They can support auxiliary rewards and penalties during training but are unavailable outside training. `events.py` defines:

| Event | Meaning |
| --- | --- |
| `e.MOVED_LEFT`, `e.MOVED_RIGHT`, `e.MOVED_UP`, `e.MOVED_DOWN` | Successfully moved in that direction. |
| `e.WAITED` | Intentionally did not act. |
| `e.INVALID_ACTION` | Selected an unavailable or non-existent action. |
| `e.BOMB_DROPPED` | Successfully dropped a bomb. |
| `e.BOMB_EXPLODED` | A bomb previously dropped by this agent exploded. |
| `e.CRATE_DESTROYED` | Own bomb destroyed a crate. |
| `e.COIN_FOUND` | Own bomb revealed a coin. |
| `e.COIN_COLLECTED` | Collected a coin. |
| `e.KILLED_OPPONENT` | Blew up an opponent. |
| `e.KILLED_SELF` | Blew up self. |
| `e.GOT_KILLED` | Was blown up by an opponent's bomb. |
| `e.OPPONENT_ELIMINATED` | An opponent was blown up. |
| `e.SURVIVED_ROUND` | Reached the end of a round alive. |

Callback order is:

```python
import callbacks
callbacks.setup(...)
if train:
    import train
    train.setup_training(...)

for round in range(n_rounds):
    ...  # Reset world
    for step in range(n_steps):
        callbacks.act(...)
        ...  # Perform agents' actions
        ...  # Evaluate bombs and explosions
        if train and not dead:
            train.game_events_occurred(...)
    if train:
        train.end_of_round(...)
```

To customize GUI appearance and the award ceremony, put 30×30-pixel PNG files named `avatar.png` and `bomb.png` in the agent directory.

## 6. Putting it all together

Create an `agent_code/<agent_name>` subdirectory; its name identifies the tournament agent. Start by copying `tpl_agent`. For example, `agent_code/my_agent/` contains `callbacks.py` and `train.py`; model parameters and other custom files must be stored there too.

Run one agent:

```bash
python main.py play --my-agent my_agent
```

It plays against three strong rule-based agents. To choose all agents explicitly:

```bash
python main.py play --agents my_agent random_agent rule_based_agent peaceful_agent
```

Multiple custom agents, including copies of the same agent, can be used for self-play. Add randomness to action choice when appropriate to avoid rapid convergence to a poor local optimum.

Train with `--train N`, where the first `N` agents passed to `--agents` train. With `--my-agent`, only `N = 0` or `N = 1` works because supplied rule-based agents cannot train:

```bash
python main.py play --agents my_agent random_agent rule_based_agent peaceful_agent --train 1
# or
python main.py play --my-agent my_agent --train 1
```

Training automatically shortens a game until the final training agent dies. Use `--continue-without-training` to prevent that behavior.

## 7. Practical hints to get you started

Use the logger on `self` to monitor and debug agent behavior:

```python
self.logger.info(f"Choosing an action for step {self.game_state['step']}...")
self.logger.debug("This is only logged if s.log_agent_code == logging.DEBUG")
```

Logs are written to a file named for the agent, for example `agent_code/my_agent/logs/my_agent.log`. Reduce agent-code log levels in `settings.py` if logs are too noisy.

`agent_code/rule_based_agent/` implements a strong rule-based player. Its `act` callback is an example of reading game state, logging, and action selection. It may be a training opponent or source of training data, but the submitted agent must itself be trained rather than rule-based.

Use `--save_replay` to store a replay for every episode in `replays/`, and view one with:

```bash
python main.py replay <stored-replay>
```

See all interface options with:

```bash
python main.py [re]play --help
```

Use `--skip-frames` to render fewer frames. For efficient training, use `--no-gui` so the game runs as quickly as agents decide.

For manual GUI control, include `user_agent`; use arrow keys to move, Space to bomb, and Return to wait. This is not efficient training data, but can build intuition for reward design or curricula. Combine it with `--turn-based` to wait for a key press before each action.

Use `--seed` to fix crate and coin placement. It does not fix agent randomness, which has a separate random state.

Predefined scenarios for Tasks 1–4:

```bash
# 1. Coin navigation
python main.py play --no-gui --agents my_agent --train 1 --scenario coin-heaven
# 2. Crates and survival
python main.py play --no-gui --agents my_agent --train 1 --scenario classic
# 3. Hunting
python main.py play --no-gui --agents my_agent peaceful_agent coin_collector_agent --train 1 --scenario classic
# 4. Competition
python main.py play --no-gui --agents my_agent rule_based_agent --train 1 --scenario classic
```

`classic` is the default. You may add scenarios in `settings.py`.

### Successful strategies from previous competition winners

Complexity alone has not won past competitions; carefully designed and trained simple models have succeeded.

**Feature engineering.** Unless a neural network learns features automatically, carefully condense important information into a low-dimensional feature vector. Useful feature classes include situational awareness (for example, a wall to the left), pathfinding (the direction that best approaches the closest coin), and lifesaving information (whether the agent is in a soon-exploding bomb path). A submitted model must learn from its features: a feature that deterministically returns the best action is not allowed.

**Deep learning.** Deep learning can be powerful and has succeeded before, but many models did not converge in time. Reserve substantial time for architecture and training fine-tuning.

**Reward shaping.** Coin and opponent rewards are sparse, so a denser reward signal is likely needed. Start with supplied events, add careful custom events, and test reward values. Reward progress aligned with features (for example, moving toward a coin) but balance it with equally large or larger penalties for opposite movement or waiting, otherwise the agent may oscillate. Many carefully useful dense custom rewards are acceptable. Avoid rewards that create poor local optima: theory says auxiliary rewards should depend only on game states, not the actions leading to them.[^1]

**Custom environment.** Complement reward shaping by changing field size, crate and coin count, timeout, and other values from `settings.py` for training. `coin-heaven` has no crates and more coins. Always test on original settings before submission.

**Hyperparameter optimization.** Do not neglect hyperparameters; performance can differ dramatically and optimization takes time.

**Symmetries.** Use rotational and mirror symmetry to accelerate and strengthen learning—for example, identify horizontally flipped equivalent states/actions or augment data with flips and rotations.

[^1]: *Policy invariance under reward transformations: Theory and application to reward shaping*, <https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf>.

## 8. Submission test

Compatibility support consists of a Docker container with the tournament Python environment and pre-runs on tournament machines. The repository's Dockerfile should be updated with `git pull`.

1. Install Docker Desktop from <https://docs.docker.com/get-docker/>.
2. Verify installation: `docker run hello-world`.
3. In the `bomberman_rl` directory containing `Dockerfile`, build the image: `docker build .`.
4. Find its image ID with `docker images`.
5. If agent code changed after the image was built, copy it into the temporary container:

   ```bash
   docker cp <AGENT_CODE_FOLDER> <CONTAINER_ID>:/home/bomberman/agent_code/<AGENT_NAME>
   ```

6. Run the image using ordinary game commands, for example:

   ```bash
   docker run <IMAGE_ID> python main.py play --agents rule_based_agent tpl_agent --no-gui
   ```

7. Run the game with the intended agent.
8. Find the previous container ID with `docker ps --all`.
9. Copy and inspect the log:

   ```bash
   docker cp <CONTAINER_ID>:/home/bomberman/logs/game.log ./game.log
   ```

Use relative, not absolute, paths for files in other directories.

For pre-runs, upload the ZIP specified in Section 1 to MaMPF. If additional packages are needed beyond the Dockerfile defaults, list them in `requirements.txt`. The organizers will unzip the submission, install those requirements, find the first directory containing `callbacks.py`, copy it into `agent_code`, and run one non-training game against three `random_agent`s. They will send console output, stack traces, game logs, and agent log output.

Submissions uploaded to the MaMPF assignment **Final project submission test** by **17 September 2026, 21:00** will be tested, leaving another weekend to correct crashes before final submission.

## 9. Report and code repository

Put all code in a public GitHub or Bitbucket repository and include its URL in the report. The MaMPF tournament ZIP must contain only the `agent_code` subdirectory holding the fully trained best player.

The report is due one week later. It should have about 4,000 words per team member (not much more), excluding title page and similar front matter. Do not use the university logo for legal reasons. Use this standard scientific structure:

1. **Introduction:** State the problem and why it is worthwhile and challenging.
2. **Background:** Describe the task and possible solution methods, including RL approaches considered.
3. **Project planning:** Describe organization, scheduling, subtask distribution, and collaboration. If using deep learning, explain training hardware allocation (for example, Google Colab).
4. **Methods:** Specialize the background methods for this task, justify crucial choices, list experiment variants, and define systematic performance evaluation (including metrics). Describe abandoned approaches and reasons if useful, cite relevant literature, and recall that at least two agents are required.
5. **Training:** Describe the training process and acceleration techniques, such as self-play, auxiliary reward design, or prioritized experience replay.
6. **Experiments and Results:** Report training-progress diagrams and performance comparisons with custom and supplied agents; discuss observations, difficulties, and resolutions. This is the most important section: systematically show whether design and implementation changes improved performance and which agent was best.
7. **Conclusion:** Summarize findings; explain how the agent and next year's game setup could improve with more time.

Scientific-publication standards and systematic experimentation are crucial for the grade. Be concise and clear enough that other students could replicate the results. Mark every chapter or section with its main author so individual contributions can be graded.
