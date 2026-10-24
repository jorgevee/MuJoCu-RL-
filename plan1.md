# Project Plan v2: Teach a Robot to Walk From Scratch

Python

MuJoCo / MJX

JAX + PPO

RTX 5090 Cloud

[Model Gallery - MuJoCo Documentation](https://images.openai.com/static-rsc-4/W5PWwB9ZXB5Mur6GmC7UMUDDsFOcTGcw_AOpXtG_z9L8kaPcYv92pTo36zzBygzIHjPqPGISxZJ5Bbm7_F8qDgOAogl29lUnL57yJ6uzTs8XEZ5jmrL1uQCrrq8LPIfboK5DymJQQGc-aQap4MdAaIv-QU3Q002atuwso72BYVM?purpose=inline)

[GitHub - alexeiplatzer/unitree-go2-mjx-rl: Reinforcement Learning quadruped locomotion for a Unitree Go2 robot in Mujoco XLA (MJX) · GitHub](https://images.openai.com/static-rsc-4/xIEj4xYUuMR_Z4xVjkrntRWYyKCpqgICyUI2M2SUoSBAF2SfXnOttd0Usbft3FX-QuQnOxiQWwPgJ_YxlRGVXugl6wnD_UdCk3edChrUwDjkqGmeCf7S_CKCxIPjsQpUnGGp8Ssv2AbWi5bz1clmRbS7l35AtqUNUlrc0MzZYKk?purpose=inline)

[アールティが挑む、国産4足歩行ロボットの未来 | 株式会社アールティのプレスリリース](https://images.openai.com/static-rsc-4/knNSF7HMUXIp13MG3f6qpfoGDf4YKHTK1ninCSeWzWkL6mCnQp44Q0lZO0s3lm1MKxx4saQvy6_1CQ5-2A4dt48dBX2RfSBpRH4tOcSzZAhDLS7b0vUFeEqhnxfSTYolzyOkBLSxKNehtEf1119n_fkPBDP3sgxXL9YVq2r1-RU?purpose=inline)

6

## 1. Project overview

Project name: Learning to Walk: Does Curriculum Learning Improve Robotic Locomotion?

We're building a reinforcement learning system that trains a four-legged robot to stand, balance, walk, and recover from unexpected disturbances using simulated physics.

Instead of hand-programming how the legs should move, we'll let a PPO agent learn joint control through trial and error.

The main difference from our original plan is that we're moving from a simple CPU-trained 2D biped to a GPU-accelerated 3D quadruped, making the project more relevant to modern robotics research.

Final technical decisions

| Component          | Selection                               |
| ------------------ | --------------------------------------- |
| Robot              | Unitree Go1 (initial target)            |
| Simulation         | MuJoCo Playground + MJX                 |
| Language           | Python                                  |
| RL algorithm       | PPO                                     |
| Learning framework | JAX + Brax PPO                          |
| Local development  | RTX 5060 8GB                            |
| Main training      | Rented RTX 5090 32GB                    |
| Initial physics    | Flat terrain, gravity, contact dynamics |
| Research design    | Direct vs fixed vs adaptive curriculum  |
| Primary outcome    | Held-out perturbation recovery          |

I'm selecting the Go1 initially because MuJoCo Playground supports established quadruped locomotion workflows, allowing us to build on a working environment rather than implement an entire robotics simulator interface. Go2 can be a later extension.

MuJoCo Playground currently supports quadruped and biped locomotion and offers JAX and Warp accelerated backends.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://playground.mujoco.org\&sz=32)

MuJoCo Playground

+1



## 2. What are we trying to discover?

Our primary hypothesis:

> A robot trained through progressively harder locomotion tasks will recover from unseen disturbances more reliably than a robot trained directly on the hardest task, given the same total environment-interaction budget.

We'll test two related questions:

1. Learning efficiency: Does curriculum learning reach reliable walking performance in fewer environment steps?
2. Robustness: After the same training budget, does the curriculum-trained policy survive stronger or unfamiliar disturbances more successfully?

The experiment must distinguish training reward from actual physical competence. A higher reward doesn't necessarily mean a better walking robot.

## 3. System architecture

All development is in Python. JAX handles neural-network operations, while MJX handles batched simulation. The RTX 5090 is primarily for accelerating large-scale training—not for designing or debugging the robot.

We should also benchmark MJX-JAX against MJX-Warp on the actual Go1 environment before choosing the production backend. Warp is designed for NVIDIA-accelerated parallel simulation, but performance depends on workload and configuration.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub

+1



## 4. Curriculum: four learning stages

[Model Gallery - MuJoCo Documentation](https://images.openai.com/static-rsc-4/Hu-DhRZ7KauygHiKglw2t3jzER55YJ3wUkHnAQmhHfARrqA1HyO16caS-bx1jcw4HwwX8-FXPAXxPftlYg2PwpkcUffI-TpauUX7OaWUeZi7xYK1rVBddwi5Sc3meoz4318o9111LlPw-SyYdJGi8Xjy5q5iQWyntCeJE2mnySk?purpose=inline)

Stage 1 — Standing

The robot learns to remain upright without falling or making excessive actuator movements.

- Target velocity: 0 m/s
- Minimal external disturbances
- Initial states close to a stable standing posture

Gate: At least 90% of evaluation episodes survive for 10 seconds.

[Model Gallery - MuJoCo Documentation](https://images.openai.com/static-rsc-4/uT0AfwrXMxpJ8AthEOb5uXIOT2Ln5w3JGN4Oi9KpItq_Sk6nt_94TTo9tTy3GtZ1xFDCAWt0nTCRP5xuH1oQWzzUAFYJ0bbaTX9sEzxU0z8kTLwSt9eS8Esocz9fsCpAN4qfEJMkgaX_hLvplme14MaH1jFYRA94skuIPLxbFhY?purpose=inline)

Stage 2 — Balance

The robot learns to correct leaning and maintain stable posture.

- Randomized initial body orientation
- Small external forces
- Mild variation in joint starting positions

Gate: At least 80% survival under the defined balance disturbances.

[Unitree Go1 trained using SAC on gymnasium/mujoco-v5 framework](https://images.openai.com/static-rsc-4/gE-K0QMofABCzv8_kV252edRzU58utt7pAjm3LWwCVSap21qoAhPP7NGKyU3XNav5-GfCtyaTckTDlyzUSU-HdlOMcaG-11pgu_D_9nEaX51XKX9QZUJhLAKPehxOxbE5egY9qQlziZNRHEH0puCs2H9zFSn6tVtsRMTPwEQgD0?purpose=inline)

Stage 3 — Walking

Introduce movement commands and reward accurate velocity tracking.

- Begin at 0.2 m/s
- Progress toward 1.0 m/s
- Randomize forward speed commands
- Penalize unnecessary control effort

Gate: Complete at least 85% of commanded-walking episodes with acceptable velocity error.

[Testing the Field Capabilities of the Unitree Go-1 | Correll lab | University of Colorado Boulder](https://images.openai.com/static-rsc-4/8ngQnfHsnc4Q3IOm_A3PPRkmKF6M8OBqpKKIseKesjFjqNPeH8HDw21fKbiNbDTT7sTJYPPvRSRwgLxeQiy9vx1FIdER3D6fjSwPDkdGCIrvX097VkRBa6jevrVCkQpGvRhc1XUkK8XjPdhyQpaGdFvSniaY-9DYuo7rqHG_zkI?purpose=inline)

Stage 4 — Robust locomotion

Train against disturbances designed to challenge walking stability.

- Random lateral and longitudinal pushes
- Friction variation
- Uneven terrain, if supported by the selected task
- Noisy proprioceptive measurements

Gate: Achieve at least 80% survival under the predefined moderate-disturbance validation suite.

These thresholds are proposed development gates. They're not claims about what the robot can already achieve.

One important design rule: after introducing harder tasks, continue sampling some easier tasks. This reduces the risk of forgetting how to stand or walk at slower speeds.

## 5. PPO agent design

We'll use a continuous-control actor–critic policy.

Policy network

The critic has its own value-output head; exact policy architecture will follow the selected training implementation.

### Observations

The robot receives base angular velocity, projected gravity/orientation, joint positions and velocities, commanded forward/lateral/yaw velocities, and previous actions.

For the Go1, there are 12 actuated leg joints. I'd have the policy output 12 normalized joint-position offsets around a default standing pose, tracked by a PD controller. This is generally a more manageable starting point than unconstrained raw motor torques.

The policy should not receive privileged simulator information such as the exact upcoming push or true ground-friction coefficient. The critic may use privileged information later, but that would be a separately documented architecture choice.

### Reward function

\\[ r_t = w_v r\_{\text{velocity}} +w_u r\_{\text{upright}} +w_h r\_{\text{height}} -w\_\tau c\_{\text{torque}} -w_s c\_{\text{action-rate}} -w_f c\_{\text{fall}} \\]

We should start from a validated Playground locomotion reward rather than invent all reward weights independently. We'll then simplify or modify the terms for the curriculum.

A critical difference from the earlier plan: use one final-task reward definition across all three primary experiment groups. Difficulty should change through command distributions, initial states, and disturbance strengths—not by quietly giving the curriculum agent a more advantageous objective.

The policy should learn a gait through interacting with the physics, not from a reference animation or demonstration. A default standing pose and joint PD control are allowed; they don't prescribe a walking sequence.

## 6. Main experiment: curriculum versus direct training

Experimental groups

A — Direct PPO

Baseline

Full walking and perturbation task from training step zero.

B — Fixed curriculum

Primary comparison

Difficulty advances according to a predetermined schedule.

C — Adaptive curriculum

Secondary comparison

Difficulty advances after reaching specified success thresholds.

D — Shuffled curriculum

Strong optional control

Same task-difficulty mix as B, but without easy-to-hard ordering.

Our primary experiment should compare A versus B. C investigates adaptive scheduling. D helps separate ordering effects from the benefit of a varied training distribution.

### Proposed training budget

Fixed curriculum — 20 million steps per run

Stand

Balance

Walk

Robust

0M

3M

7M

13M

20M

Proposed schedule to test, not a previously validated optimum.

At 20 million environment steps and five seeds per treatment:

| Groups        | Runs | Environment steps |
| ------------- | ---- | ----------------- |
| A + B         | 10   | 200M              |
| A + B + C     | 15   | 300M              |
| A + B + C + D | 20   | 400M              |

Start with a 1–2 million-step smoke test, not the full 400-million-step campaign. The final step budget should be frozen after pilot benchmarking.

Experimental controls: Same robot, PPO architecture, policy input, reward definition, optimizer parameters, action scaling, and evaluation suite. Seeds should vary between independent runs, and all results—not just the best policies—should be included.

Also record final-task exposure separately. A curriculum agent may have seen fewer hard cases than the direct agent. Comparing matched total steps is the main test; a secondary comparison should account for the number of hard-task examples.

## 7. Robustness evaluation

This is the core of the scientific analysis.

We need held-out evaluation scenarios that don't influence curriculum selection or hyperparameter tuning.

Perturbation test designer

Evaluation only

Example push force

60 N

Ground friction coefficient

0.70

Commanded forward speed

0.8 m/s

Selected example scenario

Robot walking at 0.8 m/s on a surface with friction coefficient 0.70, exposed to a 60 N lateral push.

For reproducibility, specify force duration, application point, and timing too.

&#x20;Copy scenario

The controls illustrate a test scenario, not a live simulation. Exact push magnitudes will need calibration against the robot's mass and dynamics.

We'll measure:

| Metric                   | Purpose                       |
| ------------------------ | ----------------------------- |
| Push recovery rate       | Primary robustness outcome    |
| Fall rate                | Overall stability             |
| Velocity tracking error  | Walking quality               |
| Recovery time            | How rapidly stability returns |
| Energy / torque cost     | Movement efficiency           |
| Unseen-condition success | Generalization                |

Define recovery objectively: for example, the robot must remain upright and return to a prescribed velocity-error band within three seconds after a push. An episode that merely avoids falling but never resumes walking shouldn't count as a complete recovery.

For a meaningful result, I would predeclare the primary push-test distribution, evaluate hundreds of scenarios per seed, and compare the average across independent training seeds, with uncertainty intervals. Evaluation episodes from the same trained policy aren't independent training replicates.

## 8. Hardware and training workflow

[AZURE Gaming PC - RTX 5060 + i5 14400F + 32GB DDR5 + 2TB NVMe](https://images.openai.com/static-rsc-4/EhXVMDdZXQOrPS1tjFib_1EPXanRl5sE6ASjoXNum94zXHkJ_denGJoqBDd7OlhEMuslqrMc9S4TYXfbXkOH-37YHVewQXAG41hoC47Kp7rUTjVgpRD9GJN7lCdtbiFZYp5RrdpEaaNe4NfWiaFMKXT5-ovJg6tDx8nxwb9tm8U?purpose=inline)

Phase A — Local development

RTX 5060 8GB, Windows 11 + WSL2 Ubuntu.

Use small environment batches (64–256 initially), verify reward computations, run PPO smoke tests, and render the robot. GPU memory and compilation behavior determine the actual usable batch size.

[Rackstation AI - универсальная рабочая станция с 2× RTX 5090 и AMD Threadripper PRO | AI, рендеринг, симуляции, CAD, LLM](https://images.openai.com/static-rsc-4/4iLbsdFimLyTVTzBna_f-CZFrpe0CJ29Jq--4I7bbmFmk-5naRYOKDNA6GcvI8Vn0CWXW-O-Xm8d7BlILkfd71fMCTJSuKMUGcf0nvlFJ9mKEkLRfWPuPY95p4hGS6o1IXqOQh7AbfGfAoDAXfL3b3xy1nlnn8Uyf5JSwcDg0Kk?purpose=inline)

Phase B — Cloud training

RTX 5090 32GB, rented Linux GPU instance.

Benchmark 256, 1,024, and 4,096 parallel environments, then train the experimental policies at the most efficient configuration.

Phase C — Local evaluation and analysis

Download policy checkpoints and normalization/configuration data. Re-run held-out tests in a verified physics backend, analyze results, and render comparison videos.

One thing I would change from the earlier recommendation: don't assume the entire research experiment will cost only $25–$50. Use that as an initial benchmarking allowance. Total rental cost depends on actual end-to-end training throughput, compilation, evaluation, and the number of runs.

For example, 300 million environment steps at 100,000 measured steps/second is about 50 minutes of pure simulation-equivalent work. It does not include policy optimization, compilation, evaluation, data movement, or infrastructure overhead. We must measure actual end-to-end throughput before estimating rental time.

## 9. Updated repository structure

```
robot-locomotion-rl/
├── configs/
│   ├── robot.yaml
│   ├── ppo.yaml
│   ├── experiment.yaml
│   └── curricula/
│       ├── direct.yaml
│       ├── fixed.yaml
│       ├── adaptive.yaml
│       └── shuffled.yaml
│
├── environments/
│   ├── locomotion.py
│   ├── observations.py
│   ├── rewards.py
│   ├── perturbations.py
│   └── curriculum.py
│
├── training/
│   ├── train.py
│   ├── benchmark.py
│   └── checkpoint.py
│
├── evaluation/
│   ├── evaluate.py
│   ├── push_recovery.py
│   ├── generalization.py
│   └── metrics.py
│
├── visualization/
│   ├── render_policy.py
│   └── compare_policies.py
│
├── analysis/
│   ├── learning_curves.py
│   ├── robustness_report.py
│   └── statistical_tests.py
│
├── tests/
├── results/
├── checkpoints/
└── README.md
```

Use [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground?utm_source=chatgpt.com) as the upstream foundation and [Brax PPO](https://github.com/google/brax?utm_source=chatgpt.com) for the initial training algorithm. We should pin compatible versions of MuJoCo, Playground, JAX, CUDA-dependent packages, and Brax to avoid reproducibility problems.

## 10. Development milestones

Week 1

Get the robot running locally

Load Go1, inspect the model, confirm the actuators and observations, train one baseline policy, and record the first simulation. This phase should prove the toolchain works end to end.

Week 2

Implement the custom task

Add command distributions, push disturbances, curriculum stages, reset conditions, success metrics, and unit tests. Validate physics and rewards before serious training.

Week 3

Benchmark cloud acceleration

Run small comparable training workloads on the 5060 and 5090. Measure steps/second, GPU utilization, VRAM, compilation time, and return improvement per environment step.

Week 4

Freeze and run experiments

Lock PPO settings, final reward, curriculum schedules, training budgets, seeds, and evaluation procedures. Run all primary comparison policies.

Week 5

Robustness testing

Evaluate direct and curriculum agents on the same unseen perturbations and terrain conditions. Produce learning curves and confidence intervals.

Week 6

Publish the research

Create reproducible experiment instructions, a technical write-up, plots, side-by-side walking videos, and a demonstration of perturbation recovery.

## 11. Final deliverables

Project checklist

0 of 8 complete

Reproducible MuJoCo quadruped environment

Working PPO walking policy

Fixed and adaptive curriculum implementations

Direct-training comparison with independent seeds

Held-out perturbation evaluation suite

Learning curves and robustness comparison report

Side-by-side robot walking and push-recovery videos

Open-source repository with experiment configurations

## 12. Where I would begin

The first milestone should not be building all four curriculum stages. It should be validating that we can train and evaluate one Go1 PPO locomotion policy successfully.

Our immediate tasks are:

1. Set up a clean JAX/MuJoCo Playground environment in WSL2.
2. Identify and run a supported Go1 locomotion task from the installed Playground version.
3. Train a short baseline using the upstream PPO implementation.
4. Render the policy and inspect the behavior.
5. Record end-to-end steps/second, GPU utilization, PPO metrics, and checkpoint loading behavior.

Only after that works should we modify the environment for curriculum learning.

Definition of success for the project: We should end up with a reproducible answer to whether progressive task difficulty improves held-out recovery rates, with controls that make a positive, negative, or inconclusive result scientifically useful.

My preference is to keep this as a robotics research project first, and an impressive simulation/demo second. That will make the resulting open-source project substantially more valuable than simply training a quadruped to walk.