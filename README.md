# Rule-enforced Graph Growing for Molecular Design by Lightweight Reinforcement Learning

This repository implements a reinforcement learning (RL) framework for generating chemical molecules as graphs. A custom Graph Convolutional Network (GCN) encodes the graph state, and an Actor-Critic model guides generation actions under domain-specific rules.

## Overview
- Graph-based molecule generation environment
- Node-wise generation decisions with rule-based reward shaping
- Validity and diversity evaluation using RDKit and SMILES
- Modular architecture with clear separation between model, environment, and training

## Project Structure
```
.
├── main.py                  # Entry point: launches training pipeline
├── models/
│   ├── actor_critic.py     # Actor-Critic policy network
│   └── gcn.py              # GCN module for molecular embedding
├── molecule/
│   ├── env.py              # Molecule construction logic, reward design
│   └── utils.py            # Visualization, SMILES conversion, evaluation
├── training/
│   ├── trainer.py          # Training loop, epsilon schedule, logging
│   └── buffer.py           # Experience Replay Buffer
└── README.md               # Project documentation
```

## Installation
Dependencies:
- Python >= 3.8
- PyTorch >= 1.10
- RDKit
- torch-geometric

Install required packages:
```bash
pip install -r requirements.txt
```

## How to Run
To begin training, execute:
```bash
python main.py
```
Molecules will be generated and evaluated per episode. If configured, model checkpoints and logs can be saved to a designated directory.

## Evaluation Metrics
- **Validity**: SMILES string is chemically valid (parsable by RDKit)
- **Uniqueness**: SMILES not duplicated from prior outputs
- **Novelty**: Distinct from training set or previously seen molecules

## Generation Notes
- Atom choices: [C, H, O, N, benzene, none]
- Benzene is treated as a special case and generates a 6-member aromatic ring automatically
- Each step involves choosing both atom type and whether to continue graph growth

## Model Architecture
- **GCN**: Embeds the molecular graph based on atom types and bond types
- **Actor-Critic**:
  - Policy head outputs atom type probabilities and continuation probability
  - Value head estimates state value for policy optimization

## Outputs
- `actor_critic_model_XXXX.pth`: model checkpoint examples (if saving enabled)
- `training_log.txt`: logs of episode rewards and exploration parameters
- List of valid and unique SMILES generated across episodes

## Author Notes
This project is structured for ease of experimentation. Users may consider modifying:
- The reward function in `env.py`
- The space of allowed node types or bonds
- The evaluation criteria used during training

## Entry Point Summary
```python
# main.py
from training.trainer import run_training

if __name__ == "__main__":
    run_training()
```

---
Contributions are welcome. Extensions such as alternative chemical rules, retrosynthesis integration, or GNN variants can be explored.
