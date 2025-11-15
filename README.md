# Wizardry Under Uncertainty

CS 238 Final Project - Building an AI agent to play the card game Wizard using Monte Carlo Tree Search.

**Team:** Joana Mizrahi, Jonathan Dumanski

## Project Overview

We're developing an intelligent agent to play Wizard, a trick-taking card game that requires strategic bidding and card play under partial observability. The agent uses Monte Carlo Tree Search (MCTS) with determinization to handle hidden opponent hands.

## Project Structure
```
wizard_ai/
├── game/                  # Game engine
│   ├── deck.py           # Card and deck representations
│   ├── rules.py          # Game rules (trick-taking, scoring)
│   └── simulator.py      # Main game loop
├── agents/               # AI agents
│   ├── random_agent.py   # Random baseline
│   ├── heuristic_agent.py # Hand-crafted rules baseline
│   └── mcts_simple.py    # MCTS agent
├── tests/                # Tests and evaluation
│   ├── test_deck.py
│   ├── test_rules.py
│   ├── test_simulator.py
│   ├── test_heuristic.py
│   └── test_mcts.py
└── evaluation/           # Evaluate model
    ├── ...
```

## Setup

### Prerequisites
- Python 3.8+

### Installation
```bash
# Clone the repository
git clone https://github.com/joanamizrahi-png/cs238_wizard_ai.git
cd wizard-ai

# No external dependencies needed - uses only Python standard library
```

## Running the Code

### Test the Game Simulator
```bash
python tests/test_simulator.py
```

### Test Agents
```bash
# Test heuristic agent vs random
python tests/test_heuristic.py

# Test MCTS agent vs heuristic and random
python tests/test_mcts.py
```

## Current Results

| Agent | Win Rate | Avg Score |
|-------|----------|-----------|
| MCTS (ours) | 60% | -117 |
| Heuristic | 30% | -147 |
| Random | 10% | -348 |

*Based on 10-game tournaments with 4 players each*

## Implementation Status

- Game simulator (complete)
- Random baseline (complete)
- Heuristic baseline (complete)
- MCTS for card playing (complete)
- MCTS for bidding (in progress)
- Improved determinization (planned)
- Opponent modeling (planned)

## Next Steps

1. Implement MCTS for bidding decisions
2. Weight opponent hand sampling by observed bids
3. Run comprehensive experiments (50+ games)
4. Ablation studies on simulation count and bidding strategies
5. Final report and analysis