# Mutual Influence AI Demo

This project demonstrates a simple multi-agent system with **Mutual Influence** 
(peer feedback mechanism) built on top of [autogen-agentchat](https://pypi.org/project/autogen-agentchat/).

Three agents (`planner`, `researcher`, `critic`) collaborate in a round-robin conversation.  
They adjust their reasoning based on peer feedback using *mutual influence (μ)*.

---

## Features
- **Multi-agent collaboration**: Planner, Researcher, Critic
- **Mutual Influence mechanism**:
  - Temperature modulation based on μ
  - λ-mixing between self vs peer alignment
- **Round-robin group chat** with termination condition
- **Streaming console output** for demonstration

---

## Requirements

- Python 3.10+ (tested on 3.12)
- Virtual environment (`venv` recommended)
- Dependencies:
  ```bash
  pip install -r requirements.txt
