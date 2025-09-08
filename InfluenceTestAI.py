import os
import asyncio
from math import exp
from typing import Dict

from dotenv import load_dotenv
load_dotenv()

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient


# ---------- Configuration environment ----------
api_key = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-4o")

if not api_key:
    raise RuntimeError(
        "OPENAI_API_KEY is not set. Run, for example: export OPENAI_API_KEY=sk-..."
        "or add it to the .env file and run again."
    )


# ---------- Supporting functions: T(μ), λ(μ) ----------

def temperature_from_mu(mu: float, T0: float = 0.7, alpha: float = 0.8) -> float:
    """
    Temperature modulation according to mutual influence: T = T0 / (1 + alpha * mu)
    (lower T at higher μ -> more conservative / peer-aligned)
    """
    T = T0 / (1.0 + alpha * max(0.0, mu))
    return max(0.1, min(1.5, T))  # safe limitation

def lambda_from_mu(mu: float, k: float = 6.0, tau: float = 0.5) -> float:
    """
    Sigmoid λ = 1 / (1 + exp(-k*(μ - τ))) ... degree of mixing peers vs self.
    """
    return 1.0 / (1.0 + exp(-k * (mu - tau)))


# ---------- Mutual Influence agent ----------

class MutualInfluenceAssistant(AssistantAgent):
    r"""
    AssistantAgent with mutual influence support:
    - peer_scores: s_{i←j} \in [0,1]
    - μ = mean(peer_scores)
    - receive_feedback(..., beta) = EMA update
    - run_with_influence() prefixes the prompt with μ/λ hint and modulates the temperature
    """

    def __init__(self, name: str, model_name: str, api_key: str | None = None, system_message: str | None = None):
        self._model_name = model_name
        self._api_key = api_key

        if not self._api_key:
            raise RuntimeError("OPENAI_API_KEY není nastaven (api_key=None).")

        model_client = OpenAIChatCompletionClient(
            model=self._model_name,
            api_key=self._api_key,
        )
        super().__init__(
            name=name,
            model_client=model_client,
            system_message=system_message
            or "You are a helpful assistant collaborating with peers. "
               "Be concise, factual. When mutual influence μ is high, align with peers; "
               "when low, be skeptical and justify divergences.",
        )
        self.peer_scores: Dict[str, float] = {}

    @property
    def mu(self) -> float:
        return (sum(self.peer_scores.values()) / len(self.peer_scores)) if self.peer_scores else 0.0

    def receive_feedback(self, from_peer: str, score: float, beta: float = 0.5) -> None:
        score = max(0.0, min(1.0, float(score)))
        old = self.peer_scores.get(from_peer, 0.0)
        self.peer_scores[from_peer] = (1.0 - beta) * old + beta * score

    async def run_with_influence(self, task: str, T0: float = 0.7, alphaT: float = 0.8) -> str:
        mu_val = self.mu
        lam = lambda_from_mu(mu_val)
        temp = temperature_from_mu(mu_val, T0=T0, alpha=alphaT)

        prefix = (
            f"[mutual_influence μ={mu_val:.2f}, mix λ={lam:.2f}, temp={temp:.2f}] "
            "If μ is high, prefer peer-consistent reasoning and cite their key points; "
            "if μ is low, be more skeptical and justify disagreements briefly."
        )
        prefixed_task = prefix + "\n\n" + task

        # Re-initialization of the client with a new temperature (according to library requirements)
        self._model_client = OpenAIChatCompletionClient(
            model=self._model_name,
            api_key=self._api_key,
            temperature=temp,
        )

        result = await self.run(task=prefixed_task)
        last_msg = result.messages[-1] if result.messages else None
        return getattr(last_msg, "content", "(no content)")


# ---------- Demo run ----------

async def main() -> None:
    # Agents with roles
    planner = MutualInfluenceAssistant(
        "planner",
        model_name=MODEL_NAME,
        api_key=api_key,
        system_message="Role: Planner. Produce a short, step-wise plan with clear priorities."
    )
    researcher = MutualInfluenceAssistant(
        "researcher",
        model_name=MODEL_NAME,
        api_key=api_key,
        system_message="Role: Researcher. Extract key signals/features and provide concise evidence."
    )
    critic = MutualInfluenceAssistant(
        "critic",
        model_name=MODEL_NAME,
        api_key=api_key,
        system_message="Role: Critic. Point out risks/gaps. Reply 'APPROVE' when the plan is sufficient."
    )

    # Round-robin orchestration
    termination = MaxMessageTermination(max_messages=8)
    team = RoundRobinGroupChat([planner, researcher, critic], termination_condition=termination)

    # ---------- Round 1: baseline (μ=0) ----------
    print("\n=== Round 1 (baseline μ=0) ===")
    initial_task = (
        "We need a concise 3-step plan to assess whether a network flow is malware-related, "
        "and propose one concrete real-time indicator to compute."
    )
    await Console(team.run_stream(task=initial_task), output_stats=True)

    # ---------- Peer feedback (simulation) ----------
    planner.receive_feedback(from_peer="critic", score=0.9, beta=0.6)
    planner.receive_feedback(from_peer="researcher", score=0.8, beta=0.6)

    researcher.receive_feedback(from_peer="planner", score=0.85, beta=0.6)
    researcher.receive_feedback(from_peer="critic", score=0.7, beta=0.6)

    critic.receive_feedback(from_peer="planner", score=0.8, beta=0.6)
    critic.receive_feedback(from_peer="researcher", score=0.75, beta=0.6)

    print(f"\nμ(planner)={planner.mu:.2f}, μ(researcher)={researcher.mu:.2f}, μ(critic)={critic.mu:.2f}")

    # ---------- Round 2: Mutual influence ----------
    print("\n=== Round 2 (with Mutual Influence) ===")
    await team.reset()

    refined_task = (
        "Refine the plan using peer feedback to reduce false positives. "
        "Propose a minimal set of streaming features that can be computed in real time. "
        "Critic should reply APPROVE if satisfied."
    )
    await Console(team.run_stream(task=refined_task), output_stats=True)

    # Individual calls with μ/λ/temperature prefix
    print("\n--- Individual influenced calls ---")
    msgA = await planner.run_with_influence("Give a 2-step prioritized checklist for a SOC analyst.")
    msgB = await researcher.run_with_influence("List 3 concrete streaming features to compute (names only).")
    msgC = await critic.run_with_influence("Is this implementable in 1 sprint? If yes, reply APPROVE.")

    print("\n[planner]\n", msgA)
    print("\n[researcher]\n", msgB)
    print("\n[critic]\n", msgC)


if __name__ == "__main__":
    asyncio.run(main())
