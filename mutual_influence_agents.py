# mutual_influence_agents.py
# SPDX-License-Identifier: MIT
# MIT License
# Copyright (c) 2025 Vaclav Oujezsky
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
import os
from math import exp
from typing import Dict
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

from metrics import safe_json_loads

OPENAI_MODEL = os.getenv("OPENAI_MODEL_NAME", "gpt-4o")

def temperature_from_mu(mu: float, T0: float = 0.7, alpha: float = 0.8) -> float:
    T = T0 / (1.0 + alpha * max(0.0, mu))
    return max(0.1, min(1.5, T))

def lambda_from_mu(mu: float, k: float = 6.0, tau: float = 0.5) -> float:
    return 1.0 / (1.0 + exp(-k * (mu - tau)))

class MutualInfluenceAssistant(AssistantAgent):
    def __init__(self, name: str, role_message: str, api_key: str,
                 k=6.0, tau=0.5, alpha=0.8, prior=0.5, model_name: str = OPENAI_MODEL):
        self._api_key = api_key
        self._model_name = model_name
        self.k, self.tau, self.alpha, self.prior = k, tau, alpha, prior
        self.peer_scores: Dict[str, float] = {}

        client = OpenAIChatCompletionClient(model=self._model_name, api_key=self._api_key)
        super().__init__(
            name=name,
            model_client=client,
            system_message=(
                "You are a helpful assistant collaborating with peers. "
                "Be concise and factual. When mutual influence μ is high, prefer peer-consistent reasoning "
                "and cite their key points; when low, be skeptical and justify divergences briefly. "
            ) + role_message
        )

    @property
    def mu(self) -> float:
        return (sum(self.peer_scores.values())/len(self.peer_scores)) if self.peer_scores else 0.0

    def receive_feedback(self, from_peer: str, score: float, beta: float = 0.6):
        old = self.peer_scores.get(from_peer, self.prior)
        score = max(0.0, min(1.0, float(score)))
        self.peer_scores[from_peer] = (1.0 - beta) * old + beta * score

    async def call(self, task: str, influenced: bool = False, base_temp: float = 0.2, require_keys=None) -> str:
        """Forces a JSON response and attempts self-repair once if keys are missing."""
        if influenced:
            mu_val = self.mu
            lam = lambda_from_mu(mu_val, self.k, self.tau)
            temp = temperature_from_mu(mu_val, 0.7, self.alpha)
            prefix = (
                f"[mutual_influence μ={mu_val:.2f}, mix λ={lam:.2f}, temp={temp:.2f}] "
                "If μ is high, be peer-consistent; if μ is low, justify disagreements."
            )
            task = prefix + "\n\n" + task
            temperature = temp
        else:
            temperature = base_temp

        # 1) force JSON
        self._model_client = OpenAIChatCompletionClient(
            model=self._model_name,
            api_key=self._api_key,
            temperature=temperature,
            response_format={"type": "json_object"},  # Important here
        )
        out = await self.run(task=task)
        txt = (out.messages[-1].content if out and out.messages else "") or ""

        if require_keys:
            d = safe_json_loads(txt)
            missing = [k for k in require_keys if not (isinstance(d, dict) and k in d)]
            if missing:
                # 2) one repair round
                repair = (
                    "Your previous output was not a valid JSON object with required keys. "
                    f"Required keys: {require_keys}. "
                    "Return exactly one JSON object with those keys only. No explanations."
                )
                self._model_client = OpenAIChatCompletionClient(
                    model=self._model_name,
                    api_key=self._api_key,
                    temperature=temperature,
                    response_format={"type": "json_object"},
                )
                out = await self.run(task=repair + "\n\nLast task:\n" + task)
                txt = (out.messages[-1].content if out and out.messages else "") or ""

        return txt

