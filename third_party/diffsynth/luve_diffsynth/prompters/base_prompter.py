"""Minimal T2V prompt processing from DiffSynth's BasePrompter.

Unrelated model-manager imports, prompt extenders and network-loaded refiners
are intentionally outside this portable inference subset.
"""
import torch


class BasePrompter:
    def __init__(self):
        self.refiners = []
        self.extenders = []

    @torch.no_grad()
    def process_prompt(self, prompt, positive=True):
        if isinstance(prompt, list):
            return [self.process_prompt(value, positive=positive) for value in prompt]
        for refiner in self.refiners:
            prompt = refiner(prompt, positive=positive)
        return prompt
