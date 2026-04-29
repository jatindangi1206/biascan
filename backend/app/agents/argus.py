from .base import BaseAgent


class ArgusAgent(BaseAgent):
    name = "ARGUS"
    bias_type = "confirmation_bias"
    prompt_filename = "argus_v1.0.txt"
