from .base import BaseAgent


class LensAgent(BaseAgent):
    name = "LENS"
    bias_type = "overgeneralisation"
    prompt_filename = "lens_v1.0.txt"
