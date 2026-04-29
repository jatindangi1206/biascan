from .base import BaseAgent


class VigilAgent(BaseAgent):
    name = "VIGIL"
    bias_type = "causal_inference_error"
    prompt_filename = "vigil_v1.0.txt"
