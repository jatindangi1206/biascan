from .base import BaseAgent


class QuillAgent(BaseAgent):
    name = "QUILL"
    bias_type = "framing_effect"
    prompt_filename = "quill_v1.0.txt"
