from .base import BaseAgent


class LibraAgent(BaseAgent):
    name = "LIBRA"
    bias_type = "certainty_inflation"
    prompt_filename = "libra_v1.0.txt"
