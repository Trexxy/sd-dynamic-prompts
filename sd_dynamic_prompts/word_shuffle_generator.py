import random
import re

from dynamicprompts.generators.promptgenerator import PromptGenerator

from sd_dynamic_prompts.special_syntax import (
    append_chunks,
    remove_a1111_special_syntax_chunks,
)


class WordShuffleGenerator(PromptGenerator):
    """
    Generator that randomizes words within ~[ ]~ sections.
    This runs after wildcard expansion.
    """

    def __init__(self, generator: PromptGenerator):
        self._generator = generator

    def generate(
        self,
        template: str,
        num_images: int | None = 1,
        **kwargs,
    ) -> list[str] | None:
        prompts = self._generator.generate(template, num_images, **kwargs)
        if prompts is None:
            return None
        return [self._shuffle_words(p) for p in prompts]

    def _shuffle_words(self, prompt: str) -> str:
        """
        Shuffle words within ~[ ]~ sections while preserving A1111 special syntax.
        """
        # Remove A1111 special syntax first
        prompt, special_chunks = remove_a1111_special_syntax_chunks(prompt)

        # Pattern to find ~[ ]~ sections
        pattern = r"~\[(.*?)\]~"

        def shuffle_section(match):
            content = match.group(1)
            # Split on comma or whitespace
            # First try splitting by comma
            if "," in content:
                words = [w.strip() for w in content.split(",") if w.strip()]
            else:
                # Otherwise split by whitespace
                words = content.split()

            # Shuffle the words
            random.shuffle(words)

            # Rejoin with appropriate separator
            if "," in content:
                return ", ".join(words)
            else:
                return " ".join(words)

        # Replace all ~[ ]~ sections with shuffled versions
        result = re.sub(pattern, shuffle_section, prompt)

        # Restore A1111 special syntax
        return append_chunks(result, special_chunks)
