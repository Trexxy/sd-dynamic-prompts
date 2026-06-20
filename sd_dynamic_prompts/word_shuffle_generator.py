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
    Words are split by commas, and anything inside parentheses is treated as a single word.
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

    def _split_by_comma_respecting_parens(self, text: str) -> list[str]:
        """
        Split text by commas, but treat anything inside parentheses as a single unit.
        For example: "happy, (very, very sad), joyful" -> ["happy", "(very, very sad)", "joyful"]
        """
        words = []
        current_word = ""
        paren_depth = 0

        for char in text:
            if char == "(":
                paren_depth += 1
                current_word += char
            elif char == ")":
                paren_depth -= 1
                current_word += char
            elif char == "," and paren_depth == 0:
                # We're at a comma outside of parentheses, so split here
                if current_word.strip():
                    words.append(current_word.strip())
                current_word = ""
            else:
                current_word += char

        # Don't forget the last word
        if current_word.strip():
            words.append(current_word.strip())

        return words

    def _shuffle_segment(self, text: str) -> str:
        """
        Shuffle a single comma-separated segment, respecting parentheses and priority prefixes.
        Returns the shuffled words joined by ", " with no trailing comma, or "" if empty.
        """
        words = self._split_by_comma_respecting_parens(text)
        if not words:
            return ""

        priority_pattern = r"^¤(\d+)(.*)$"
        prioritized = {}
        unprioritized = []

        for word in words:
            m = re.match(priority_pattern, word)
            if m:
                priority = int(m.group(1))
                prioritized.setdefault(priority, []).append(m.group(2))
            else:
                unprioritized.append(word)

        result = []
        for priority in sorted(prioritized.keys()):
            group = prioritized[priority]
            random.shuffle(group)
            result.extend(group)

        random.shuffle(unprioritized)
        result.extend(unprioritized)

        return ", ".join(result)

    def _shuffle_words(self, prompt: str) -> str:
        """
        Shuffle words within ~[ ]~ sections while preserving A1111 special syntax.
        Words are split by commas only, and parentheses are respected.
        Supports multiline sections.
        Words with priority prefix (¤1, ¤2, etc.) are ordered by priority,
        with words of the same priority shuffled among themselves.
        BREAK splits a section into independently shuffled segments.
        """
        prompt, special_chunks = remove_a1111_special_syntax_chunks(prompt)

        pattern = r"~\[(.*?)\]~"

        def shuffle_section(match):
            content = match.group(1)

            # Split on BREAK tokens, keeping them as delimiters
            parts = re.split(r"(\bBREAK\b)", content, flags=re.IGNORECASE)

            output_parts = []
            for part in parts:
                if part.strip().upper() == "BREAK":
                    output_parts.append("BREAK")
                else:
                    shuffled = self._shuffle_segment(part)
                    if shuffled:
                        output_parts.append(shuffled)

            return ", ".join(output_parts) + ","

        result = re.sub(pattern, shuffle_section, prompt, flags=re.DOTALL)
        return append_chunks(result, special_chunks)
