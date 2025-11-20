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

    def _shuffle_words(self, prompt: str) -> str:
        """
        Shuffle words within ~[ ]~ sections while preserving A1111 special syntax.
        Words are split by commas only, and parentheses are respected.
        Supports multiline sections.
        Words with priority prefix (¤1, ¤2, etc.) are ordered by priority,
        with words of the same priority shuffled among themselves.
        """
        # Remove A1111 special syntax first
        prompt, special_chunks = remove_a1111_special_syntax_chunks(prompt)

        # Pattern to find ~[ ]~ sections (DOTALL flag allows matching across newlines)
        pattern = r"~\[(.*?)\]~"

        def shuffle_section(match):
            content = match.group(1)

            # Split by comma while respecting parentheses
            words = self._split_by_comma_respecting_parens(content)

            # Pattern to match priority prefix like ¤1, ¤2, etc.
            priority_pattern = r"^¤(\d+)(.*)$"

            # Group words by priority
            prioritized = {}  # {priority_number: [words with that priority]}
            unprioritized = []  # words without priority

            for word in words:
                match_priority = re.match(priority_pattern, word)
                if match_priority:
                    priority = int(match_priority.group(1))
                    word_without_prefix = match_priority.group(2)
                    if priority not in prioritized:
                        prioritized[priority] = []
                    prioritized[priority].append(word_without_prefix)
                else:
                    unprioritized.append(word)

            # Build result by processing priorities in order
            result = []

            # Sort priority keys and process each group
            for priority in sorted(prioritized.keys()):
                group = prioritized[priority]
                random.shuffle(group)
                result.extend(group)

            # Shuffle and add unprioritized words last
            random.shuffle(unprioritized)
            result.extend(unprioritized)

            # Rejoin with commas
            return ", ".join(result) + ","

        # Replace all ~[ ]~ sections with shuffled versions
        # re.DOTALL makes . match newlines too
        result = re.sub(pattern, shuffle_section, prompt, flags=re.DOTALL)

        # Restore A1111 special syntax
        return append_chunks(result, special_chunks)
