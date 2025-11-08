from __future__ import annotations

import random
import re

from dynamicprompts.generators.promptgenerator import PromptGenerator

from sd_dynamic_prompts.special_syntax import (
    append_chunks,
    remove_a1111_special_syntax_chunks,
)


def split_respecting_parentheses(text: str, separator: str = ",") -> list[str]:
    """
    Split text by separator, but respect parentheses boundaries.

    Anything inside parentheses (including nested) is treated as a single unit,
    even if it contains the separator character.

    Examples:
        "a, b, c" -> ["a", "b", "c"]
        "a, (b, c), d" -> ["a", "(b, c)", "d"]
        "a, ((b, c):1.5), d" -> ["a", "((b, c):1.5)", "d"]

    Args:
        text: The text to split
        separator: The separator character (default: ",")

    Returns:
        List of parts, with parenthesized content kept intact
    """
    parts = []
    current_part = []
    paren_depth = 0

    for char in text:
        if char == "(":
            paren_depth += 1
            current_part.append(char)
        elif char == ")":
            paren_depth -= 1
            current_part.append(char)
        elif char == separator and paren_depth == 0:
            # We're at a separator outside of parentheses
            part = "".join(current_part).strip()
            if part:  # Don't add empty parts
                parts.append(part)
            current_part = []
        else:
            current_part.append(char)

    # Add the last part
    part = "".join(current_part).strip()
    if part:
        parts.append(part)

    return parts


class WordShuffleGenerator(PromptGenerator):
    """
    Generator that shuffles marked sections of prompts.

    Sections to shuffle are marked with configurable delimiters (default: $[...]$).
    Content is split by commas, but parenthesized content is kept intact.

    Examples:
        Input: "a portrait of $[a cat, wearing a hat, in the park]$, highly detailed"
        Output: "a portrait of in the park, a cat, wearing a hat, highly detailed"

        Input: "$[red, (blue and green:1.5), yellow]$ car"
        Output: "yellow, red, (blue and green:1.5) car"
    """

    def __init__(
        self,
        generator: PromptGenerator,
        seed: int | None = None,
        shuffle_start: str = "$[",
        shuffle_end: str = "]$",
        separator: str = ",",
    ):
        """
        Initialize the WordShuffleGenerator.

        Args:
            generator: The wrapped prompt generator
            seed: Random seed for reproducibility (will be overridden by generate() seeds)
            shuffle_start: Starting delimiter for shuffle sections
            shuffle_end: Ending delimiter for shuffle sections
            separator: Character to split on (default: comma)
        """
        self._generator = generator
        self._seed = seed
        self._shuffle_start = shuffle_start
        self._shuffle_end = shuffle_end
        self._separator = separator

        # Create pattern to match shuffle sections
        # Use re.escape to handle special regex characters in delimiters
        escaped_start = re.escape(shuffle_start)
        escaped_end = re.escape(shuffle_end)
        self._shuffle_pattern = re.compile(
            f"{escaped_start}(.*?){escaped_end}",
            re.DOTALL,  # Allow matching across newlines
        )

    def generate(
        self,
        template: str,
        num_images: int | None = 1,
        **kwargs,
    ) -> list[str]:
        """
        Generate prompts with shuffled sections.

        Args:
            template: The prompt template
            num_images: Number of prompts to generate
            **kwargs: Additional arguments (including seeds)

        Returns:
            List of prompts with shuffled sections
        """
        # Generate base prompts using wrapped generator
        prompts = self._generator.generate(template, num_images, **kwargs)

        # Get seeds for each prompt (use provided seeds or None)
        seeds = kwargs.get("seeds", [None] * len(prompts))

        # Shuffle each prompt with its corresponding seed
        shuffled_prompts = [
            self._shuffle_prompt(prompt, seed)
            for prompt, seed in zip(prompts, seeds)
        ]

        return shuffled_prompts

    def _shuffle_prompt(self, prompt: str, seed: int | None) -> str:
        """
        Shuffle marked sections in a single prompt.

        Args:
            prompt: The prompt to process
            seed: Random seed for this prompt

        Returns:
            Prompt with shuffled sections
        """
        # Remove A1111 special syntax (LoRA, hypernet tags) before processing
        prompt, special_chunks = remove_a1111_special_syntax_chunks(prompt)

        # Find and shuffle all marked sections
        def shuffle_section(match):
            content = match.group(1)

            # Split by separator, respecting parentheses
            parts = split_respecting_parentheses(content, self._separator)

            # Shuffle the parts
            # Create a new Random instance for each shuffle to ensure reproducibility
            rng = random.Random(seed if seed is not None else self._seed)
            rng.shuffle(parts)

            # Join back with the separator (add space after for readability)
            return (self._separator + " ").join(parts)

        # Replace all shuffle sections with their shuffled versions
        shuffled = self._shuffle_pattern.sub(shuffle_section, prompt)

        # Re-add A1111 special syntax
        return append_chunks(shuffled, special_chunks)
