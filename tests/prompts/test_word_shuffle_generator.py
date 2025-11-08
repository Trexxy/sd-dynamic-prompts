from dynamicprompts.generators import DummyGenerator

from sd_dynamic_prompts.word_shuffle_generator import (
    WordShuffleGenerator,
    split_respecting_parentheses,
)


def test_split_respecting_parentheses_basic():
    """Test basic comma splitting without parentheses."""
    result = split_respecting_parentheses("a, b, c")
    assert result == ["a", "b", "c"]


def test_split_respecting_parentheses_with_parens():
    """Test that content inside parentheses is kept intact."""
    result = split_respecting_parentheses("a, (b, c), d")
    assert result == ["a", "(b, c)", "d"]


def test_split_respecting_parentheses_with_nested():
    """Test that nested parentheses are handled correctly."""
    result = split_respecting_parentheses("a, ((b, c):1.5), d")
    assert result == ["a", "((b, c):1.5)", "d"]


def test_split_respecting_parentheses_complex():
    """Test complex attention syntax."""
    result = split_respecting_parentheses("cat, (wearing a hat:1.5), (in the park, sunny:1.2)")
    assert result == ["cat", "(wearing a hat:1.5)", "(in the park, sunny:1.2)"]


def test_split_respecting_parentheses_empty_parts():
    """Test that empty parts are filtered out."""
    result = split_respecting_parentheses("a,, b,,, c")
    assert result == ["a", "b", "c"]


def test_basic_shuffle():
    """Test that shuffle sections are identified and shuffled."""
    generator = WordShuffleGenerator(DummyGenerator(), seed=42)
    prompt = "a portrait of $[a cat, wearing a hat, in the park]$, highly detailed"

    result = generator.generate(prompt, 1)[0]

    # Check that the shuffle section exists
    assert "a portrait of" in result
    assert "highly detailed" in result

    # Check that all parts are present
    assert "a cat" in result
    assert "wearing a hat" in result
    assert "in the park" in result

    # The order should be different from the original
    # With seed=42, the order will be deterministic but shuffled
    assert result != prompt


def test_shuffle_preserves_parentheses():
    """Test that parenthesized content is kept as a single unit."""
    generator = WordShuffleGenerator(DummyGenerator(), seed=42)
    prompt = "$[red, (blue and green:1.5), yellow]$ car"

    result = generator.generate(prompt, 1)[0]

    # Check that the parenthesized part is intact
    assert "(blue and green:1.5)" in result
    assert "red" in result
    assert "yellow" in result
    assert "car" in result


def test_multiple_shuffle_sections():
    """Test that multiple shuffle sections work independently."""
    generator = WordShuffleGenerator(DummyGenerator(), seed=42)
    prompt = "$[a, b, c]$ and $[x, y, z]$"

    result = generator.generate(prompt, 1)[0]

    # Check all elements are present
    assert "a" in result and "b" in result and "c" in result
    assert "x" in result and "y" in result and "z" in result
    assert " and " in result


def test_no_shuffle_sections():
    """Test that prompts without shuffle sections are unchanged."""
    generator = WordShuffleGenerator(DummyGenerator())
    prompt = "a normal prompt without shuffle sections"

    result = generator.generate(prompt, 1)[0]

    assert result == prompt


def test_shuffle_preserves_special_syntax():
    """Test that A1111 special syntax (LoRA, hypernet) is preserved."""
    generator = WordShuffleGenerator(DummyGenerator(), seed=42)
    prompt = "$[red car, blue car, green car]$ <lora:carmodel:0.8> <hypernet:test:1>"

    result = generator.generate(prompt, 1)[0]

    # Special syntax should be preserved
    assert "<lora:carmodel:0.8>" in result
    assert "<hypernet:test:1>" in result

    # All car colors should be present
    assert "red car" in result
    assert "blue car" in result
    assert "green car" in result


def test_shuffle_with_seeds():
    """Test that different seeds produce different shuffles."""
    generator = WordShuffleGenerator(DummyGenerator())
    prompt = "$[a, b, c, d, e]$"

    # Generate with different seeds
    result1 = generator.generate(prompt, 1, seeds=[123])[0]
    result2 = generator.generate(prompt, 1, seeds=[456])[0]

    # Both should have all elements
    for char in ["a", "b", "c", "d", "e"]:
        assert char in result1
        assert char in result2

    # But they should be in different orders
    assert result1 != result2


def test_shuffle_same_seed_same_result():
    """Test that the same seed produces the same shuffle."""
    generator = WordShuffleGenerator(DummyGenerator())
    prompt = "$[a, b, c, d, e]$"

    # Generate multiple times with the same seed
    result1 = generator.generate(prompt, 1, seeds=[999])[0]
    result2 = generator.generate(prompt, 1, seeds=[999])[0]

    # Should produce identical results
    assert result1 == result2


def test_custom_delimiters():
    """Test that custom shuffle delimiters work."""
    generator = WordShuffleGenerator(
        DummyGenerator(),
        seed=42,
        shuffle_start="<<",
        shuffle_end=">>",
    )
    prompt = "a <<red, blue, green>> car"

    result = generator.generate(prompt, 1)[0]

    assert "red" in result
    assert "blue" in result
    assert "green" in result
    assert "car" in result


def test_empty_shuffle_section():
    """Test that empty shuffle sections are handled gracefully."""
    generator = WordShuffleGenerator(DummyGenerator())
    prompt = "test $[]$ end"

    result = generator.generate(prompt, 1)[0]

    # Should handle gracefully without crashing
    assert "test" in result
    assert "end" in result
