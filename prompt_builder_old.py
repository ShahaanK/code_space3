"""
CAMEL Annotation Pipeline - Prompt Builder
============================================
Loads label definitions from config and assembles prompts by conditionally
including/excluding markers and examples sections based on prompt settings.

When a section is disabled, it is removed cleanly — no blank lines or
orphaned headers are left behind.
"""

import random


def select_examples(examples_list, num_examples, method="fixed", rng=None):
    """
    Select examples from a label's examples list.

    Args:
        examples_list: Full list of example strings for this label.
        num_examples: How many to select (ignored when method is "all").
        method: "fixed" (first N), "random" (randomly sampled), or "all".
        rng: random.Random instance for reproducible random selection.

    Returns:
        List of selected example strings.
    """
    if not examples_list:
        return []

    if method == "all":
        return list(examples_list)

    if num_examples <= 0:
        return []

    num_examples = min(num_examples, len(examples_list))

    if method == "random" and rng:
        return rng.sample(examples_list, num_examples)
    else:
        return examples_list[:num_examples]


def build_markers_section(label_config):
    """Build the key markers section string."""
    markers = label_config.get("markers", "").strip()
    if not markers:
        return ""
    return f"Key markers: {markers}"


def build_examples_section(label_name, examples):
    """Build the examples section string from a list of selected examples."""
    if not examples:
        return ""

    lines = [f"Example of {label_name}:"]
    for ex in examples:
        lines.append(f"  - {ex}")

    return "\n".join(lines)


def clean_prompt(text):
    """
    Clean up a prompt by collapsing multiple blank lines into at most one,
    and stripping leading/trailing whitespace.
    """
    import re
    # Collapse 3+ newlines into 2 (one blank line)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def build_prompt(prompt_config, label_name, label_config, text,
                 example_selection="fixed", rng=None,
                 annotation_guidelines=""):
    """
    Build a complete prompt string for one (prompt, label, text) combination.

    Args:
        prompt_config: Dict from config prompts list (id, name, template, etc.)
        label_name: String label name (e.g. "Loyalty").
        label_config: Dict with definition, markers, examples.
        text: The social media post to annotate.
        example_selection: "fixed", "random", or "all".
        rng: random.Random instance for reproducible selection.
        annotation_guidelines: Optional guidelines text to inject.

    Returns:
        Fully assembled prompt string ready to send to the model.
    """
    template = prompt_config["template"]
    definition = label_config.get("definition", "").strip()

    # --- Conditionally build sections ---

    # Markers section
    markers_section = ""
    if prompt_config.get("include_markers", False):
        markers_section = build_markers_section(label_config)

    # Examples section
    examples_section = ""
    if prompt_config.get("include_examples", False):
        num_examples = prompt_config.get("num_examples", 0)
        selected = select_examples(
            label_config.get("examples", []),
            num_examples,
            method=example_selection,
            rng=rng,
        )
        examples_section = build_examples_section(label_name, selected)

    # --- Fill template ---

    prompt = template.format(
        text=text,
        label_name=label_name,
        label_definition=definition,
        markers_section=markers_section,
        examples_section=examples_section,
        annotation_guidelines=annotation_guidelines,
    )

    return clean_prompt(prompt)
