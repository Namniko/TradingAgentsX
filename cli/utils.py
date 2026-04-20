import questionary
from typing import List, Optional, Tuple, Dict

from rich.console import Console

from cli.models import AnalystType
from tradingagents.llm_clients.model_catalog import get_model_options

console = Console()

TICKER_INPUT_EXAMPLES = "Examples: SPY, CNC.TO, 7203.T, 0700.HK"

ANALYST_ORDER = [
    ("Market Analyst", AnalystType.MARKET),
    ("Social Media Analyst", AnalystType.SOCIAL),
    ("News Analyst", AnalystType.NEWS),
    ("Fundamentals Analyst", AnalystType.FUNDAMENTALS),
]

PROVIDERS = [
    ("OpenAI", "openai", "https://api.openai.com/v1"),
    ("Google", "google", None),
    ("Anthropic", "anthropic", "https://api.anthropic.com/"),
    ("xAI", "xai", "https://api.x.ai/v1"),
    ("DeepSeek", "deepseek", "https://api.deepseek.com"),
    ("Qwen", "qwen", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    ("GLM", "glm", "https://open.bigmodel.cn/api/paas/v4/"),
    ("OpenRouter", "openrouter", "https://openrouter.ai/api/v1"),
    ("Azure OpenAI", "azure", None),
    ("Ollama", "ollama", "http://localhost:11434/v1"),
]


def _move_preferred_choice_first(options, preferred_value, value_index=1):
    """Move a preferred option to the first slot so Enter selects it by default."""
    if not preferred_value:
        return list(options)

    preferred_value = preferred_value.strip()
    prioritized = [item for item in options if item[value_index] == preferred_value]
    remaining = [item for item in options if item[value_index] != preferred_value]
    return prioritized + remaining


def _build_model_choices(
    provider: str, mode: str, default_model: Optional[str] = None
):
    """Build model choices, prepending a configured default when needed."""
    options = list(get_model_options(provider, mode))
    if default_model:
        default_model = default_model.strip()
        if any(value == default_model for _, value in options):
            options = _move_preferred_choice_first(options, default_model)
        else:
            options = [
                (f"Configured default ({default_model})", default_model),
                *options,
            ]

    return [questionary.Choice(display, value=value) for display, value in options]


def get_ticker() -> str:
    """Prompt the user to enter a ticker symbol."""
    ticker = questionary.text(
        f"Enter the exact ticker symbol to analyze ({TICKER_INPUT_EXAMPLES}):",
        validate=lambda x: len(x.strip()) > 0 or "Please enter a valid ticker symbol.",
        style=questionary.Style(
            [
                ("text", "fg:green"),
                ("highlighted", "noinherit"),
            ]
        ),
    ).ask()

    if not ticker:
        console.print("\n[red]No ticker symbol provided. Exiting...[/red]")
        exit(1)

    return normalize_ticker_symbol(ticker)


def normalize_ticker_symbol(ticker: str) -> str:
    """Normalize ticker input while preserving exchange suffixes."""
    return ticker.strip().upper()


def get_analysis_date() -> str:
    """Prompt the user to enter a date in YYYY-MM-DD format."""
    import re
    from datetime import datetime

    def validate_date(date_str: str) -> bool:
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", date_str):
            return False
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
            return True
        except ValueError:
            return False

    date = questionary.text(
        "Enter the analysis date (YYYY-MM-DD):",
        validate=lambda x: validate_date(x.strip())
        or "Please enter a valid date in YYYY-MM-DD format.",
        style=questionary.Style(
            [
                ("text", "fg:green"),
                ("highlighted", "noinherit"),
            ]
        ),
    ).ask()

    if not date:
        console.print("\n[red]No date provided. Exiting...[/red]")
        exit(1)

    return date.strip()


def select_analysts() -> List[AnalystType]:
    """Select analysts using an interactive checkbox."""
    choices = questionary.checkbox(
        "Select Your [Analysts Team]:",
        choices=[
            questionary.Choice(display, value=value) for display, value in ANALYST_ORDER
        ],
        instruction="\n- Press Space to select/unselect analysts\n- Press 'a' to select/unselect all\n- Press Enter when done",
        validate=lambda x: len(x) > 0 or "You must select at least one analyst.",
        style=questionary.Style(
            [
                ("checkbox-selected", "fg:green"),
                ("selected", "fg:green noinherit"),
                ("highlighted", "noinherit"),
                ("pointer", "noinherit"),
            ]
        ),
    ).ask()

    if not choices:
        console.print("\n[red]No analysts selected. Exiting...[/red]")
        exit(1)

    return choices


def select_research_depth() -> int:
    """Select research depth using an interactive selection."""
    depth_options = [
        ("Shallow - Quick research, few debate and strategy discussion rounds", 1),
        ("Medium - Middle ground, moderate debate rounds and strategy discussion", 3),
        ("Deep - Comprehensive research, in depth debate and strategy discussion", 5),
    ]

    choice = questionary.select(
        "Select Your [Research Depth]:",
        choices=[
            questionary.Choice(display, value=value)
            for display, value in depth_options
        ],
        instruction="\n- Use arrow keys to navigate\n- Press Enter to select",
        style=questionary.Style(
            [
                ("selected", "fg:yellow noinherit"),
                ("highlighted", "fg:yellow noinherit"),
                ("pointer", "fg:yellow noinherit"),
            ]
        ),
    ).ask()

    if choice is None:
        console.print("\n[red]No research depth selected. Exiting...[/red]")
        exit(1)

    return choice


def _fetch_openrouter_models() -> List[Tuple[str, str]]:
    """Fetch available models from the OpenRouter API."""
    import requests

    try:
        resp = requests.get("https://openrouter.ai/api/v1/models", timeout=10)
        resp.raise_for_status()
        models = resp.json().get("data", [])
        return [(m.get("name") or m["id"], m["id"]) for m in models]
    except Exception as e:
        console.print(f"\n[yellow]Could not fetch OpenRouter models: {e}[/yellow]")
        return []


def select_openrouter_model(default_model: Optional[str] = None) -> str:
    """Select an OpenRouter model from the newest available, or enter a custom ID."""
    models = _fetch_openrouter_models()
    model_choices = models[:5]

    if default_model:
        default_model = default_model.strip()
        if any(mid == default_model for _, mid in model_choices):
            model_choices = _move_preferred_choice_first(model_choices, default_model)
        else:
            model_choices = [
                (f"Configured default ({default_model})", default_model),
                *model_choices,
            ]

    choices = [questionary.Choice(name, value=mid) for name, mid in model_choices]
    choices.append(questionary.Choice("Custom model ID", value="custom"))

    choice = questionary.select(
        "Select OpenRouter Model (latest available):",
        choices=choices,
        instruction="\n- Use arrow keys to navigate\n- Press Enter to select",
        style=questionary.Style(
            [
                ("selected", "fg:magenta noinherit"),
                ("highlighted", "fg:magenta noinherit"),
                ("pointer", "fg:magenta noinherit"),
            ]
        ),
    ).ask()

    if choice is None or choice == "custom":
        return questionary.text(
            "Enter OpenRouter model ID (e.g. google/gemma-4-26b-a4b-it):",
            validate=lambda x: len(x.strip()) > 0 or "Please enter a model ID.",
            default=default_model or "",
        ).ask().strip()

    return choice


def _prompt_custom_model_id(default_model: Optional[str] = None) -> str:
    """Prompt user to type a custom model ID."""
    return questionary.text(
        "Enter model ID:",
        validate=lambda x: len(x.strip()) > 0 or "Please enter a model ID.",
        default=default_model or "",
    ).ask().strip()


def _select_model(
    provider: str, mode: str, default_model: Optional[str] = None
) -> str:
    """Select a model for the given provider and mode (quick/deep)."""
    if provider.lower() == "openrouter":
        return select_openrouter_model(default_model=default_model)

    if provider.lower() == "azure":
        return questionary.text(
            f"Enter Azure deployment name ({mode}-thinking):",
            validate=lambda x: len(x.strip()) > 0 or "Please enter a deployment name.",
            default=default_model or "",
        ).ask().strip()

    choice = questionary.select(
        f"Select Your [{mode.title()}-Thinking LLM Engine]:",
        choices=_build_model_choices(provider, mode, default_model=default_model),
        instruction="\n- Use arrow keys to navigate\n- Press Enter to select",
        style=questionary.Style(
            [
                ("selected", "fg:magenta noinherit"),
                ("highlighted", "fg:magenta noinherit"),
                ("pointer", "fg:magenta noinherit"),
            ]
        ),
    ).ask()

    if choice is None:
        console.print(f"\n[red]No {mode} thinking llm engine selected. Exiting...[/red]")
        exit(1)

    if choice == "custom":
        return _prompt_custom_model_id(default_model=default_model)

    return choice


def select_shallow_thinking_agent(
    provider, default_model: Optional[str] = None
) -> str:
    """Select shallow thinking llm engine using an interactive selection."""
    return _select_model(provider, "quick", default_model=default_model)


def select_deep_thinking_agent(provider, default_model: Optional[str] = None) -> str:
    """Select deep thinking llm engine using an interactive selection."""
    return _select_model(provider, "deep", default_model=default_model)


def select_llm_provider(
    default_provider: Optional[str] = None,
) -> tuple[str, str | None]:
    """Select the LLM provider and its API endpoint."""
    provider_options = _move_preferred_choice_first(
        PROVIDERS,
        default_provider.lower() if default_provider else None,
    )

    choice = questionary.select(
        "Select your LLM Provider:",
        choices=[
            questionary.Choice(display, value=(provider_key, url))
            for display, provider_key, url in provider_options
        ],
        instruction="\n- Use arrow keys to navigate\n- Press Enter to select",
        style=questionary.Style(
            [
                ("selected", "fg:magenta noinherit"),
                ("highlighted", "fg:magenta noinherit"),
                ("pointer", "fg:magenta noinherit"),
            ]
        ),
    ).ask()

    if choice is None:
        console.print("\n[red]No LLM provider selected. Exiting...[/red]")
        exit(1)

    provider, url = choice
    return provider, url


def ask_openai_reasoning_effort(default_effort: Optional[str] = None) -> str:
    """Ask for OpenAI reasoning effort level."""
    choices = [
        questionary.Choice("Medium (Default)", "medium"),
        questionary.Choice("High (More thorough)", "high"),
        questionary.Choice("Low (Faster)", "low"),
    ]
    if default_effort:
        choices = sorted(
            choices,
            key=lambda choice: choice.value != default_effort,
        )

    return questionary.select(
        "Select Reasoning Effort:",
        choices=choices,
        style=questionary.Style(
            [
                ("selected", "fg:cyan noinherit"),
                ("highlighted", "fg:cyan noinherit"),
                ("pointer", "fg:cyan noinherit"),
            ]
        ),
    ).ask()


def ask_anthropic_effort(default_effort: Optional[str] = None) -> str | None:
    """Ask for Anthropic effort level.

    Controls token usage and response thoroughness on Claude 4.5+ and 4.6 models.
    """
    choices = [
        questionary.Choice("High (recommended)", "high"),
        questionary.Choice("Medium (balanced)", "medium"),
        questionary.Choice("Low (faster, cheaper)", "low"),
    ]
    if default_effort:
        choices = sorted(
            choices,
            key=lambda choice: choice.value != default_effort,
        )

    return questionary.select(
        "Select Effort Level:",
        choices=choices,
        style=questionary.Style(
            [
                ("selected", "fg:cyan noinherit"),
                ("highlighted", "fg:cyan noinherit"),
                ("pointer", "fg:cyan noinherit"),
            ]
        ),
    ).ask()


def ask_gemini_thinking_config(
    default_thinking_level: Optional[str] = None,
) -> str | None:
    """Ask for Gemini thinking configuration."""
    choices = [
        questionary.Choice("Enable Thinking (recommended)", "high"),
        questionary.Choice("Minimal/Disable Thinking", "minimal"),
    ]
    if default_thinking_level:
        choices = sorted(
            choices,
            key=lambda choice: choice.value != default_thinking_level,
        )

    return questionary.select(
        "Select Thinking Mode:",
        choices=choices,
        style=questionary.Style(
            [
                ("selected", "fg:green noinherit"),
                ("highlighted", "fg:green noinherit"),
                ("pointer", "fg:green noinherit"),
            ]
        ),
    ).ask()


def ask_output_language(default_language: Optional[str] = None) -> str:
    """Ask for report output language."""
    language_options = [
        ("English (default)", "English"),
        ("Chinese", "Chinese"),
        ("Japanese", "Japanese"),
        ("Korean", "Korean"),
        ("Hindi", "Hindi"),
        ("Spanish", "Spanish"),
        ("Portuguese", "Portuguese"),
        ("French", "French"),
        ("German", "German"),
        ("Arabic", "Arabic"),
        ("Russian", "Russian"),
        ("Custom language", "custom"),
    ]

    custom_default = ""
    if default_language:
        default_language = default_language.strip()
        known_language_values = {value for _, value in language_options}
        if default_language in known_language_values:
            language_options = _move_preferred_choice_first(
                language_options, default_language
            )
        else:
            custom_default = default_language
            language_options = [
                (f"Configured default ({default_language})", default_language),
                *language_options,
            ]

    choice = questionary.select(
        "Select Output Language:",
        choices=[
            questionary.Choice(label, value=value)
            for label, value in language_options
        ],
        style=questionary.Style(
            [
                ("selected", "fg:yellow noinherit"),
                ("highlighted", "fg:yellow noinherit"),
                ("pointer", "fg:yellow noinherit"),
            ]
        ),
    ).ask()

    if choice == "custom":
        return questionary.text(
            "Enter language name (e.g. Turkish, Vietnamese, Thai, Indonesian):",
            validate=lambda x: len(x.strip()) > 0 or "Please enter a language name.",
            default=custom_default,
        ).ask().strip()

    return choice
