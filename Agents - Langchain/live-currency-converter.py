"""
Live Currency Converter
-----------------------
Uses LangChain + Meta Llama 3.1 (via HuggingFace) and the ExchangeRate-API
to answer natural-language currency conversion queries.

Setup
-----
1. Install dependencies:
       pip install langchain langchain-core langchain-huggingface \
                   huggingface_hub requests python-dotenv

2. Create a .env file in the same directory with your HuggingFace token:
       HUGGINGFACEHUB_API_TOKEN=hf_...

3. Run:
       python live_currency_converter.py
"""

import json
from typing import Annotated

import requests
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import InjectedToolArg, tool
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# ---------------------------------------------------------------------------
# Environment & model setup
# ---------------------------------------------------------------------------

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="conversational",
)

model = ChatHuggingFace(llm=llm)

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

EXCHANGE_API_KEY = "d61b36cab6dff181877e9c5c"  # replace with your own key if needed


@tool
def get_conversion_factor(base_currency: str, target_currency: str) -> dict:
    """
    Fetches the currency conversion factor between a given base currency
    and a target currency using the ExchangeRate-API.
    """
    url = (
        f"https://v6.exchangerate-api.com/v6/{EXCHANGE_API_KEY}"
        f"/pair/{base_currency}/{target_currency}"
    )
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return response.json()


@tool
def convert(
    base_currency_value: float,
    conversion_rate: Annotated[float, InjectedToolArg],
) -> float:
    """
    Given a currency conversion rate, calculates the target currency value
    from a given base currency value.
    """
    return base_currency_value * conversion_rate


# ---------------------------------------------------------------------------
# Bind tools to the model
# ---------------------------------------------------------------------------

model_with_tools = model.bind_tools([get_conversion_factor, convert])

# ---------------------------------------------------------------------------
# Core conversion logic
# ---------------------------------------------------------------------------


def run_conversion(query: str) -> str:
    """
    Takes a natural-language currency query (e.g. '10 dollars in INR')
    and returns the model's final answer as a string.
    """
    messages = [HumanMessage(query)]

    # First model call — may produce tool calls
    response = model_with_tools.invoke(messages)
    messages.append(response)

    if not response.tool_calls:
        return response.content

    # Execute tool calls
    conversion_rate: float | None = None

    for tool_call in response.tool_calls:
        if tool_call["name"] == "get_conversion_factor":
            tool_message = get_conversion_factor.invoke(tool_call)
            data = json.loads(tool_message.content)
            conversion_rate = data.get("conversion_rate")
            messages.append(tool_message)

        elif tool_call["name"] == "convert":
            if conversion_rate is None:
                # Fallback: try to get rate from args if already provided
                conversion_rate = tool_call["args"].get("conversion_rate", 1.0)
            tool_call["args"]["conversion_rate"] = conversion_rate
            tool_message = convert.invoke(tool_call)
            messages.append(tool_message)

    # Final model call with tool results
    final_response = model_with_tools.invoke(messages)
    return final_response.content


# ---------------------------------------------------------------------------
# Interactive CLI loop
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 55)
    print("  Live Currency Converter  (type 'quit' to exit)")
    print("=" * 55)
    print("Examples: '100 USD to EUR', '50 pounds in rupees'\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break

        try:
            answer = run_conversion(user_input)
            print(f"Bot: {answer}\n")
        except requests.RequestException as exc:
            print(f"[Network error] Could not fetch exchange rates: {exc}\n")
        except Exception as exc:  # noqa: BLE001
            print(f"[Error] {exc}\n")


if __name__ == "__main__":
    main()