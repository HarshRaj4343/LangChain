from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


def load_chat_history(file_path: Path):
    role_map = {
        "customer": HumanMessage,
        "human": HumanMessage,
        "user": HumanMessage,
        "amazon": AIMessage,
        "assistant": AIMessage,
    }

    history = []

    with file_path.open("r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line or ":" not in line:
                continue

            role, content = line.split(":", 1)
            role = role.strip().lower()
            content = content.strip()

            if role == "system":
                continue

            message_class = role_map.get(role)
            if message_class is not None:
                history.append(message_class(content=content))

    return history


chat_temp = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)

history_file = Path(__file__).with_name("chathistory.txt")
chat_history = load_chat_history(history_file)

chat_prompt = chat_temp.invoke(
    {"history": chat_history, "input": "What should I do if my refund is delayed?"}
)
print(chat_prompt)


# this code was altered so that it can work with the current sample chat history file, which did not contain Langchain Messages format. The original code was designed to work with a chat history file that contained messages in the format of "role: content", where role could be "system", "human", "user", "amazon", or "assistant". The altered code maps these roles to the appropriate Langchain message classes (SystemMessage, HumanMessage, AIMessage) and constructs the chat history accordingly.