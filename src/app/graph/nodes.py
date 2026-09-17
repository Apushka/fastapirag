from src.app.graph.state import State
from src.settings import settings
from src.app.graph.tools import tools
from langchain_core.messages import SystemMessage
from langchain_ollama import ChatOllama


async def call_model(state: State) -> State:
    llm = ChatOllama(model=settings.llm_model)
    llm_with_tools = llm.bind_tools(tools)

    system = SystemMessage(
        content=(
            "Ты — агент управления задачами и работы с документацией.\n"
            "\n"
            "Если пользователь задаёт вопрос по документации:\n"
            "1. Сначала вызови search_documentation для поиска.\n"
            "2. Если результаты найдены — ответь на основе них.\n"
            "3. Если результаты пусты или нет ответа — скажи: 'Я не могу ответить на этот вопрос'.\n"
            "\n"
            "Инструменты:\n"
            "1. search_documentation — поиск по документации\n"
            "2. create_task — создание задачи\n"
            "3. add_comment — добавление комментария\n"
            "\n"
            f"Текущий task_id: {state.task_id or 'не задан'}."
        )
    )
    messages = [system] + state.messages
    return {"messages": [await llm_with_tools.ainvoke(messages)]}
