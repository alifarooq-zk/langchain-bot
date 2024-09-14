from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_community.chat_message_histories import FileChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_openai.chat_models import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()


memory = InMemoryChatMessageHistory()


def get_history():
    return FileChatMessageHistory('messages.json')


prompt = ChatPromptTemplate(messages=[
    MessagesPlaceholder(variable_name='messages'),
    ('human', '{content}'),
]
)

lcel_chain = prompt | model | StrOutputParser()

chain = RunnableWithMessageHistory(
    lcel_chain,
    get_history,
    input_messages_key="content",
    history_messages_key="messages"
)


while True:
    content = input(">> ")
    result = chain.invoke(
        {'content': content})

    print(result)
