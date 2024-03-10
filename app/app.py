import chainlit as cl
from chainlit.playground.config import add_llm_provider
from chainlit.playground.providers.langchain import LangchainGenericProvider
from langchain.llms.ollama import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable, RunnablePassthrough
from langchain.schema.runnable.config import RunnableConfig
from langchain_community.utilities.sql_database import SQLDatabase

db = SQLDatabase.from_uri(
    database_uri="mysql://user:password@host.docker.internal:3306/classicmodels",
    sample_rows_in_table_info=0
)

def get_schema(_):
    return db.get_table_info()

def run_query(query):
    return db.run(query)

# Instantiate the LLM
llm = Ollama(
    model="mistral:7b-instruct-v0.2-q5_K_M",
    base_url="http://host.docker.internal:11434"
)

# Add the LLM provider
add_llm_provider(
    LangchainGenericProvider(
        # It is important that the id of the provider matches the _llm_type
        id=llm._llm_type,
        # The name is not important. It will be displayed in the UI.
        name="Database Chat using LangChain",
        # This should always be a Langchain llm instance (correctly configured)
        llm=llm,
        # If the LLM works with messages, set this to True
        is_chat=False,
    )
)


@cl.on_chat_start
async def on_chat_start():
    prompt = ChatPromptTemplate.from_template(
        """
        I want you to return just a SQL query in your response to below prompt. Do not return any text other than a SQL query enclosed in triple back-ticks (```).

        Generate a SQL query that answers the question "{question}".

        This query should be able to run on a database that has the below schema:
        {schema}
        """
    )
    runnable = RunnablePassthrough.assign(schema=get_schema) | prompt | llm | StrOutputParser()

    cl.user_session.set("runnable", runnable)


@cl.on_message
async def on_message(message: cl.Message):
    runnable = cl.user_session.get("runnable")  # type: Runnable

    msg = cl.Message(content="")

    async for chunk in runnable.astream(
        {"question": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ): await msg.stream_token(chunk)

    await msg.send()
