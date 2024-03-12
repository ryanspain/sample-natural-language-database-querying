import chainlit as cl
from operator import itemgetter
from chainlit.playground.config import add_llm_provider
from chainlit.playground.providers.langchain import LangchainGenericProvider
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain.chains import create_sql_query_chain
from langchain.llms.ollama import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable, RunnablePassthrough
from langchain_core.runnables import RunnableLambda
from langchain.schema.runnable.config import RunnableConfig
from langchain_community.utilities.sql_database import SQLDatabase
import os

db = SQLDatabase.from_uri(
    database_uri=os.getenv('DATABASE_URI'),
    sample_rows_in_table_info=0
)

def get_schema(_):
    return db.get_table_info()

def run_query(query):
    print(query)
    return db.run(query)

# Instantiate the LLM
llm = Ollama(
    model=os.getenv('LLM_MODEL'),
    base_url=os.getenv('LLM_URL'),
    verbose=1
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
    query_writer_prompt = ChatPromptTemplate.from_template(
        """
        I want you to return just a plain-text SQL query (No comments) in your response to below prompt. Do not return any text other than a SQL query.

        Generate a SQL query that answers the question "{question}".

        This query should be able to run on a database that has the below schema:
        {schema}
        """
    )

    query_writer_chain = (
        {
            "question": itemgetter("question"),
            "schema": get_schema
        }
        | query_writer_prompt 
        | llm 
        | StrOutputParser()
    )

    query_executor_chain = (
        query_writer_chain
        | RunnableLambda(run_query)
    )

    result_interpreter_prompt = ChatPromptTemplate.from_template(
        """
        Summarize the below data that was found which answers the question "{question}" using tables as you deem fit.

        {result}
        """
    )

    result_interpreter_chain = (
        RunnablePassthrough.assign(result=query_executor_chain)
        | result_interpreter_prompt
        | llm
        | StrOutputParser()
    )

    # chain = (
    #     query_writer_chain | query_executor_chain
    # )

    cl.user_session.set("runnable", result_interpreter_chain)


@cl.on_message
async def on_message(message: cl.Message):
    runnable = cl.user_session.get("runnable")  # type: Runnable

    msg = cl.Message(content="")

    async for chunk in runnable.astream(
        {"question": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ): await msg.stream_token(chunk)

    await msg.send()
