import chainlit as cl
from operator import itemgetter
from chainlit.playground.config import add_llm_provider
from chainlit.playground.providers.langchain import LangchainGenericProvider
from langchain.llms.ollama import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.runnables import RunnableLambda
from langchain.schema.runnable.config import RunnableConfig
from langchain_community.utilities.sql_database import SQLDatabase
import os

# Create a connection to the database
db = SQLDatabase.from_uri(
    database_uri=os.getenv('DATABASE_URI'),
    sample_rows_in_table_info=0
)

# Create a connection to the LLM
llm = Ollama(
    model=os.getenv('LLM_MODEL'),
    base_url=os.getenv('LLM_URL'),
    verbose=1
)

# The prompt to generate a valid SQL query using the LLM
query_writer_prompt = ChatPromptTemplate.from_template(
    """
    I want you to return just a plain-text SQL query (No comments) in your response to below prompt. Do not return any text other than a SQL query.

    Generate a SQL query that answers the question "{question}".

    This query should be able to run on a database that has the below schema:
    {schema}
    """
)

# The prompt to interpret the SQL query result and form a formatted answer
result_interpreter_prompt = ChatPromptTemplate.from_template(
    """
    Without mentioning the "query" below, summarise the below "data" which answers the question "{question}". Use tables for formatting as you deem fit.

    Data:
    {result}

    Query:
    {query}
    """
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
        is_chat=False
    )
)

@cl.on_chat_start
async def on_chat_start():
    
    # Passes the {question} and database {schema} to the LLM an produces a query
    query_writer_chain = query_writer_prompt | llm | StrOutputParser()

    # Executes the generated {query} against the database and returns the data
    query_executor_chain = itemgetter("query") | RunnableLambda(db.run)

    # Summarises the {result} data with reference to the {question} and {query}
    query_interpreter_chain = result_interpreter_prompt | llm | StrOutputParser()
    
    # Takes in the {question} and produces a natural language answer using other chains
    question_to_answer_chain = (
        {
            "question": itemgetter("question"),
            "schema": RunnableLambda(lambda x: db.get_table_info())
        }
        | RunnablePassthrough.assign(query=query_writer_chain)
        | RunnablePassthrough.assign(result=query_executor_chain)
        | query_interpreter_chain
    )

    # Save the entrypoint chain to the session to be used in the on_message handler
    cl.user_session.set("runnable", question_to_answer_chain)


@cl.on_message
async def on_message(message: cl.Message):

    # Get the entrypoint chain from the session
    runnable = cl.user_session.get("runnable")

    msg = cl.Message(content="")

    # Stream the response from the chains to the UI
    async for chunk in runnable.astream(
        {"question":message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ): await msg.stream_token(chunk)

    await msg.send()
