import os
from langchain.prompts.prompt import PromptTemplate
from langchain.llms import VertexAI, OpenAI
from langchain.embeddings.vertexai import VertexAIEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
import tempfile
import shutil
from langchain.document_loaders.csv_loader import CSVLoader



class ConversationalChain():

    def __init__(self, model_name, **kwargs):
        if model_name == "VertexAI":
            self.embeddings = VertexAIEmbeddings()
            self.llm = VertexAI()
            self.vector_store = self.get_vector_store(model_name)
        elif model_name == "OpenAI":
            self.embeddings = OpenAIEmbeddings(openai_api_key=kwargs.get("api_key"))
            self.llm = OpenAI(openai_api_key=kwargs.get("api_key"))
            self.vector_store = self.get_vector_store(model_name)
        self.load_sql_to_db()
        self.chain = self.get_chain()

    def get_vector_store(self, model_name):
        if os.path.exists(f'{tempfile.gettempdir()}/chroma'):
            shutil.rmtree(f'{tempfile.gettempdir()}/chroma')

        if model_name == "VertexAI":
            return Chroma(
                collection_name="Vertex_data",
                embedding_function=self.embeddings,
                persist_directory=f'{tempfile.gettempdir()}/chroma'
            )
        elif model_name == "OpenAI":
            return Chroma(
                collection_name="OpenAI_data",
                embedding_function=self.embeddings,
                persist_directory=f'{tempfile.gettempdir()}/chroma'
            )

    def load_sql_to_db(self):
        DATA_FOLDER = './data'
        try:
            # loader = DirectoryLoader(DATA_FOLDER, glob="*.*")
            loader = DirectoryLoader(DATA_FOLDER, glob='*.csv', loader_cls=CSVLoader)
            documents = loader.load()
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            docs = text_splitter.split_documents(documents)
            texts = [doc.page_content for doc in docs]
            metadatas = [doc.metadata for doc in docs]
            self.vector_store.add_texts(texts=texts, metadatas=metadatas)

        except Exception as ex:
            raise ex

    def get_chain(self):
        condense_question_template = """Considering the provided chat history and a subsequent question, rewrite the follow-up question to be an independent query. Alternatively, conclude the conversation if it appears to be complete.
Chat History:\"""
{chat_history}
\"""
Follow Up Input: \"""
{question}
\"""
Standalone question:"""

        qa_template = """
You're an AI assistant specializing in data analysis with Google SQL.
When providing responses, strive to exhibit friendliness and adopt a conversational tone,
similar to how a friend or tutor would communicate.

When asked about your capabilities, provide a general overview of your ability to assist with data analysis tasks using Google SQL,
instead of performing specific SQL queries.

Based on the question provided, if it pertains to data analysis or Google SQL tasks, generate SQL code that is compatible with the Google SQL environment.
Additionally, offer a brief explanation about how you arrived at the SQL code. If the required column isn't explicitly stated in the context, suggest an alternative using available columns,
but do not assume the existence of any columns that are not mentioned.
Also, do not modify the database in any way (no insert, update, or delete operations).
You are only allowed to query the database. Refrain from using the information schema.

Do use the data below that is provided to generate accurate Google SQL.
{context}


**You are only required to write one SQL query per question.**

If the question or context does not clearly involve Google SQL or data analysis tasks, respond appropriately without generating Google SQL queries.

When the user expresses gratitude or says "Thanks", interpret it as a signal to conclude the conversation. Respond with an appropriate closing statement without generating further SQL queries.

If you don't know the answer, simply state, "I'm sorry, I don't know the answer to your question."

Write your response in markdown format and code in ```sql  ```.

Do use the DDL schema to generate accurate Google SQL.

Question: ```{question}```

Answer:
"""

        CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(condense_question_template)
        QA_PROMPT = PromptTemplate(template=qa_template, input_variables=["question", "context"])
        question_generator = LLMChain(llm=self.llm, prompt=CONDENSE_QUESTION_PROMPT)
        doc_chain = load_qa_chain(llm=self.llm, chain_type="stuff", prompt=QA_PROMPT)
        conv_chain = ConversationalRetrievalChain(
            retriever=self.vector_store.as_retriever(),
            combine_docs_chain=doc_chain,
            question_generator=question_generator,
        )
        return conv_chain
