import pinecone
import time
import streamlit as st
from langchain import FewShotPromptTemplate
from langchain.vectorstores import Pinecone
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains.conversation.memory import ConversationBufferMemory
                                                    
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.embeddings.openai import OpenAIEmbeddings


def get_vectorstore():
    ## Embedding
    model_name = 'text-embedding-ada-002'

    embed = OpenAIEmbeddings(
        model=model_name,
        openai_api_key=st.secrets["open_api_key"]
    )

    ## Create vector database
    index_name = 'ph-dpo'

    pinecone.init(
        api_key=st.secrets["pinecone_api_key"],
        environment='us-west4-gcp-free'
    )

    ## Call the existing index
    docsearch = Pinecone.from_existing_index(index_name, embed)

    return docsearch

## Fewshotprompt examples
# create our examples
examples = [
    {
        "question": "is age spi",
        "answer": """Yes, age is considered sensitive personal information according to the provided context. 
        Sensitive personal information includes personal information about an individual's age.

        Source: Data Privacy Act of 2012 - SEC.3(l)"""
    },
    {
        "question": "question",
        "answer": """ answer

        Sources: Data Privacy Act of 2012 - SEC.16, SEC.17"""
    }
]

# create a example template
example_template = """
User: {question}
AI: {answer}
"""

# create a prompt example from above template
example_prompt = PromptTemplate(
    input_variables=["question", "answer"],
    template=example_template
)

# now break our previous prompt into a prefix and suffix
# the prefix is our instructions
prefix = """As an AI assisstant, your task is to provide answers in laymanized term from the provided
CONTEXT of data privacy laws. 

1. Please analyze before giving an answer according to the given legal approach. 
   Approach: Issue, rule, application, conclusion.
2. Avoid legal jargon words
2. DO NOT give any information not mentioned in the CONTEXT INFORMATION.
3. If the question is out-of-context. Remind the user that you are only answering about data privacy laws.
4. BE HONEST if you are not sure or do not know the answer, advise to consult with legal 
   experts or the relevant authorities to ensure compliance with data privacy laws.


{context}

Moreover, the following are exerpts from conversations with an AI
assistant. Do not fabricate references, Make sure that you give the correct source or sources.
Only state a source when it's needed. Include the Title, section number/letter of the source.
Here are few examples:
"""
# and the suffix our user input and output indicator
suffix = """
User: {question}
AI: """

# now create the few shot prompt template
few_shot_prompt_template = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix=prefix,
    suffix=suffix,
    input_variables=["context", "question"],
    example_separator="\n" # What will separate each one of those examples within the prompt that we're building
)


def get_conversation_chain(vectorstore):
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True
    )


    # completion llm
    llm = ChatOpenAI(
        openai_api_key=st.secrets["open_api_key"],
        model_name='gpt-3.5-turbo',
        temperature=0.0
    )

    combine_docs_chain_kwargs = {"prompt": few_shot_prompt_template}
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        combine_docs_chain_kwargs=combine_docs_chain_kwargs,
        memory=memory
    )

    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    # st.write(response)
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            with st.chat_message("user"):
                st.markdown(message.content)
        else:
            with st.chat_message("assistant"):
                st.markdown(message.content)

def main():
    st.set_page_config(page_title="Chat with Data Privacy Laws", page_icon=":books:")

    st.markdown('Trained with '+'***Republic Act 10173 - Data Privacy Act of 2012***')

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state: 
        st.session_state.chat_history = None

    st.header('Chat with Data Privacy Laws :books:')
    
    user_question = st.chat_input('Ask a question about data privacy laws')
    if user_question:
        handle_userinput(user_question)

    # st.write(bot_template.replace("{{MSG}}", "hello human"), unsafe_allow_html=True)

    st.markdown('Click on **"Process"** to begin or restart the conversation')
    if st.button("Process"): # Becomes true when the user clicks
        with st.spinner("Processing"):
            docsearch = get_vectorstore()

            # Conversation chain
            st.session_state.conversation = get_conversation_chain(docsearch)

    # st.write(st.session_state)
    
if __name__ == "__main__":
    main()
