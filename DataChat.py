import streamlit as st
from utils.bigqueryddl import Bigqueryddl
from utils.chain import ConversationalChain
from utils.query import get_dataframe, get_sql

st.set_page_config(
    page_title='DA Penguin Chat',
    page_icon='https://static.wixstatic.com/media/d194d0_43975bd8ec1a43dc9a9483f4f9632d2a~mv2.png/v1/fill/w_1516,h_1330,al_c/d194d0_43975bd8ec1a43dc9a9483f4f9632d2a~mv2.png'
)


avatar_map = {
    "user": "https://cdn-icons-png.flaticon.com/512/6813/6813541.png",
    "assistant": "https://cdn-icons-png.flaticon.com/512/826/826963.png",
}


bigqueryddl = Bigqueryddl()

st.title("🐧 PenguinChat")

st.caption("Ask Data Anything")

def get_chain():
    print("Initialize LLM")
    if st.session_state['model'] == "OpenAI":
        api_key = st.sidebar.text_input(label="api_key", type="password")

        if not api_key.startswith("sk-"):
            st.toast("Please put a valid api key", icon="⚠️")
        else:
            st.session_state["chain"] = ConversationalChain(model_name=st.session_state["model"], api_key=api_key).chain
    else:
        st.session_state["chain"] = ConversationalChain(model_name=st.session_state["model"]).chain
    return st.session_state["chain"]


model = st.radio(
    label="model",
    options=["VertexAI", "OpenAI"],
    index=0,
    horizontal=True,
    on_change=get_chain,
    key='model'
)


INITIAL_MESSAGE = [
    {
        "role": "assistant",
        "content": "Hey there, I'm PenguinBot, your DA Data Expert, ready to address any data-related questions that you enquire 🔍",
    },
]

with open("ui/sidebar.md", "r") as sidebar_file:
    sidebar_content = sidebar_file.read()

st.sidebar.markdown(sidebar_content)

selected_table = st.sidebar.selectbox(
    "Select a table:", options=list(bigqueryddl.ddl_dict.keys())
)
st.sidebar.markdown(f"### Schema for `{selected_table}` table")
st.sidebar.code(bigqueryddl.ddl_dict[selected_table], language="sql")

st.session_state["schema"] = bigqueryddl.ddl_dict[selected_table]

# Add a reset button
if st.sidebar.button("Reset Chat"):
    for key in st.session_state.keys():
        del st.session_state[key]
    st.session_state["messages"] = INITIAL_MESSAGE
    st.session_state["history"] = []

# Initialize the chat messages history
if "messages" not in st.session_state:
    st.session_state["messages"] = INITIAL_MESSAGE

if "history" not in st.session_state:
    st.session_state["history"] = []

if "chain" not in st.session_state:
    st.session_state["chain"] = get_chain()

for message in st.session_state["messages"]:
    with st.chat_message(message["role"], avatar=avatar_map[message["role"]]):
        st.markdown(message["content"])


if prompt := st.chat_input():
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=avatar_map["user"]):
        st.markdown(prompt)

    if isinstance(prompt, str):
        chain = st.session_state["chain"]
        qa_prompt = prompt + f"The DDL schema is: \n {st.session_state['schema']}"
        result = chain({"question": qa_prompt,
                        "chat_history": st.session_state["history"]})["answer"]
        st.session_state["history"].append((prompt, result))
        st.session_state["messages"].append({"role": "assistant", "content": result})

        # extract_sql = get_sql(result)
        # result_df = get_dataframe(extract_sql)
        # st.session_state["assistant_messages"].append({"role": "assistant", "content": result_df})

    with st.chat_message("assistant", avatar=avatar_map["assistant"]):
        st.markdown(result)
        # df_response = st.session_state["assistant_messages"][-1]["content"]
        # st.markdown("\n" + df_response.to_markdown())
