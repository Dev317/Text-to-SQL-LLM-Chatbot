import os
import streamlit as st
import logging
from utils.chromadb_connector import ChromaDBConnection
import pandas as pd
import tempfile


FOLDER = f"{tempfile.gettempdir()}/upload"
CHROMA_PATH = f"{tempfile.gettempdir()}/chroma"
st.session_state['UPLOAD_FOLDER'] = FOLDER
st.session_state["CHROMA_PATH"] = CHROMA_PATH
if not os.path.exists(FOLDER):
    logging.info("Creating upload folder!")
    os.makedirs(FOLDER)


st.header("ChromaDB Connection")

client_type = st.selectbox(
    label="Client Type",
    options=['PersistentClient']
)

if not client_type:
    st.warning("Please select a Client Type")


st.session_state['configuration'] = {}
st.session_state['configuration']['client_type'] = client_type

if client_type == "PersistentClient":
    persistent_path = st.text_input(
        label="Persistent directory",
        value=CHROMA_PATH
    )
    st.session_state['configuration']['path'] = persistent_path

def connectChroma():
    try:
        st.session_state['conn'] = st.experimental_connection(name="chromadb",
                                                              type=ChromaDBConnection,
                                                              **st.session_state['configuration'])

        st.session_state['is_connected'] = True
        st.toast(body='Connection established!', icon='✅')

        st.session_state['chroma_collections'] = st.session_state['conn'].get_collection_names()

    except Exception as ex:
        logging.error(f"Error: {str(ex)}")
        st.toast(body="Failed to establish connection",
                icon="❌")

def get_vis(df: pd.DataFrame):
    import plotly.express as px
    from sklearn.decomposition import PCA

    embeddings = df['embeddings'].tolist()
    source = df['metadatas'].tolist()
    cols = [str(i) for i in range(len(embeddings[0]))]
    new_data = {}
    new_data['source'] = [s['source'] for s in source]
    for col in cols:
        new_data[col] = []

    for embedding in embeddings:
        for idx, value in enumerate(embedding):
            new_data[str(idx)].append(value)

    embedding_df = pd.DataFrame(data=new_data)

    X = embedding_df[cols]

    pca = PCA(n_components=3)
    components = pca.fit_transform(X)

    return px.scatter_3d(
        components, x=0, y=1, z=2,
        labels={'0': 'x', '1': 'y', '2': 'z'},
        color=embedding_df['source']
    )


st.button(label="Connect", on_click=connectChroma)

if 'chroma_collections' in st.session_state:
    selected_collection_placeholder = st.empty()
    st.session_state['selected_collection'] = selected_collection_placeholder.selectbox(
            label="Chroma collections",
            options=st.session_state['chroma_collections']
        )

    delete_button_placeholder = st.empty()

    if 'selected_collection' in st.session_state:
        if st.session_state['selected_collection']:
            df = st.session_state["conn"].get_collection_data(collection_name=st.session_state['selected_collection'])
            embedding_data_placeholder = st.empty()
            with embedding_data_placeholder.container():
                st.subheader("Embedding data")
                st.markdown("Dataframe:")

                dataframe_placeholder = st.empty()
                dataframe_placeholder.dataframe(df)

                vis_placeholder = st.empty()
                if len(df):
                    vis_placeholder.plotly_chart(get_vis(df), theme="streamlit", use_container_width=True)


    if len(st.session_state['chroma_collections']) != 0 \
            and delete_button_placeholder.button(label="❗ Delete collection", type="primary"):
        st.cache_resource.clear()
        try:
            st.session_state["conn"].delete_collection(collection_name=st.session_state['selected_collection'])
            st.toast(body='Collection deleted!', icon='✅')
        except Exception as ex:
            st.toast(body=f"{str(ex)}", icon="⚠️")
            st.toast(body="Failed to delete connection", icon="😢")

        st.session_state['chroma_collections'] = st.session_state["conn"].get_collection_names()
        if len(st.session_state['chroma_collections']) == 0:
            delete_button_placeholder.empty()
            embedding_data_placeholder.empty()
            vis_placeholder.empty()
            st.session_state.pop('selected_collection')
        else:
            with embedding_data_placeholder.container():
                st.subheader("Embedding data")
                st.markdown("Dataframe:")

                dataframe_placeholder = st.empty()
                dataframe_placeholder.dataframe(df)

        selected_collection_placeholder.empty()
        st.session_state['selected_collection'] = selected_collection_placeholder.selectbox(
            label="Chroma collections",
            options=st.session_state['chroma_collections'],
            key="new_select_box_after_delete"
        )

