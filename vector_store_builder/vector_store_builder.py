class VectorStoreBuilder:
    """
    Builds and saves a FAISS vector store from a list of text file paths.
    """

    def __init__(self, model_name='all-MiniLM-L6-v2'):
        import warnings
        import os

        # Suppress Python warnings
        warnings.filterwarnings("ignore")

        # Suppress TensorFlow logs
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        from langchain_huggingface import HuggingFaceEmbeddings
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)

    def build_and_save(self, doc_paths, save_path):
        """
        Builds a FAISS vector store from a list of text file paths and saves it.

        Parameters:
        - doc_paths (list of str): Paths to the text files.
        - save_path (str): Directory where the FAISS index will be saved.
        """
        from langchain_community.vectorstores import FAISS

        # Load documents
        documents = []
        for path in doc_paths:
            with open(path, 'r', encoding='utf-8') as f:
                documents.append(f.read())

        # Build and save FAISS index
        vector_store = FAISS.from_texts(documents, self.embeddings)
        vector_store.save_local(save_path)
