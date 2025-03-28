from typing import Any, Iterable, List, Optional
from utils import audio_embedding
import uuid
import LanceDB

class MultimodalLanceDB(LanceDB):
    """`LanceDB` vector store to process multimodal data

    Args:
        connection: LanceDB connection to use. If not provided, a new connection
                    will be created.
        embedding: Embedding to use for the vectorstore.
        vector_key: Key to use for the vector in the database. Defaults to ``vector``.
        id_key: Key to use for the id in the database. Defaults to ``id``.
        text_key: Key to use for the text in the database. Defaults to ``text``.
        audio_path_key: Key to use for the path to image in the database. Defaults to ``image_path``.
        table_name: Name of the table to use. Defaults to ``vectorstore``.
        mode: Mode to use for adding data to the table. Defaults to ``overwrite``.

    Example:
        .. code-block:: python
            vectorstore = MultimodalLanceDB(uri='/lancedb', embedding_function)
            vectorstore.add_texts(['text1', 'text2'])
            result = vectorstore.similarity_search('text1')
    """
    
    def __init__(
        self,
        connection: Optional[Any] = None,
        embedding: Optional[function] = None,
        uri: Optional[str] = "/tmp/lancedb",
        vector_key: Optional[str] = "vector",
        id_key: Optional[str] = "id",
        text_key: Optional[str] = "text",
        audio_path_key: Optional[str] = "audio_path", 
        table_name: Optional[str] = "vectorstore",
        mode: Optional[str] = "append",
    ):
        super(MultimodalLanceDB, self).__init__(connection, embedding, uri, vector_key, id_key, text_key, table_name, mode)
        self._audio_path_key = audio_path_key
        
    def add_text_audio_pairs(
        self,
        texts: Iterable[str],
        audio_paths: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Turn text-audio pairs into embedding and add it to the database

        Args:
            texts: Iterable of strings to add as descriptive metadata (not used for embeddings).
            audios: Iterable of path-to-audio files to generate embeddings.
            metadatas: Optional list of metadatas associated with the texts.
            ids: Optional list of ids to associate with the texts.

        Returns:
            List of ids of the added audio embeddings.
        """
        # the length of texts must be equal to the length of audios
        assert len(texts)==len(audio_paths), "the len of transcripts should be equal to the len of audios"

        # Embed texts and create documents
        docs = []
        ids = ids or [str(uuid.uuid4()) for _ in texts]
        embeddings = self._embedding.audio_embedding(audio_paths=list(audio_paths))  # type: ignore
        for idx, text in enumerate(texts):
            embedding = embeddings[idx]
            metadata = metadatas[idx] if metadatas else {"id": ids[idx]}
            docs.append(
                {
                    self._vector_key: embedding,
                    self._id_key: ids[idx],
                    self._text_key: text,
                    self._audio_path_key: audio_paths[idx],
                    "metadata": metadata,
                }
            )

        if 'mode' in kwargs:
            mode = kwargs['mode']
        else:
            mode = self.mode
        if self._table_name in self._connection.table_names():
            tbl = self._connection.open_table(self._table_name)
            if self.api_key is None:
                tbl.add(docs, mode=mode)
            else:
                tbl.add(docs)
        else:
            self._connection.create_table(self._table_name, data=docs)
        return ids

    @classmethod
    def from_text_audio_pairs(
        cls,
        texts: List[str],
        image_paths: List[str],
        embedding: audio_embedding,
        metadatas: Optional[List[dict]] = None,
        connection: Any = None,
        vector_key: Optional[str] = "vector",
        id_key: Optional[str] = "id",
        text_key: Optional[str] = "text",
        audio_path_key: Optional[str] = "audio_path",
        table_name: Optional[str] = "vectorstore",
        **kwargs: Any,
    ):

        instance = MultimodalLanceDB(
            connection=connection,
            embedding=embedding,
            vector_key=vector_key,
            id_key=id_key,
            text_key=text_key,
            audio_path_key=audio_path_key,
            table_name=table_name,
        )
        instance.add_text_image_pairs(texts, image_paths, metadatas=metadatas, **kwargs)

        return instance