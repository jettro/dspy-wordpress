from typing import List

from rag4p.rag.embedding.embedder import Embedder
from rag4p.rag.model.chunk import Chunk
from rag4p.rag.store.content_store import ContentStore

from dspy_wordpress.integrations.rockset.access_rockset import AccessRockset


class RocksetContentStore(ContentStore):

    def __init__(self, rockset_access: AccessRockset, collection_name: str, workspace_name: str, embedder: Embedder):
        self.rockset_access = rockset_access
        self.collection_name = collection_name
        self.workspace_name = workspace_name
        self.embedder = embedder

    def store(self, chunks: List[Chunk]):
        results = []
        for chunk in chunks:
            properties = {
                "document_id": chunk.document_id,
                "chunk_id": chunk.chunk_id,
                "text": chunk.chunk_text,
                "total_chunks": len(chunks),
            }

            for key, value in chunk.properties.items():
                properties[key] = value

            properties["embedding"] = self.embedder.embed(chunk.chunk_text)

            response = self.rockset_access.add_document(
                workspace=self.workspace_name,
                collection=self.collection_name,
                document=properties,
            )
            results.append(response)

        print(f"Stored {len(chunks)} chunks in Rockset with {len([r for r in results if r])} successful responses.")
