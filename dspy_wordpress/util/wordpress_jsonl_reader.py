import json
from pathlib import Path

from rag4p.indexing.content_reader import ContentReader
from rag4p.indexing.input_document import InputDocument


class WordpressJsonlReader(ContentReader):
    def __init__(self, file: Path):
        if not isinstance(file, Path):
            file = Path(file)
        self.file = file

    def read(self):
        with open(self.file, 'r') as file:
            for line in file:
                data = json.loads(line)
                properties = {
                    "url": data["url"],
                    "title": data["title"],
                    "updated_at": data["updated_at"],
                    "tags": data["tags"],
                    "categories": data["categories"]
                }
                document = InputDocument(
                    document_id=str(data["post_id"]),
                    text=data["body"],
                    properties=properties
                )
                yield document
