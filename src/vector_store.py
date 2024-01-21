# import numpy as np
# import shortuuid
from typing import Optional, Any
from configuration import Settings
# import logging
DEFAULT_MILVUS_CONNECTION = {
    "host": Settings().MILVUS_HOST,
    "port": Settings().MILVUS_PORT,
    # "user": "",
    # "password": "",
    # "secure": False,
}
DEFAULT_SEARCH_PARAMS={"metric_type": "COSINE"}

DEFAULT_INDEX_PARAMS={
    "metric_type": "COSINE",
    "index_type": "HNSW",
    "params": {"M": 8, "efConstruction": 64},
}
index_params = {
        'metric_type': 'COSINE',
        'index_type': "IVF_FLAT",
        'params': {"nlist": 128}
    }

class Milvus(object):
    def __init__(self, connection_args=DEFAULT_MILVUS_CONNECTION, search_params=DEFAULT_SEARCH_PARAMS, index_params=index_params,  collection_name="voice_db") -> None:
        from pymilvus import Collection, utility
        self.alias = self._create_connection(connection_args)
        self.index_params = index_params
        self.search_params = search_params
        self._primary_field = "pk"
        self._vector_field = "vector"
        self.consistency_level = "Session"
        self.fields: list[str] = []
        self.col: Optional[Collection] = None
        self.collection_name = collection_name
        if utility.has_collection(collection_name, self.alias):
            self.col= Collection(collection_name, using=self.alias)
        self._init()

    def _get_index(self) -> Optional[dict[str, Any]]:
        """Return the vector index information if it exists"""
        from pymilvus import Collection

        if isinstance(self.col, Collection):
            for x in self.col.indexes:
                if x.field_name == self._vector_field:
                    return x.to_dict()
        return None

    def add_embeddings(self, embeddings: list, metadatas: Optional[list[dict]] = None, batch_size=1000, timeout=None, **kwargs):
        from pymilvus import Collection
        if not isinstance(self.col, Collection):
            self._init(embeddings, metadatas)
                # Dict to hold all insert columns
        insert_dict: dict[str, list] = {
            self._vector_field: embeddings,
        }

        # Collect the metadata into the insert dict.
        if metadatas is not None:
            for d in metadatas:
                for key, value in d.items():
                    if key in self.fields:
                        insert_dict.setdefault(key, []).append(value)

        total_count = len(embeddings)
        pks: list[str] = []

        for i in range(0, total_count, batch_size):
            # Grab end index
            end = min(i + batch_size, total_count)
            # Convert dict to list of lists batch for insertion
            insert_list = [insert_dict[x][i:end] for x in self.fields]
            # Insert into the collection.
            res: Collection
            res=self.col.insert(insert_list, timeout=timeout, **kwargs)
            pks.extend(res.primary_keys)
        return pks

    def search(self, embedding, k=1, timeout=None, **kwargs):
        output_fields = self.fields[:]
        output_fields.remove(self._vector_field)

        # print(output_fields)
        # print(self.search_params)

        res = self.col.search(
            data=[embedding],
            anns_field=self._vector_field,
            param=self.search_params,
            limit=k,
            output_fields=output_fields,
            timeout=timeout,
            **kwargs,
        )
        # Organize results.
        ret = []
        for result in res[0]:
            meta = {x: result.entity.get(x) for x in output_fields}
            pair = (meta, result.score)
            ret.append(pair)
        return ret


    def _create_connection(self, connection_args):
        from pymilvus import connections
        # alias = shortuuid.uuid()
        alias = "default"
        connections.connect( alias=alias, **connection_args)
        return alias

    def _init(
        self, embeddings: Optional[list] = None, metadatas: Optional[list[dict]] = None
    ) -> None:
        from pymilvus import Collection
        if embeddings is not None:
            self._create_collection(embeddings, metadatas)

        # Extract the fields from the schema
        if isinstance(self.col, Collection):
            schema = self.col.schema
            for x in schema.fields:
                self.fields.append(x.name)
            # Since primary field is auto-id, no need to track it
            self.fields.remove(self._primary_field)

        # Create index if it doesn't exist
        if isinstance(self.col, Collection) and self._get_index() is None:
            self.col.create_index(self._vector_field, index_params=self.index_params, using=self.alias)

        # Load the collection if it exists
        if isinstance(self.col, Collection) and self._get_index() is not None:
            self.col.load()

    
    def _create_collection(self, embeddings, metadatas: Optional[list[dict]] = None):
        from pymilvus import (
            Collection,
            CollectionSchema,
            DataType,
            FieldSchema,
            MilvusException,
        )
        from pymilvus.orm.types import infer_dtype_bydata

        # Determine embedding dim
        dim = len(embeddings[0])
        fields = []

        if metadatas:
            # Create FieldSchema for each entry in metadata.
            for key, value in metadatas[0].items():
                # Infer the corresponding datatype of the metadata
                dtype = infer_dtype_bydata(value)
                # Datatype isn't compatible
                if dtype == DataType.UNKNOWN or dtype == DataType.NONE:
                    raise ValueError(f"Unrecognized datatype for {key}.")
                # Dataype is a string/varchar equivalent
                elif dtype == DataType.VARCHAR:
                    fields.append(FieldSchema(key, DataType.VARCHAR, max_length=65_535))
                else:
                    fields.append(FieldSchema(key, dtype))

        # Create the primary key field
        fields.append(
            FieldSchema(
                self._primary_field, DataType.INT64, is_primary=True, auto_id=True
            )
        )
        # Create the vector field, supports binary or float vectors
        fields.append(
            FieldSchema(self._vector_field, infer_dtype_bydata(embeddings[0]), dim=dim)
        )

        # Create the schema for the collection
        schema = CollectionSchema(fields)

        # Create the collection
        self.col = Collection(
            name=self.collection_name,
            schema=schema,
            consistency_level=self.consistency_level,
            using=self.alias,
        )
