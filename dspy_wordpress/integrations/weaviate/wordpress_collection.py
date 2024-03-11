import weaviate.classes.config as wvc


def wordpress_collection_properties():
    return [
        wvc.Property(name="title",
                     data_type=wvc.DataType.TEXT,
                     vectorize_property_name=False,
                     skip_vectorization=True),
        wvc.Property(name="url",
                     data_type=wvc.DataType.TEXT,
                     vectorize_property_name=False,
                     skip_vectorization=True),
        wvc.Property(name="updated_at",
                     data_type=wvc.DataType.TEXT,
                     vectorize_property_name=False,
                     skip_vectorization=True),
        wvc.Property(name="tags",
                     data_type=wvc.DataType.TEXT_ARRAY,
                     vectorize_property_name=False,
                     skip_vectorization=True),
        wvc.Property(name="categories",
                     data_type=wvc.DataType.TEXT_ARRAY,
                     vectorize_property_name=False,
                     skip_vectorization=True),
    ]