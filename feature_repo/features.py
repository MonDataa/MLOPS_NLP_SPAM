from feast import FeatureStore
from feast import Entity, FeatureView, Feature, ValueType
from feast.infra.offline_stores.file import FileOfflineStore
from feast.infra.online_stores.redis import RedisOnlineStore
import pandas as pd
from datetime import datetime

# Define entity
message_entity = Entity(name="message_id", value_type=ValueType.INT64, description="Message ID")

# Define feature view
message_features = FeatureView(
    name="message_features",
    entities=["message_id"],
    ttl=None,
    features=[
        Feature(name="text_length", dtype=ValueType.INT64),
        Feature(name="word_count", dtype=ValueType.INT64),
        Feature(name="has_url", dtype=ValueType.BOOL),
        Feature(name="has_email", dtype=ValueType.BOOL),
    ],
    online=True,
    input=None,  # Will be set later
)

# Create store
store = FeatureStore(repo_path=".")

# To populate, load data and materialize
# But for now, define