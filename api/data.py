import uuid
from typing import List, Optional

import pandas as pd
from faker import Faker

from .types import Interest, UserFeatureStoreRecord


class FeatureStore:
    user_features_file_path = "./api/datasets/users.csv"
    content_activity_file_path = (
        "./api/datasets/content_activity.csv"
    )


    def __init__(self):

        self.users_df = pd.read_csv(self.user_features_file_path)
        self.content_activity_df = pd.read_csv(self.content_activity_file_path)
        self.merged_df = pd.merge(self.users_df, self.content_activity_df)

    def get_user_features(self, user_id: uuid.UUID) -> Optional[UserFeatureStoreRecord]:
        user_data = self.merged_df.loc[self.merged_df["id"] == str(user_id)]
        if not user_data.size:
            return None

        return UserFeatureStoreRecord(**user_data.to_dict(orient="records")[0])

    def get_interests_labels(self, ids: List[uuid.UUID], probabilities: List[float]) -> List[Interest]:
        
        return [Interest(id=id, label=Faker().job(), probability=probability) for id, probability in  zip(ids, probabilities)]


    
