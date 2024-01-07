from pydantic import BaseModel, model_validator
# from typing import List
import json

class PersonMetaData(BaseModel):
    id: str

# class AddFaceRequest(BaseModel):
#     people_metadatas:List[PersonMetaData]

#     @model_validator(mode='before')
#     @classmethod
#     def validate_to_json(cls, value):
#         if isinstance(value, str):
#             return cls(**json.loads(value))
#         return value
    
class AddVoiceRequest(BaseModel):
    # people_metadatas:List[PersonMetaData]
    name: str
    
    @model_validator(mode='before')
    @classmethod
    def validate_to_json(cls, value):
        if isinstance(value, str):
            return cls(**json.loads(value))
        return value
