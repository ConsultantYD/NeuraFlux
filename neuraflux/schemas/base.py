from pydantic import BaseModel, ConfigDict


class BaseSchema(BaseModel):
    model_config = ConfigDict(
        validate_assignment=True,
        populate_by_name=True,
    )
