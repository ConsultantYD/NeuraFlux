from pydantic import BaseModel, ConfigDict


class BaseSchema(BaseModel):
    ConfigDict(validate_assignment=True, populate_by_name=True)
