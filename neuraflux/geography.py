from enum import Enum, unique


@unique
class CityEnum(str, Enum):
    NEW_YORK = "New York"
    LOS_ANGELES = "Los Angeles"
    CHICAGO = "Chicago"
    TORONTO = "Toronto"
    MEXICO_CITY = "Mexico City"
    HOUSTON = "Houston"
    VANCOUVER = "Vancouver"

    @classmethod
    def get_all_available_city_values(cls) -> list[str]:
        cities = [member.value for member in cls]
        cities.sort()
        return cities

    @classmethod
    def get_all_available_cities(cls) -> list[str]:
        cities = [member.value for member in cls]
        cities.sort()
        return cities

    @classmethod
    def validate_city(cls, city: str) -> str:
        for member in cls:
            if member.value == city:
                return member.value
        raise ValueError(
            f"Invalid city: {city}. Available cities are: {cls.get_all_available_city_values()}"
        )

    @classmethod
    def from_string(cls, city: str):
        for member in cls:
            if member.value == city:
                return member
        raise ValueError(
            f"Invalid city: {city}. Available cities are: {cls.get_all_available_city_values()}"
        )
