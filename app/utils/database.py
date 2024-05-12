from motor.motor_asyncio import AsyncIOMotorClient
from decouple import config


class Database:
    _instances = {}

    def __new__(cls, db_name="deb8"):
        if db_name not in cls._instances:
            print(f"Connecting to {db_name}")
            instance = super().__new__(cls)
            instance.client = AsyncIOMotorClient(config("MONGODB_API_KEY"))
            instance._db = instance.client[db_name]
            cls._instances[db_name] = instance
        return cls._instances[db_name]

    @property
    def db(self):
        return self._db

    @db.setter
    def db(self, new_db):
        self._db = new_db
