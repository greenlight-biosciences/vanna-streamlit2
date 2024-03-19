import snowflake.connector as sfc
from VannaLogger import VannaLogger
from vanna.base import VannaBase
from vanna.exceptions import DependencyError, ImproperlyConfigured, ValidationError
from typing import Union, List
import os 
# Context manager for Snowflake connection
class SnowflakeConnection():
    def __init__(self, config=None):
        self.user=None,
        self.password=None,
        self.account=None,
        self.warehouse=None,
        self.database=None,
        self.schema=None
        self.role=None 
        self.vannaLogger = VannaLogger()

    def setConnection(
        self,
        account: str,
        username: str,
        password: str,
        database: str,
        schema: str,
        role: str,
        warehouse: str
    ):

        if username == "my-username":
            username_env = os.environ.get("SNOWFLAKE_USERNAME")

            if username_env is not None:
                username = username_env
            else:
                raise ImproperlyConfigured("Please set your Snowflake username.")

        if password == "my-password":
            password_env = os.environ.get("SNOWFLAKE_PASSWORD")

            if password_env is not None:
                password = password_env
            else:
                raise ImproperlyConfigured("Please set your Snowflake password.")

        if account == "my-account":
            account_env = os.environ.get("SNOWFLAKE_ACCOUNT")

            if account_env is not None:
                account = account_env
            else:
                raise ImproperlyConfigured("Please set your Snowflake account.")

        if database == "my-database":
            database_env = os.environ.get("SNOWFLAKE_DATABASE")

            if database_env is not None:
                database = database_env
            else:
                raise ImproperlyConfigured("Please set your Snowflake database.")

       
        self.user=username
        self.password=password
        self.account=account
        self.database=database
        self.schema=schema
        self.role=role
        self.warehouse=warehouse
        self.vannaLogger.logInfo('Ready to connect to Snowflake')
        
    def set_schema(self,schema:str=None):
        self.schema= schema
        self.vannaLogger.logInfo(f'Updated Snowflake Schema: {self.schema}')

    def __enter__(self):
        self.conn = sfc.connect(
            user=self.user,
            password=self.password,
            account=self.account,
            warehouse=self.warehouse,
            database=self.database,
            schema=self.schema
        )
        if self.role:
            self.conn.cursor().execute(f"USE ROLE {self.role}")
        return self.conn

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.conn.close()
