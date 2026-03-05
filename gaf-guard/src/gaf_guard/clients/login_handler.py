from typing import Optional

import httpx
from acp_sdk.client import Client


class LoginHandler:

    def __init__(self, host: str, port: Optional[int] = None):
        self.host = host
        self.port = port
        self.base_url = self.host + (f":{int(self.port)}" if self.port else "")

    async def health_check(self):
        try:
            await self.create_client().ping()
        except Exception as e:
            if isinstance(e, httpx.ConnectTimeout):
                message = "Connection Timeout."
            else:
                message = str(e)
            raise Exception(f"Failed to connect: {self.base_url}, {message}")

    def create_client(self):
        try:
            return Client(base_url=self.base_url, verify=True)
        except Exception as e:
            if isinstance(e, httpx.ConnectTimeout):
                message = "Connection Timeout."
            else:
                message = str(e)
            raise Exception(f"Failed to connect: {self.base_url}, {message}")
