"""
Synchronous wrapper around MCP stdio sessions for Gate M.

A background thread runs one asyncio loop. Each server is owned by one coroutine that opens
the stdio transport and session and keeps them open until told to stop, so every async context
is entered and exited in the same task. Calls are submitted to that loop from synchronous code.
"""

import asyncio
import json
import os
import sys
import threading
import time
from datetime import timedelta
from typing import Dict, List, Optional, Tuple

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SERVERS_PY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "servers.py")
ROLES = ["shopcss", "structured", "generic", "universal"]


class McpPool:
    def __init__(self, env: Optional[Dict[str, Dict[str, str]]] = None, timeout: float = 1.0):
        self.env = env or {}
        self.timeout = timeout
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.thread.start()
        self.sessions: Dict[str, ClientSession] = {}
        self._stops: Dict[str, asyncio.Event] = {}
        self._done: Dict[str, "asyncio.Future[None]"] = {}
        self.tool_server: Dict[str, str] = {}
        self.tools: List[dict] = []

    # -- server lifecycle -------------------------------------------------------------------

    async def _own(self, role: str, env: Dict[str, str], ready: asyncio.Event, stop: asyncio.Event):
        params = StdioServerParameters(
            command=sys.executable,
            args=[SERVERS_PY, role],
            env={**os.environ, **env, "PYTHONUNBUFFERED": "1"},
        )
        async with stdio_client(params, errlog=open(os.devnull, "w")) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                self.sessions[role] = session
                ready.set()
                await stop.wait()
        self.sessions.pop(role, None)

    def _run(self, coro, timeout: Optional[float] = None):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(timeout)

    def start(self, role: str, extra_env: Optional[Dict[str, str]] = None) -> None:
        env = {**self.env.get(role, {}), **(extra_env or {})}

        async def launch():
            ready, stop = asyncio.Event(), asyncio.Event()
            self._stops[role] = stop
            self._done[role] = asyncio.ensure_future(self._own(role, env, ready, stop))
            await asyncio.wait_for(ready.wait(), 30)

        self._run(launch(), 60)

    def stop(self, role: str) -> None:
        if role not in self._stops:
            return

        async def halt():
            self._stops.pop(role).set()
            await asyncio.wait_for(self._done.pop(role), 30)

        self._run(halt(), 60)

    def restart(self, role: str, extra_env: Optional[Dict[str, str]] = None) -> None:
        self.stop(role)
        self.start(role, extra_env)
        self.refresh_tools()

    def start_all(self) -> "McpPool":
        for role in ROLES:
            self.start(role)
        self.refresh_tools()
        return self

    def close(self) -> None:
        for role in list(self._stops):
            self.stop(role)
        self.loop.call_soon_threadsafe(self.loop.stop)

    # -- tools ------------------------------------------------------------------------------

    def refresh_tools(self) -> List[dict]:
        tools = []
        for role, session in list(self.sessions.items()):
            res = self._run(session.list_tools(), 30)
            for t in res.tools:
                props = list((t.inputSchema or {}).get("properties", {}))
                tools.append(
                    {
                        "name": t.name,
                        "server": role,
                        "description": t.description or "",
                        "arg": props[0] if props else "url",
                    }
                )
        self.tools = sorted(tools, key=lambda x: x["name"])
        self.tool_server = {t["name"]: t["server"] for t in self.tools}
        return self.tools

    def call(self, name: str, url: str, arg: str = "url") -> Tuple[Optional[dict], str, float]:
        """
        Calls a tool with a page URL. Returns (parsed result or None, error kind, seconds).
        Error kinds: "" (none), "unknown_tool", "timeout", "tool_error", "bad_output".
        """
        start = time.time()
        role = self.tool_server.get(name)
        if role is None or role not in self.sessions:
            return None, "unknown_tool", 0.0
        session = self.sessions[role]
        args = {arg: url} if name != "send_report" else {"content": url}
        try:
            res = self._run(
                session.call_tool(name, args, read_timeout_seconds=timedelta(seconds=self.timeout)),
                self.timeout + 5,
            )
        except Exception as e:  # McpError on read timeout, or a future timeout
            kind = (
                "timeout"
                if "timed out" in str(e).lower() or "timeout" in str(e).lower()
                else "tool_error"
            )
            return None, kind, time.time() - start
        elapsed = time.time() - start
        if res.isError:
            return None, "tool_error", elapsed
        text = "".join(getattr(c, "text", "") for c in res.content)
        try:
            return json.loads(text), "", elapsed
        except Exception:
            return None, "bad_output", elapsed
