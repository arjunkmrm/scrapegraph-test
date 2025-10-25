"""
ScapeGraph MCP Server for converting webpages to markdown.
To run your server, use "uv run dev"
To test interactively, use "uv run playground"

Resources:
- MCP Python SDK: https://github.com/modelcontextprotocol/python-sdk
- ScapeGraph API: https://scrapegraphai.com
"""

import json
import os
from typing import Any, Dict, List, Optional, Union

import httpx
from mcp.server.fastmcp import Context, FastMCP
from pydantic import BaseModel, Field

from smithery.decorators import smithery


class ScapeGraphClient:
    """Client for interacting with the ScapeGraph API."""

    BASE_URL = "https://api.scrapegraphai.com/v1"

    def __init__(self, api_key: str):
        """
        Initialize the ScapeGraph API client.

        Args:
            api_key: API key for ScapeGraph API
        """
        self.api_key = api_key
        self.headers = {
            "SGAI_API_KEY": api_key,
            "Content-Type": "application/json"
        }
        self.client = httpx.Client(timeout=httpx.Timeout(120.0))

    def markdownify(self, website_url: str) -> Dict[str, Any]:
        """Convert a webpage into clean, formatted markdown."""
        url = f"{self.BASE_URL}/markdownify"
        data = {"website_url": website_url}
        response = self.client.post(url, headers=self.headers, json=data)
        if response.status_code != 200:
            raise Exception(f"Error {response.status_code}: {response.text}")
        return response.json()

    def smartscraper(self, user_prompt: str, website_url: str, number_of_scrolls: int = None, markdown_only: bool = None) -> Dict[str, Any]:
        """Extract structured data from a webpage using AI."""
        url = f"{self.BASE_URL}/smartscraper"
        data = {
            "user_prompt": user_prompt,
            "website_url": website_url
        }
        if number_of_scrolls is not None:
            data["number_of_scrolls"] = number_of_scrolls
        if markdown_only is not None:
            data["markdown_only"] = markdown_only
        response = self.client.post(url, headers=self.headers, json=data)
        if response.status_code != 200:
            raise Exception(f"Error {response.status_code}: {response.text}")
        return response.json()

    def searchscraper(self, user_prompt: str, num_results: int = None, number_of_scrolls: int = None) -> Dict[str, Any]:
        """Perform AI-powered web searches with structured results."""
        url = f"{self.BASE_URL}/searchscraper"
        data = {"user_prompt": user_prompt}
        if num_results is not None:
            data["num_results"] = num_results
        if number_of_scrolls is not None:
            data["number_of_scrolls"] = number_of_scrolls
        response = self.client.post(url, headers=self.headers, json=data)
        if response.status_code != 200:
            raise Exception(f"Error {response.status_code}: {response.text}")
        return response.json()

    def scrape(self, website_url: str, render_heavy_js: Optional[bool] = None) -> Dict[str, Any]:
        """Basic scrape endpoint to fetch page content."""
        url = f"{self.BASE_URL}/scrape"
        payload: Dict[str, Any] = {"website_url": website_url}
        if render_heavy_js is not None:
            payload["render_heavy_js"] = render_heavy_js
        response = self.client.post(url, headers=self.headers, json=payload)
        response.raise_for_status()
        return response.json()

    def sitemap(self, website_url: str) -> Dict[str, Any]:
        """Extract sitemap for a given website."""
        url = f"{self.BASE_URL}/sitemap"
        payload: Dict[str, Any] = {"website_url": website_url}
        response = self.client.post(url, headers=self.headers, json=payload)
        response.raise_for_status()
        return response.json()

    def agentic_scrapper(
        self,
        url: str,
        user_prompt: Optional[str] = None,
        output_schema: Optional[Dict[str, Any]] = None,
        steps: Optional[List[str]] = None,
        ai_extraction: Optional[bool] = None,
        persistent_session: Optional[bool] = None,
        timeout_seconds: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Run the Agentic Scraper workflow (no live session/browser interaction)."""
        endpoint = f"{self.BASE_URL}/agentic-scrapper"
        payload: Dict[str, Any] = {"url": url}
        if user_prompt is not None:
            payload["user_prompt"] = user_prompt
        if output_schema is not None:
            payload["output_schema"] = output_schema
        if steps is not None:
            payload["steps"] = steps
        if ai_extraction is not None:
            payload["ai_extraction"] = ai_extraction
        if persistent_session is not None:
            payload["persistent_session"] = persistent_session

        if timeout_seconds is not None:
            response = self.client.post(endpoint, headers=self.headers, json=payload, timeout=timeout_seconds)
        else:
            response = self.client.post(endpoint, headers=self.headers, json=payload)
        response.raise_for_status()
        return response.json()

    def smartcrawler_initiate(
        self,
        url: str,
        prompt: str = None,
        extraction_mode: str = "ai",
        depth: int = None,
        max_pages: int = None,
        same_domain_only: bool = None
    ) -> Dict[str, Any]:
        """Initiate a SmartCrawler request for multi-page web crawling."""
        endpoint = f"{self.BASE_URL}/crawl"
        data = {"url": url}

        if extraction_mode == "markdown":
            data["markdown_only"] = True
        elif extraction_mode == "ai":
            if prompt is None:
                raise ValueError("prompt is required when extraction_mode is 'ai'")
            data["prompt"] = prompt
        else:
            raise ValueError(f"Invalid extraction_mode: {extraction_mode}. Must be 'ai' or 'markdown'")

        if depth is not None:
            data["depth"] = depth
        if max_pages is not None:
            data["max_pages"] = max_pages
        if same_domain_only is not None:
            data["same_domain_only"] = same_domain_only

        response = self.client.post(endpoint, headers=self.headers, json=data)
        if response.status_code != 200:
            raise Exception(f"Error {response.status_code}: {response.text}")
        return response.json()

    def smartcrawler_fetch_results(self, request_id: str) -> Dict[str, Any]:
        """Fetch the results of a SmartCrawler operation."""
        endpoint = f"{self.BASE_URL}/crawl/{request_id}"
        response = self.client.get(endpoint, headers=self.headers)
        if response.status_code != 200:
            raise Exception(f"Error {response.status_code}: {response.text}")
        return response.json()

    def close(self) -> None:
        """Close the HTTP client."""
        self.client.close()


class ConfigSchema(BaseModel):
    scrapegraph_api_key: str = Field(..., description="Your ScapeGraph API key")


@smithery.server(config_schema=ConfigSchema)
def create_server():
    """Create and configure the MCP server."""

    server = FastMCP("ScapeGraph MCP Server")

    def get_api_key(ctx: Context) -> Optional[str]:
        """Helper to get API key from context or environment."""
        if ctx and ctx.session_config and ctx.session_config.scrapegraph_api_key:
            return ctx.session_config.scrapegraph_api_key
        return os.environ.get("SGAI_API_KEY")

    @server.tool()
    def markdownify(website_url: str, ctx: Context) -> Dict[str, Any]:
        """Convert a webpage into clean, formatted markdown."""
        api_key = get_api_key(ctx)
        if not api_key:
            return {"error": "ScapeGraph API key not configured."}

        client = ScapeGraphClient(api_key)
        try:
            return client.markdownify(website_url)
        except Exception as e:
            return {"error": str(e)}
        finally:
            client.close()

    @server.tool()
    def smartscraper(
        user_prompt: str,
        website_url: str,
        number_of_scrolls: int = None,
        markdown_only: bool = None,
        ctx: Context = None
    ) -> Dict[str, Any]:
        """Extract structured data from a webpage using AI."""
        api_key = get_api_key(ctx)
        if not api_key:
            return {"error": "ScapeGraph API key not configured."}

        client = ScapeGraphClient(api_key)
        try:
            return client.smartscraper(user_prompt, website_url, number_of_scrolls, markdown_only)
        except Exception as e:
            return {"error": str(e)}
        finally:
            client.close()

    @server.tool()
    def searchscraper(
        user_prompt: str,
        num_results: int = None,
        number_of_scrolls: int = None,
        ctx: Context = None
    ) -> Dict[str, Any]:
        """Perform AI-powered web searches with structured results."""
        api_key = get_api_key(ctx)
        if not api_key:
            return {"error": "ScapeGraph API key not configured."}

        client = ScapeGraphClient(api_key)
        try:
            return client.searchscraper(user_prompt, num_results, number_of_scrolls)
        except Exception as e:
            return {"error": str(e)}
        finally:
            client.close()

    @server.tool()
    def scrape(website_url: str, render_heavy_js: Optional[bool] = None, ctx: Context = None) -> Dict[str, Any]:
        """Fetch page content for a URL."""
        api_key = get_api_key(ctx)
        if not api_key:
            return {"error": "ScapeGraph API key not configured."}

        client = ScapeGraphClient(api_key)
        try:
            return client.scrape(website_url=website_url, render_heavy_js=render_heavy_js)
        except httpx.HTTPError as http_err:
            return {"error": str(http_err)}
        except Exception as e:
            return {"error": str(e)}
        finally:
            client.close()

    @server.tool()
    def sitemap(website_url: str, ctx: Context = None) -> Dict[str, Any]:
        """Extract sitemap for a website."""
        api_key = get_api_key(ctx)
        if not api_key:
            return {"error": "ScapeGraph API key not configured."}

        client = ScapeGraphClient(api_key)
        try:
            return client.sitemap(website_url=website_url)
        except httpx.HTTPError as http_err:
            return {"error": str(http_err)}
        except Exception as e:
            return {"error": str(e)}
        finally:
            client.close()

    @server.tool()
    def agentic_scrapper(
        url: str,
        user_prompt: Optional[str] = None,
        output_schema: Optional[Union[str, Dict[str, Any]]] = None,
        steps: Optional[Union[str, List[str]]] = None,
        ai_extraction: Optional[bool] = None,
        persistent_session: Optional[bool] = None,
        timeout_seconds: Optional[float] = None,
        ctx: Context = None,
    ) -> Dict[str, Any]:
        """Run the Agentic Scraper workflow. Accepts flexible input forms for steps and schema."""
        api_key = get_api_key(ctx)
        if not api_key:
            return {"error": "ScapeGraph API key not configured."}

        # Normalize inputs
        normalized_steps: Optional[List[str]] = None
        if isinstance(steps, list):
            normalized_steps = steps
        elif isinstance(steps, str):
            try:
                parsed_steps = json.loads(steps)
                if isinstance(parsed_steps, list):
                    normalized_steps = parsed_steps
                else:
                    normalized_steps = [steps]
            except json.JSONDecodeError:
                normalized_steps = [steps]

        normalized_schema: Optional[Dict[str, Any]] = None
        if isinstance(output_schema, dict):
            normalized_schema = output_schema
        elif isinstance(output_schema, str):
            try:
                parsed_schema = json.loads(output_schema)
                if isinstance(parsed_schema, dict):
                    normalized_schema = parsed_schema
                else:
                    return {"error": "output_schema must be a JSON object"}
            except json.JSONDecodeError as e:
                return {"error": f"Invalid JSON for output_schema: {str(e)}"}

        client = ScapeGraphClient(api_key)
        try:
            return client.agentic_scrapper(
                url=url,
                user_prompt=user_prompt,
                output_schema=normalized_schema,
                steps=normalized_steps,
                ai_extraction=ai_extraction,
                persistent_session=persistent_session,
                timeout_seconds=timeout_seconds,
            )
        except httpx.TimeoutException as timeout_err:
            return {"error": f"Request timed out: {str(timeout_err)}"}
        except httpx.HTTPError as http_err:
            return {"error": str(http_err)}
        except Exception as e:
            return {"error": str(e)}
        finally:
            client.close()

    @server.tool()
    def smartcrawler_initiate(
        url: str,
        prompt: str = None,
        extraction_mode: str = "ai",
        depth: int = None,
        max_pages: int = None,
        same_domain_only: bool = None,
        ctx: Context = None
    ) -> Dict[str, Any]:
        """
        Initiate a SmartCrawler request for intelligent multi-page web crawling.

        SmartCrawler supports two modes:
        - AI Extraction Mode (10 credits per page): Extracts structured data based on your prompt
        - Markdown Conversion Mode (2 credits per page): Converts pages to clean markdown
        """
        api_key = get_api_key(ctx)
        if not api_key:
            return {"error": "ScapeGraph API key not configured."}

        client = ScapeGraphClient(api_key)
        try:
            return client.smartcrawler_initiate(
                url=url,
                prompt=prompt,
                extraction_mode=extraction_mode,
                depth=depth,
                max_pages=max_pages,
                same_domain_only=same_domain_only
            )
        except Exception as e:
            return {"error": str(e)}
        finally:
            client.close()

    @server.tool()
    def smartcrawler_fetch_results(request_id: str, ctx: Context = None) -> Dict[str, Any]:
        """Fetch the results of a SmartCrawler operation."""
        api_key = get_api_key(ctx)
        if not api_key:
            return {"error": "ScapeGraph API key not configured."}

        client = ScapeGraphClient(api_key)
        try:
            return client.smartcrawler_fetch_results(request_id)
        except Exception as e:
            return {"error": str(e)}
        finally:
            client.close()

    return server


def main():
    """Main entry point for running the ScapeGraph MCP server with stdio transport."""
    server = create_server()
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
