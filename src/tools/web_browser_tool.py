"""
Web browsing tool for local information retrieval.
Provides web scraping capabilities for the research agent.
"""

import asyncio
import logging
import urllib.parse
from typing import Optional, Dict, Any, List
import aiohttp
from bs4 import BeautifulSoup
from crewai.tools.base_tool import BaseTool

try:
    from .query_parser import query_parser
except ImportError:
    from query_parser import query_parser

logger = logging.getLogger(__name__)

class WebBrowserTool(BaseTool):
    """Tool for browsing web content locally without external APIs."""

    name: str = "web_browser"
    description: str = (
        "Browse web pages and extract content. Provide a URL to fetch and parse "
        "the webpage content. Returns the main text content and basic metadata."
    )

    def __init__(self):
        super().__init__()
        self._timeout = aiohttp.ClientTimeout(total=30)
        self._headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    
    async def _fetch_page(self, url: str) -> Optional[Dict[str, Any]]:
        """Fetch and parse a web page."""
        try:
            async with aiohttp.ClientSession(timeout=self._timeout) as session:
                async with session.get(url, headers=self._headers) as response:
                    if response.status != 200:
                        return {
                            "error": f"HTTP {response.status}: Failed to fetch {url}",
                            "content": "",
                            "title": "",
                            "url": url
                        }
                    
                    html_content = await response.text()
                    soup = BeautifulSoup(html_content, 'html.parser')
                    
                    # Extract title
                    title = soup.find('title')
                    title_text = title.get_text().strip() if title else "No title"
                    
                    # Remove script and style elements
                    for script in soup(["script", "style", "nav", "footer", "header"]):
                        script.decompose()
                    
                    # Extract main content
                    # Try to find main content areas
                    main_content = soup.find('main') or soup.find('article') or soup.find('div', class_='content')
                    if not main_content:
                        main_content = soup.find('body')
                    
                    if main_content:
                        text_content = main_content.get_text()
                    else:
                        text_content = soup.get_text()
                    
                    # Clean up text
                    lines = (line.strip() for line in text_content.splitlines())
                    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                    text_content = ' '.join(chunk for chunk in chunks if chunk)
                    
                    # Limit content length
                    if len(text_content) > 5000:
                        text_content = text_content[:5000] + "... [Content truncated]"
                    
                    return {
                        "title": title_text,
                        "content": text_content,
                        "url": url,
                        "length": len(text_content),
                        "status": "success"
                    }
                    
        except asyncio.TimeoutError:
            return {
                "error": f"Timeout while fetching {url}",
                "content": "",
                "title": "",
                "url": url
            }
        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")
            return {
                "error": f"Error fetching {url}: {str(e)}",
                "content": "",
                "title": "",
                "url": url
            }
    
    def _run(self, url: str) -> str:
        """Synchronous wrapper for the async fetch method."""
        try:
            # Run the async function
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(self._fetch_page(url))
            loop.close()
            
            if "error" in result:
                return f"Error: {result['error']}"
            
            return f"""
Title: {result['title']}
URL: {result['url']}
Content Length: {result['length']} characters

Content:
{result['content']}
"""
        except Exception as e:
            logger.error(f"Error in web browser tool: {e}")
            return f"Error browsing {url}: {str(e)}"

class WebSearchTool(BaseTool):
    """Tool for performing web searches and retrieving results."""

    name: str = "web_search"
    description: str = (
        "Perform web searches to find current information. Provide search terms "
        "and get relevant web results with summaries."
    )

    def __init__(self):
        super().__init__()
        self._browser_tool = WebBrowserTool()
        self._timeout = aiohttp.ClientTimeout(total=30)
        self._headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

    def _extract_search_subject(self, query: str) -> str:
        """Extract the actual subject from image search queries using AI model."""
        try:
            # Use the intelligent query parser
            parse_result = query_parser.parse_image_query(query)
            subject = parse_result.get("subject", query)
            method = parse_result.get("method", "unknown")
            confidence = parse_result.get("confidence", 0.0)

            logger.info(f"Query parsing: '{query}' -> '{subject}' (method: {method}, confidence: {confidence})")

            return subject

        except Exception as e:
            logger.error(f"Query parsing failed: {e}")
            # Fallback to original query if parsing fails
            return query

    def _build_search_urls(self, query: str) -> List[str]:
        """Build search URLs for different search engines."""
        # For image searches, extract the actual subject
        if any(term in query.lower() for term in ["picture", "image", "photo", "pic"]):
            search_subject = self._extract_search_subject(query)
            encoded_query = urllib.parse.quote_plus(search_subject)
            return [
                f"https://www.google.com/search?q={encoded_query}&tbm=isch",
                f"https://www.bing.com/images/search?q={encoded_query}",
                f"https://duckduckgo.com/?q={encoded_query}&iax=images&ia=images"
            ]

        # For non-image searches, use original query
        encoded_query = urllib.parse.quote_plus(query)

        # For stock prices, use specific financial sites
        if any(term in query.lower() for term in ["stock price", "stock", "share price", "nasdaq", "nyse"]):
            return [
                f"https://finance.yahoo.com/quote/{self._extract_ticker(query)}",
                f"https://www.google.com/search?q={encoded_query}+stock+price",
                f"https://www.marketwatch.com/investing/stock/{self._extract_ticker(query)}"
            ]

        # General search URLs
        return [
            f"https://www.google.com/search?q={encoded_query}",
            f"https://duckduckgo.com/?q={encoded_query}",
            f"https://www.bing.com/search?q={encoded_query}"
        ]

    def _extract_ticker(self, query: str) -> str:
        """Extract stock ticker from query."""
        query_lower = query.lower()

        # Common stock tickers
        tickers = {
            "nvidia": "NVDA",
            "apple": "AAPL",
            "microsoft": "MSFT",
            "google": "GOOGL",
            "amazon": "AMZN",
            "tesla": "TSLA",
            "meta": "META",
            "facebook": "META"
        }

        for company, ticker in tickers.items():
            if company in query_lower:
                return ticker

        # Default to NVDA if no specific ticker found but stock-related
        return "NVDA"

    async def _search_web(self, query: str) -> str:
        """Perform web search and return results."""
        try:
            # For image searches, extract the subject and log it
            if any(term in query.lower() for term in ["picture", "image", "photo", "pic"]):
                search_subject = self._extract_search_subject(query)
                logger.info(f"Image search: Original query '{query}' -> Extracted subject '{search_subject}'")

            search_urls = self._build_search_urls(query)
            results = []

            for url in search_urls[:2]:  # Try first 2 URLs
                try:
                    result = await self._browser_tool._fetch_page(url)
                    if result and "error" not in result:
                        # Extract relevant information
                        content = result.get("content", "")
                        title = result.get("title", "")

                        # For image searches, look for image URLs
                        if any(term in query.lower() for term in ["picture", "image", "photo", "pic"]):
                            image_info = self._extract_image_info(content, title, url)
                            if image_info:
                                results.append(f"Image Search Results from {url}:\n{image_info}")
                        # For stock searches, look for price information
                        elif "stock" in query.lower() or "price" in query.lower():
                            price_info = self._extract_price_info(content, title)
                            if price_info:
                                results.append(f"From {url}:\n{price_info}")
                        else:
                            # For general searches, get summary
                            summary = content[:500] + "..." if len(content) > 500 else content
                            results.append(f"From {title}:\n{summary}")

                        break  # Stop after first successful result

                except Exception as e:
                    logger.error(f"Error searching {url}: {e}")
                    continue

            if results:
                return "\n\n".join(results)
            else:
                return f"I couldn't find current information about '{query}'. The search may have encountered technical difficulties."

        except Exception as e:
            logger.error(f"Web search error: {e}")
            return f"Web search failed: {str(e)}"

    def _extract_price_info(self, content: str, title: str) -> str:
        """Extract price information from web content."""
        content_lower = content.lower()

        # Look for price patterns
        import re

        # Common price patterns
        price_patterns = [
            r'\$[\d,]+\.?\d*',  # $123.45 or $1,234
            r'[\d,]+\.?\d*\s*(?:usd|dollars?)',  # 123.45 USD
            r'price[:\s]*\$?[\d,]+\.?\d*',  # Price: $123.45
            r'[\d,]+\.?\d*\s*per\s*share',  # 123.45 per share
        ]

        found_prices = []
        for pattern in price_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            found_prices.extend(matches)

        if found_prices:
            # Get the first few price mentions
            price_info = f"Stock price information found:\n"
            for price in found_prices[:3]:
                price_info += f"- {price}\n"

            # Add context from title
            if title:
                price_info += f"\nSource: {title}"

            return price_info

        # If no specific price found, return relevant excerpt
        if any(term in content_lower for term in ["stock", "price", "trading", "market"]):
            # Find relevant sentences
            sentences = content.split('.')
            relevant_sentences = []
            for sentence in sentences[:10]:  # Check first 10 sentences
                if any(term in sentence.lower() for term in ["price", "stock", "trading", "market", "$"]):
                    relevant_sentences.append(sentence.strip())
                    if len(relevant_sentences) >= 3:
                        break

            if relevant_sentences:
                return "Relevant information found:\n" + "\n".join(relevant_sentences)

        return ""

    def _extract_image_info(self, content: str, title: str, url: str) -> str:
        """Extract image URLs and information from web content."""
        import re

        # Look for image URLs in the content
        image_patterns = [
            r'https?://[^\s<>"]+\.(?:jpg|jpeg|png|gif|webp|svg)',  # Direct image URLs
            r'src="([^"]+\.(?:jpg|jpeg|png|gif|webp|svg))"',  # img src attributes
            r'data-src="([^"]+\.(?:jpg|jpeg|png|gif|webp|svg))"',  # lazy loading images
            r'background-image:\s*url\(["\']?([^"\']+\.(?:jpg|jpeg|png|gif|webp|svg))["\']?\)'  # CSS background images
        ]

        found_images = []
        for pattern in image_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            found_images.extend(matches)

        # Remove duplicates and filter out small/icon images
        unique_images = []
        seen = set()
        for img_url in found_images:
            # Extract URL from tuple if needed (from capture groups)
            if isinstance(img_url, tuple):
                img_url = img_url[0] if img_url else ""

            # Skip if already seen or if it looks like an icon/small image
            if (img_url not in seen and
                not any(skip in img_url.lower() for skip in ['icon', 'logo', 'favicon', 'thumb']) and
                len(img_url) > 10):
                unique_images.append(img_url)
                seen.add(img_url)

                # Limit to first 5 images
                if len(unique_images) >= 5:
                    break

        if unique_images:
            image_info = f"Found {len(unique_images)} image(s):\n\n"
            for i, img_url in enumerate(unique_images, 1):
                # Make sure URL is complete
                if img_url.startswith('//'):
                    img_url = 'https:' + img_url
                elif img_url.startswith('/'):
                    # Relative URL - need to construct full URL
                    base_url = '/'.join(url.split('/')[:3])
                    img_url = base_url + img_url

                image_info += f"**Image {i}:**\n{img_url}\n\n"

            # Add source information
            if title:
                image_info += f"Source: {title}\n"
            image_info += f"Search URL: {url}"

            return image_info

        # If no images found in content, provide helpful guidance
        return f"""No direct image URLs found on this page. This might be because:
- The page uses JavaScript to load images dynamically
- Images are behind authentication or paywalls
- The search results page doesn't contain direct image links

**Alternative approaches:**
1. **Direct Image Search URLs:**
   - Google Images: https://www.google.com/search?q={urllib.parse.quote_plus(title or 'search query')}&tbm=isch
   - Bing Images: https://www.bing.com/images/search?q={urllib.parse.quote_plus(title or 'search query')}

2. **Try searching for specific terms** like "Paris Hilton official photos" or "Paris Hilton red carpet"

3. **Check official sources** like verified social media accounts or official websites

Note: Always respect copyright and usage rights when using images."""

    def _run(self, query: str) -> str:
        """Synchronous wrapper for web search."""
        try:
            # For image searches, try to get actual image URLs using a simpler approach
            if any(term in query.lower() for term in ["picture", "image", "photo", "pic"]):
                return self._search_images_sync(query)

            # Check if we're already in an async context
            try:
                asyncio.get_running_loop()
                # In async context, provide fallback response
                logger.warning("Web search skipped due to async context conflict")
                return self._provide_search_guidance(query)
            except RuntimeError:
                # Not in async context, safe to create new loop
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(self._search_web(query))
                loop.close()
                return result

        except Exception as e:
            logger.error(f"Error in web search tool: {e}")
            return f"Error searching for '{query}': {str(e)}"

    def _search_images_sync(self, query: str) -> str:
        """Synchronous image search that actually returns image URLs."""
        try:
            import requests
            from urllib.parse import quote_plus

            # Extract the subject
            search_subject = self._extract_search_subject(query)
            logger.info(f"Searching for images of: {search_subject}")

            # Use a simple image search API approach
            encoded_query = quote_plus(search_subject)

            # Try to get images from a simple search
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }

            # Search using DuckDuckGo's instant answer API (no rate limits)
            try:
                ddg_url = f"https://duckduckgo.com/?q={encoded_query}&format=json&no_html=1&skip_disambig=1"
                response = requests.get(ddg_url, headers=headers, timeout=10)

                if response.status_code == 200:
                    data = response.json()
                    # Look for image results in the response
                    if 'Image' in data and data['Image']:
                        image_url = data['Image']

                        # Convert relative URLs to absolute
                        if image_url.startswith('/'):
                            image_url = 'https://duckduckgo.com' + image_url
                        elif image_url.startswith('//'):
                            image_url = 'https:' + image_url

                        return f"""Found image of **{search_subject}**:

**Direct Image URL:**
{image_url}

**Additional Search Options:**
- [Google Images](https://www.google.com/search?q={encoded_query}&tbm=isch)
- [Bing Images](https://www.bing.com/images/search?q={encoded_query})
- [DuckDuckGo Images](https://duckduckgo.com/?q={encoded_query}&iax=images&ia=images)

**Note:** Always respect copyright and usage rights when using images."""
            except Exception as e:
                logger.error(f"DuckDuckGo search failed: {e}")

            # Fallback: Try to scrape Google Images (simplified)
            try:
                google_url = f"https://www.google.com/search?q={encoded_query}&tbm=isch&safe=active"
                response = requests.get(google_url, headers=headers, timeout=10)

                if response.status_code == 200:
                    # Simple regex to find image URLs in the HTML
                    import re
                    img_pattern = r'"(https://[^"]*\.(?:jpg|jpeg|png|gif|webp))"'
                    matches = re.findall(img_pattern, response.text)

                    if matches:
                        # Get first few unique images
                        unique_images = list(dict.fromkeys(matches))[:5]

                        result = f"""Found images of **{search_subject}**:

"""
                        for i, img_url in enumerate(unique_images, 1):
                            result += f"**Image {i}:**\n{img_url}\n\n"

                        result += f"""**Search Links for More:**
- [Google Images](https://www.google.com/search?q={encoded_query}&tbm=isch)
- [Bing Images](https://www.bing.com/images/search?q={encoded_query})

**Note:** Always respect copyright and usage rights when using images."""

                        return result
            except Exception as e:
                logger.error(f"Google Images scraping failed: {e}")

            # Final fallback with better messaging
            return f"""I found search options for **{search_subject}** images:

**Direct Image Search Links:**
🔍 [Google Images - {search_subject}](https://www.google.com/search?q={encoded_query}&tbm=isch)
🔍 [Bing Images - {search_subject}](https://www.bing.com/images/search?q={encoded_query})
🔍 [DuckDuckGo Images - {search_subject}](https://duckduckgo.com/?q={encoded_query}&iax=images&ia=images)

**Quick Tips:**
- Click any link above to see actual images
- Use "Tools" → "Usage Rights" for free-to-use images
- Try specific terms like "{search_subject} official" or "{search_subject} high quality"

**Note:** I cannot directly display images in Discord, but these links will take you to the actual image results!"""

        except Exception as e:
            logger.error(f"Image search failed: {e}")
            return f"Sorry, I encountered an error searching for images of {query}. Please try the search links manually."

    def _provide_search_guidance(self, query: str) -> str:
        """Provide search guidance when direct web search isn't available."""

        # For image searches, extract the subject first
        if any(term in query.lower() for term in ["picture", "image", "photo", "pic"]):
            search_subject = self._extract_search_subject(query)
            encoded_query = urllib.parse.quote_plus(search_subject)

            return f"""I can help you find images of **{search_subject}**! Here are direct search links:

**Google Images:**
https://www.google.com/search?q={encoded_query}&tbm=isch

**Bing Images:**
https://www.bing.com/images/search?q={encoded_query}

**DuckDuckGo Images:**
https://duckduckgo.com/?q={encoded_query}&iax=images&ia=images

**Tips for finding images:**
- Use specific search terms for better results
- Check the "Usage Rights" filter for images you can use
- Look for official sources or verified accounts
- Always respect copyright and attribution requirements

Click any of the links above to search for images directly!"""

        # For stock searches
        elif any(term in query.lower() for term in ["stock", "price"]):
            ticker = self._extract_ticker(query)
            return f"""Here are direct links to get current stock information:

**Yahoo Finance:**
https://finance.yahoo.com/quote/{ticker}

**Google Finance:**
https://www.google.com/search?q={encoded_query}+stock+price

**MarketWatch:**
https://www.marketwatch.com/investing/stock/{ticker}

These links will give you real-time stock prices and market data."""

        # For general searches
        else:
            encoded_query = urllib.parse.quote_plus(query)
            return f"""Here are search links to find current information:

**Google Search:**
https://www.google.com/search?q={encoded_query}

**Bing Search:**
https://www.bing.com/search?q={encoded_query}

**DuckDuckGo Search:**
https://duckduckgo.com/?q={encoded_query}

Click any link above to search for the latest information!"""

class LocalSearchTool(BaseTool):
    """Tool for searching local information and cached content."""
    
    name: str = "local_search"
    description: str = (
        "Search for information in local knowledge base and cached content. "
        "Provide search terms to find relevant local information."
    )
    
    def __init__(self):
        super().__init__()
        # This could be expanded to include local file search, cached web content, etc.
        self._local_knowledge = {
            "programming": {
                "python": "Python is a high-level programming language known for its simplicity and readability.",
                "javascript": "JavaScript is a programming language commonly used for web development.",
                "discord.py": "discord.py is a Python library for creating Discord bots.",
                "crewai": "CrewAI is a framework for orchestrating role-playing, autonomous AI agents."
            },
            "general": {
                "ai": "Artificial Intelligence (AI) refers to computer systems that can perform tasks typically requiring human intelligence.",
                "machine learning": "Machine Learning is a subset of AI that enables computers to learn and improve from experience.",
                "discord": "Discord is a communication platform designed for creating communities."
            }
        }
    
    def _run(self, query: str) -> str:
        """Search local knowledge base."""
        query_lower = query.lower()
        results = []
        
        # Search through local knowledge
        for category, items in self._local_knowledge.items():
            for key, value in items.items():
                if query_lower in key.lower() or any(term in value.lower() for term in query_lower.split()):
                    results.append(f"**{key.title()}** ({category}): {value}")
        
        if results:
            return f"Local search results for '{query}':\n\n" + "\n\n".join(results)
        else:
            return f"No local information found for '{query}'. You may want to search the web for more information."
