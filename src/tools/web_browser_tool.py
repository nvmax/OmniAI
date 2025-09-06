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
        """Build generic search URLs for any query."""
        # Always use the original query - no hardcoded patterns for specific topics
        encoded_query = urllib.parse.quote_plus(query)

        # For image searches, use image search engines
        if any(term in query.lower() for term in ["picture", "image", "photo", "pic"]):
            search_subject = self._extract_search_subject(query)
            encoded_subject = urllib.parse.quote_plus(search_subject)
            return [
                f"https://www.google.com/search?q={encoded_subject}&tbm=isch",
                f"https://www.bing.com/images/search?q={encoded_subject}",
                f"https://duckduckgo.com/?q={encoded_subject}&iax=images&ia=images"
            ]

        # For all other searches, use general search engines
        # Try DuckDuckGo first (more scraping-friendly), then others as fallback
        return [
            f"https://duckduckgo.com/html/?q={encoded_query}",
            f"https://www.bing.com/search?q={encoded_query}",
            f"https://www.google.com/search?q={encoded_query}"
        ]

    # Removed hardcoded ticker extraction - using generic search approach

    async def _search_web(self, query: str) -> str:
        """Perform web search and return results."""
        try:
            # For image searches, extract the subject and log it
            if any(term in query.lower() for term in ["picture", "image", "photo", "pic"]):
                search_subject = self._extract_search_subject(query)
                logger.info(f"Image search: Original query '{query}' -> Extracted subject '{search_subject}'")

            search_urls = self._build_search_urls(query)
            results = []

            for url in search_urls[:2]:  # Try first 2 search engines
                try:
                    result = await self._browser_tool._fetch_page(url)
                    if result and "error" not in result:
                        content = result.get("content", "")
                        title = result.get("title", "")

                        # Use LLM to intelligently extract and format information
                        if any(term in query.lower() for term in ["picture", "image", "photo", "pic"]):
                            # For image searches, look for image URLs
                            image_info = self._extract_image_info(content, title, url)
                            if image_info:
                                results.append(f"Image Search Results from {url}:\n{image_info}")
                        else:
                            # For stock/financial queries, try following links first (search results pages don't have actual prices)
                            if any(term in query.lower() for term in ["stock", "price", "share", "nasdaq", "nyse"]):
                                link_results = await self._follow_search_result_links(content, query)
                                if link_results:
                                    results.extend(link_results)
                                else:
                                    # Fallback to processing search results page
                                    processed_content = await self._llm_process_web_content(content, title, query)
                                    if processed_content:
                                        results.append(processed_content)
                            else:
                                # For non-financial queries, try processing search results first
                                processed_content = await self._llm_process_web_content(content, title, query)
                                if processed_content:
                                    results.append(processed_content)
                                else:
                                    # If search results page doesn't have the info, try to follow links
                                    link_results = await self._follow_search_result_links(content, query)
                                    if link_results:
                                        results.extend(link_results)

                        if results:  # Stop after getting some results
                            break

                except Exception as e:
                    logger.error(f"Error searching {url}: {e}")
                    continue

            if results:
                return "\n\n".join(results)
            else:
                # Provide helpful fallback information
                return self._provide_search_fallback(query)

        except Exception as e:
            logger.error(f"Web search error: {e}")
            return self._provide_search_fallback(query)

    async def _llm_process_web_content(self, content: str, title: str, query: str) -> str:
        """Use LLM to intelligently extract and format relevant information from web content."""
        try:
            # Import LLM manager
            try:
                from llm_integration import llm_manager
            except ImportError:
                from .llm_integration import llm_manager

            # Truncate content if too long (LLM token limits)
            max_content_length = 2000
            if len(content) > max_content_length:
                content = content[:max_content_length] + "..."

            prompt = f"""You are helping extract relevant information from a web page to answer a user's question.

USER QUESTION: "{query}"

WEB PAGE TITLE: "{title}"

WEB PAGE CONTENT:
{content}

INSTRUCTIONS:
- Extract ONLY the information that directly answers the user's question
- Format the response clearly and concisely
- For stock prices: show the current price and any relevant details
- For weather: show temperature, conditions, and location
- For general info: provide the most relevant facts
- IMPORTANT: If this appears to be a search results page (with links to other sites) rather than actual content, say "SEARCH_RESULTS_PAGE"
- If no relevant information is found, say "No relevant information found on this page"
- Keep the response under 200 words
- Use a natural, conversational tone

RESPONSE:"""

            response = await llm_manager.generate_response(
                user_message=prompt,
                system_message="You are an expert at extracting and formatting relevant information from web content.",
                max_tokens=300
            )

            if response and "no relevant information found" not in response.lower() and "search_results_page" not in response.lower():
                return f"From {title}:\n{response}"
            else:
                return None

        except Exception as e:
            logger.error(f"Error processing web content with LLM: {e}")
            return None

    async def _follow_search_result_links(self, search_content: str, query: str) -> List[str]:
        """Extract and follow links from search results to get actual content."""
        try:
            import re
            from urllib.parse import urljoin, urlparse

            # Extract ALL URLs from search results - follow any relevant links!
            url_patterns = [
                r'(https?://[^\s<>"]+)',  # Any complete HTTP/HTTPS URLs
                r'(www\.[^\s<>"]+)',  # www. URLs
                r'([a-zA-Z0-9.-]+\.com/[^\s<>"]*)',  # .com URLs with paths
                r'([a-zA-Z0-9.-]+\.org/[^\s<>"]*)',  # .org URLs with paths
                r'([a-zA-Z0-9.-]+\.gov/[^\s<>"]*)',  # .gov URLs with paths
                r'([a-zA-Z0-9.-]+\.net/[^\s<>"]*)',  # .net URLs with paths
            ]

            found_urls = []
            for pattern in url_patterns:
                matches = re.findall(pattern, search_content)
                found_urls.extend(matches)

            # Clean and construct complete URLs - follow ANY relevant links
            filtered_urls = []
            for url in found_urls:
                # Skip obviously irrelevant URLs
                skip_domains = [
                    'duckduckgo.com', 'google.com', 'bing.com',  # Search engines
                    'facebook.com', 'twitter.com', 'instagram.com',  # Social media
                    'youtube.com', 'tiktok.com',  # Video platforms
                    'amazon.com/dp/', 'ebay.com',  # Shopping (unless specifically searching for products)
                ]

                if any(skip in url.lower() for skip in skip_domains):
                    continue

                # Make sure URL is complete
                if not url.startswith('http'):
                    if url.startswith('www.'):
                        url = 'https://' + url
                    else:
                        url = 'https://' + url

                # Remove any trailing punctuation
                url = url.rstrip('.,;!?)')

                filtered_urls.append(url)

            # Try to fetch content from the filtered URLs - follow multiple links for comprehensive results
            if filtered_urls:
                results = []
                for url in filtered_urls[:5]:  # Try first 5 relevant links for better coverage
                    try:
                        # Make sure URL is complete
                        if url.startswith('/'):
                            continue  # Skip relative URLs for now

                        logger.info(f"Following search result link: {url}")
                        result = await self._browser_tool._fetch_page(url)

                        if result and "error" not in result:
                            content = result.get("content", "")
                            title = result.get("title", "")

                            # Process the actual content page with LLM
                            processed = await self._llm_process_web_content(content, title, query)
                            if processed:
                                results.append(processed)
                                # Continue to get more results instead of stopping after first one
                                if len(results) >= 3:  # Limit to 3 good results to avoid overwhelming
                                    break

                    except Exception as e:
                        logger.error(f"Error following link {url}: {e}")
                        continue

                return results

            return []

        except Exception as e:
            logger.error(f"Error following search result links: {e}")
            return []

    def _provide_search_fallback(self, query: str) -> str:
        """Provide helpful fallback when web search fails."""
        query_lower = query.lower()

        # Provide helpful guidance based on query type
        if any(term in query_lower for term in ["stock", "price", "nasdaq", "nyse"]):
            # Extract company name for direct links
            company_name = self._extract_company_name(query)

            result = "I'm having trouble accessing real-time stock data. Here are direct links to check stock prices:\n\n"

            if company_name:
                # Provide direct search links for the specific company
                encoded_company = urllib.parse.quote_plus(f"{company_name} stock price")
                result += f"📈 **{company_name.title()} Stock Price:**\n"
                result += f"• [Yahoo Finance](https://finance.yahoo.com/lookup?s={encoded_company})\n"
                result += f"• [Google Finance](https://www.google.com/search?q={encoded_company})\n"
                result += f"• [MarketWatch](https://www.marketwatch.com/tools/quotes/lookup.asp?siteID=mktw&Lookup={encoded_company})\n\n"

            result += """📈 **General Financial Sites:**
• [Yahoo Finance](https://finance.yahoo.com)
• [Google Finance](https://www.google.com/finance)
• [MarketWatch](https://www.marketwatch.com)
• [Bloomberg](https://www.bloomberg.com)"""

            return result

        elif any(term in query_lower for term in ["weather", "temperature", "forecast"]):
            # Extract location for direct weather links
            location = self._extract_location_from_query(query)

            result = "I'm unable to access current weather data. Here are direct links to check the weather:\n\n"

            if location:
                # Provide direct weather links for the specific location
                encoded_location = urllib.parse.quote_plus(f"{location} weather")
                result += f"🌤️ **Weather for {location}:**\n"
                result += f"• [Weather.com](https://weather.com/search/enhancedlocalsearch?where={encoded_location})\n"
                result += f"• [AccuWeather](https://www.accuweather.com/en/search-locations?query={encoded_location})\n"
                result += f"• [Google Weather](https://www.google.com/search?q={encoded_location})\n\n"

            result += """🌤️ **General Weather Sites:**
• [Weather.com](https://weather.com)
• [AccuWeather.com](https://www.accuweather.com)
• [National Weather Service](https://weather.gov)"""

            return result

        elif any(term in query_lower for term in ["news", "latest", "recent"]):
            return """I'm having trouble accessing current news. For the latest information, check:

📰 **News Sources:**
• Google News (news.google.com)
• BBC News (bbc.com/news)
• Reuters (reuters.com)
• Your preferred news website"""

        else:
            return f"""I'm unable to search for current information about "{query}" right now due to technical difficulties.

🔍 **Alternative Search Options:**
• Google.com
• DuckDuckGo.com
• Bing.com

Try searching directly on these sites for the most up-to-date information."""

    def _extract_relevant_content(self, content: str, title: str, query: str) -> str:
        """Extract relevant content from web page based on query (generic approach)."""
        try:
            query_lower = query.lower()

            # For stock price queries, try to extract actual prices first
            if any(term in query_lower for term in ["stock", "price", "share"]):
                price_info = self._extract_price_data(content, query)
                if price_info:
                    return price_info

            # For weather queries, try to extract weather data
            if any(term in query_lower for term in ["weather", "temperature", "forecast", "rain", "snow"]):
                weather_info = self._extract_weather_data(content, query)
                if weather_info:
                    return weather_info

            # Extract query keywords for relevance matching
            query_words = set(query_lower.split())
            # Remove common stop words
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'what', 'is', 'are', 'was', 'were', 'how', 'when', 'where', 'why', 'who'}
            query_words = query_words - stop_words

            if not query_words:
                # If no meaningful query words, return summary
                return content[:500] + "..." if len(content) > 500 else content

            # Split content into sentences
            sentences = content.split('.')
            relevant_sentences = []

            # Score sentences based on query word matches
            for sentence in sentences[:20]:  # Check first 20 sentences
                sentence_lower = sentence.lower()
                matches = sum(1 for word in query_words if word in sentence_lower)

                if matches > 0:
                    relevant_sentences.append((sentence.strip(), matches))

            if relevant_sentences:
                # Sort by relevance score and take top sentences
                relevant_sentences.sort(key=lambda x: x[1], reverse=True)
                top_sentences = [sent[0] for sent in relevant_sentences[:3]]  # Reduced to 3 for brevity

                result = "Found information:\n"
                for sentence in top_sentences:
                    if sentence and len(sentence) > 10:  # Skip very short sentences
                        result += f"• {sentence}\n"

                return result.strip()
            else:
                # No relevant sentences found, return summary
                return content[:300] + "..." if len(content) > 300 else content

        except Exception as e:
            logger.error(f"Error extracting relevant content: {e}")
            return content[:300] + "..." if len(content) > 300 else content

    def _extract_price_data(self, content: str, query: str) -> str:
        """Extract price data from content for stock/financial queries."""
        import re

        # Look for price patterns in the content
        price_patterns = [
            r'\$[\d,]+\.?\d*',  # $123.45 or $123
            r'[\d,]+\.\d{2}',   # 123.45
            r'Price[:\s]*\$?[\d,]+\.?\d*',  # Price: $123.45
            r'[\d,]+\.?\d*\s*USD',  # 123.45 USD
        ]

        found_prices = []
        for pattern in price_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            found_prices.extend(matches)

        if found_prices:
            # Filter reasonable stock prices (between $1 and $2000)
            reasonable_prices = []
            for price in found_prices:
                try:
                    # Clean the price string
                    clean_price = re.sub(r'[^\d.]', '', price)
                    if clean_price:
                        price_value = float(clean_price)
                        if 1 <= price_value <= 2000:  # Reasonable stock price range
                            reasonable_prices.append(price)
                except:
                    continue

            if reasonable_prices:
                # Return the first few reasonable prices found
                unique_prices = list(set(reasonable_prices))[:3]
                result = "Stock price information found:\n"
                for price in unique_prices:
                    result += f"• {price}\n"
                return result.strip()

        return None  # No price data found

    def _extract_weather_data(self, content: str, query: str) -> str:
        """Extract weather data from content for weather queries."""
        import re

        # Look for temperature patterns
        temp_patterns = [
            r'(\d+)°[CF]',  # 75°F or 23°C
            r'(\d+)\s*degrees?',  # 75 degrees
            r'Temperature[:\s]*(\d+)',  # Temperature: 75
            r'High[:\s]*(\d+)',  # High: 75
            r'Low[:\s]*(\d+)',   # Low: 45
        ]

        # Look for weather condition patterns
        condition_patterns = [
            r'(sunny|cloudy|rainy|snowy|clear|overcast|partly cloudy|mostly sunny)',
            r'(thunderstorms?|showers?|drizzle|fog|mist)',
            r'(fair|pleasant|hot|cold|warm|cool|mild)'
        ]

        found_temps = []
        found_conditions = []

        # Extract temperatures
        for pattern in temp_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                try:
                    temp = int(match)
                    if -50 <= temp <= 150:  # Reasonable temperature range
                        found_temps.append(f"{temp}°")
                except:
                    continue

        # Extract weather conditions
        for pattern in condition_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            found_conditions.extend([match.title() for match in matches])

        # Look for location information
        location_info = self._extract_location_from_query(query)

        if found_temps or found_conditions:
            result = "🌤️ **Weather Information"
            if location_info:
                result += f" for {location_info}"
            result += ":**\n\n"

            if found_temps:
                unique_temps = list(set(found_temps))[:3]
                result += f"🌡️ **Temperature:** {', '.join(unique_temps)}\n"

            if found_conditions:
                unique_conditions = list(set(found_conditions))[:3]
                result += f"☁️ **Conditions:** {', '.join(unique_conditions)}\n"

            return result.strip()

        return None  # No weather data found

    def _extract_location_from_query(self, query: str) -> str:
        """Extract location from weather query."""
        query_lower = query.lower()

        # Remove weather-related words to isolate location
        weather_words = ['weather', 'temperature', 'forecast', 'in', 'for', 'what', 'is', 'the']
        words = query_lower.split()
        location_words = [word for word in words if word not in weather_words and len(word) > 1]

        if location_words:
            return ' '.join(location_words).title()

        return None

    def _extract_company_name(self, query: str) -> str:
        """Extract company name from stock price query."""
        query_lower = query.lower()

        # Remove common stock-related words to isolate company name
        stock_words = ['stock', 'price', 'share', 'current', 'latest', 'what', 'is', 'the', 'of', 'whats']
        words = query_lower.split()
        company_words = [word for word in words if word not in stock_words and len(word) > 2]

        if company_words:
            return ' '.join(company_words)

        return None

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
                loop = asyncio.get_running_loop()
                # In async context, use asyncio.create_task to run the search
                logger.info("Running web search in existing async context")

                # Create a task to run the search
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(self._run_sync_search, query)
                    result = future.result(timeout=30)  # 30 second timeout
                    return result

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

    def _run_sync_search(self, query: str) -> str:
        """Run search in a separate thread to avoid async conflicts."""
        try:
            # Create a new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(self._search_web(query))
            loop.close()
            return result
        except Exception as e:
            logger.error(f"Error in sync search: {e}")
            return self._provide_search_guidance(query)

    async def run_async(self, query: str) -> str:
        """Async version of web search that can be called from async contexts."""
        try:
            # For image searches, try to get actual image URLs using a simpler approach
            is_image_query = any(term in query.lower() for term in ["picture", "image", "photo", "pic"])

            if is_image_query:
                try:
                    result = self._search_images_sync(query)
                    return result
                except Exception as e:
                    logger.error(f"Image search failed, falling back to regular search: {e}")
                    # Fall through to regular search

            # Run the async search directly
            logger.info(f"Calling _search_web for non-image query: {query}")
            result = await self._search_web(query)
            return result

        except Exception as e:
            logger.error(f"Error in async web search: {e}")
            return f"Error searching for '{query}': {str(e)}"

    def _search_images_sync(self, query: str) -> str:
        """Synchronous image search that returns reliable, clickable image search links."""
        try:
            from urllib.parse import quote_plus

            # Extract the subject
            search_subject = self._extract_search_subject(query)
            logger.info(f"Providing image search links for: {search_subject}")

            # Use a simple image search API approach
            encoded_query = quote_plus(search_subject)

            # Return reliable, working image search links that definitely work in Discord
            return f"""**Images of {search_subject}:**

**Google Images:**
https://www.google.com/search?q={encoded_query}&tbm=isch

**High Quality Photos:**
https://unsplash.com/s/photos/{encoded_query}

**Bing Images:**
https://www.bing.com/images/search?q={encoded_query}

**Pinterest Images:**
https://www.pinterest.com/search/pins/?q={encoded_query}

Click any link above to see images of {search_subject}!"""

        except Exception as e:
            logger.error(f"Image search failed: {e}")
            return f"Sorry, I encountered an error searching for images of {search_subject}. Please try searching manually on Google Images or Unsplash."

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
            if ticker:
                return f"""Here are direct links to get current stock information for {ticker}:

**Yahoo Finance:**
https://finance.yahoo.com/quote/{ticker}

**Google Finance:**
https://www.google.com/search?q={ticker}+stock+price

**MarketWatch:**
https://www.marketwatch.com/investing/stock/{ticker}

These links will give you real-time stock prices and market data."""
            else:
                return f"""Here are links to search for stock information:

**Google Finance Search:**
https://www.google.com/search?q={encoded_query}+stock+price

**Yahoo Finance Search:**
https://finance.yahoo.com/lookup?s={encoded_query}

**MarketWatch Search:**
https://www.marketwatch.com/tools/quotes/lookup.asp?siteID=mktw&Lookup={encoded_query}

These links will help you find the stock information you're looking for."""

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



    def _run(self, query: str) -> str:
        """Synchronous wrapper for web search."""
        try:
            # Check if we're already in an async context
            try:
                loop = asyncio.get_running_loop()
                # In async context, use asyncio.create_task to run the search
                logger.info("Running web search in existing async context")

                # Create a task to run the search
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(self._run_sync_search, query)
                    result = future.result(timeout=30)  # 30 second timeout
                    return result

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

    def _run_sync_search(self, query: str) -> str:
        """Run search in a separate thread to avoid async conflicts."""
        try:
            # Create a new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(self._search_web(query))
            loop.close()
            return result
        except Exception as e:
            logger.error(f"Error in sync search: {e}")
            return self._provide_search_guidance(query)

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
