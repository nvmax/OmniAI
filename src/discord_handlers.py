"""
Discord event handlers and message processing for Omni-Assistant.
Manages Discord interactions, message filtering, and response coordination.
"""

import asyncio
import logging
import re
from typing import Optional, List
import discord
from discord.ext import commands

try:
    from .config import config
    from .crew_orchestrator import crew_orchestrator
    from .personalities import personality_manager
    from .memory_manager import memory_manager
except ImportError:
    from config import config
    from crew_orchestrator import crew_orchestrator
    from personalities import personality_manager
    from memory_manager import memory_manager

logger = logging.getLogger(__name__)

class DiscordBot(commands.Bot):
    """Enhanced Discord bot with AI agent integration."""
    
    def __init__(self):
        # Set up intents
        intents = discord.Intents.default()
        intents.message_content = True
        intents.guilds = True
        intents.guild_messages = True
        
        super().__init__(
            command_prefix='!',
            intents=intents,
            help_command=None
        )
        
        self.orchestrator = crew_orchestrator
        self.allowed_channels = set(config.discord_channel_ids) if config.discord_channel_ids else set()
        self.allowed_server = config.discord_server_id
        
        # Rate limiting
        self.user_cooldowns = {}
        self.cooldown_duration = 3  # seconds between messages per user
        
    async def on_ready(self):
        """Called when the bot is ready."""
        logger.info(f'{self.user} has connected to Discord!')
        logger.info(f'Bot is in {len(self.guilds)} guilds')

        if self.allowed_channels:
            logger.info(f'Monitoring channels: {self.allowed_channels}')
        else:
            logger.info('Monitoring all channels (no channel restriction)')

        # Sync slash commands
        try:
            synced = await self.tree.sync()
            logger.info(f"Synced {len(synced)} slash command(s)")
        except Exception as e:
            logger.error(f"Failed to sync slash commands: {e}")

        # Set bot status
        activity = discord.Activity(
            type=discord.ActivityType.listening,
            name="conversations | /personality for settings"
        )
        await self.change_presence(activity=activity)
    
    async def on_message(self, message):
        """Handle incoming messages."""
        # Process commands first
        await self.process_commands(message)
        
        # Skip if message is from a bot
        if message.author.bot:
            return
        
        # Skip if not in allowed server (if specified)
        if self.allowed_server and str(message.guild.id) != self.allowed_server:
            return
        
        # Skip if not in allowed channels (if specified)
        if self.allowed_channels and str(message.channel.id) not in self.allowed_channels:
            return
        
        # Skip if message contains @mentions (except possibly the bot itself)
        if self._contains_user_mentions(message):
            return
        
        # Check rate limiting
        if not self._check_rate_limit(message.author.id):
            return

        # Check for image attachments first
        if message.attachments:
            await self._process_image_message(message)
            return

        # Process text message
        await self._process_user_message(message)
    
    def _contains_user_mentions(self, message) -> bool:
        """Check if message contains @mentions of users (excluding the bot)."""
        # Check for @everyone or @here
        if message.mention_everyone:
            return True
        
        # Check for user mentions (excluding the bot itself)
        user_mentions = [mention for mention in message.mentions if mention.id != self.user.id]
        if user_mentions:
            return True
        
        # Check for role mentions
        if message.role_mentions:
            return True
        
        return False
    
    def _check_rate_limit(self, user_id: int) -> bool:
        """Check if user is rate limited."""
        import time
        current_time = time.time()
        
        if user_id in self.user_cooldowns:
            if current_time - self.user_cooldowns[user_id] < self.cooldown_duration:
                return False
        
        self.user_cooldowns[user_id] = current_time
        return True

    async def _process_image_message(self, message):
        """Process a message with image attachments."""
        import uuid
        request_id = str(uuid.uuid4())[:8]

        try:
            # Show typing indicator
            async with message.channel.typing():
                user_id = str(message.author.id)

                logger.info(f"[{request_id}] Processing image message from user {user_id}")

                # Import image analysis agent
                try:
                    from .agents.image_analysis_agent import image_analysis_executor
                except ImportError:
                    from agents.image_analysis_agent import image_analysis_executor

                responses = []

                # Process each image attachment
                for attachment in message.attachments:
                    if any(attachment.filename.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.gif', '.webp']):
                        logger.info(f"[{request_id}] Analyzing image: {attachment.filename}")

                        # Create context for the analysis
                        context = {
                            "filename": attachment.filename,
                            "size": attachment.size,
                            "message_content": message.content if message.content else None
                        }

                        # Analyze the image
                        analysis = await image_analysis_executor.analyze_image(
                            image_url=attachment.url,
                            user_id=user_id,
                            context=context
                        )

                        responses.append(f"**{attachment.filename}**\n{analysis}")
                    else:
                        logger.info(f"[{request_id}] Skipping non-image attachment: {attachment.filename}")

                if responses:
                    # Combine all image analyses
                    if len(responses) == 1:
                        final_response = responses[0]
                    else:
                        final_response = "\n\n---\n\n".join(responses)

                    # Add user message context if provided
                    if message.content:
                        final_response = f"*{message.content}*\n\n{final_response}"

                    logger.info(f"[{request_id}] Sending image analysis response")

                    # Split long responses if needed
                    if len(final_response) > 2000:
                        chunks = self._split_message(final_response)
                        for i, chunk in enumerate(chunks, 1):
                            if len(chunks) > 1:
                                part_indicator = f"**[Part {i}/{len(chunks)}]**\n\n"
                                if len(part_indicator) + len(chunk) <= 2000:
                                    chunk = part_indicator + chunk
                            await message.channel.send(chunk)
                            if i < len(chunks):
                                await asyncio.sleep(0.5)
                    else:
                        await message.channel.send(final_response)
                else:
                    await message.channel.send("I can see you uploaded files, but none appear to be images I can analyze. I support PNG, JPG, JPEG, GIF, and WebP formats.")

                logger.info(f"[{request_id}] Image message processing completed")

        except Exception as e:
            logger.error(f"[{request_id}] Error processing image message: {e}")
            await message.channel.send("I encountered an error while analyzing your image(s). Please try again.")

    async def _process_user_message(self, message):
        """Process a user message and generate a response."""
        import uuid
        request_id = str(uuid.uuid4())[:8]  # Short unique ID for this request

        try:
            # Show typing indicator
            async with message.channel.typing():
                # Get user ID as string for consistency
                user_id = str(message.author.id)

                logger.info(f"[{request_id}] Processing message from user {user_id}: {message.content[:50]}...")

                # Process the message through the orchestrator
                response = await self.orchestrator.process_request(
                    user_message=message.content,
                    user_id=user_id,
                    channel_id=str(message.channel.id)
                )

                # Debug: Log the raw response to see what we're getting
                logger.info(f"[{request_id}] Raw response length: {len(response)} chars")
                logger.info(f"[{request_id}] Raw response preview: {response[:200]}...")

                # Split long responses if needed and send all parts
                if len(response) > 2000:
                    chunks = self._split_message(response)
                    logger.info(f"[{request_id}] Splitting long response into {len(chunks)} parts for user {user_id}")
                    logger.info(f"[{request_id}] Chunks preview: {[chunk[:50] + '...' for chunk in chunks]}")

                    for i, chunk in enumerate(chunks, 1):
                        # Add part indicator for multi-part messages
                        if len(chunks) > 1:
                            part_indicator = f"**[Part {i}/{len(chunks)}]**\n\n"
                            # Check if adding indicator would exceed limit
                            if len(part_indicator) + len(chunk) <= 2000:
                                chunk = part_indicator + chunk

                        logger.info(f"[{request_id}] Sending chunk {i}: {chunk[:100]}...")
                        await message.channel.send(chunk)

                        # Small delay between parts to ensure proper ordering
                        if i < len(chunks):
                            await asyncio.sleep(0.5)
                else:
                    logger.info(f"[{request_id}] Sending single response: {response[:100]}...")
                    await message.channel.send(response)

                logger.info(f"[{request_id}] Message processing completed")
                    
        except Exception as e:
            logger.error(f"Error processing message from {message.author}: {e}")
            await message.channel.send(
                "I apologize, but I encountered an error while processing your message. Please try again."
            )
    
    def _split_message(self, message: str, max_length: int = 1950) -> List[str]:
        """
        Split a long message into chunks intelligently.
        Uses 1950 chars to leave room for part indicators.
        """
        if len(message) <= max_length:
            return [message]

        chunks = []
        current_chunk = ""

        # First, try to split by code blocks to keep them intact
        code_block_pattern = r'```[\s\S]*?```'
        parts = re.split(f'({code_block_pattern})', message)

        for part in parts:
            if not part:
                continue

            # If this part is a code block, handle it specially
            if re.match(code_block_pattern, part):
                if len(current_chunk) + len(part) <= max_length:
                    current_chunk += part
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                        current_chunk = ""

                    # If code block is too long, split it but preserve structure
                    if len(part) > max_length:
                        code_chunks = self._split_code_block(part, max_length)
                        chunks.extend(code_chunks[:-1])
                        current_chunk = code_chunks[-1] if code_chunks else ""
                    else:
                        current_chunk = part
            else:
                # Regular text - split by paragraphs, then sentences
                paragraphs = part.split('\n\n')

                for paragraph in paragraphs:
                    if not paragraph.strip():
                        continue

                    if len(current_chunk) + len(paragraph) + 2 <= max_length:
                        if current_chunk:
                            current_chunk += '\n\n' + paragraph
                        else:
                            current_chunk = paragraph
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())

                        # If paragraph itself is too long, split by sentences
                        if len(paragraph) > max_length:
                            sentence_chunks = self._split_by_sentences(paragraph, max_length)
                            chunks.extend(sentence_chunks[:-1])
                            current_chunk = sentence_chunks[-1] if sentence_chunks else ""
                        else:
                            current_chunk = paragraph

        if current_chunk and current_chunk.strip():
            chunks.append(current_chunk.strip())

        # Final safety check - if any chunk is still too long, force split
        final_chunks = []
        for chunk in chunks:
            if len(chunk) <= max_length:
                final_chunks.append(chunk)
            else:
                final_chunks.extend(self._force_split(chunk, max_length))

        return final_chunks

    def _split_code_block(self, code_block: str, max_length: int) -> List[str]:
        """Split a code block while preserving structure."""
        lines = code_block.split('\n')
        first_line = lines[0]  # ```language
        last_line = lines[-1] if lines[-1].strip() == '```' else '```'
        code_lines = lines[1:-1] if lines[-1].strip() == '```' else lines[1:]

        chunks = []
        current_chunk = first_line + '\n'

        for line in code_lines:
            if len(current_chunk) + len(line) + len(last_line) + 2 <= max_length:
                current_chunk += line + '\n'
            else:
                current_chunk += last_line
                chunks.append(current_chunk)
                current_chunk = first_line + '\n' + line + '\n'

        if current_chunk != first_line + '\n':
            current_chunk += last_line
            chunks.append(current_chunk)

        return chunks

    def _split_by_sentences(self, text: str, max_length: int) -> List[str]:
        """Split text by sentences."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 <= max_length:
                if current_chunk:
                    current_chunk += ' ' + sentence
                else:
                    current_chunk = sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk)

                if len(sentence) > max_length:
                    # Force split long sentences
                    chunks.extend(self._force_split(sentence, max_length))
                    current_chunk = ""
                else:
                    current_chunk = sentence

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def _force_split(self, text: str, max_length: int) -> List[str]:
        """Force split text at word boundaries."""
        chunks = []
        words = text.split()
        current_chunk = ""

        for word in words:
            if len(current_chunk) + len(word) + 1 <= max_length:
                if current_chunk:
                    current_chunk += ' ' + word
                else:
                    current_chunk = word
            else:
                if current_chunk:
                    chunks.append(current_chunk)

                # If single word is too long, split it
                if len(word) > max_length:
                    for i in range(0, len(word), max_length):
                        chunks.append(word[i:i + max_length])
                    current_chunk = ""
                else:
                    current_chunk = word

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

class PersonalitySelectionView(discord.ui.View):
    """View for personality selection with dropdown."""

    def __init__(self):
        super().__init__(timeout=300)  # 5 minute timeout

        # Create dropdown with all available personalities
        self.add_item(PersonalityDropdown())

class PersonalityDropdown(discord.ui.Select):
    """Dropdown for selecting personalities."""

    def __init__(self):
        # Get all available personalities
        try:
            from .personalities import personality_manager
        except ImportError:
            from personalities import personality_manager

        personalities = personality_manager.list_personalities()

        # Create options for each personality
        options = []
        for name, description in personalities.items():
            # Limit description length for dropdown
            short_desc = description[:50] + "..." if len(description) > 50 else description

            options.append(discord.SelectOption(
                label=name.title(),
                value=name,
                description=short_desc,
                emoji=self._get_personality_emoji(name)
            ))

        super().__init__(
            placeholder="Choose your AI personality...",
            min_values=1,
            max_values=1,
            options=options
        )

    def _get_personality_emoji(self, personality_name: str) -> str:
        """Get emoji for personality type."""
        emoji_map = {
            "default": "🤖",
            "friendly": "😊",
            "professional": "💼",
            "casual": "😎",
            "enthusiastic": "🎉",
            "sarcastic": "😏",
            "angry": "😤",
            "helpful": "🤝",
            "creative": "🎨",
            "analytical": "🔬"
        }
        return emoji_map.get(personality_name.lower(), "🎭")

    async def callback(self, interaction: discord.Interaction):
        """Handle personality selection."""
        try:
            from .personalities import personality_manager
        except ImportError:
            from personalities import personality_manager

        selected_personality = self.values[0]
        user_id = str(interaction.user.id)

        # Set the user's personality
        success = personality_manager.set_user_personality(user_id, selected_personality)

        if success:
            personality = personality_manager.get_personality(selected_personality)

            # Create success embed
            embed = discord.Embed(
                title="✅ Personality Updated!",
                description=f"You've switched to the **{personality.name.title()}** personality.",
                color=0x00ff00
            )

            embed.add_field(
                name="Your New Personality",
                value=f"**{personality.name.title()}** - {personality.description}",
                inline=False
            )

            embed.add_field(
                name="What This Means",
                value="I'll now respond to you with this personality style. Other users aren't affected by your choice.",
                inline=False
            )

            embed.set_footer(text="You can change this anytime with /personality")

            await interaction.response.edit_message(embed=embed, view=None)

        else:
            # Error embed
            embed = discord.Embed(
                title="❌ Error",
                description=f"Failed to set personality to {selected_personality}. Please try again.",
                color=0xff0000
            )

            await interaction.response.edit_message(embed=embed, view=None)

class DiscordCommands:
    """Discord slash commands and text commands for the bot."""
    
    def __init__(self, bot: DiscordBot):
        self.bot = bot
        self._setup_commands()
    
    def _setup_commands(self):
        """Set up bot commands."""
        
        @self.bot.command(name='help')
        async def help_command(ctx):
            """Show help information."""
            embed = discord.Embed(
                title="Omni-Assistant Help",
                description="I'm an AI assistant powered by local LLM and CrewAI agents!",
                color=0x00ff00
            )
            
            embed.add_field(
                name="How I Work",
                value=(
                    "I read messages in this channel and respond naturally without needing @mentions. "
                    "I ignore messages that mention other users to respect conversations."
                ),
                inline=False
            )
            
            embed.add_field(
                name="Commands",
                value=(
                    "`!help` - Show this help message\n"
                    "`/personality` - Select your personal AI personality (recommended)\n"
                    "`!personality [name]` - Change my personality (text command)\n"
                    "`!personalities` - List available personalities\n"
                    "`!memory clear` - Clear our conversation history\n"
                    "`!memory stats` - Show memory usage statistics\n"
                    "`!status` - Show bot status"
                ),
                inline=False
            )
            
            embed.add_field(
                name="Capabilities",
                value=(
                    "🔍 Research and information gathering\n"
                    "💻 Code generation and debugging\n"
                    "🧠 Conversation memory and context\n"
                    "🎭 Multiple personalities\n"
                    "🌐 Web browsing (local)\n"
                    "📁 File analysis\n"
                    "🖼️ Image analysis and commentary"
                ),
                inline=False
            )
            
            await ctx.send(embed=embed)
        
        @self.bot.command(name='personality')
        async def personality_command(ctx, *, personality_name: str = None):
            """Change bot personality."""
            user_id = str(ctx.author.id)

            if not personality_name:
                current = personality_manager.get_user_personality(user_id)
                await ctx.send(f"Current personality: **{current.name}** - {current.description}")
                return

            success = personality_manager.set_user_personality(user_id, personality_name)
            if success:
                personality = personality_manager.get_personality(personality_name)
                await ctx.send(f"✅ Switched to **{personality.name}** personality! {personality.description}")
            else:
                await ctx.send(f"❌ Unknown personality: {personality_name}. Use `!personalities` to see available options.")

        # Add slash command for personality selection
        @self.bot.tree.command(name="personality", description="Select your personal AI personality")
        async def slash_personality_command(interaction: discord.Interaction):
            """Slash command to select personality with dropdown."""
            view = PersonalitySelectionView()

            embed = discord.Embed(
                title="🎭 Select Your AI Personality",
                description="Choose how I should behave and respond to you personally:",
                color=0x9932cc
            )

            # Show current personality
            user_id = str(interaction.user.id)
            current = personality_manager.get_user_personality(user_id)
            embed.add_field(
                name="Current Personality",
                value=f"**{current.name.title()}** - {current.description}",
                inline=False
            )

            embed.set_footer(text="This setting is personal to you and won't affect other users.")

            await interaction.response.send_message(embed=embed, view=view, ephemeral=True)
        
        @self.bot.command(name='personalities')
        async def personalities_command(ctx):
            """List available personalities."""
            personalities = personality_manager.list_personalities()
            
            embed = discord.Embed(
                title="Available Personalities",
                description="Choose how I should behave and respond:",
                color=0x9932cc
            )
            
            for name, description in personalities.items():
                embed.add_field(
                    name=name.title(),
                    value=description,
                    inline=False
                )
            
            embed.set_footer(text="Use !personality <name> to switch")
            await ctx.send(embed=embed)
        
        @self.bot.command(name='memory')
        async def memory_command(ctx, action: str = None):
            """Memory management commands."""
            user_id = str(ctx.author.id)
            
            if action == "clear":
                memory_manager.clear_short_term_memory(user_id)
                await ctx.send("🧹 Cleared our conversation history!")
            
            elif action == "stats":
                # Get user-specific stats
                user_stats = memory_manager.get_memory_stats(user_id)
                global_stats = memory_manager.get_memory_stats()

                embed = discord.Embed(
                    title="Your Memory Statistics",
                    description=f"Personal memory data for <@{ctx.author.id}>",
                    color=0x3498db
                )

                # User-specific stats
                embed.add_field(name="Your Conversation Messages", value=user_stats["conversation_messages"], inline=True)
                embed.add_field(name="Your Short-term Memories", value=user_stats["short_term_memories"], inline=True)
                embed.add_field(name="Your Long-term Memories", value=user_stats.get("long_term_memories", "unknown"), inline=True)
                embed.add_field(name="Your Preferences Stored", value=user_stats["user_preferences"], inline=True)
                embed.add_field(name="Facts About You", value=user_stats["user_facts"], inline=True)
                embed.add_field(name="Vector DB Available", value="✅" if user_stats["vector_db_available"] else "❌", inline=True)

                # Global stats
                embed.add_field(name="Total Bot Users", value=global_stats["total_users"], inline=True)
                embed.add_field(name="Total User Collections", value=global_stats["user_collections"], inline=True)
                embed.add_field(name="Global Cache Entries", value=global_stats["cache_entries"], inline=True)

                await ctx.send(embed=embed)
            
            else:
                await ctx.send("Memory commands: `!memory clear` or `!memory stats`")
        
        @self.bot.command(name='status')
        async def status_command(ctx):
            """Show bot status."""
            embed = discord.Embed(
                title="Omni-Assistant Status",
                color=0x00ff00
            )

            # Check LLM connection
            try:
                # Import from correct location
                try:
                    from .llm_integration import llm_manager
                except ImportError:
                    from llm_integration import llm_manager

                health = await llm_manager.health_check()
                llm_status = "✅ Connected" if health else "❌ Disconnected"
            except Exception as e:
                llm_status = f"❌ Error: {str(e)[:50]}"

            # Check vector database
            try:
                try:
                    from .memory_manager import memory_manager
                except ImportError:
                    from memory_manager import memory_manager

                db_status = "✅ Active" if memory_manager.chroma_client else "❌ Inactive"
            except Exception as e:
                db_status = f"❌ Error: {str(e)[:30]}"

            # Bot info
            channel_count = len(self.bot.allowed_channels) if self.bot.allowed_channels else "All"
            guild_count = len(self.bot.guilds)

            embed.add_field(name="LM Studio Connection", value=llm_status, inline=True)
            embed.add_field(name="Vector Database", value=db_status, inline=True)
            embed.add_field(name="Guilds", value=str(guild_count), inline=True)
            embed.add_field(name="Monitored Channels", value=str(channel_count), inline=True)
            embed.add_field(name="Bot Latency", value=f"{round(self.bot.latency * 1000)}ms", inline=True)
            embed.add_field(name="Uptime", value="Running", inline=True)

            # Add personality info
            try:
                try:
                    from .personalities import personality_manager
                except ImportError:
                    from personalities import personality_manager

                user_id = str(ctx.author.id)
                current_personality = personality_manager.get_user_personality(user_id)
                embed.add_field(
                    name="Your Personality",
                    value=f"{current_personality.name.title()}",
                    inline=True
                )
            except Exception:
                embed.add_field(name="Your Personality", value="Unknown", inline=True)

            await ctx.send(embed=embed)

def create_discord_bot() -> DiscordBot:
    """Create and configure the Discord bot."""
    bot = DiscordBot()
    commands_handler = DiscordCommands(bot)
    return bot
