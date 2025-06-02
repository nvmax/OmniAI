# Omni-Assistant Discord Bot

A sophisticated Discord AI bot powered by CrewAI multi-agent framework and local LLM inference through LM Studio. Features intelligent conversation, research capabilities, coding assistance, image analysis, and advanced memory management - all running 100% locally.

## 🌟 Features

### 🤖 Core AI Capabilities
- **Multi-Agent AI System**: Powered by CrewAI with specialized agents for different tasks
- **Advanced Memory**: Short-term, long-term, and contextual memory using vector database
- **Per-User Personalities**: Each user can select their own AI personality with just saying "switch to [personality name]"
- **Intelligent Task Routing**: Automatically routes requests to appropriate specialized agents
- **Local Operation**: No external API dependencies for core AI functionality

### 🎯 Specialized Agents
- **🔍 Research Agent**: In-depth research, analysis, and knowledge synthesis
- **🌐 Web Search Agent**: Quick web searches for current information and facts
- **💻 Coding Agent**: Code generation, debugging, file analysis, and safe execution
- **🖼️ Image Analysis Agent**: AI-powered image analysis with vision transformers and commentary

### 📱 Discord Integration
- **Natural Conversation**: Responds to messages without requiring @mentions
- **Smart Filtering**: Ignores messages with @mentions to respect private conversations
- **Channel Control**: Configure specific channels for bot interaction
- **Rate Limiting**: Built-in cooldown system to prevent spam
- **Modern UI**: Slash commands with interactive dropdowns and rich embeds
- **Image Upload Support**: Automatic detection and analysis of uploaded images

## 🏗️ Architecture

```
omni_assistant/
├── src/
│   ├── main.py                    # Bot entry point
│   ├── config.py                  # Configuration management
│   ├── llm_integration.py         # LM Studio client with response cleaning
│   ├── memory_manager.py          # Vector DB & memory management
│   ├── personalities.py           # Per-user personality system
│   ├── crew_orchestrator.py       # CrewAI coordination & task routing
│   ├── discord_handlers.py        # Discord integration & slash commands
│   ├── agents/
│   │   ├── base_agent.py          # Common agent configuration
│   │   ├── research_agent.py      # In-depth research specialist
│   │   ├── coding_agent.py        # Code generation & debugging
│   │   ├── web_search_agent.py    # Quick web search specialist
│   │   └── image_analysis_agent.py # AI image analysis & commentary
│   └── tools/
│       ├── web_browser_tool.py    # Web browsing capabilities
│       ├── coding_tool.py         # Safe code execution
│       ├── vector_db_tool.py      # Memory retrieval
│       └── query_parser.py        # Query parsing & classification
├── data/
│   └── vector_db/                 # Local vector database storage
├── logs/
│   └── bot.log                   # Application logs
├── requirements.txt               # Python dependencies
├── .env                          # Environment variables
└── README.md                     # This file
```

## 🚀 Quick Start

### Prerequisites

1. **Python 3.8+** installed
2. **LM Studio** downloaded and running
3. **Discord Bot Token** from Discord Developer Portal

### Installation

1. **Clone or download** this project to your local machine

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up LM Studio**:
   - Download and install [LM Studio](https://lmstudio.ai/)
   - Load a compatible model (recommended: Llama 2, Mistral, or similar)
    - very capable with qwen/qwen3-14b
    - model must support tool calls
   - Start the local server (default: http://127.0.0.1:1234)

4. **Configure environment variables**:
   ```bash
   cp .env.example .env
   ```

   Edit `.env` with your settings:
   ```env
   # Discord Configuration
   DISCORD_BOT_TOKEN=your_discord_bot_token_here
   DISCORD_SERVER_ID=your_server_id_here
   DISCORD_CHANNEL_ID=your_channel_id_here

   # LM Studio Configuration (defaults should work)
   LM_STUDIO_HOST=127.0.0.1
   LM_STUDIO_PORT=1234

   # Bot Personality (choose default personality for new users)
   DEFAULT_BOT_PERSONALITY=default

   # Logging
   LOG_LEVEL=INFO
   ```

5. **Run the bot**:
   ```bash
   python src/main.py
   ```

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DISCORD_BOT_TOKEN` | Your Discord bot token | Required |
| `DISCORD_SERVER_ID` | Server ID to monitor (optional) | All servers |
| `DISCORD_CHANNEL_ID` | Channel ID(s) to monitor | All channels |
| `LM_STUDIO_HOST` | LM Studio server host | 127.0.0.1 |
| `LM_STUDIO_PORT` | LM Studio server port | 1234 |
| `DEFAULT_BOT_PERSONALITY` | Default personality mode | default |
| `LOG_LEVEL` | Logging level | INFO |

### Discord Setup

1. **Create a Discord Application**:
   - Go to [Discord Developer Portal](https://discord.com/developers/applications)
   - Create a new application
   - Go to "Bot" section and create a bot
   - Copy the bot token

2. **Set Bot Permissions**:
   - Read Messages
   - Send Messages
   - Use Slash Commands
   - Read Message History

3. **Invite Bot to Server**:
   - Go to "OAuth2" > "URL Generator"
   - Select "bot" scope and required permissions
   - Use generated URL to invite bot

### LM Studio Setup

1. **Download Models**:
   - Open LM Studio
   - Browse and download a compatible model
   - Recommended: `microsoft/DialoGPT-medium` or similar conversational models

2. **Start Server**:
   - Load your chosen model
   - Go to "Local Server" tab
   - Click "Start Server"
   - Verify it's running on http://127.0.0.1:1234

## 🎭 Per-User Personality System

Each user can select their own AI personality that affects only their interactions. The bot supports multiple personalities that change its behavior and response style:

### Available Personalities
- **Default**: A balanced, professional assistant suitable for most interactions
- **😎 Casual**: Relaxed and informal - laid-back conversation style
- **🔬 Technical**: Logical and detail-oriented - methodical and precise
- **🎉 creative**: creative and imaginative - creative and vivid language
- **😏 Sarcastic**: Witty and sarcastic - dry humor and clever remarks
- **🤝 Menotor**: Extremely supportive - patient and thorough assistance
- **😊 pirate**: pirate-themed - witty and humorous
- **😤 Angry**: Insulting but helpful - sarcastic and blunt responses


### How to Change Your Personality
- Use the `/personality` slash command for a list of available personalities
- Switch to a specific personality by saying "switch to [personality name]"

### Personal Settings
- Each user has their own personality setting
- Your choice doesn't affect other users
- Settings persist across bot restarts

## 🤖 Commands & Interaction

### Slash Commands (Recommended)
- `/personality` - list of personalities

### Text Commands
- `!help` - Show comprehensive help information
- `!personality [name]` - Change bot personality (alternative to slash command)
- `!personalities` - List all available personalities with descriptions
- `!memory clear` - Clear your conversation history
- `!memory stats` - Show detailed memory usage statistics
- `!status` - Show bot status, health, and connection info

### Image Analysis
- **Upload any image** - Bot automatically detects and analyzes images
- **Supported formats**: PNG, JPG, JPEG, GIF, WebP
- **AI Analysis**: Uses vision transformers to understand image content
- **Personality Commentary**: Provides commentary in your selected personality style

### Natural Interaction Examples
The bot responds to natural conversation without commands:
- **Research**: "Tell me about quantum computing"
- **Web Search**: "What's the current stock price of NVIDIA?"
- **Coding**: "Help me debug this Python function"
- **Image Analysis**: Upload any image for AI commentary
- **Memory**: "Remember that I prefer TypeScript over JavaScript"
- **Current Info**: "What's happening in AI news today?"

## 🧠 Advanced Memory System

### Per-User Memory Management
- **Individual Memory**: Each user has their own conversation history
- **Context Awareness**: Bot remembers your preferences and past conversations
- **Cross-Session Persistence**: Memory survives bot restarts and updates

### Short-term Memory
- Stores recent conversation turns for immediate context
- Maintains conversation flow within sessions
- Automatically managed with configurable size limits

### Long-term Memory
- **Vector Database**: Uses ChromaDB for semantic storage
- **Permanent Storage**: Important information stored indefinitely
- **Semantic Search**: Finds relevant past information using AI embeddings
- **Smart Retrieval**: Automatically surfaces relevant context

### Memory Features
- **Preference Learning**: Remembers your coding preferences, interests, etc.
- **Context Enhancement**: Uses past conversations to improve responses
- **Memory Commands**: Clear or view your memory statistics
- **Privacy**: Each user's memory is completely separate

## �️ Image Analysis Features

### AI-Powered Image Understanding
- **Vision Transformer Models**: Uses BLIP (Salesforce/blip-image-captioning-base) for image analysis
- **Automatic Detection**: Bot automatically detects when you upload images
- **Multi-Format Support**: PNG, JPG, JPEG, GIF, WebP formats supported
- **GPU Acceleration**: Uses GPU if available for faster processing

### Image Analysis Capabilities
- **Content Recognition**: Identifies objects, people, scenes, and activities
- **Technical Analysis**: Reports image dimensions, aspect ratio, and format
- **Artistic Commentary**: Provides insights about composition and visual elements
- **Personality Integration**: Commentary matches your selected personality style

### Example Image Analysis Response
```
**sunset_photo.jpg**

📸 Content: A beautiful sunset over the ocean with vibrant orange clouds
📏 Size: 1920x1080
🖼️ Format: landscape

[Personality-based commentary about the image's composition, colors, and artistic elements]
```

## �🔧 Development & Customization

### Adding New Agents
1. Create agent file in `src/agents/`
2. Inherit from `BaseAgentConfig`
3. Define tools and capabilities
4. Register in `crew_orchestrator.py`

### Adding New Tools
1. Create tool file in `src/tools/`
2. Inherit from `crewai_tools.BaseTool`
3. Implement `_run` method
4. Add to appropriate agent

### Custom Personalities
Edit `src/personalities.py` to add new personality types:

```python
"custom": Personality(
    name="custom",
    system_message="Your custom system prompt here...",
    description="Description of the personality",
    response_style="How it responds",
    emoji_usage="Emoji usage pattern"
)
```

### Response Cleaning System
The bot includes advanced response cleaning to ensure high-quality outputs:
- **Thinking Tag Removal**: Removes internal reasoning from responses
- **Duplicate Detection**: Prevents repeated content
- **Content Validation**: Ensures responses meet quality standards

## 🐛 Troubleshooting

### Common Issues

1. **Bot doesn't respond**:
   - Check if LM Studio is running and model is loaded
   - Verify Discord token is correct in `.env`
   - Check bot permissions in Discord server
   - Ensure bot is in the correct channel (if channel restriction is set)

2. **LM Studio connection failed**:
   - Ensure LM Studio server is started (Local Server tab)
   - Check host/port configuration in `.env`
   - Verify model is loaded and responding
   - Test connection: `curl http://127.0.0.1:1234/v1/models`

3. **Slash commands not working**:
   - Wait a few minutes after bot startup for command sync
   - Check bot has "Use Slash Commands" permission
   - Try restarting the bot to re-sync commands

4. **Image analysis not working**:
   - Check if image format is supported (PNG, JPG, JPEG, GIF, WebP)
   - Verify Pillow and transformers are installed
   - Check logs for vision model loading errors

5. **Memory/personality issues**:
   - Check disk space for vector database storage
   - Verify ChromaDB installation and permissions
   - Clear memory if corrupted: `!memory clear`

6. **Response quality issues**:
   - Try different LM Studio models
   - Adjust model parameters in LM Studio
   - Check if thinking tags appear (indicates cleaning system issues)

### Logs & Debugging

- **Main log**: Check `logs/bot.log` for detailed error information
- **Console output**: Monitor terminal for real-time status
- **Debug mode**: Set `LOG_LEVEL=DEBUG` in `.env` for verbose logging
- **Health check**: Use `!status` command to verify all systems

## 📝 License

This project is open source. Feel free to modify and distribute according to your needs.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.



## 🔮 Future Enhancements

- [ ] Voice channel integration with speech-to-text
- [ ] Plugin system for custom tools and agents
- [ ] Web dashboard for bot management and analytics
- [ ] Multi-server memory isolation
- [ ] Advanced conversation analytics and insights
- [ ] File upload analysis (PDFs, documents, code files)
- [ ] Integration with more AI models and services
- [ ] Custom agent creation through Discord interface


**Omni-Assistant** - Your advanced local AI companion for Discord! 🚀🤖✨
Created by Jerrod Linderman
Support Dev [PayPal](https://paypal.me/nvmaxx]