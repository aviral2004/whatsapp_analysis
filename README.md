# WhatsApp Chat Analyzer

An interactive web application built with Streamlit to analyze WhatsApp chat exports. The app provides detailed insights into chat patterns, participant statistics, sentiment analysis, and more.

## Features

- 📊 Basic chat statistics
- 👥 Participant analysis
- 💬 Conversation segmentation
- ⏱️ Response time analysis
- 😊 Sentiment analysis
- 📈 Activity patterns visualization

## Demo

You can try the app at: [Your Streamlit Cloud URL after deployment]

## Local Development

1. Clone the repository:
```bash
git clone [your-repo-url]
cd whatsapp-chat-analyzer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the app:
```bash
python run.py
```

Or use the shell script:
```bash
chmod +x run.sh
./run.sh
```

The app will be available at:
- Local: http://localhost:8501
- Network: http://[your-ip]:8501

## How to Use

1. Export your WhatsApp chat:
   - Open WhatsApp chat
   - Click on three dots (menu)
   - More > Export chat
   - Choose "Without Media"
   - Save the text file

2. Upload the exported chat file to the app
3. Explore the various analytics sections
4. Adjust conversation break threshold in sidebar if needed

## Deployment

To deploy on Streamlit Cloud:

1. Fork this repository
2. Sign up at [share.streamlit.io](https://share.streamlit.io)
3. Create a new app and connect it to your forked repository
4. Deploy!

## Privacy Note

This application processes WhatsApp chat data locally in your browser. No chat data is stored or transmitted to any external servers. When deploying, ensure you don't commit any personal chat files to the repository.

## License

MIT License - feel free to use this project as you wish.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 