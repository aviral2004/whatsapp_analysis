import pandas as pd
import numpy as np
from datetime import datetime
import re
from typing import List, Dict, Tuple
from collections import defaultdict
import matplotlib.pyplot as plt
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class WhatsAppChatParser:
    def __init__(self, file_path: str):
        """Initialize the parser with the chat file path."""
        self.file_path = file_path
        self.messages = []
        self.df = None
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self._parse_chat()
    
    def _parse_chat(self):
        """Parse the WhatsApp chat file and create a structured DataFrame."""
        with open(self.file_path, 'r', encoding='utf-8') as file:
            pattern = r'(\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}(?::\d{2})?(?:\s|[\u202f])(?:AM|PM|am|pm)?) - (.*?): (.*)'
            
            for line in file:
                match = re.match(pattern, line.strip())
                if match:
                    timestamp_str, sender, message = match.groups()
                    # Clean the timestamp string by replacing Unicode spaces
                    timestamp_str = timestamp_str.replace('\u202f', ' ')
                    
                    try:
                        # Try parsing with DD/MM/YY format first
                        timestamp = datetime.strptime(timestamp_str, '%d/%m/%y, %I:%M:%S %p')
                    except ValueError:
                        try:
                            # Try without seconds
                            timestamp = datetime.strptime(timestamp_str, '%d/%m/%y, %I:%M %p')
                        except ValueError:
                            # Skip malformed timestamps
                            continue
                    
                    self.messages.append({
                        'timestamp': timestamp,
                        'sender': sender,
                        'message': message
                    })
        
        if not self.messages:
            raise ValueError("No valid messages found in the chat file. Please check the file format.")
            
        self.df = pd.DataFrame(self.messages)
    
    def get_basic_stats(self) -> Dict:
        """Calculate basic statistics about the chat."""
        stats = {
            'total_messages': len(self.df),
            'unique_senders': len(self.df['sender'].unique()),
            'avg_message_length': self.df['message'].str.len().mean(),
            'total_words': self.df['message'].str.split().str.len().sum(),
            'time_span': (self.df['timestamp'].max() - self.df['timestamp'].min()).days,
            'messages_per_day': len(self.df) / (self.df['timestamp'].max() - self.df['timestamp'].min()).days
        }
        return stats
    
    def get_sender_stats(self) -> pd.DataFrame:
        """Get message statistics per sender."""
        return self.df.groupby('sender').agg({
            'message': ['count', lambda x: x.str.len().mean()],
            'timestamp': [lambda x: (x.max() - x.min()).days]
        }).round(2)
    
    def calculate_response_times(self) -> Dict[Tuple[str, str], float]:
        """Calculate average response times between pairs of senders."""
        response_times = defaultdict(list)
        
        for i in range(1, len(self.df)):
            prev_msg = self.df.iloc[i-1]
            curr_msg = self.df.iloc[i]
            
            if prev_msg['sender'] != curr_msg['sender']:
                time_diff = (curr_msg['timestamp'] - prev_msg['timestamp']).total_seconds() / 60  # in minutes
                if time_diff < 60:  # Only consider responses within 60 minutes
                    response_times[(prev_msg['sender'], curr_msg['sender'])].append(time_diff)
        
        return {k: np.mean(v) for k, v in response_times.items() if v}
    
    def analyze_message_sentiment(self, message: str) -> Dict:
        """Analyze sentiment of a single message using VADER."""
        scores = self.sentiment_analyzer.polarity_scores(message)
        return {
            'compound': scores['compound'],  # Overall sentiment (-1 to 1)
            'pos': scores['pos'],           # Positive component (0 to 1)
            'neu': scores['neu'],           # Neutral component (0 to 1)
            'neg': scores['neg']            # Negative component (0 to 1)
        }

    def sentiment_analysis(self) -> Dict:
        """Perform sentiment analysis on messages."""
        # Add sentiment scores to DataFrame
        self.df['sentiment_scores'] = self.df['message'].apply(self.analyze_message_sentiment)
        self.df['sentiment'] = self.df['sentiment_scores'].apply(lambda x: x['compound'])
        
        # Calculate overall sentiment statistics
        sentiment_stats = {
            'overall_sentiment': self.df['sentiment'].mean(),
            'sentiment_by_sender': self.df.groupby('sender')['sentiment'].mean().to_dict(),
            'most_positive_msg': self.df.loc[self.df['sentiment'].idxmax()].to_dict(),
            'most_negative_msg': self.df.loc[self.df['sentiment'].idxmin()].to_dict(),
            'sentiment_distribution': {
                'positive': len(self.df[self.df['sentiment'] > 0.05]) / len(self.df),
                'negative': len(self.df[self.df['sentiment'] < -0.05]) / len(self.df),
                'neutral': len(self.df[(self.df['sentiment'] >= -0.05) & (self.df['sentiment'] <= 0.05)]) / len(self.df)
            }
        }
        
        return sentiment_stats

    def analyze_conversation_sentiment(self, conversation_df: pd.DataFrame) -> Dict:
        """Analyze sentiment patterns within a conversation."""
        # Add sentiment scores if not already present
        if 'sentiment_scores' not in conversation_df.columns:
            conversation_df['sentiment_scores'] = conversation_df['message'].apply(self.analyze_message_sentiment)
            conversation_df['sentiment'] = conversation_df['sentiment_scores'].apply(lambda x: x['compound'])
        
        # Calculate various sentiment metrics
        sentiment_analysis = {
            'overall_sentiment': conversation_df['sentiment'].mean(),
            'sentiment_by_sender': conversation_df.groupby('sender')['sentiment'].agg(['mean', 'std']).to_dict('index'),
            'sentiment_distribution': {
                'positive': len(conversation_df[conversation_df['sentiment'] > 0.05]) / len(conversation_df),
                'negative': len(conversation_df[conversation_df['sentiment'] < -0.05]) / len(conversation_df),
                'neutral': len(conversation_df[(conversation_df['sentiment'] >= -0.05) & (conversation_df['sentiment'] <= 0.05)]) / len(conversation_df)
            },
            'sentiment_progression': conversation_df[['timestamp', 'sentiment']].values.tolist(),
            'most_positive_msg': conversation_df.loc[conversation_df['sentiment'].idxmax()].to_dict(),
            'most_negative_msg': conversation_df.loc[conversation_df['sentiment'].idxmin()].to_dict(),
            'sentiment_stats': {
                'mean': conversation_df['sentiment'].mean(),
                'median': conversation_df['sentiment'].median(),
                'std': conversation_df['sentiment'].std(),
                'max': conversation_df['sentiment'].max(),
                'min': conversation_df['sentiment'].min()
            }
        }
        
        # Calculate sentiment shifts (how sentiment changes between messages)
        sentiment_shifts = conversation_df['sentiment'].diff()
        sentiment_analysis['sentiment_volatility'] = sentiment_shifts.std()
        
        # Identify significant sentiment changes
        significant_shifts = conversation_df[abs(sentiment_shifts) > 0.5]
        if not significant_shifts.empty:
            sentiment_analysis['significant_shifts'] = significant_shifts[['timestamp', 'sender', 'message', 'sentiment']].to_dict('records')
        
        return sentiment_analysis
    
    def plot_activity_patterns(self, save_path: str = None):
        """Plot daily and hourly activity patterns."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Daily activity
        self.df['date'] = self.df['timestamp'].dt.date
        daily_counts = self.df.groupby('date').size()
        daily_counts.plot(ax=ax1, title='Daily Message Activity')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Number of Messages')
        
        # Hourly activity
        hourly_counts = self.df['timestamp'].dt.hour.value_counts().sort_index()
        hourly_counts.plot(kind='bar', ax=ax2, title='Hourly Message Distribution')
        ax2.set_xlabel('Hour of Day')
        ax2.set_ylabel('Number of Messages')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.close()
    
    def segment_conversations(self, time_threshold_minutes: int = 60) -> List[pd.DataFrame]:
        """Segment messages into separate conversations based on time gaps."""
        if len(self.df) == 0:
            return []
        
        # Use original message order
        df_sorted = self.df.copy()  # Changed from sort_values('timestamp')
        
        # Calculate time differences between consecutive messages
        time_diffs = df_sorted['timestamp'].diff()
        
        # Mark start of new conversations where time gap exceeds threshold
        conversation_breaks = time_diffs > pd.Timedelta(minutes=time_threshold_minutes)
        
        # Assign conversation IDs
        conversation_ids = conversation_breaks.cumsum()
        
        # Group messages by conversation ID while maintaining original order
        conversations = []
        for conv_id, conv_messages in df_sorted.groupby(conversation_ids):
            conv_data = conv_messages.copy()
            conv_data['conversation_id'] = conv_id
            conv_data['time_gap_minutes'] = time_diffs[conv_messages.index].dt.total_seconds() / 60
            conversations.append(conv_data)
        
        return conversations
    
    def get_conversation_stats(self, time_threshold_minutes: int = 60) -> Dict:
        """
        Get statistics about the segmented conversations.
        
        Args:
            time_threshold_minutes: Time gap (in minutes) to consider as a conversation break
            
        Returns:
            Dictionary containing conversation statistics
        """
        conversations = self.segment_conversations(time_threshold_minutes)
        
        stats = {
            'total_conversations': len(conversations),
            'conversation_details': []
        }
        
        for i, conv in enumerate(conversations):
            conv_stats = {
                'conversation_id': i,
                'start_time': conv['timestamp'].min(),
                'end_time': conv['timestamp'].max(),
                'duration_minutes': (conv['timestamp'].max() - conv['timestamp'].min()).total_seconds() / 60,
                'num_messages': len(conv),
                'participants': conv['sender'].unique().tolist(),
                'num_participants': len(conv['sender'].unique()),
                'avg_message_length': conv['message'].str.len().mean(),
                'most_active_sender': conv['sender'].value_counts().index[0]
            }
            stats['conversation_details'].append(conv_stats)
        
        # Add summary statistics
        stats['avg_conversation_duration'] = np.mean([c['duration_minutes'] for c in stats['conversation_details']])
        stats['avg_messages_per_conversation'] = np.mean([c['num_messages'] for c in stats['conversation_details']])
        stats['avg_participants_per_conversation'] = np.mean([c['num_participants'] for c in stats['conversation_details']])
        
        return stats
    
    def plot_conversation_patterns(self, time_threshold_minutes: int = 60, save_path: str = None):
        """
        Plot patterns of conversations including their distribution and gaps.
        
        Args:
            time_threshold_minutes: Time gap (in minutes) to consider as a conversation break
            save_path: Path to save the plot
        """
        conversations = self.segment_conversations(time_threshold_minutes)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot conversation lengths
        conv_lengths = [len(conv) for conv in conversations]
        ax1.hist(conv_lengths, bins=20)
        ax1.set_title('Distribution of Conversation Lengths')
        ax1.set_xlabel('Number of Messages')
        ax1.set_ylabel('Frequency')
        
        # Plot conversation gaps
        all_gaps = []
        for conv in conversations:
            gaps = conv['time_gap_minutes'].dropna().tolist()
            all_gaps.extend(gaps)
        
        ax2.hist(all_gaps, bins=50, range=(0, time_threshold_minutes))
        ax2.set_title('Distribution of Time Gaps Between Messages')
        ax2.set_xlabel('Gap Duration (minutes)')
        ax2.set_ylabel('Frequency')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.close()

    def add_chat_file(self, file_path: str) -> None:
        """
        Add messages from another chat file to the existing parser.
        
        Args:
            file_path: Path to the additional chat file
        """
        # Store original messages
        original_messages = self.messages.copy()
        original_df = self.df.copy() if self.df is not None else None
        
        try:
            # Temporarily clear messages for parsing new file
            self.messages = []
            self.file_path = file_path
            self._parse_chat()
            
            # Combine messages from both files
            all_messages = original_messages + self.messages
            
            # Sort messages by timestamp
            all_messages.sort(key=lambda x: x['timestamp'])
            
            # Update instance variables
            self.messages = all_messages
            self.df = pd.DataFrame(all_messages)
            
            print(f"Successfully added {len(self.messages) - len(original_messages)} messages from {file_path}")
            
        except Exception as e:
            # Restore original state if there's an error
            self.messages = original_messages
            self.df = original_df
            raise ValueError(f"Error adding chat file: {str(e)}")

    def format_basic_stats(self) -> str:
        """Format basic statistics in a readable way."""
        stats = self.get_basic_stats()
        return f"""
📊 Basic Chat Statistics
------------------------
Total Messages: {stats['total_messages']:,}
Unique Participants: {stats['unique_senders']}
Average Message Length: {stats['avg_message_length']:.2f} characters
Total Words: {stats['total_words']:,}
Time Span: {stats['time_span']} days
Messages per Day: {stats['messages_per_day']:.1f}
"""

    def format_sender_stats(self) -> str:
        """Format sender statistics in a readable way."""
        df = self.get_sender_stats()
        
        # Rename columns for better readability
        df.columns = ['Total Messages', 'Avg Length', 'Active Days']
        
        # Calculate percentage of total messages
        total_messages = df['Total Messages'].sum()
        df['Message %'] = (df['Total Messages'] / total_messages * 100).round(1)
        
        # Format the output
        output = "\n👥 Participant Statistics\n-------------------------\n"
        for sender in df.index:
            stats = df.loc[sender]
            output += f"\n{sender}:\n"
            output += f"  Messages: {stats['Total Messages']:,} ({stats['Message %']}%)\n"
            output += f"  Avg Length: {stats['Avg Length']:.1f} characters\n"
            output += f"  Active Days: {stats['Active Days']}\n"
        
        return output

    def format_conversation_stats(self, time_threshold_minutes: int = 30) -> str:
        """Format conversation statistics in a readable way."""
        stats = self.get_conversation_stats(time_threshold_minutes)
        
        output = f"""
💬 Conversation Analysis (>{time_threshold_minutes}min gap = new conversation)
---------------------------------------------------------------------
Total Conversations: {stats['total_conversations']:,}
Average Duration: {stats['avg_conversation_duration']:.1f} minutes
Average Messages per Conversation: {stats['avg_messages_per_conversation']:.1f}

Top 5 Longest Conversations:
"""
        # Sort conversations by duration and get top 5
        sorted_convs = sorted(stats['conversation_details'], 
                            key=lambda x: x['duration_minutes'], 
                            reverse=True)[:5]
        
        for i, conv in enumerate(sorted_convs, 1):
            output += f"\n{i}. Duration: {conv['duration_minutes']:.1f} minutes"
            output += f"\n   Messages: {conv['num_messages']}"
            output += f"\n   Participants: {len(conv['participants'])}"
            output += f"\n   Most Active: {conv['most_active_sender']}"
            output += f"\n   Time: {conv['start_time'].strftime('%Y-%m-%d %H:%M')} to {conv['end_time'].strftime('%H:%M')}\n"
        
        return output

    def format_response_times(self) -> str:
        """Format response time statistics in a readable way."""
        response_times = self.calculate_response_times()
        
        output = "\n⏱️ Average Response Times\n------------------------\n"
        
        # Group by responder
        responder_times = defaultdict(list)
        for (sender, responder), time in response_times.items():
            responder_times[responder].append((sender, time))
        
        # Format for each person
        for responder in sorted(responder_times.keys()):
            output += f"\n{responder} responds to:\n"
            for sender, time in sorted(responder_times[responder], key=lambda x: x[1]):
                output += f"  {sender}: {time:.1f} minutes\n"
        
        return output

    def format_sentiment_stats(self) -> str:
        """Format sentiment analysis statistics in a readable way."""
        stats = self.sentiment_analysis()
        
        output = """
    😊 Sentiment Analysis
    -------------------
    """
        output += f"Overall Chat Sentiment: {stats['overall_sentiment']:.3f}\n"
        output += "\nSentiment by Participant:\n"
        
        # Sort by sentiment score
        sorted_sentiments = sorted(stats['sentiment_by_sender'].items(), 
                                 key=lambda x: x[1], 
                                 reverse=True)
        
        for sender, sentiment in sorted_sentiments:
            output += f"  {sender}: {sentiment:.3f}\n"
        
        # Most positive and negative messages
        pos_msg = stats['most_positive_msg']
        neg_msg = stats['most_negative_msg']
        
        output += "\nMost Positive Message:\n"
        output += f"  From: {pos_msg['sender']}\n"
        output += f"  Time: {pos_msg['timestamp'].strftime('%Y-%m-%d %H:%M')}\n"
        output += f"  Message: {pos_msg['message']}\n"
        output += f"  Sentiment: {pos_msg['sentiment']:.3f}\n"
        
        output += "\nMost Negative Message:\n"
        output += f"  From: {neg_msg['sender']}\n"
        output += f"  Time: {neg_msg['timestamp'].strftime('%Y-%m-%d %H:%M')}\n"
        output += f"  Message: {neg_msg['message']}\n"
        output += f"  Sentiment: {neg_msg['sentiment']:.3f}\n"
        
        return output

    def analyze_all_conversations_sentiment(self, time_threshold_minutes: int = 60) -> Dict:
        """Analyze sentiment patterns across all conversations."""
        conversations = self.segment_conversations(time_threshold_minutes)
        all_conv_sentiments = []
        
        for conv in conversations:
            # Get sentiment for this conversation
            conv_sentiment = self.analyze_conversation_sentiment(conv)
            conv_data = {
                'conversation_id': conv['conversation_id'].iloc[0],
                'start_time': conv['timestamp'].min(),
                'duration_minutes': (conv['timestamp'].max() - conv['timestamp'].min()).total_seconds() / 60,
                'num_messages': len(conv),
                'mean_sentiment': conv_sentiment['sentiment_stats']['mean'],
                'sentiment_volatility': conv_sentiment['sentiment_volatility'],
                'positive_ratio': conv_sentiment['sentiment_distribution']['positive'],
                'negative_ratio': conv_sentiment['sentiment_distribution']['negative'],
                'neutral_ratio': conv_sentiment['sentiment_distribution']['neutral'],
                'most_positive_msg': conv_sentiment['most_positive_msg'],
                'most_negative_msg': conv_sentiment['most_negative_msg']
            }
            all_conv_sentiments.append(conv_data)
        
        # Convert to DataFrame for easier analysis
        conv_df = pd.DataFrame(all_conv_sentiments)
        
        # Calculate aggregate statistics
        sentiment_stats = {
            'total_conversations': len(conv_df),
            'sentiment_distribution': {
                'very_positive': len(conv_df[conv_df['mean_sentiment'] > 0.3]) / len(conv_df),
                'positive': len(conv_df[(conv_df['mean_sentiment'] > 0.05) & (conv_df['mean_sentiment'] <= 0.3)]) / len(conv_df),
                'neutral': len(conv_df[(conv_df['mean_sentiment'] >= -0.05) & (conv_df['mean_sentiment'] <= 0.05)]) / len(conv_df),
                'negative': len(conv_df[(conv_df['mean_sentiment'] < -0.05) & (conv_df['mean_sentiment'] >= -0.3)]) / len(conv_df),
                'very_negative': len(conv_df[conv_df['mean_sentiment'] < -0.3]) / len(conv_df)
            },
            'overall_stats': {
                'mean_sentiment': conv_df['mean_sentiment'].mean(),
                'median_sentiment': conv_df['mean_sentiment'].median(),
                'std_sentiment': conv_df['mean_sentiment'].std(),
                'mean_volatility': conv_df['sentiment_volatility'].mean()
            },
            'most_positive_conversation': conv_df.loc[conv_df['mean_sentiment'].idxmax()].to_dict() if not conv_df.empty else None,
            'most_negative_conversation': conv_df.loc[conv_df['mean_sentiment'].idxmin()].to_dict() if not conv_df.empty else None,
            'most_volatile_conversation': conv_df.loc[conv_df['sentiment_volatility'].idxmax()].to_dict() if not conv_df.empty else None,
            'sentiment_progression': conv_df[['start_time', 'mean_sentiment']].values.tolist(),
            'all_conversations': conv_df.to_dict('records')
        }
        
        return sentiment_stats

# Example usage
if __name__ == "__main__":
    # Initialize parser with first chat file
    parser = WhatsAppChatParser("WhatsApp Chat with circlejerk 3/WhatsApp Chat with circlejerk 3.txt")
    
    # Add another chat file
    parser.add_chat_file("WhatsApp Chat with Lucknow/WhatsApp Chat with Lucknow.txt")
    
    # Print formatted statistics
    print(parser.format_basic_stats())
    print(parser.format_sender_stats())
    print(parser.format_conversation_stats(time_threshold_minutes=30))
    print(parser.format_response_times())
    print(parser.format_sentiment_stats())
    
    # Generate plots
    parser.plot_conversation_patterns(time_threshold_minutes=30, save_path="conversation_patterns.png")
    parser.plot_activity_patterns("activity_patterns.png")
    print("\n📊 Plots have been saved as 'conversation_patterns.png' and 'activity_patterns.png'")
