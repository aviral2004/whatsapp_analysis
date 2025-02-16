import streamlit as st
import pandas as pd
from parser import WhatsAppChatParser
import os
import tempfile
import colorsys
import hashlib
import plotly.express as px  # Add this import

st.set_page_config(page_title="Conversation Viewer", page_icon="💭", layout="wide")

def generate_user_colors(users):
    """Generate distinct colors for each user."""
    colors = {}
    for i, user in enumerate(users):
        # Generate a hash of the username for consistency
        hash_value = int(hashlib.md5(user.encode()).hexdigest(), 16)
        # Use the hash to generate HSV color (using golden ratio for better distribution)
        hue = (hash_value % 1000) / 1000.0
        saturation = 0.3 + (hash_value % 400) / 1000.0  # 0.3-0.7 range for saturation
        value = 0.9  # Keep brightness constant for readability
        # Convert to RGB
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        # Convert to hex
        colors[user] = '#{:02x}{:02x}{:02x}'.format(
            int(rgb[0] * 255),
            int(rgb[1] * 255),
            int(rgb[2] * 255)
        )
    return colors

def display_chat_message(msg, align="left", show_date=False, user_colors=None):
    """Display a single chat message in a chat-like interface."""
    styles = {
        "left": {
            "bg_color": user_colors.get(msg['sender'], "#f0f0f0") if user_colors else "#f0f0f0",
            "text_color": "#000000",
            "meta_color": "#666666"
        },
        "right": {
            "bg_color": user_colors.get(msg['sender'], "#DCF8C6") if user_colors else "#DCF8C6",
            "text_color": "#000000",
            "meta_color": "#666666"
        }
    }
    
    with st.container():
        if show_date:
            st.markdown(
                f"""
                <div style="text-align: center; margin: 20px 0; color: #666666;">
                    <hr style="border: 0; border-top: 1px solid #e0e0e0;"/>
                    <span style="background-color: white; padding: 0 10px;">
                        {msg['timestamp'].strftime('%A, %B %d, %Y')}
                    </span>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        cols = st.columns([1, 4, 1])
        col_idx = 1
        
        with cols[col_idx]:
            st.markdown(
                f"""
                <div style="
                    background-color: {styles[align]['bg_color']};
                    padding: 10px;
                    border-radius: 10px;
                    margin: 5px;
                    max-width: 100%;
                    display: inline-block;
                    box-shadow: 0 1px 2px rgba(0,0,0,0.1);
                ">
                    <p style="color: {styles[align]['meta_color']}; font-size: 0.8em; margin: 0;">
                        {msg['sender']}
                    </p>
                    <p style="margin: 5px 0; color: {styles[align]['text_color']};">
                        {msg['message']}
                    </p>
                    <p style="color: {styles[align]['meta_color']}; font-size: 0.7em; text-align: right; margin: 0;">
                        {msg['timestamp'].strftime('%I:%M %p')}
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

def view_conversation(conversation_df):
    """Display a specific conversation in a chat-like interface."""
    # Sort messages by index instead of timestamp
    conv_messages = conversation_df.sort_index()  # Changed from sort_values('timestamp')
    
    # Get unique participants and generate colors
    participants = conv_messages['sender'].unique()
    user_colors = generate_user_colors(participants)
    
    # Create tabs for different views
    stats_tab, sentiment_tab, chat_tab = st.tabs(["Statistics", "Sentiment Analysis", "Chat View"])
    
    with stats_tab:
        # Basic stats in columns
        col1, col2, col3 = st.columns(3)
        duration = (conv_messages['timestamp'].max() - conv_messages['timestamp'].min()).total_seconds() / 60
        
        col1.metric("Total Messages", len(conv_messages))
        col2.metric("Duration (min)", f"{duration:.1f}")
        col3.metric("Messages/Min", f"{len(conv_messages)/duration:.1f}" if duration > 0 else "N/A")
        
        # Message distribution pie chart
        st.subheader("Message Distribution")
        msg_dist = conv_messages['sender'].value_counts()
        fig = px.pie(
            values=msg_dist.values,
            names=msg_dist.index,
            title="Messages by Participant"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Message timeline
        st.subheader("Conversation Timeline")
        timeline = conv_messages.set_index('timestamp')['sender'].reset_index()
        fig = px.scatter(
            timeline,
            x='timestamp',
            y='sender',
            color='sender',
            title="Message Timeline",
            height=200,
            color_discrete_map={sender: color for sender, color in user_colors.items()}
        )
        fig.update_traces(marker=dict(size=8))
        st.plotly_chart(fig, use_container_width=True)
        
        # Time gaps between messages
        gaps = conv_messages['timestamp'].diff().dt.total_seconds() / 60
        gaps = gaps[gaps <= 30]  # Filter gaps less than 30 minutes
        
        st.subheader("Response Patterns")
        fig = px.histogram(
            gaps,
            title="Distribution of Time Gaps Between Messages",
            labels={'value': 'Gap Duration (minutes)', 'count': 'Frequency'},
            nbins=30
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with sentiment_tab:
        # Calculate sentiment for this conversation
        parser = st.session_state.parser
        sentiment_results = parser.analyze_conversation_sentiment(conversation_df)
        
        # Show sentiment metrics
        col1, col2 = st.columns(2)
        
        # Use correct keys from sentiment analysis results
        sentiment_stats = sentiment_results['sentiment_stats']
        col1.metric("Overall Sentiment", f"{sentiment_stats['mean']:.3f}")
        col1.metric("Sentiment Volatility", f"{sentiment_results['sentiment_volatility']:.3f}")
        
        # Use correct distribution keys
        dist = sentiment_results['sentiment_distribution']
        col2.metric("Positive Messages", f"{dist['positive']:.1%}")
        col2.metric("Negative Messages", f"{dist['negative']:.1%}")
        
        # Show sentiment timeline
        st.subheader("Sentiment Timeline")
        sentiment_progression = pd.DataFrame(
            sentiment_results['sentiment_progression'],
            columns=['timestamp', 'sentiment']
        )
        fig = px.scatter(
            sentiment_progression,
            x='timestamp',
            y='sentiment',
            color='sentiment',
            title="Message Sentiments Over Time",
            color_continuous_scale=['red', 'gray', 'green']
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig, use_container_width=True)
        
        # Show most emotional messages with correct keys
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Most Positive Message")
            msg = sentiment_results.get('most_positive_msg')
            if msg:
                st.info(
                    f"**From:** {msg['sender']}\n\n"
                    f"**Message:** {msg['message']}\n\n"
                    f"**Sentiment:** {msg['sentiment']:.3f}"
                )
        
        with col2:
            st.subheader("Most Negative Message")
            msg = sentiment_results.get('most_negative_msg')
            if msg:
                st.error(
                    f"**From:** {msg['sender']}\n\n"
                    f"**Message:** {msg['message']}\n\n"
                    f"**Sentiment:** {msg['sentiment']:.3f}"
                )

    with chat_tab:
        # Display participants legend
        st.sidebar.markdown("### Participants")
        for user, color in user_colors.items():
            st.sidebar.markdown(
                f"""
                <div style="display: flex; align-items: center; margin: 5px 0;">
                    <div style="width: 20px; height: 20px; background-color: {color}; 
                              margin-right: 10px; border-radius: 3px;"></div>
                    <span>{user}</span>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        # Display messages
        current_date = None
        for _, msg in conv_messages.iterrows():
            msg_date = msg['timestamp'].date()
            show_date = current_date != msg_date
            if show_date:
                current_date = msg_date
            
            align = "left"
            display_chat_message(msg, align, show_date, user_colors)

def main():
    st.title("💭 Conversation Viewer")
    
    # Check for parser
    if 'parser' not in st.session_state:
        st.warning("Please upload chat files in the main page first!")
        st.stop()
    
    # Use the stored parser to get conversations
    parser = st.session_state.parser
    time_threshold = st.session_state.get('time_threshold', 30)
    
    # Get conversations
    if 'conversations' not in st.session_state:
        st.session_state.conversations = parser.segment_conversations(time_threshold)
    
    # Sidebar controls
    st.sidebar.header("Conversation Selection")
    
    # Sort options - add sentiment-based sorting
    sort_by = st.sidebar.selectbox(
        "Sort conversations by",
        ["Recent", "Longest", "Most Messages", "Most Positive", "Most Negative", "Most Volatile"]
    )
    
    # Prepare conversation list with sentiment info
    conv_list = []
    for i, conv_df in enumerate(st.session_state.conversations):
        start_time = conv_df['timestamp'].min()
        duration = (conv_df['timestamp'].max() - start_time).total_seconds() / 60
        
        # Calculate sentiment for each conversation
        sentiment = parser.analyze_conversation_sentiment(conv_df)
        sentiment_stats = sentiment['sentiment_stats']
        
        conv_list.append({
            'id': i,
            'start_time': start_time,
            'duration': duration,
            'messages': len(conv_df),
            'participants': ', '.join(conv_df['sender'].unique()),
            'sentiment': sentiment_stats['mean'],  # Use mean sentiment
            'volatility': sentiment['sentiment_volatility']  # Use volatility
        })
    
    # Sort conversations
    if sort_by == "Recent":
        conv_list.sort(key=lambda x: x['start_time'], reverse=True)
    elif sort_by == "Longest":
        conv_list.sort(key=lambda x: x['duration'], reverse=True)
    elif sort_by == "Most Messages":
        conv_list.sort(key=lambda x: x['messages'], reverse=True)
    elif sort_by == "Most Positive":
        conv_list.sort(key=lambda x: x['sentiment'], reverse=True)
    elif sort_by == "Most Negative":
        conv_list.sort(key=lambda x: x['sentiment'])
    else:  # Most Volatile
        conv_list.sort(key=lambda x: x['volatility'], reverse=True)
    
    # Update conversation selection display to include sentiment
    selected_conv = st.sidebar.selectbox(
        "Select conversation",
        range(len(conv_list)),
        format_func=lambda x: (
            f"{conv_list[x]['start_time'].strftime('%Y-%m-%d %H:%M')} "
            f"({conv_list[x]['duration']:.1f}min, {conv_list[x]['messages']} msgs, "
            f"sentiment: {conv_list[x]['sentiment']:.2f})"
        )
    )
    
    # Display selected conversation
    if selected_conv is not None:
        conv_data = st.session_state.conversations[conv_list[selected_conv]['id']]
        view_conversation(conv_data)

if __name__ == "__main__":
    main()
