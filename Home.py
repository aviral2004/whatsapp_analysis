import streamlit as st
import pandas as pd
from parser import WhatsAppChatParser
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
import tempfile

st.set_page_config(
    page_title="WhatsApp Chat Analyzer",
    page_icon="💬",
    layout="wide"
)

def plot_daily_activity(df):
    """Create an interactive daily activity plot using plotly."""
    daily_counts = df.groupby(df['timestamp'].dt.date).size().reset_index()
    daily_counts.columns = ['date', 'count']
    
    fig = px.line(daily_counts, x='date', y='count',
                  title='Daily Message Activity',
                  labels={'date': 'Date', 'count': 'Number of Messages'})
    return fig

def plot_hourly_activity(df):
    """Create an interactive hourly activity plot using plotly."""
    hourly_counts = df['timestamp'].dt.hour.value_counts().sort_index()
    
    fig = px.bar(x=hourly_counts.index, y=hourly_counts.values,
                 title='Hourly Message Distribution',
                 labels={'x': 'Hour of Day', 'y': 'Number of Messages'})
    return fig

def plot_conversation_patterns(conversations, time_threshold):
    """Create interactive conversation pattern plots using plotly."""
    # Conversation lengths
    conv_lengths = [len(conv) for conv in conversations]
    fig1 = px.histogram(x=conv_lengths, 
                       title='Distribution of Conversation Lengths',
                       labels={'x': 'Number of Messages', 'y': 'Frequency'})
    
    # Time gaps
    all_gaps = []
    for conv in conversations:
        gaps = conv['time_gap_minutes'].dropna().tolist()
        all_gaps.extend([gap for gap in gaps if gap <= time_threshold])
    
    fig2 = px.histogram(all_gaps, 
                       title='Distribution of Time Gaps Between Messages',
                       labels={'value': 'Gap Duration (minutes)', 'count': 'Frequency'},
                       nbins=50)
    
    return fig1, fig2

def display_chat_message(msg, align="left", show_date=False):
    """Display a single chat message in a chat-like interface."""
    # Define colors for different alignments
    styles = {
        "left": {
            "bg_color": "#f0f0f0",
            "text_color": "#000000",
            "meta_color": "#666666"
        },
        "right": {
            "bg_color": "#DCF8C6",  # WhatsApp green color
            "text_color": "#000000",
            "meta_color": "#666666"
        }
    }
    
    # Create a container for the message
    with st.container():
        # Show date separator if requested
        if show_date:
            st.markdown(
                f"""
                <div style="
                    text-align: center;
                    margin: 20px 0;
                    color: #666666;
                    font-size: 0.9em;
                    position: relative;
                ">
                    <hr style="
                        border: 0;
                        border-top: 1px solid #e0e0e0;
                        margin: 10px 0;
                    "/>
                    <span style="
                        background-color: white;
                        padding: 0 10px;
                        position: absolute;
                        top: 50%;
                        left: 50%;
                        transform: translate(-50%, -50%);
                    ">{msg['timestamp'].strftime('%A, %B %d, %Y')}</span>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        cols = st.columns([1, 4, 1])
        
        # Choose which column to place the message in
        col_idx = 1 if align == "left" else 1
        
        with cols[col_idx]:
            # Message container with styling
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
                    <p style="
                        color: {styles[align]['meta_color']};
                        font-size: 0.8em;
                        margin: 0;
                        font-weight: 500;
                    ">{msg['sender']}</p>
                    <p style="
                        margin: 5px 0;
                        color: {styles[align]['text_color']};
                        font-size: 1em;
                        line-height: 1.4;
                        word-wrap: break-word;
                    ">{msg['message']}</p>
                    <p style="
                        color: {styles[align]['meta_color']};
                        font-size: 0.7em;
                        text-align: right;
                        margin: 0;
                    ">{msg['timestamp'].strftime('%I:%M %p')}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

def view_conversation(conversation_df, num_messages=50):
    """Display a conversation in a chat-like interface."""
    # Sort messages by timestamp
    conv_messages = conversation_df.sort_values('timestamp')
    
    # Get unique participants
    participants = conv_messages['sender'].unique()
    
    # Display basic conversation info
    st.write(f"Total Messages: {len(conv_messages)}")
    st.write(f"Participants: {', '.join(participants)}")
    st.write(f"Duration: {(conv_messages['timestamp'].max() - conv_messages['timestamp'].min()).total_seconds() / 60:.1f} minutes")
    
    # Create a container for the chat
    chat_container = st.container()
    
    with chat_container:
        current_date = None
        for _, msg in conv_messages.iterrows():
            # Check if the date has changed
            msg_date = msg['timestamp'].date()
            show_date = current_date != msg_date
            if show_date:
                current_date = msg_date
            
            # Determine message alignment
            align = "right" if msg['sender'] == participants[0] else "left"
            display_chat_message(msg, align, show_date)

def main():
    st.title("📱 WhatsApp Chat Analyzer")
    st.write("Upload your WhatsApp chat export file to analyze conversations and patterns.")
    
    if 'parser' not in st.session_state:
        # Show file upload only if parser isn't in session state
        uploaded_files = st.file_uploader("Choose WhatsApp chat export file(s)", 
                                        accept_multiple_files=True,
                                        type=['txt'])
        
        if uploaded_files:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_paths = []
                for uploaded_file in uploaded_files:
                    temp_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(temp_path, 'wb') as f:
                        f.write(uploaded_file.getvalue())
                    temp_paths.append(temp_path)
                
                try:
                    # Initialize parser with first file
                    parser = WhatsAppChatParser(temp_paths[0])
                    
                    # Add additional files if any
                    for path in temp_paths[1:]:
                        parser.add_chat_file(path)
                    
                    # Store parser in session state
                    st.session_state.parser = parser
                except Exception as e:
                    st.error(f"Error processing chat file: {str(e)}")
                    st.stop()
    
    # If we have a parser, show the analysis
    if 'parser' in st.session_state:
        parser = st.session_state.parser
        
        # Add a button to clear the state and upload new files
        if st.sidebar.button("Upload New Files"):
            del st.session_state.parser
            st.rerun()  # Changed from st.experimental_rerun()
        
        # Sidebar controls
        st.sidebar.header("⚙️ Analysis Settings")
        time_threshold = st.sidebar.slider(
            "Conversation Break Threshold (minutes)",
            min_value=5,
            max_value=120,
            value=30,
            step=5
        )
        
        # Store data in session state for chat viewer
        st.session_state.chat_data = parser.df
        st.session_state.conversations = parser.segment_conversations(time_threshold)
        st.session_state.time_threshold = time_threshold
        
        # Basic Stats
        st.header("📊 Basic Statistics")
        col1, col2, col3 = st.columns(3)
        stats = parser.get_basic_stats()
        
        col1.metric("Total Messages", f"{stats['total_messages']:,}")
        col1.metric("Total Words", f"{stats['total_words']:,}")
        
        col2.metric("Unique Participants", stats['unique_senders'])
        col2.metric("Messages per Day", f"{stats['messages_per_day']:.1f}")
        
        col3.metric("Time Span (days)", stats['time_span'])
        col3.metric("Avg Message Length", f"{stats['avg_message_length']:.1f}")
        
        # Participant Statistics
        st.header("👥 Participant Statistics")
        df_stats = parser.get_sender_stats()
        df_stats.columns = ['Total Messages', 'Avg Length', 'Active Days']
        total_messages = df_stats['Total Messages'].sum()
        df_stats['Message %'] = (df_stats['Total Messages'] / total_messages * 100).round(1)
        
        # Create pie chart for message distribution
        fig_pie = px.pie(values=df_stats['Total Messages'], 
                       names=df_stats.index,
                       title='Message Distribution by Participant')
        st.plotly_chart(fig_pie)
        
        # Show participant stats table
        st.dataframe(df_stats)
        
        # Conversation Analysis
        st.header("💬 Conversation Analysis")
        conv_stats = parser.get_conversation_stats(time_threshold)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Conversations", f"{conv_stats['total_conversations']:,}")
        col2.metric("Avg Duration (min)", f"{conv_stats['avg_conversation_duration']:.1f}")
        col3.metric("Avg Messages/Conv", f"{conv_stats['avg_messages_per_conversation']:.1f}")
        
        # Get all conversations first
        conversations = parser.segment_conversations(time_threshold)
        
        # Top 5 longest conversations
        st.subheader("Longest Conversations")
        sorted_convs = sorted(conv_stats['conversation_details'], 
                            key=lambda x: x['duration_minutes'], 
                            reverse=True)[:5]
        
        for i, conv in enumerate(sorted_convs, 1):
            start_time = conv['start_time']
            # Format the date in a more readable way
            date_str = start_time.strftime('%B %d, %Y')
            time_str = start_time.strftime('%I:%M %p')
            with st.expander(f"{i}. {date_str} at {time_str} ({conv['duration_minutes']:.1f} min)"):
                # Basic stats
                st.write(f"Messages: {conv['num_messages']}")
                st.write(f"Participants: {', '.join(conv['participants'])}")
                st.write(f"Most Active: {conv['most_active_sender']}")
                
                # Add a button to view the conversation
                if st.button(f"View Conversation #{i}", key=f"view_conv_{i}"):
                    # Get the conversation messages
                    conv_messages = conversations[conv['conversation_id']]
                    
                    # Create tabs for different views
                    stats_tab, view_tab = st.tabs(["Statistics", "View Messages"])
                    
                    with stats_tab:
                        # Show message distribution
                        msg_dist = conv_messages['sender'].value_counts()
                        fig = px.pie(
                            values=msg_dist.values,
                            names=msg_dist.index,
                            title="Message Distribution"
                        )
                        st.plotly_chart(fig)
                        
                        # Show message timeline
                        timeline = conv_messages.set_index('timestamp')['sender'].reset_index()
                        fig = px.scatter(
                            timeline,
                            x='timestamp',
                            y='sender',
                            title="Message Timeline",
                            height=200
                        )
                        st.plotly_chart(fig)
                    
                    with view_tab:
                        view_conversation(conv_messages)
        
        # Conversation Patterns
        st.header("🔄 Conversation Patterns")
        fig1, fig2 = plot_conversation_patterns(conversations, time_threshold)
        
        tab1, tab2 = st.tabs(["Message Distribution", "Time Gaps"])
        with tab1:
            st.plotly_chart(fig1)
        with tab2:
            st.plotly_chart(fig2)
        
        # Activity Patterns
        st.header("📈 Activity Patterns")
        tab1, tab2 = st.tabs(["Daily Activity", "Hourly Activity"])
        
        with tab1:
            st.plotly_chart(plot_daily_activity(parser.df))
        
        with tab2:
            st.plotly_chart(plot_hourly_activity(parser.df))
        
        # Response Times
        st.header("⏱️ Response Times")
        response_times = parser.calculate_response_times()
        
        # Create a heatmap of response times
        senders = list(set([k[0] for k in response_times.keys()]))
        responders = list(set([k[1] for k in response_times.keys()]))
        
        response_matrix = pd.DataFrame(index=senders, columns=responders)
        for (sender, responder), time in response_times.items():
            response_matrix.loc[sender, responder] = time
        
        fig = go.Figure(data=go.Heatmap(
            z=response_matrix.values,
            x=response_matrix.columns,
            y=response_matrix.index,
            colorscale='Viridis',
            text=[[f'{val:.1f}' if pd.notnull(val) else '' 
                   for val in row] for row in response_matrix.values],
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title='Average Response Times (minutes)',
            xaxis_title='Responder',
            yaxis_title='Original Sender'
        )
        
        st.plotly_chart(fig)
        
        # Sentiment Analysis
        st.header("😊 Sentiment Analysis")
        
        # Calculate both sentiment stats first
        sentiment_stats = parser.sentiment_analysis()
        conv_sentiment_stats = parser.analyze_all_conversations_sentiment(time_threshold)
        
        # Store both in session state right after calculation
        st.session_state.sentiment_stats = sentiment_stats
        st.session_state.conv_sentiment_stats = conv_sentiment_stats
        
        # Overall Sentiment Analysis
        st.subheader("Overall Chat Sentiment")
        col1, col2, col3 = st.columns(3)
        
        # Compound score with color coding
        sentiment_score = sentiment_stats['overall_sentiment']
        sentiment_delta_color = "normal" if sentiment_score > 0 else "inverse" if sentiment_score < 0 else "off"
        col1.metric("Overall Sentiment", 
                  f"{sentiment_score:.3f}",
                  delta=None,
                  delta_color=sentiment_delta_color)
        
        # Sentiment distribution
        dist = sentiment_stats['sentiment_distribution']
        col2.metric("Positive Messages", f"{dist['positive']:.1%}")
        col2.metric("Neutral Messages", f"{dist['neutral']:.1%}")
        col2.metric("Negative Messages", f"{dist['negative']:.1%}")
        
        # Participant sentiment
        st.subheader("Sentiment by Participant")
        sentiment_df = pd.DataFrame.from_dict(
            sentiment_stats['sentiment_by_sender'], 
            orient='index', 
            columns=['Sentiment']
        ).sort_values('Sentiment', ascending=False)
        
        # Create a bar chart with color gradient
        fig = px.bar(sentiment_df, 
                   title='Participant Sentiment Scores',
                   labels={'index': 'Participant', 'Sentiment': 'Sentiment Score'},
                   y='Sentiment',
                   color='Sentiment',
                   color_continuous_scale=['red', 'gray', 'green'])
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig)
        
        # Most Emotional Messages
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Most Positive Message")
            pos_msg = sentiment_stats['most_positive_msg']
            st.info(
                f"**From:** {pos_msg['sender']}\n\n"
                f"**Time:** {pos_msg['timestamp'].strftime('%B %d, %Y at %I:%M %p')}\n\n"
                f"**Message:** {pos_msg['message']}\n\n"
                f"**Sentiment Score:** {pos_msg['sentiment']:.3f}"
            )
        
        with col2:
            st.subheader("Most Negative Message")
            neg_msg = sentiment_stats['most_negative_msg']
            st.error(
                f"**From:** {neg_msg['sender']}\n\n"
                f"**Time:** {neg_msg['timestamp'].strftime('%B %d, %Y at %I:%M %p')}\n\n"
                f"**Message:** {neg_msg['message']}\n\n"
                f"**Sentiment Score:** {neg_msg['sentiment']:.3f}"
            )
        
        # Conversation Sentiment Analysis
        st.header("💭 Conversation Sentiment Analysis")
        
        # Overall Conversation Sentiment Stats
        st.subheader("Overall Conversation Sentiment")
        col1, col2, col3 = st.columns(3)
        
        stats = conv_sentiment_stats['overall_stats']
        col1.metric("Mean Conversation Sentiment", f"{stats['mean_sentiment']:.3f}")
        col1.metric("Median Conversation Sentiment", f"{stats['median_sentiment']:.3f}")
        col2.metric("Sentiment Variation", f"{stats['std_sentiment']:.3f}")
        col2.metric("Average Volatility", f"{stats['mean_volatility']:.3f}")
        
        # Sentiment Distribution
        dist = conv_sentiment_stats['sentiment_distribution']
        col3.metric("Very Positive Conversations", f"{dist['very_positive']:.1%}")
        col3.metric("Positive Conversations", f"{dist['positive']:.1%}")
        col3.metric("Neutral Conversations", f"{dist['neutral']:.1%}")
        col3.metric("Negative Conversations", f"{dist['negative']:.1%}")
        col3.metric("Very Negative Conversations", f"{dist['very_negative']:.1%}")
        
        # Sentiment Progression Over Time
        st.subheader("Conversation Sentiment Progression")
        progression = pd.DataFrame(
            conv_sentiment_stats['sentiment_progression'],
            columns=['timestamp', 'sentiment']
        ).sort_values('timestamp')
        
        fig = px.scatter(progression,
                       x='timestamp',
                       y='sentiment',
                       title='Conversation Sentiments Over Time',
                       labels={'sentiment': 'Mean Sentiment'},
                       trendline="lowess")
        st.plotly_chart(fig)
        
        # Notable Conversations
        st.subheader("Notable Conversations")
        
        tab1, tab2, tab3 = st.tabs(["Most Positive", "Most Negative", "Most Volatile"])
        
        with tab1:
            conv = conv_sentiment_stats['most_positive_conversation']
            if conv:
                st.metric("Sentiment Score", f"{conv['mean_sentiment']:.3f}")
                st.write(f"**Date:** {conv['start_time'].strftime('%B %d, %Y at %I:%M %p')}")
                st.write(f"**Duration:** {conv['duration_minutes']:.1f} minutes")
                st.write(f"**Messages:** {conv['num_messages']}")
                if 'most_positive_msg' in conv:
                    st.info(
                        f"**Most Positive Message:**\n\n"
                        f"From: {conv['most_positive_msg']['sender']}\n\n"
                        f"Message: {conv['most_positive_msg']['message']}\n\n"
                        f"Sentiment: {conv['most_positive_msg']['sentiment']:.3f}"
                    )
        
        with tab2:
            conv = conv_sentiment_stats['most_negative_conversation']
            if conv:
                st.metric("Sentiment Score", f"{conv['mean_sentiment']:.3f}")
                st.write(f"**Date:** {conv['start_time'].strftime('%B %d, %Y at %I:%M %p')}")
                st.write(f"**Duration:** {conv['duration_minutes']:.1f} minutes")
                st.write(f"**Messages:** {conv['num_messages']}")
                if 'most_negative_msg' in conv:
                    st.error(
                        f"**Most Negative Message:**\n\n"
                        f"From: {conv['most_negative_msg']['sender']}\n\n"
                        f"Message: {conv['most_negative_msg']['message']}\n\n"
                        f"Sentiment: {conv['most_negative_msg']['sentiment']:.3f}"
                    )
        
        with tab3:
            conv = conv_sentiment_stats['most_volatile_conversation']
            if conv:
                st.metric("Volatility Score", f"{conv['sentiment_volatility']:.3f}")
                st.write(f"**Date:** {conv['start_time'].strftime('%B %d, %Y at %I:%M %p')}")
                st.write(f"**Duration:** {conv['duration_minutes']:.1f} minutes")
                st.write(f"**Messages:** {conv['num_messages']}")
                st.write(f"**Mean Sentiment:** {conv['mean_sentiment']:.3f}")
        
        # All Conversations Table
        st.subheader("All Conversations")
        conv_df = pd.DataFrame(conv_sentiment_stats['all_conversations'])
        if not conv_df.empty:
            # Format the DataFrame for display
            display_df = conv_df[[
                'start_time', 'duration_minutes', 'num_messages',
                'mean_sentiment', 'sentiment_volatility'
            ]].copy()
            display_df['start_time'] = display_df['start_time'].dt.strftime('%Y-%m-%d %H:%M')
            display_df.columns = ['Start Time', 'Duration (min)', 'Messages',
                                'Mean Sentiment', 'Volatility']
            st.dataframe(
                display_df.style.background_gradient(
                    subset=['Mean Sentiment'],
                    cmap='RdYlGn',
                    vmin=-1,
                    vmax=1
                )
            )
        
if __name__ == "__main__":
    main()