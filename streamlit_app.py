import streamlit as st
import pandas as pd
from parser import WhatsAppChatParser
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
import tempfile

import pandas as pd
import numpy as np
from datetime import datetime
import re
from typing import List, Dict, Tuple
from collections import defaultdict
import matplotlib.pyplot as plt
from textblob import TextBlob

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
        all_gaps.extend(gaps)
    
    fig2 = px.histogram(x=all_gaps, 
                       title='Distribution of Time Gaps Between Messages',
                       labels={'x': 'Gap Duration (minutes)', 'y': 'Frequency'},
                       range_x=[0, time_threshold])
    
    return fig1, fig2

def main():
    st.title("📱 WhatsApp Chat Analyzer")
    st.write("Upload your WhatsApp chat export file to analyze conversations and patterns.")
    
    uploaded_files = st.file_uploader("Choose WhatsApp chat export file(s)", 
                                    accept_multiple_files=True,
                                    type=['txt'])
    
    if uploaded_files:
        # Use tempfile to handle uploads securely
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
                
                # Sidebar controls
                st.sidebar.header("⚙️ Analysis Settings")
                time_threshold = st.sidebar.slider(
                    "Conversation Break Threshold (minutes)",
                    min_value=5,
                    max_value=120,
                    value=30,
                    step=5
                )
                
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
                
                # Top 5 longest conversations
                st.subheader("Longest Conversations")
                sorted_convs = sorted(conv_stats['conversation_details'], 
                                    key=lambda x: x['duration_minutes'], 
                                    reverse=True)[:5]
                
                for i, conv in enumerate(sorted_convs, 1):
                    with st.expander(f"{i}. {conv['start_time'].strftime('%Y-%m-%d %H:%M')} ({conv['duration_minutes']:.1f} min)"):
                        st.write(f"Messages: {conv['num_messages']}")
                        st.write(f"Participants: {', '.join(conv['participants'])}")
                        st.write(f"Most Active: {conv['most_active_sender']}")
                
                # Activity Patterns
                st.header("📈 Activity Patterns")
                tab1, tab2 = st.tabs(["Daily Activity", "Hourly Activity"])
                
                with tab1:
                    st.plotly_chart(plot_daily_activity(parser.df))
                
                with tab2:
                    st.plotly_chart(plot_hourly_activity(parser.df))
                
                # Conversation Patterns
                st.header("🔄 Conversation Patterns")
                conversations = parser.segment_conversations(time_threshold)
                fig1, fig2 = plot_conversation_patterns(conversations, time_threshold)
                
                tab1, tab2 = st.tabs(["Message Distribution", "Time Gaps"])
                with tab1:
                    st.plotly_chart(fig1)
                with tab2:
                    st.plotly_chart(fig2)
                
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
                sentiment_stats = parser.sentiment_analysis()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Overall Chat Sentiment", f"{sentiment_stats['overall_sentiment']:.3f}")
                    
                    st.subheader("Sentiment by Participant")
                    sentiment_df = pd.DataFrame.from_dict(
                        sentiment_stats['sentiment_by_sender'], 
                        orient='index', 
                        columns=['Sentiment']
                    ).sort_values('Sentiment', ascending=False)
                    
                    fig = px.bar(sentiment_df, 
                               title='Participant Sentiment Scores',
                               labels={'index': 'Participant', 'value': 'Sentiment Score'})
                    st.plotly_chart(fig)
                
                with col2:
                    st.subheader("Most Positive Message")
                    pos_msg = sentiment_stats['most_positive_msg']
                    st.info(
                        f"From: {pos_msg['sender']}\n\n"
                        f"Time: {pos_msg['timestamp'].strftime('%Y-%m-%d %H:%M')}\n\n"
                        f"Message: {pos_msg['message']}\n\n"
                        f"Sentiment: {pos_msg['sentiment']:.3f}"
                    )
                    
                    st.subheader("Most Negative Message")
                    neg_msg = sentiment_stats['most_negative_msg']
                    st.error(
                        f"From: {neg_msg['sender']}\n\n"
                        f"Time: {neg_msg['timestamp'].strftime('%Y-%m-%d %H:%M')}\n\n"
                        f"Message: {neg_msg['message']}\n\n"
                        f"Sentiment: {neg_msg['sentiment']:.3f}"
                    )
                
            except Exception as e:
                st.error(f"Error processing chat file: {str(e)}")
                st.stop()

if __name__ == "__main__":
    main() 