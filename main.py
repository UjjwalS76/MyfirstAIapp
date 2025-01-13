from langchain_google_genai import ChatGoogleGenerativeAI
from langchain import LLMChain
from langchain import PromptTemplate
import streamlit as st
import json
import os

# Set up API key
os.environ['GOOGLE_API_KEY'] = st.secrets['GOOGLE_API_KEY']

# Enhanced prompt template for better structured outputs
tweet_template = """Generate {number} professional tweets about {topic}. 
Each tweet should:
- Be engaging and informative
- Include relevant hashtags
- Stay within Twitter's character limit
- Be uniquely different from other tweets
- Follow proper formatting

Format each tweet as a complete, ready-to-post message.
If generating multiple tweets, separate them with unique numbers.

Example format:
1. [Tweet content with #hashtags]
2. [Another tweet content with #different #hashtags]

Remember to keep each tweet concise and impactful."""

tweet_prompt = PromptTemplate(template=tweet_template, input_variables=['number', 'topic'])

# Initialize Google's Gemini model
gemini_model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.7)

# Create LLM chain
tweet_chain = tweet_prompt | gemini_model

def create_copy_button(text, key):
    if st.button(f"📋 Copy", key=f"copy_{key}"):
        st.toast("Tweet copied to clipboard! 📋")
        st.write(f'<textarea style="position: absolute; left: -9999px;">{text}</textarea>', unsafe_allow_html=True)
        st.write('<script>document.querySelector("textarea").select();document.execCommand("copy");</script>', unsafe_allow_html=True)

# Streamlit UI
st.header("📱 Tweet Generator")
st.subheader("Generate engaging tweets using AI")

# Add some spacing and styling
st.markdown("""
<style>
.tweet-box {
    background-color: #f8f9fa;
    border-radius: 10px;
    padding: 15px;
    margin: 10px 0;
}
</style>
""", unsafe_allow_html=True)

# Input fields
col1, col2 = st.columns(2)
with col1:
    topic = st.text_input("📝 Topic", placeholder="Enter your topic...")
with col2:
    number = st.number_input("🔢 Number of tweets", min_value=1, max_value=10, value=1, step=1)

# Generate button with styling
if st.button("🚀 Generate Tweets", type="primary"):
    if topic:
        with st.spinner("Generating tweets..."):
            try:
                tweets = tweet_chain.invoke({"number": number, "topic": topic})
                tweet_content = tweets.content.strip()
                
                # Split tweets if multiple
                individual_tweets = tweet_content.split('\n\n') if '\n\n' in tweet_content else [tweet_content]
                
                # Display each tweet in a custom container with copy button
                for i, tweet in enumerate(individual_tweets, 1):
                    tweet = tweet.strip()
                    if tweet:
                        st.markdown(f"""
                        <div class="tweet-box">
                            {tweet}
                        </div>
                        """, unsafe_allow_html=True)
                        create_copy_button(tweet, i)
                        st.markdown("---")
            
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please enter a topic first! 🎯")

# Add footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit and Gemini")
