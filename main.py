import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os

# Set page config
st.set_page_config(page_title="Tweet Generator", page_icon="🐦")

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
tweet_chain = LLMChain(llm=gemini_model, prompt=tweet_prompt)

# Custom CSS
st.markdown("""
<style>
.tweet-box {
    background-color: #f8f9fa;
    border-radius: 10px;
    padding: 15px;
    margin: 10px 0;
    border: 1px solid #e1e8ed;
}
.stButton>button {
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

# Streamlit UI
st.header("🐦 Tweet Generator")
st.subheader("Generate engaging tweets using AI")

# Input fields
col1, col2 = st.columns(2)
with col1:
    topic = st.text_input("📝 Topic", placeholder="Enter your topic...")
with col2:
    number = st.number_input("🔢 Number of tweets", min_value=1, max_value=10, value=1, step=1)

# Generate button
if st.button("🚀 Generate Tweets", type="primary"):
    if topic:
        with st.spinner("Generating tweets..."):
            try:
                # Get response from LLM
                response = tweet_chain.run({"number": number, "topic": topic})
                
                # Split tweets if multiple
                tweets = [tweet.strip() for tweet in response.split('\n') if tweet.strip()]
                
                # Display each tweet
                for i, tweet in enumerate(tweets, 1):
                    if tweet:
                        # Remove numbering if present
                        tweet_text = tweet[2:].strip() if tweet.startswith(str(i)+'.') else tweet
                        
                        # Create unique key for each tweet's container
                        tweet_container = st.container()
                        with tweet_container:
                            st.markdown(f'<div class="tweet-box">{tweet_text}</div>', unsafe_allow_html=True)
                            if st.button('📋 Copy', key=f'copy_{i}'):
                                st.toast(f'Tweet {i} copied! 📋')
                                # Add hidden textarea for copying
                                st.markdown(f'<textarea style="position: absolute; left: -9999px;">{tweet_text}</textarea>', unsafe_allow_html=True)
                                st.markdown('<script>document.querySelector("textarea").select();document.execCommand("copy");</script>', unsafe_allow_html=True)
            
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please enter a topic first! 🎯")

# Add footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit and Gemini")
