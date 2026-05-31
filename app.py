
import streamlit as st

st.set_page_config(page_title="AI News Intelligence Platform", page_icon="🧠", layout="wide")

with open("assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown('''
<div class="hero">
<h1>🧠 AI News Intelligence Platform</h1>
<p>Enterprise Deep Learning • Real-Time Classification • Analytics Dashboard</p>
</div>
''', unsafe_allow_html=True)

c1,c2,c3,c4=st.columns(4)
c1.metric("Accuracy","96.2%")
c2.metric("Categories","4")
c3.metric("Model","LSTM")
c4.metric("Status","🟢 Online")

st.markdown("### Welcome")
st.write("Navigate using the sidebar to access Dashboard, Sample Data, History and About sections.")
