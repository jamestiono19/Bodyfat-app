import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(layout="centered")
st.title("Vanta Background Test")

components.html("""
<style>
  body { margin: 0; overflow: hidden; background: transparent; }
  #vanta-bg { width: 100vw; height: 100vh; position: absolute; top: 0; left: 0; }
</style>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r134/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/vanta@latest/dist/vanta.net.min.js"></script>
<div id="vanta-bg"></div>
<script>
  VANTA.NET({
    el: "#vanta-bg",
    mouseControls: true,
    touchControls: true,
    gyroControls: false,
    minHeight: 200.00,
    minWidth: 200.00,
    scale: 1.00,
    scaleMobile: 1.00,
    color: 0x60a5fa,
    backgroundColor: 0x0f172a,
    points: 12.00,
    maxDistance: 22.00,
    spacing: 16.00
  })

  // Hack to make iframe fullscreen background
  const frame = window.frameElement;
  if (frame) {
      frame.style.position = 'fixed';
      frame.style.top = '0';
      frame.style.left = '0';
      frame.style.width = '100vw';
      frame.style.height = '100vh';
      frame.style.zIndex = '-1';
      frame.style.border = 'none';
  }
  const parentDoc = window.parent.document;
  const stApp = parentDoc.querySelector('.stApp');
  const stHeader = parentDoc.querySelector('[data-testid="stHeader"]');
  if(stApp) {
      stApp.style.background = 'transparent';
  }
  if(stHeader) {
      stHeader.style.background = 'transparent';
  }
</script>
""", height=0)

st.write("This text should be on top of the interactive background.")
