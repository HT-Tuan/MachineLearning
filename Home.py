import streamlit as st

st.set_page_config(
    page_title="Trang chủ",
    page_icon="🚀"
)

css = """
    <style>
        .css-6qob1r {
            background-color: #98EECC;
        }
    </style>
"""
st.markdown(css, unsafe_allow_html=True)

st.image("./images/banner.jpg")

st.markdown(
    """
    
    **Môn học: Học máy**
    
    **Giảng viên hướng dẫn: Trần Tiến Đức**
    
    **Thực hiện:**
    
    **1. Huỳnh Thanh Tuấn - 20110120**
    
    **2. Trần Văn Dân - 20110451**
    
"""
)