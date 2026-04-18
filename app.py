"""Minimal Streamlit UI for DataDetective.

Features:
- Single drag-and-drop CSV uploader
- Background EDA run after upload
- Chat interface powered by agent_core.run_agent
"""

from __future__ import annotations

import streamlit as st
import pandas as pd

from agent_core import run_agent
from tools_engine import run_automated_eda


st.set_page_config(page_title="DataDetective", page_icon="🕵️", layout="centered")

# Minimal, clean styling.
st.markdown(
    """
    <style>
        .block-container {
            max-width: 820px;
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .dd-title {
            font-size: 2rem;
            font-weight: 650;
            letter-spacing: -0.02em;
            margin-bottom: 0.2rem;
        }
        .dd-subtitle {
            color: #6b7280;
            margin-bottom: 1.2rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="dd-title">DataDetective</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="dd-subtitle">CSV dosyanı yükle, verinle sohbet et.</div>',
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "CSV dosyası yükle",
    type=["csv"],
    accept_multiple_files=False,
    help="Dosyayı sürükleyip bırakabilir veya tıklayarak seçebilirsin.",
)

if "df" not in st.session_state:
    st.session_state.df = None
if "eda_result" not in st.session_state:
    st.session_state.eda_result = None
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df
    except Exception as exc:  # noqa: BLE001
        st.error(f"CSV okunurken hata oluştu: {exc}")
        st.stop()

    # Run EDA in the background section on first load or file change.
    file_sig = (uploaded_file.name, uploaded_file.size)
    if st.session_state.get("uploaded_sig") != file_sig:
        st.session_state.uploaded_sig = file_sig
        with st.spinner("Veri analiz ediliyor..."):
            try:
                st.session_state.eda_result = run_automated_eda(df)
            except Exception as exc:  # noqa: BLE001
                st.session_state.eda_result = {"error": str(exc)}
        st.session_state.chat_messages = []

    st.success(
        f"Yüklendi: {uploaded_file.name}  •  {st.session_state.df.shape[0]} satır, "
        f"{st.session_state.df.shape[1]} sütun"
    )

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_prompt = st.chat_input(
        "Örn: Verimi özetle ve kredi_onay sütununa göre bir model kur"
    )
    if user_prompt:
        st.session_state.chat_messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        with st.chat_message("assistant"):
            with st.spinner("Ajan düşünüyor..."):
                try:
                    answer = run_agent(user_message=user_prompt, df=st.session_state.df)
                except Exception as exc:  # noqa: BLE001
                    answer = f"Ajan çalışırken hata oluştu: {exc}"
                st.markdown(answer)

        st.session_state.chat_messages.append({"role": "assistant", "content": answer})
else:
    st.info("Başlamak için sadece bir CSV dosyası yükle.")
