import streamlit as st
import pandas as pd
import os
from pathlib import Path
from functools import partial
from importlib.resources import files
# from geniusai.constants import SENDER_ID
SENDER_ID = 'sender_id'

# Spinner context
with st.spinner("Loading packages..."):

    # imports (these take very long to import the first time)
    from deepeval import evaluate
    from deepeval.evaluate import CacheConfig
    from deepeval.evaluate.configs import AsyncConfig, DisplayConfig
    from deepeval.test_case import LLMTestCase
    from src.auto_tester import TruthfulnessMetric, TruthfulnessTemplate, DeepEvalJudgeLLM
    from src.auto_tester.utils import SilentConsole, results_to_df
    from src.auto_tester.api_call import call_llama, suppress_insecure_request_warning

    display_config = DisplayConfig(show_indicator=False, print_results= False, verbose_mode=False)
    async_config=AsyncConfig(run_async=False)
    cache_config=CacheConfig(write_cache=False, use_cache=False)

    import os
    os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = '1'

st.success("Packages loaded!")

# Initialize session state
if "columns" not in st.session_state:
    st.session_state.columns = ["Select a column"]

################################################################################
# Automated Test (GEval) TAB HERE #
################################################################################
st.header("GeniusAI Claw")
st.subheader("👋 BYOB Auto Tester")
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

# file upload
if uploaded_file is not None:
    df_input = pd.read_csv(uploaded_file)
    st.session_state.columns = ["Select a column"] + df_input.columns.tolist()
    st.subheader("Uploaded File Preview (Editable)")
    df_input = st.data_editor(df_input, num_rows="dynamic", height=200)
    st.success("New Change saved...")
else:
    # Reset columns if file is deleted
    st.session_state.columns = ["Select a column"]

# --- Selectbox bound to session state ---
query_colname = st.selectbox("Reference query column name", options=st.session_state.columns, index=0)
reference_colname = st.selectbox("Reference answer column name", options=st.session_state.columns, index=0)
byob_colname = st.selectbox("BYOB response column name", options=st.session_state.columns, index=0)

default_name = "DeepEvalTest"
if uploaded_file is not None:
    default_name = uploaded_file.name.replace(".csv", "")
output_filename = st.text_input("Output file name", value=default_name)
sender_id = st.text_input("Sender ID", value=SENDER_ID)

with st.expander("🔍 Advanced Options"):
    enable_verbose = st.checkbox("Enable verbose logs", value=False)

st.markdown("") # small spacer

st.info("Ready when you are - run the automated test to generate the report.")
run_test = st.button(
    "⚡ Run Automated Test",
    use_container_width=True
)

if run_test:
    # Set up api (str -> str only)
    partial_llama = partial(call_llama, senderId=sender_id)
    quiet_partial_llama = suppress_insecure_request_warning(partial_llama)

    # Set up judge
    judge = DeepEvalJudgeLLM(fn=quiet_partial_llama)

    # assign synthetic key for joining back judge responses
    df_input = (
        df_input
        .reset_index(drop=True)
        .pipe(lambda df: df.assign(key=df.index))
    )

    # build test cases
    test_cases = [
        LLMTestCase(
            input=row.get(query_colname),
            actual_output=row.get(byob_colname),
            expected_output=row.get(reference_colname),
            additional_metadata={'key': row['key']}
        )
        for _, row in df_input.iterrows()
    ]

    # build metric
    metric = TruthfulnessMetric(
        threshold=0.5,
        model=judge,
        include_reason=True,
        verbose_mode=True,
        async_mode=False,
        require_all=True,
        evaluation_template=TruthfulnessTemplate
    )

    # Get results
    total_records = len(df_input)
    progress_bar = st.progress(0)
    status_text = st.empty()

    results = []

    for i, tc in enumerate(test_cases):

        status_text.text(f"Processing record {i+1} of {total_records}...")
        
        # with SilentConsole():
        results.append(
            evaluate(
                test_cases=[tc],
                metrics=[metric],
                display_config=display_config,
                async_config=async_config,
                cache_config=cache_config
            )
        )

        progress_bar.progress((i + 1) / total_records)

    # Compile results
    eval_df = results_to_df(results, verbose_logs=enable_verbose)

    # Merge back with original dataframe
    score_df = (
        df_input
        .merge(eval_df, on='key')
        .drop(['input', 'key'], axis=1)
    )

    #output the result
    timestamp = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M")
    download_name = f"DeepEvalTest_{output_filename}_{timestamp}.csv"
    csv_bytes = score_df.to_csv(index=False).encode("utf-8")
    st.success("Automated test completed. Click below to download.")

    st.download_button(
        label="Download Result CSV",
        data=csv_bytes,
        file_name=download_name,
        mime="text/csv",
    )
