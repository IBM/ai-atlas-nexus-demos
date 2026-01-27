import streamlit as st


# Hide the prompt suggestions buttons
def hide_buttons():
    st.markdown(
        """
        <style>
        button[class="st-key-tr_TEMP"] {
            display: none;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def show_buttons():
    st.markdown(
        """
        <style>
        button[data-testid="stBaseButton-secondary"] {
            display: block;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


risks = ["Hallucination", "Toxic output"]
priority = ["low", "medium", "high"]
cont = st.container()

if "dynamic_risks" not in st.session_state:
    st.session_state.setdefault("dynamic_risks", {}).update(
        {
            "0": {
                "risk": risks[0],
                "priority": "low",
                "threshold": 0,
            }
        }
    )


def add_row():
    st.session_state.dynamic_risks[len(st.session_state.dynamic_risks)] = {
        "risk": risks[0],
        "priority": "low",
        "threshold": 0,
    }


@st.dialog("Add Dynamic Risks", width="medium")
def vote():
    st.text_input(
        "Enter you text",
        st.selectbox(
            "Risk",
            tuple(risks),
            key=f"col133",
            index=0,
        ),
    )
    st.button("Add New Row", type="primary", on_click=add_row)
    with st.form("input_form"):

        # Create columns for the form inputs
        col1, col2, col3 = st.columns(3)

        for key, dynamic_risk in st.session_state.dynamic_risks.items():
            with col1:
                value = st.selectbox(
                    "Risk" if key == "0" else "",
                    tuple(risks),
                    key=f"col1{key}",
                    index=risks.index(dynamic_risk["risk"]),
                )
                st.session_state.dynamic_risks[key].update({"risk": value})
            with col2:
                value = st.selectbox(
                    "Priority" if key == "0" else "",
                    tuple(priority),
                    key=f"col2{key}",
                    index=priority.index(dynamic_risk["priority"]),
                )
                st.session_state.dynamic_risks[key].update({"priority": value})
            with col3:
                threshold = st.number_input(
                    "Threshold" if key == "0" else "",
                    key=f"col3{key}",
                    value=dynamic_risk["threshold"],
                )
                st.session_state.dynamic_risks[key].update({"threshold": threshold})

        # Every form must have a submit button
        submitted = st.form_submit_button("Submit")

    if submitted:
        st.write(st.session_state.dynamic_risks)


with st.empty():
    cont.button("Add Dynamic Risks", on_click=vote)
