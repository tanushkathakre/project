import streamlit as st
from prob import *
from summ import *
from sec import *

# CSS styling with background image and translucent overlay
st.markdown(
    """
    <style>
        body {
            background-image: url('https://img.freepik.com/free-photo/photorealistic-lawyer-environment_23-2151151893.jpg'); /* Add your image URL here */
            background-size: cover;
            background-position: center;
        }
        .overlay {
            background-color: rgba(0, 0, 0, 0.01); /* Adjust the alpha value (fourth parameter) for transparency */
            padding-top: 20px;
            padding-bottom: 20px;
        }
        .title {
            color: #1f77b4;
            text-align: center;
        }
        .section {
            padding: 10px;
            margin-top: 20px;
            margin-bottom: 20px;
            border-radius: 10px;
        }
        .warning {
            color: #d62728;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit app
def main():
    st.markdown("<div class='overlay'>", unsafe_allow_html=True)
    st.markdown("<h1 class='title'>Legal Assistant</h1>", unsafe_allow_html=True)

    # Input Section
    st.markdown("<div class='section'><h2>Case Information</h2></div>", unsafe_allow_html=True)
    petitioner = st.text_input("Enter the petitioner:")
    respondent = st.text_input("Enter the respondent:")
    case_facts = st.text_area("Enter the case facts:")

    # Predict Button
    if st.button("Predict"):
        if case_facts:
            # Feature 1: Summary Generation
            st.markdown("<div class='section'><h3>Case Summary</h3></div>", unsafe_allow_html=True)
            summary = generate_summary(case_facts)
            st.write(summary)

            # Feature 3: Winning Probability Prediction
            st.markdown("<div class='section'><h3>Winning Probability Prediction</h3></div>", unsafe_allow_html=True)
            probabilities = predict_winning_probability(petitioner, respondent, case_facts)
            st.write(f"Winning Probability for Petitioner: {probabilities[1]}")
            st.write(f"Winning Probability for Respondent: {probabilities[0]}")

            # Feature 2: Section Prediction and Punishment
            # Feature 2: Section Prediction and Punishment
            # Feature 2: Section Prediction and Punishment
            st.markdown("<div class='section'><h3>Section Prediction and Punishment</h3></div>", unsafe_allow_html=True)
            sections, probabilities = predict_section_and_punishment(case_facts)
            
            all_sections = ", ".join(sections)
            all_punishments = ", ".join([punishments[section] for section in sections])
            
            st.write(f"In this particular case, the following sections of the Indian Penal Code are applicable: {all_sections}.")
            st.write(f"The party found guilty will be subject to the specified charges: {all_punishments}.")
            
            

        else:
            st.warning("Please enter case facts.")

    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
