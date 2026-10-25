"""
Disease Knowledge Base for Chest X-ray Classification.
Provides detailed disease information, symptoms, remedies, and care guidance
for each of the four pulmonary classes supported by the CNN classifier.

All content is general educational information. It is NOT medical advice,
diagnosis, or treatment. Always consult a qualified healthcare professional.
"""
from typing import Dict, List


# ---------------------------------------------------------------------------
# Disease detail schema per class key (lowercase, matches IDX_TO_CLASS keys)
# ---------------------------------------------------------------------------
# Required keys for every entry:
#   disease_name          - Display name
#   overview              - Plain-language description
#   key_symptoms          - List of common symptoms
#   causes_risk_factors   - List of causes / risk factors
#   remedies_treatment    - List of general care / treatment options
#   prevention            - List of prevention tips
#   when_to_seek_urgent_care - List of red-flag symptoms
#   disclaimer            - Standard medical disclaimer for this entry
# ---------------------------------------------------------------------------

DISEASE_KNOWLEDGE_BASE: Dict[str, Dict] = {
    "normal": {
        "disease_name": "Normal Lung (Healthy Findings)",
        "overview": (
            "A normal chest X-ray shows clear, well-aerated lung fields with sharp costophrenic angles, "
            "a normal heart size, and no signs of consolidation, fluid, masses, or abnormal opacities. "
            "The lung parenchyma appears healthy with typical bronchovascular markings."
        ),
        "key_symptoms": [
            "No respiratory symptoms (asymptomatic)",
            "Normal breathing without shortness of breath",
            "No cough, fever, or chest pain",
        ],
        "causes_risk_factors": [
            "Not a disease — indicates absence of detectable pulmonary pathology on X-ray",
            "Maintained by healthy lifestyle, avoiding smoking, and minimizing exposure to air pollutants",
        ],
        "remedies_treatment": [
            "No treatment required — findings are consistent with healthy lungs",
            "Continue routine preventive health care and wellness checkups",
            "Avoid smoking and secondhand smoke to maintain lung health",
        ],
        "prevention": [
            "Get regular physical exercise to support cardiovascular and respiratory fitness",
            "Avoid tobacco smoke, vaping, and occupational dust or chemical exposures",
            "Stay current on vaccinations (influenza, COVID-19, pneumococcal as recommended)",
            "Practice good hand hygiene to reduce respiratory infection risk",
        ],
        "when_to_seek_urgent_care": [
            "New or worsening shortness of breath",
            "Persistent cough, chest pain, or coughing up blood",
            "Unexplained fever or unexplained weight loss",
        ],
        "disclaimer": (
            "This is an AI-generated academic output, not a medical diagnosis. "
            "A normal AI result does not replace a radiologist's reading or a physician's evaluation. "
            "Consult a qualified healthcare professional for any health concerns."
        ),
    },

    "covid": {
        "disease_name": "COVID-19 Associated Pneumonitis",
        "overview": (
            "COVID-19 is a respiratory illness caused by the SARS-CoV-2 virus. On chest X-ray it may "
            "show bilateral peripheral ground-glass opacities, consolidations, or crazy-paving patterns, "
            "often in the lower lobes. It can range from mild upper-respiratory symptoms to severe "
            "pneumonia and acute respiratory distress."
        ),
        "key_symptoms": [
            "Fever or chills",
            "Dry cough",
            "Shortness of breath or difficulty breathing",
            "Fatigue, muscle or body aches",
            "New loss of taste or smell",
            "Sore throat, congestion, or runny nose",
        ],
        "causes_risk_factors": [
            "Infection with SARS-CoV-2 virus via respiratory droplets or aerosols",
            "Close contact with an infected person",
            "Higher risk in older adults and people with underlying conditions (diabetes, heart disease, obesity, immunocompromised state)",
        ],
        "remedies_treatment": [
            "Isolate from others to prevent spread; follow local public health guidance",
            "Rest, hydration, and over-the-counter fever reducers (e.g., acetaminophen) as appropriate",
            "Monitor oxygen saturation (SpO2) with a pulse oximeter if advised by a clinician",
            "Seek medical care for prescription antivirals or other therapies if eligible and early in illness",
            "Oxygen support or hospitalization may be needed for severe or critical cases",
        ],
        "prevention": [
            "Stay up to date with COVID-19 vaccination and boosters",
            "Wear a well-fitting mask in crowded or high-risk indoor settings",
            "Improve indoor ventilation and practice hand hygiene",
            "Avoid close contact with people who have confirmed or suspected COVID-19",
        ],
        "when_to_seek_urgent_care": [
            "Trouble breathing or persistent chest pain/pressure",
            "New confusion or inability to stay awake",
            "Pale, gray, or blue-colored skin, lips, or nail beds",
            "Oxygen saturation persistently below 94% (if monitoring)",
        ],
        "disclaimer": (
            "This is an AI-generated academic output, not a medical diagnosis. "
            "COVID-19 cannot be confirmed by chest X-ray alone — RT-PCR or other approved testing is required. "
            "Consult a qualified healthcare professional for testing, treatment, and medical advice."
        ),
    },

    "lung_opacity": {
        "disease_name": "Pulmonary Opacity / Alveolar Consolidation",
        "overview": (
            "Pulmonary opacity refers to an area of increased whiteness (density) on a chest X-ray, "
            "which can represent alveolar consolidation, infection, inflammation, or other processes. "
            "It may be focal (one area) or diffuse, and the underlying cause must be determined by "
            "clinical correlation and sometimes further imaging such as CT."
        ),
        "key_symptoms": [
            "Cough (may produce mucus or sputum)",
            "Fever or chills",
            "Shortness of breath",
            "Chest pain that may worsen with breathing",
            "Fatigue",
        ],
        "causes_risk_factors": [
            "Bacterial pneumonia (common cause of focal consolidation)",
            "Viral or fungal pneumonia",
            "Pulmonary edema, hemorrhage, or aspiration",
            "Recent surgery, immobilization, or swallowing difficulty (aspiration risk)",
            "Smoking and chronic lung disease increase susceptibility",
        ],
        "remedies_treatment": [
            "Treatment depends on the underlying cause; bacterial pneumonia is treated with antibiotics",
            "Viral causes may require supportive care or antivirals in selected cases",
            "Rest, fluids, fever control, and breathing exercises as advised",
            "Follow-up chest X-ray or CT to confirm resolution",
            "Hospitalization may be needed for severe cases or difficulty breathing",
        ],
        "prevention": [
            "Vaccination against influenza, pneumococcus, and COVID-19 as recommended",
            "Avoid smoking and exposure to secondhand smoke or airborne irritants",
            "Treat underlying conditions such as diabetes, heart disease, or COPD",
            "Practice good oral hygiene and swallowing precautions if aspiration risk exists",
        ],
        "when_to_seek_urgent_care": [
            "Difficulty breathing or worsening shortness of breath",
            "High fever not responding to medication",
            "Chest pain, confusion, or bluish lips/fingernails",
            "Coughing up blood",
        ],
        "disclaimer": (
            "This is an AI-generated academic output, not a medical diagnosis. "
            "Pulmonary opacity is a radiological finding with many possible causes and requires "
            "correlation with symptoms, history, and often additional testing. Consult a physician."
        ),
    },

    "viral_pneumonia": {
        "disease_name": "Viral Pneumonia / Interstitial Infiltration",
        "overview": (
            "Viral pneumonia is a lung infection caused by viruses (such as influenza, RSV, adenovirus, "
            "or SARS-CoV-2) that inflames the air sacs and interstitial tissue of the lungs. On chest "
            "X-ray it often appears as diffuse bilateral interstitial markings, ground-glass opacities, "
            "or patchy infiltrates, typically more widespread than focal bacterial consolidation."
        ),
        "key_symptoms": [
            "Dry or productive cough",
            "Fever and chills",
            "Shortness of breath or rapid breathing",
            "Muscle aches, headache, and fatigue",
            "Wheezing or chest tightness",
        ],
        "causes_risk_factors": [
            "Respiratory viruses including influenza, respiratory syncytial virus (RSV), and adenovirus",
            "Close contact with infected individuals",
            "Higher risk in young children, older adults, and people with chronic lung or heart disease",
            "Weakened immune system",
        ],
        "remedies_treatment": [
            "Antibiotics do not treat viruses; treatment is mainly supportive (rest, fluids, fever control)",
            "Antiviral medications may be prescribed for influenza or COVID-19 if started early",
            "Inhaled bronchodilators or oxygen may be used for wheezing or low oxygen",
            "Hospitalization for severe cases with respiratory distress",
            "Follow-up imaging to ensure the infection resolves",
        ],
        "prevention": [
            "Annual influenza vaccination and COVID-19 vaccination as recommended",
            "RSV vaccination for eligible older adults and infants",
            "Hand hygiene and avoiding close contact with sick individuals",
            "Avoid smoking and maintain good nutrition and sleep to support immune health",
        ],
        "when_to_seek_urgent_care": [
            "Severe shortness of breath or bluish lips/face",
            "Chest pain or pressure",
            "Confusion, dizziness, or fainting",
            "Oxygen saturation below 92–94% if monitoring at home",
            "Symptoms that worsen after initial improvement",
        ],
        "disclaimer": (
            "This is an AI-generated academic output, not a medical diagnosis. "
            "Viral pneumonia requires clinical evaluation and may need laboratory testing or imaging "
            "confirmation. Consult a qualified healthcare professional for treatment decisions."
        ),
    },
}


def get_disease_details(class_key: str) -> Dict:
    """
    Retrieve disease details for a given class key.

    Args:
        class_key: Lowercase class identifier (e.g., 'normal', 'covid',
            'lung_opacity', 'viral_pneumonia').

    Returns:
        A dictionary with disease details, or an empty dict if the key is
        not found (graceful degradation — no crash).
    """
    return DISEASE_KNOWLEDGE_BASE.get(class_key.lower().strip(), {})


def get_all_class_keys() -> List[str]:
    """Return the list of supported class keys in the knowledge base."""
    return list(DISEASE_KNOWLEDGE_BASE.keys())
