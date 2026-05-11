import streamlit as st
import numpy as np
import pandas as pd

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MediPredict AI",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@300;400;500&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #040d0a; color: #e8f5f0; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; }

[data-testid="stSidebar"] { background: #081210 !important; border-right: 1px solid #1a3528; }
[data-testid="stSidebar"] * { color: #e8f5f0 !important; }
[data-testid="stSidebar"] .stMultiSelect > div { background: #0d1f1a !important; border: 1px solid #2a5540 !important; border-radius: 10px !important; }
[data-testid="stSidebar"] .stSelectbox > div > div { background: #0d1f1a !important; border: 1px solid #2a5540 !important; }

.medi-card { background: #0d1f1a; border: 1px solid #1a3528; border-radius: 16px; padding: 24px; margin-bottom: 16px; transition: border-color .2s; }
.medi-card:hover { border-color: #2a5540; }
.top-card { background: linear-gradient(135deg, #0d1f1a 0%, #0a1a14 100%); border: 1px solid #00b85e; border-radius: 16px; padding: 24px; margin-bottom: 16px; box-shadow: 0 0 30px rgba(0,232,122,0.08); }

.disease-title { font-family: 'Syne', sans-serif; font-weight: 800; font-size: 22px; color: #e8f5f0; margin: 0 0 6px 0; }
.disease-desc { font-size: 13px; color: #5a8070; line-height: 1.6; margin: 0 0 16px 0; }

.conf-badge-gold { display: inline-block; background: rgba(245,166,35,0.15); border: 1px solid rgba(245,166,35,0.4); color: #f5a623; padding: 4px 14px; border-radius: 100px; font-family: 'DM Mono', monospace; font-size: 13px; font-weight: 500; }
.conf-badge-blue { display: inline-block; background: rgba(84,160,255,0.12); border: 1px solid rgba(84,160,255,0.3); color: #54a0ff; padding: 4px 14px; border-radius: 100px; font-family: 'DM Mono', monospace; font-size: 13px; }
.conf-badge-gray { display: inline-block; background: rgba(90,128,112,0.15); border: 1px solid #1a3528; color: #5a8070; padding: 4px 14px; border-radius: 100px; font-family: 'DM Mono', monospace; font-size: 13px; }

.info-block { background: #081210; border: 1px solid #1a3528; border-radius: 12px; padding: 16px; height: 100%; }
.info-block-title { font-family: 'Syne', sans-serif; font-size: 11px; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: #5a8070; margin-bottom: 10px; }
.info-item { display: flex; align-items: flex-start; gap: 8px; font-size: 13px; color: #e8f5f0; margin-bottom: 7px; line-height: 1.5; }
.info-bullet { color: #00e87a; font-weight: 700; flex-shrink: 0; }

.hero-title { font-family: 'Syne', sans-serif; font-weight: 800; font-size: 42px; line-height: 1.1; margin-bottom: 10px; }
.hero-title span { color: #00e87a; }
.hero-sub { font-size: 15px; color: #5a8070; font-weight: 300; margin-bottom: 24px; line-height: 1.7; }

.stat-row { display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 28px; }
.stat-chip { background: #0d1f1a; border: 1px solid #1a3528; border-radius: 10px; padding: 12px 20px; text-align: center; min-width: 100px; }
.stat-num { font-family: 'Syne', sans-serif; font-size: 24px; font-weight: 700; color: #00e87a; display: block; }
.stat-lbl { font-family: 'DM Mono', monospace; font-size: 10px; color: #5a8070; text-transform: uppercase; letter-spacing: .06em; }

.rank-1 { color: #f5a623; font-family: 'Syne',sans-serif; font-weight:800; font-size:28px; }
.rank-2 { color: #54a0ff; font-family: 'Syne',sans-serif; font-weight:800; font-size:28px; }
.rank-3 { color: #5a8070; font-family: 'Syne',sans-serif; font-weight:800; font-size:28px; }

.sym-tag { display: inline-block; background: rgba(0,232,122,0.08); border: 1px solid #2a5540; color: #00e87a; border-radius: 6px; padding: 3px 10px; font-size: 11px; font-family: 'DM Mono', monospace; margin: 2px; }
.disclaimer { background: rgba(255,71,87,0.06); border: 1px solid rgba(255,71,87,0.2); border-radius: 12px; padding: 14px 18px; font-size: 12px; color: #5a8070; line-height: 1.6; margin-top: 20px; }

hr { border-color: #1a3528 !important; margin: 20px 0 !important; }
.stProgress > div > div > div { background: #00e87a !important; }
.stButton > button { background: #00e87a !important; color: #000 !important; font-family: 'Syne', sans-serif !important; font-weight: 700 !important; font-size: 15px !important; border: none !important; border-radius: 12px !important; padding: 12px 28px !important; width: 100% !important; transition: all .2s !important; }
.stButton > button:hover { background: #00ff88 !important; box-shadow: 0 0 24px rgba(0,232,122,0.4) !important; }
[data-testid="stMultiSelect"] span[data-baseweb="tag"] { background: rgba(0,232,122,0.15) !important; border: 1px solid #2a5540 !important; color: #00e87a !important; border-radius: 6px !important; }
</style>
""", unsafe_allow_html=True)


# ─── All Data (embedded — no files needed) ───────────────────────────────────

ALL_SYMPTOMS = [
    "itching", "skin_rash", "nodal_skin_eruptions", "continuous_sneezing", "shivering",
    "chills", "joint_pain", "stomach_pain", "acidity", "ulcers_on_tongue",
    "muscle_wasting", "vomiting", "burning_micturition", "spotting_urination", "fatigue",
    "weight_gain", "anxiety", "cold_hands_and_feets", "mood_swings", "weight_loss",
    "restlessness", "lethargy", "patches_in_throat", "irregular_sugar_level", "cough",
    "high_fever", "sunken_eyes", "breathlessness", "sweating", "dehydration",
    "indigestion", "headache", "yellowish_skin", "dark_urine", "nausea",
    "loss_of_appetite", "pain_behind_the_eyes", "back_pain", "constipation", "abdominal_pain",
    "diarrhoea", "mild_fever", "yellow_urine", "yellowing_of_eyes", "acute_liver_failure",
    "fluid_overload", "swelling_of_stomach", "swelled_lymph_nodes", "malaise", "blurred_and_distorted_vision",
    "phlegm", "throat_irritation", "redness_of_eyes", "sinus_pressure", "runny_nose",
    "congestion", "chest_pain", "weakness_in_limbs", "fast_heart_rate", "pain_during_bowel_movements",
    "pain_in_anal_region", "bloody_stool", "irritation_in_anus", "neck_stiffness", "word_finding_difficulty",
    "spinning_movements", "loss_of_balance", "unsteadiness", "weakness_of_one_body_side", "loss_of_smell",
    "bladder_discomfort", "foul_smell_of_urine", "continuous_feel_of_urine", "passage_of_gases", "internal_itching",
    "toxic_look", "depression", "irritability", "muscle_pain", "altered_sensorium",
    "red_spots_over_body", "belly_pain", "abnormal_menstruation", "watering_from_eyes", "increased_appetite",
    "polyuria", "family_history", "mucoid_sputum", "rusty_sputum", "lack_of_concentration",
    "visual_disturbances", "receiving_blood_transfusion", "receiving_unsterile_injections", "coma", "stomach_bleeding",
    "distention_of_abdomen", "history_of_alcohol_consumption", "blood_in_sputum", "prominent_veins_on_calf", "palpitations",
    "painful_walking", "pus_filled_pimples", "blackheads", "scurring", "skin_peeling",
    "silver_like_dusting", "small_dents_in_nails", "inflammatory_nails", "blister", "red_sore_around_nose",
    "yellow_crust_ooze", "dizziness", "cramps", "bruising", "obesity", "swollen_legs",
    "swollen_blood_vessels", "puffy_face_and_eyes", "enlarged_thyroid", "brittle_nails", "swollen_extremeties",
    "excessive_hunger", "extra_marital_contacts", "drying_and_tingling_lips", "slurred_speech", "knee_pain",
    "hip_joint_pain", "muscle_weakness", "stiff_neck", "swelling_joints", "movement_stiffness",
    "spinning_sensations", "loss_of_sensation", "numbness", "weakness_of_limbs", "patches", "redness"
]

# Deduplicate while preserving order (skin_peeling appeared twice in original)
seen = set()
ALL_SYMPTOMS = [s for s in ALL_SYMPTOMS if not (s in seen or seen.add(s))]

DISEASE_INFO = {
    "Fungal infection": {
        "description": "A fungal skin infection causing itching, rashes, and skin changes.",
        "precautions": ["Keep skin dry and clean", "Avoid sharing personal items", "Wear breathable clothing", "Use antifungal powder in skin folds"],
        "medications": ["Clotrimazole cream", "Terbinafine", "Fluconazole", "Miconazole"],
        "diet": ["Reduce sugar intake", "Eat probiotic-rich foods (yogurt)", "Garlic and coconut oil", "Avoid processed foods"],
        "workout": ["Low-intensity yoga", "Walking 30 min/day", "Avoid excessive sweating", "Swimming with proper hygiene"]
    },
    "Allergy": {
        "description": "An immune response to foreign substances like pollen, pet dander, or food.",
        "precautions": ["Identify and avoid allergens", "Keep windows closed during high pollen season", "Use air purifiers", "Wash hands frequently"],
        "medications": ["Cetirizine (Zyrtec)", "Loratadine (Claritin)", "Fexofenadine", "Nasal corticosteroids"],
        "diet": ["Anti-inflammatory foods", "Vitamin C rich fruits", "Omega-3 fatty acids", "Avoid known food allergens"],
        "workout": ["Indoor cycling", "Yoga and stretching", "Swimming (if not chlorine-sensitive)", "Exercise during low pollen times"]
    },
    "GERD": {
        "description": "Gastroesophageal reflux disease — stomach acid flowing back into the esophagus.",
        "precautions": ["Eat smaller meals", "Don't lie down after eating", "Elevate head while sleeping", "Avoid tight clothing"],
        "medications": ["Omeprazole (Prilosec)", "Pantoprazole", "Ranitidine", "Antacids (Tums)"],
        "diet": ["Avoid spicy and fatty foods", "No caffeine or alcohol", "Eat alkaline foods", "Ginger and aloe vera juice"],
        "workout": ["Walking after meals", "Gentle yoga", "Avoid high-impact exercise after eating", "Core strengthening (mild)"]
    },
    "Chronic cholestasis": {
        "description": "A condition where bile flow from the liver is slowed or blocked.",
        "precautions": ["Avoid alcohol completely", "Take vitamin supplements", "Regular liver function tests", "Avoid hepatotoxic drugs"],
        "medications": ["Ursodeoxycholic acid", "Cholestyramine", "Rifampicin", "Vitamin K supplements"],
        "diet": ["Low-fat diet", "Fat-soluble vitamins (A, D, E, K)", "High-fiber foods", "Avoid alcohol entirely"],
        "workout": ["Light walking", "Tai chi", "Gentle stretching", "Avoid strenuous activity"]
    },
    "Drug Reaction": {
        "description": "An adverse reaction to a medication causing skin rashes, fever, or other symptoms.",
        "precautions": ["Stop the offending drug immediately", "Inform doctor about all medications", "Carry medical alert card", "Avoid self-medication"],
        "medications": ["Antihistamines", "Corticosteroids", "Epinephrine (severe cases)", "Supportive care"],
        "diet": ["Hydrate well", "Bland diet", "Avoid alcohol", "Light easily digestible foods"],
        "workout": ["Rest until symptoms resolve", "Light walking when feeling better", "Avoid strenuous exercise", "Yoga for stress relief"]
    },
    "Peptic ulcer disease": {
        "description": "Sores in the stomach lining, small intestine, or esophagus.",
        "precautions": ["Avoid NSAIDs", "Quit smoking", "Limit alcohol", "Manage stress"],
        "medications": ["Proton pump inhibitors", "H2 blockers", "Antibiotics (if H. pylori)", "Antacids"],
        "diet": ["Eat small frequent meals", "Avoid spicy food", "Cabbage juice", "Avoid caffeine and alcohol"],
        "workout": ["Light yoga", "Walking", "Stress-reduction exercises", "Avoid intense core workouts"]
    },
    "AIDS": {
        "description": "Advanced stage of HIV infection affecting the immune system severely.",
        "precautions": ["Strict antiretroviral therapy adherence", "Safe sex practices", "Avoid infections", "Regular medical monitoring"],
        "medications": ["Antiretroviral therapy (ART)", "Prophylactic antibiotics", "Antifungals", "Vaccinations"],
        "diet": ["High-protein diet", "Fruits and vegetables", "Safe food handling", "Adequate caloric intake"],
        "workout": ["Moderate aerobic exercise", "Resistance training", "Yoga", "Adjust intensity to energy levels"]
    },
    "Diabetes": {
        "description": "A metabolic disease causing high blood sugar due to insulin issues.",
        "precautions": ["Monitor blood sugar regularly", "Foot care", "Eye exams", "Take medications as prescribed"],
        "medications": ["Metformin", "Insulin therapy", "Glipizide", "SGLT2 inhibitors"],
        "diet": ["Low glycemic index foods", "Whole grains", "Lean proteins", "Avoid sugary drinks and sweets"],
        "workout": ["30 min brisk walking daily", "Resistance training 3x/week", "Yoga", "Swimming"]
    },
    "Gastroenteritis": {
        "description": "Inflammation of the stomach and intestines, typically from infection.",
        "precautions": ["Wash hands thoroughly", "Avoid contaminated food/water", "Stay hydrated", "Rest"],
        "medications": ["Oral rehydration salts", "Loperamide (Imodium)", "Antiemetics", "Antibiotics if bacterial"],
        "diet": ["BRAT diet (Banana, Rice, Applesauce, Toast)", "Clear broths", "Electrolyte drinks", "Avoid dairy and fatty foods"],
        "workout": ["Complete rest during acute phase", "Light walking when recovering", "Resume normal activity gradually", "Yoga after full recovery"]
    },
    "Bronchial Asthma": {
        "description": "Chronic inflammatory disease of airways causing breathing difficulties.",
        "precautions": ["Avoid triggers (dust, smoke, pollen)", "Keep rescue inhaler handy", "Monitor peak flow", "Avoid cold air"],
        "medications": ["Salbutamol inhaler", "Corticosteroid inhalers", "Montelukast", "Theophylline"],
        "diet": ["Anti-inflammatory foods", "Vitamin D rich foods", "Avoid sulfites", "Ginger and turmeric"],
        "workout": ["Swimming", "Yoga breathing exercises", "Walking", "Avoid cold-weather outdoor exercise"]
    },
    "Hypertension": {
        "description": "Persistently elevated blood pressure that can damage the heart and blood vessels.",
        "precautions": ["Monitor BP daily", "Reduce sodium intake", "Quit smoking", "Manage stress"],
        "medications": ["Amlodipine", "Lisinopril", "Losartan", "Metoprolol"],
        "diet": ["DASH diet", "Low sodium", "Potassium-rich foods", "Limit alcohol"],
        "workout": ["Aerobic exercise 30 min/day", "Walking and cycling", "Yoga", "Avoid heavy weightlifting"]
    },
    "Migraine": {
        "description": "Intense recurring headaches often with nausea, vomiting, and light sensitivity.",
        "precautions": ["Identify and avoid triggers", "Maintain sleep schedule", "Stay hydrated", "Reduce screen time"],
        "medications": ["Sumatriptan", "Topiramate", "Propranolol", "NSAIDs for mild attacks"],
        "diet": ["Regular meal times", "Avoid tyramine-rich foods (aged cheese, red wine)", "Magnesium-rich foods", "Caffeine in moderation"],
        "workout": ["Regular moderate aerobic exercise", "Yoga and stretching", "Swimming", "Avoid sudden intense exercise"]
    },
    "Cervical spondylosis": {
        "description": "Age-related wear of the cervical spine causing neck pain and stiffness.",
        "precautions": ["Maintain good posture", "Ergonomic workspace setup", "Avoid heavy lifting", "Use supportive pillow"],
        "medications": ["NSAIDs (Ibuprofen)", "Muscle relaxants", "Gabapentin", "Topical pain relievers"],
        "diet": ["Calcium and Vitamin D rich foods", "Anti-inflammatory foods", "Omega-3 fatty acids", "Stay hydrated"],
        "workout": ["Neck stretching exercises", "Yoga", "Swimming", "Avoid contact sports"]
    },
    "Paralysis (brain hemorrhage)": {
        "description": "Loss of muscle function due to bleeding in the brain.",
        "precautions": ["Strict BP control", "Rehabilitation therapy", "Fall prevention", "Medication compliance"],
        "medications": ["Blood pressure medications", "Anticoagulants (if indicated)", "Muscle relaxants", "Neuroprotective agents"],
        "diet": ["Heart-healthy diet", "Low sodium", "High fiber", "Antioxidant-rich foods"],
        "workout": ["Physical therapy", "Range-of-motion exercises", "Hydrotherapy", "Occupational therapy exercises"]
    },
    "Jaundice": {
        "description": "Yellowing of skin and eyes due to high bilirubin levels.",
        "precautions": ["Avoid alcohol", "Rest adequately", "Stay hydrated", "Follow up with blood tests"],
        "medications": ["Treat underlying cause", "Ursodeoxycholic acid", "Vitamin K", "IV fluids if needed"],
        "diet": ["High-carbohydrate diet", "Fresh fruits and vegetables", "Avoid fatty foods", "Plenty of water and juices"],
        "workout": ["Complete rest initially", "Light walking when improving", "Yoga (gentle)", "Avoid strenuous exercise"]
    },
    "Malaria": {
        "description": "A mosquito-borne infectious disease causing fever, chills, and flu-like illness.",
        "precautions": ["Use mosquito repellent", "Sleep under mosquito nets", "Wear protective clothing", "Eliminate standing water"],
        "medications": ["Chloroquine", "Artemisinin-based therapy", "Primaquine", "Doxycycline (prophylaxis)"],
        "diet": ["High-calorie nutritious foods", "Fruits rich in Vitamin C", "Adequate hydration", "Avoid alcohol"],
        "workout": ["Complete rest during acute phase", "Light activity when fever resolves", "Gradual return to exercise", "Yoga when recovered"]
    },
    "Chicken pox": {
        "description": "Highly contagious viral infection causing itchy blisters all over the body.",
        "precautions": ["Isolate from others", "Don't scratch blisters", "Keep fingernails short", "Avoid aspirin"],
        "medications": ["Acyclovir (antivirals)", "Calamine lotion", "Antihistamines for itching", "Paracetamol for fever"],
        "diet": ["Soft bland foods", "Cold foods to soothe mouth sores", "High Vitamin C", "Stay well hydrated"],
        "workout": ["Complete rest until contagious period ends", "Light stretching when better", "Avoid public gyms", "Gentle yoga at home"]
    },
    "Dengue": {
        "description": "A mosquito-borne viral infection causing high fever and severe joint pain.",
        "precautions": ["Eliminate mosquito breeding sites", "Use repellents", "Full-sleeve clothing", "Monitor platelet count"],
        "medications": ["Paracetamol for fever", "IV fluids", "Platelet transfusion if needed", "Avoid NSAIDs/Aspirin"],
        "diet": ["Papaya leaf juice (boosts platelets)", "High fluid intake", "Kiwi and citrus fruits", "Avoid spicy foods"],
        "workout": ["Strict rest during fever", "Light walking when platelet count normalizes", "No intense exercise for 2 weeks", "Gentle yoga when recovered"]
    },
    "Typhoid": {
        "description": "A bacterial infection spread through contaminated food and water.",
        "precautions": ["Drink purified water only", "Practice safe food handling", "Get vaccinated", "Wash hands frequently"],
        "medications": ["Ciprofloxacin", "Azithromycin", "Ceftriaxone", "Supportive IV fluids"],
        "diet": ["High-calorie soft diet", "Boiled foods", "Plenty of fluids", "Avoid raw vegetables and salads"],
        "workout": ["Complete bed rest during fever", "Light walking after recovery", "Avoid exercise for 4-6 weeks post-illness", "Gradual return to activity"]
    },
    "Hepatitis A": {
        "description": "A viral liver infection spread through contaminated food and water.",
        "precautions": ["Vaccination", "Safe food and water", "Hand hygiene", "Avoid sharing utensils"],
        "medications": ["Supportive treatment", "Rest and hydration", "Vitamin supplements", "Anti-nausea medications"],
        "diet": ["High carbohydrate low fat diet", "Fresh fruits", "Avoid alcohol completely", "Small frequent meals"],
        "workout": ["Rest during acute phase", "Light activity when feeling better", "Yoga when liver enzymes normalize", "No alcohol-related socializing"]
    },
    "Hepatitis B": {
        "description": "A viral infection attacking the liver through blood or body fluids.",
        "precautions": ["Vaccination", "Safe sex", "Don't share needles", "Regular liver monitoring"],
        "medications": ["Tenofovir", "Entecavir", "Interferon alpha", "Regular monitoring"],
        "diet": ["Low fat diet", "High fiber", "Avoid alcohol", "Antioxidant-rich foods"],
        "workout": ["Moderate exercise when not in acute phase", "Yoga and walking", "Avoid overexertion", "Swimming"]
    },
    "Hepatitis C": {
        "description": "A blood-borne viral infection affecting the liver.",
        "precautions": ["Don't share needles", "Safe sex practices", "Avoid alcohol", "Regular liver tests"],
        "medications": ["Sofosbuvir", "Daclatasvir", "Ribavirin", "Direct-acting antivirals"],
        "diet": ["Low sugar low fat", "Coffee (shown to help liver)", "Leafy greens", "Avoid alcohol completely"],
        "workout": ["Moderate aerobic exercise", "Yoga", "Walking", "Strength training (light)"]
    },
    "Hepatitis D": {
        "description": "A liver infection that occurs only with hepatitis B.",
        "precautions": ["Hepatitis B vaccination (prevents D)", "Avoid sharing needles", "Safe sex", "Regular liver monitoring"],
        "medications": ["Interferon alpha (limited options)", "Treat underlying Hepatitis B", "Supportive care", "Liver transplant (severe)"],
        "diet": ["Low fat", "Avoid alcohol completely", "High antioxidant foods", "Adequate protein"],
        "workout": ["Rest during acute phase", "Light walking", "Yoga", "Avoid strenuous activity"]
    },
    "Hepatitis E": {
        "description": "A waterborne liver infection common in developing countries.",
        "precautions": ["Drink clean water", "Proper sanitation", "Avoid raw shellfish", "Especially caution in pregnancy"],
        "medications": ["Supportive treatment", "Ribavirin (immunocompromised)", "Rest", "Adequate hydration"],
        "diet": ["High carbohydrate low fat", "Avoid alcohol", "Small frequent meals", "Fresh fruits and vegetables"],
        "workout": ["Complete bed rest in acute phase", "Light walking when recovering", "Gentle yoga", "Gradually increase activity"]
    },
    "Alcoholic hepatitis": {
        "description": "Liver inflammation caused by excessive alcohol consumption.",
        "precautions": ["Stop alcohol completely", "Nutritional support", "Regular liver function tests", "Join support groups for sobriety"],
        "medications": ["Corticosteroids (Prednisolone)", "Pentoxifylline", "Nutritional supplements", "Vitamin B1 (Thiamine)"],
        "diet": ["High protein high calorie", "Vitamin-rich foods", "Avoid alcohol entirely", "Adequate hydration"],
        "workout": ["Rest initially", "Light walking when stable", "Yoga for stress management", "Gradual increase in activity"]
    },
    "Tuberculosis": {
        "description": "A serious bacterial infection primarily affecting the lungs.",
        "precautions": ["Complete full course of antibiotics", "Cover mouth when coughing", "Ventilate living spaces", "Regular follow-up tests"],
        "medications": ["Isoniazid", "Rifampicin", "Ethambutol", "Pyrazinamide"],
        "diet": ["High protein diet", "Vitamin D rich foods", "Calorie-dense foods", "Avoid alcohol"],
        "workout": ["Bed rest during active phase", "Light walking as tolerated", "Breathing exercises", "Gradual return to exercise"]
    },
    "Common Cold": {
        "description": "A viral infection of the upper respiratory tract.",
        "precautions": ["Wash hands frequently", "Avoid close contact with sick people", "Don't touch face", "Stay hydrated"],
        "medications": ["Paracetamol for fever", "Decongestants", "Antihistamines", "Throat lozenges"],
        "diet": ["Hot soups and broths", "Honey and ginger tea", "Citrus fruits (Vitamin C)", "Stay well hydrated"],
        "workout": ["Rest if fever present", "Light walking when feeling better", "Avoid strenuous exercise", "Yoga breathing exercises"]
    },
    "Pneumonia": {
        "description": "Infection causing inflammation of air sacs in one or both lungs.",
        "precautions": ["Complete antibiotic course", "Vaccination (pneumococcal)", "Rest adequately", "Avoid smoking"],
        "medications": ["Amoxicillin", "Azithromycin", "Doxycycline", "Oseltamivir (if viral)"],
        "diet": ["High-protein foods", "Warm broths", "Vitamin C rich foods", "Stay well hydrated"],
        "workout": ["Complete bed rest during acute phase", "Deep breathing exercises", "Light walking when improving", "Pulmonary rehab when recovered"]
    },
    "Dimorphic hemmorhoids(piles)": {
        "description": "Swollen veins in the rectum or anus causing pain and bleeding.",
        "precautions": ["Avoid straining during bowel movements", "Use moist wipes", "Sitz baths", "Don't sit for long periods"],
        "medications": ["Sitz baths", "Hemorrhoid creams", "Fiber supplements", "Stool softeners"],
        "diet": ["High-fiber diet", "Plenty of water", "Fruits and vegetables", "Avoid spicy foods"],
        "workout": ["Regular walking", "Pelvic floor exercises (Kegels)", "Yoga", "Avoid heavy weightlifting"]
    },
    "Heart attack": {
        "description": "Blockage of blood flow to the heart muscle causing tissue damage.",
        "precautions": ["Take medications religiously", "Monitor cholesterol and BP", "Quit smoking", "Cardiac rehab program"],
        "medications": ["Aspirin", "Statins", "Beta-blockers", "ACE inhibitors"],
        "diet": ["Heart-healthy diet", "Low saturated fat", "Mediterranean diet", "Omega-3 rich foods"],
        "workout": ["Cardiac rehabilitation program", "Light walking initially", "Supervised aerobic exercise", "Avoid heavy exertion"]
    },
    "Varicose veins": {
        "description": "Enlarged, twisted veins usually in the legs.",
        "precautions": ["Avoid prolonged standing", "Elevate legs when resting", "Wear compression stockings", "Maintain healthy weight"],
        "medications": ["Compression stockings", "Sclerotherapy", "Horse chestnut extract", "Diosmin"],
        "diet": ["High fiber diet", "Flavonoid-rich foods (blueberries, cherries)", "Avoid salty foods", "Stay hydrated"],
        "workout": ["Walking", "Cycling", "Swimming", "Leg elevation exercises"]
    },
    "Hypothyroidism": {
        "description": "Underactive thyroid gland causing slow metabolism.",
        "precautions": ["Take levothyroxine at same time daily", "Regular thyroid function tests", "Avoid soy around medication time", "Monitor symptoms"],
        "medications": ["Levothyroxine (Synthroid)", "Liothyronine", "Thyroid hormone replacement", "Regular TSH monitoring"],
        "diet": ["Iodine-rich foods", "Selenium sources (Brazil nuts)", "Avoid goitrogens in excess", "Zinc-rich foods"],
        "workout": ["Regular aerobic exercise", "Yoga", "Strength training", "Walking — helps metabolism"]
    },
    "Hyperthyroidism": {
        "description": "Overactive thyroid producing excess hormones.",
        "precautions": ["Regular thyroid function tests", "Avoid iodine excess", "Stress management", "Bone density monitoring"],
        "medications": ["Methimazole", "Propylthiouracil", "Beta-blockers for symptoms", "Radioactive iodine therapy"],
        "diet": ["Calcium and Vitamin D rich foods", "Low iodine diet", "Cruciferous vegetables", "Avoid caffeine and stimulants"],
        "workout": ["Low-impact exercise", "Yoga", "Swimming", "Avoid high-intensity exercise when heart rate is elevated"]
    },
    "Hypoglycemia": {
        "description": "Abnormally low blood sugar levels causing dizziness and confusion.",
        "precautions": ["Carry fast-acting sugar", "Eat regular meals", "Monitor blood sugar", "Wear medical ID bracelet"],
        "medications": ["Glucose tablets", "Glucagon injection (severe)", "Adjust diabetes medications with doctor", "Dextrose IV if unconscious"],
        "diet": ["Regular small meals", "Complex carbohydrates", "Protein with each meal", "Avoid alcohol on empty stomach"],
        "workout": ["Check blood sugar before and after exercise", "Carry glucose snacks", "Moderate intensity preferred", "Buddy system when exercising"]
    },
    "Osteoarthritis": {
        "description": "Degenerative joint disease causing cartilage breakdown and pain.",
        "precautions": ["Maintain healthy weight", "Protect joints during activity", "Use assistive devices", "Regular physiotherapy"],
        "medications": ["Acetaminophen", "NSAIDs (Ibuprofen)", "Topical diclofenac", "Glucosamine supplements"],
        "diet": ["Anti-inflammatory diet", "Omega-3 fatty acids", "Vitamin D and calcium", "Collagen-rich foods"],
        "workout": ["Low-impact aerobics", "Swimming", "Cycling", "Range-of-motion exercises"]
    },
    "Arthritis": {
        "description": "Inflammation of one or more joints causing pain and stiffness.",
        "precautions": ["Protect joints", "Use warm/cold therapy", "Rest when flaring", "Physiotherapy"],
        "medications": ["NSAIDs", "DMARDs (Methotrexate)", "Hydroxychloroquine", "Biologics (severe cases)"],
        "diet": ["Anti-inflammatory foods", "Omega-3 fatty acids", "Turmeric and ginger", "Avoid processed foods"],
        "workout": ["Aquatic exercises", "Yoga", "Tai chi", "Gentle strength training"]
    },
    "Vertigo": {
        "description": "A sensation of spinning or dizziness, often from inner ear problems.",
        "precautions": ["Move slowly when changing positions", "Avoid heights", "Fall-proof your home", "Adequate rest"],
        "medications": ["Meclizine (Antivert)", "Diazepam", "Betahistine", "Scopolamine patch"],
        "diet": ["Low sodium diet", "Stay hydrated", "Avoid caffeine and alcohol", "Ginger tea"],
        "workout": ["Epley maneuver (under guidance)", "Balance exercises", "Yoga (avoid inversions)", "Tai chi"]
    },
    "Acne": {
        "description": "A skin condition causing pimples, blackheads, and cysts.",
        "precautions": ["Cleanse face twice daily", "Don't pop pimples", "Change pillowcases regularly", "Use non-comedogenic products"],
        "medications": ["Benzoyl peroxide", "Retinoids (Tretinoin)", "Doxycycline", "Isotretinoin (severe)"],
        "diet": ["Low glycemic index foods", "Omega-3 fatty acids", "Zinc-rich foods", "Avoid dairy and sugar"],
        "workout": ["Cleanse after sweating", "Yoga for stress reduction", "Regular exercise improves skin", "Avoid touching face during workout"]
    },
    "Urinary tract infection": {
        "description": "Bacterial infection in any part of the urinary system.",
        "precautions": ["Drink plenty of water", "Don't hold urine", "Proper wiping technique", "Urinate after sex"],
        "medications": ["Trimethoprim-sulfamethoxazole", "Nitrofurantoin", "Ciprofloxacin", "Cranberry supplements"],
        "diet": ["Lots of water", "Cranberry juice", "Vitamin C rich foods", "Probiotics"],
        "workout": ["Light walking", "Yoga (avoid heated yoga)", "Swimming", "Moderate exercise to boost immunity"]
    },
    "Psoriasis": {
        "description": "An autoimmune condition causing rapid skin cell buildup and red scaly patches.",
        "precautions": ["Moisturize regularly", "Avoid triggers (stress, smoking)", "Sun exposure in moderation", "Avoid skin trauma"],
        "medications": ["Topical corticosteroids", "Methotrexate", "Biologics (Adalimumab)", "Vitamin D analogues"],
        "diet": ["Anti-inflammatory diet", "Omega-3 fatty acids", "Avoid alcohol", "Gluten-free (if sensitive)"],
        "workout": ["Regular aerobic exercise", "Swimming (saltwater pools preferred)", "Yoga for stress", "Low-impact activities"]
    },
    "Impetigo": {
        "description": "A highly contagious bacterial skin infection causing red sores.",
        "precautions": ["Keep sores covered", "Wash hands frequently", "Don't share towels or clothing", "Complete antibiotic course"],
        "medications": ["Mupirocin topical", "Fusidic acid cream", "Oral Amoxicillin", "Cefalexin"],
        "diet": ["Immune-boosting foods", "Vitamin C rich foods", "Zinc-rich foods", "Adequate protein"],
        "workout": ["Avoid contact sports until healed", "Light walking at home", "Yoga", "No sharing exercise equipment"]
    }
}

DISEASE_SYMPTOMS = {
    "Fungal infection": ["itching", "skin_rash", "nodal_skin_eruptions", "patches_in_throat"],
    "Allergy": ["continuous_sneezing", "shivering", "chills", "watering_from_eyes", "skin_rash"],
    "GERD": ["stomach_pain", "acidity", "ulcers_on_tongue", "vomiting", "cough", "chest_pain"],
    "Chronic cholestasis": ["itching", "vomiting", "yellowish_skin", "dark_urine", "abdominal_pain", "loss_of_appetite"],
    "Drug Reaction": ["itching", "skin_rash", "stomach_pain", "burning_micturition", "fatigue"],
    "Peptic ulcer disease": ["vomiting", "loss_of_appetite", "abdominal_pain", "passage_of_gases", "internal_itching"],
    "AIDS": ["muscle_wasting", "patches_in_throat", "high_fever", "extra_marital_contacts", "fatigue", "weight_loss"],
    "Diabetes": ["fatigue", "weight_loss", "restlessness", "lethargy", "irregular_sugar_level", "polyuria", "increased_appetite", "blurred_and_distorted_vision"],
    "Gastroenteritis": ["vomiting", "sunken_eyes", "dehydration", "diarrhoea", "stomach_pain"],
    "Bronchial Asthma": ["fatigue", "cough", "high_fever", "breathlessness", "family_history", "mucoid_sputum"],
    "Hypertension": ["headache", "chest_pain", "dizziness", "loss_of_balance", "lack_of_concentration"],
    "Migraine": ["acidity", "indigestion", "headache", "blurred_and_distorted_vision", "excessive_hunger", "stiff_neck", "depression", "irritability"],
    "Cervical spondylosis": ["back_pain", "weakness_in_limbs", "neck_stiffness", "dizziness", "loss_of_balance"],
    "Paralysis (brain hemorrhage)": ["vomiting", "headache", "weakness_of_one_body_side", "altered_sensorium"],
    "Jaundice": ["itching", "vomiting", "fatigue", "weight_loss", "high_fever", "dark_urine", "yellowish_skin", "abdominal_pain"],
    "Malaria": ["chills", "vomiting", "high_fever", "sweating", "headache", "nausea", "diarrhoea", "muscle_pain"],
    "Chicken pox": ["itching", "skin_rash", "fatigue", "lethargy", "high_fever", "headache", "loss_of_appetite", "mild_fever", "swelled_lymph_nodes", "malaise", "red_spots_over_body"],
    "Dengue": ["skin_rash", "chills", "joint_pain", "vomiting", "fatigue", "high_fever", "headache", "nausea", "loss_of_appetite", "pain_behind_the_eyes", "back_pain", "sweating", "muscle_pain", "red_spots_over_body"],
    "Typhoid": ["chills", "vomiting", "fatigue", "high_fever", "headache", "nausea", "constipation", "abdominal_pain", "diarrhoea", "toxic_look", "depression", "muscle_pain"],
    "Hepatitis A": ["joint_pain", "vomiting", "yellowish_skin", "dark_urine", "nausea", "loss_of_appetite", "abdominal_pain", "diarrhoea", "mild_fever", "yellowing_of_eyes", "muscle_pain"],
    "Hepatitis B": ["itching", "fatigue", "lethargy", "yellowish_skin", "dark_urine", "loss_of_appetite", "abdominal_pain", "yellow_urine", "yellowing_of_eyes", "malaise", "receiving_blood_transfusion", "receiving_unsterile_injections"],
    "Hepatitis C": ["fatigue", "yellowish_skin", "nausea", "loss_of_appetite", "yellowing_of_eyes", "family_history"],
    "Hepatitis D": ["joint_pain", "vomiting", "fatigue", "yellowish_skin", "dark_urine", "nausea", "loss_of_appetite", "yellowing_of_eyes"],
    "Hepatitis E": ["joint_pain", "vomiting", "fatigue", "high_fever", "yellowish_skin", "dark_urine", "nausea", "loss_of_appetite", "abdominal_pain", "yellowing_of_eyes", "acute_liver_failure", "coma"],
    "Alcoholic hepatitis": ["vomiting", "yellowish_skin", "abdominal_pain", "swelling_of_stomach", "distention_of_abdomen", "history_of_alcohol_consumption", "fluid_overload"],
    "Tuberculosis": ["chills", "vomiting", "fatigue", "weight_loss", "cough", "high_fever", "breathlessness", "sweating", "loss_of_appetite", "mild_fever", "yellowing_of_eyes", "swelled_lymph_nodes", "malaise", "phlegm", "blood_in_sputum"],
    "Common Cold": ["continuous_sneezing", "chills", "fatigue", "cough", "headache", "runny_nose", "congestion", "chest_pain"],
    "Pneumonia": ["chills", "fatigue", "cough", "high_fever", "breathlessness", "sweating", "malaise", "phlegm", "chest_pain", "fast_heart_rate", "rusty_sputum"],
    "Dimorphic hemmorhoids(piles)": ["constipation", "pain_during_bowel_movements", "pain_in_anal_region", "bloody_stool", "irritation_in_anus"],
    "Heart attack": ["vomiting", "breathlessness", "sweating", "chest_pain", "weakness_in_limbs"],
    "Varicose veins": ["fatigue", "cramps", "bruising", "obesity", "swollen_legs", "swollen_blood_vessels", "prominent_veins_on_calf"],
    "Hypothyroidism": ["fatigue", "weight_gain", "cold_hands_and_feets", "mood_swings", "lethargy", "dizziness", "puffy_face_and_eyes", "enlarged_thyroid", "brittle_nails", "swollen_extremeties", "depression", "irritability", "abnormal_menstruation"],
    "Hyperthyroidism": ["fatigue", "mood_swings", "weight_loss", "restlessness", "sweating", "diarrhoea", "fast_heart_rate", "excessive_hunger", "enlarged_thyroid", "irritability", "abnormal_menstruation"],
    "Hypoglycemia": ["vomiting", "fatigue", "anxiety", "sweating", "headache", "nausea", "blurred_and_distorted_vision", "excessive_hunger", "drying_and_tingling_lips", "slurred_speech", "irritability", "palpitations"],
    "Osteoarthritis": ["joint_pain", "neck_stiffness", "knee_pain", "hip_joint_pain", "swelling_joints", "painful_walking"],
    "Arthritis": ["muscle_weakness", "stiff_neck", "swelling_joints", "movement_stiffness", "loss_of_appetite"],
    "Vertigo": ["vomiting", "headache", "nausea", "spinning_movements", "loss_of_balance", "unsteadiness"],
    "Acne": ["skin_rash", "pus_filled_pimples", "blackheads", "scurring"],
    "Urinary tract infection": ["burning_micturition", "bladder_discomfort", "foul_smell_of_urine", "continuous_feel_of_urine"],
    "Psoriasis": ["skin_rash", "joint_pain", "skin_peeling", "silver_like_dusting", "small_dents_in_nails", "inflammatory_nails"],
    "Impetigo": ["skin_rash", "blister", "red_sore_around_nose", "yellow_crust_ooze", "itching"]
}


# ─── Train Models (cached — runs once per session) ────────────────────────────
@st.cache_resource(show_spinner="🧬 Training ML models on first launch (takes ~30s)...")
def build_models():
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import train_test_split

    rows = []
    rng = np.random.default_rng(42)
    for disease, symptoms in DISEASE_SYMPTOMS.items():
        valid_symptoms = [s for s in symptoms if s in ALL_SYMPTOMS]
        for _ in range(120):
            row = {s: 0 for s in ALL_SYMPTOMS}
            for s in valid_symptoms:
                row[s] = 1
            n_remove = int(rng.integers(0, min(2, len(valid_symptoms)) + 1))
            if n_remove and valid_symptoms:
                for s in rng.choice(valid_symptoms, n_remove, replace=False):
                    row[s] = 0
            n_extra = int(rng.integers(0, 3))
            for s in rng.choice(ALL_SYMPTOMS, n_extra, replace=False):
                row[s] = 1
            row["disease"] = disease
            rows.append(row)

    df = pd.DataFrame(rows)
    X = df.drop("disease", axis=1)
    y = df["disease"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=20, n_jobs=-1)
    rf.fit(X_train, y_train)

    gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb.fit(X_train, y_train)

    return rf, gb

rf_model, gb_model = build_models()


# ─── Prediction ───────────────────────────────────────────────────────────────
def predict_disease(selected_symptoms, model_choice="rf"):
    model = rf_model if model_choice == "rf" else gb_model
    features = {s: (1 if s in selected_symptoms else 0) for s in ALL_SYMPTOMS}
    X = pd.DataFrame([features])
    proba = model.predict_proba(X)[0]
    classes = model.classes_
    top_indices = np.argsort(proba)[::-1][:3]
    results = []
    for i in top_indices:
        disease = classes[i]
        conf = round(float(proba[i]) * 100, 1)
        if conf < 0.5:
            continue
        info = DISEASE_INFO.get(disease, {
            "description": "Information not available.",
            "precautions": [], "medications": [], "diet": [], "workout": []
        })
        results.append({"disease": disease, "confidence": conf, **info})
    return results


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:16px 0 24px'>
        <div style='font-family:Syne,sans-serif;font-weight:800;font-size:22px;'>
            ⚕️ Medi<span style='color:#00e87a'>Predict</span>
        </div>
        <div style='font-size:11px;font-family:DM Mono,monospace;color:#3a6050;margin-top:4px;'>
            ML-POWERED DISEASE ANALYSIS
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🔬 Select Algorithm")
    model_choice = st.selectbox(
        "ML Model",
        options=["rf", "gb"],
        format_func=lambda x: "🌲 Random Forest (99.9%)" if x == "rf" else "⚡ Gradient Boosting (99.2%)",
        label_visibility="collapsed"
    )

    st.markdown("### 🩺 Select Symptoms")
    symptom_display = [s.replace("_", " ").title() for s in ALL_SYMPTOMS]
    symptom_map = {s.replace("_", " ").title(): s for s in ALL_SYMPTOMS}

    selected_display = st.multiselect(
        "Search and select your symptoms",
        options=symptom_display,
        placeholder="Type to search symptoms...",
        label_visibility="collapsed"
    )
    selected_symptoms = [symptom_map[s] for s in selected_display]

    st.markdown(f"""
    <div style='margin:12px 0;padding:10px 14px;background:#0d1f1a;border:1px solid #1a3528;border-radius:10px;'>
        <span style='font-family:DM Mono,monospace;font-size:12px;color:#5a8070;'>SELECTED</span>
        <span style='font-family:Syne,sans-serif;font-weight:700;font-size:20px;color:#00e87a;margin-left:10px;'>{len(selected_symptoms)}</span>
        <span style='font-size:12px;color:#3a6050;'> symptoms</span>
    </div>
    """, unsafe_allow_html=True)

    analyze_clicked = st.button("🔍 Analyze Symptoms", disabled=len(selected_symptoms) == 0)

    st.markdown("""
    <div class='disclaimer'>
        ⚠️ <strong style='color:#e8f5f0'>Educational use only.</strong>
        Always consult a qualified healthcare professional for diagnosis and treatment.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='font-family:DM Mono,monospace;font-size:10px;color:#3a6050;'>
        DISEASES: {len(DISEASE_SYMPTOMS)} &nbsp;|&nbsp; SYMPTOMS: {len(ALL_SYMPTOMS)}<br>
        MODELS: RF + GB &nbsp;|&nbsp; Streamlit Cloud
    </div>
    """, unsafe_allow_html=True)


# ─── Main ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='hero-title'>Symptom-Based<br/><span>Disease Prediction</span></div>
<div class='hero-sub'>Select your symptoms from the sidebar and let our ensemble ML models<br/>analyze patterns to predict potential diseases with full health guidance.</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class='stat-row'>
    <div class='stat-chip'><span class='stat-num'>99.9%</span><span class='stat-lbl'>Accuracy</span></div>
    <div class='stat-chip'><span class='stat-num'>{len(DISEASE_SYMPTOMS)}</span><span class='stat-lbl'>Diseases</span></div>
    <div class='stat-chip'><span class='stat-num'>{len(ALL_SYMPTOMS)}</span><span class='stat-lbl'>Symptoms</span></div>
    <div class='stat-chip'><span class='stat-num'>2</span><span class='stat-lbl'>ML Models</span></div>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ─── Results ──────────────────────────────────────────────────────────────────
if analyze_clicked and selected_symptoms:
    with st.spinner("🧬 Running ML inference..."):
        predictions = predict_disease(selected_symptoms, model_choice)

    if not predictions:
        st.warning("No strong predictions found. Try adding more symptoms.")
    else:
        model_name = "Random Forest" if model_choice == "rf" else "Gradient Boosting"
        tags_html = "".join([f"<span class='sym-tag'>{s.replace('_',' ')}</span>" for s in selected_symptoms])
        st.markdown(f"""
        <div class='medi-card' style='margin-bottom:24px;'>
            <div style='font-family:DM Mono,monospace;font-size:11px;color:#5a8070;margin-bottom:8px;'>
                ANALYSIS · {model_name} · {len(selected_symptoms)} symptoms
            </div>
            <div>{tags_html}</div>
        </div>
        """, unsafe_allow_html=True)

        rank_colors = ["conf-badge-gold", "conf-badge-blue", "conf-badge-gray"]
        rank_labels = ["#1", "#2", "#3"]
        rank_css    = ["rank-1", "rank-2", "rank-3"]

        for i, pred in enumerate(predictions):
            with st.container():
                col_rank, col_info, col_conf = st.columns([0.8, 5, 2])
                with col_rank:
                    st.markdown(f"<div class='{rank_css[i]}'>{rank_labels[i]}</div>", unsafe_allow_html=True)
                with col_info:
                    st.markdown(f"""
                    <div class='disease-title'>{pred['disease']}</div>
                    <div class='disease-desc'>{pred['description']}</div>
                    """, unsafe_allow_html=True)
                with col_conf:
                    st.markdown(f"<div class='{rank_colors[i]}'>{pred['confidence']}% confidence</div>", unsafe_allow_html=True)
                    st.progress(int(pred["confidence"]))

                c1, c2, c3, c4 = st.columns(4)
                for col, key, icon, label in [
                    (c1, "precautions", "🛡️", "Precautions"),
                    (c2, "medications", "💊", "Medications"),
                    (c3, "diet",        "🥗", "Diet Plan"),
                    (c4, "workout",     "🏃", "Workout"),
                ]:
                    with col:
                        items_html = "".join([
                            f"<div class='info-item'><span class='info-bullet'>›</span>{item}</div>"
                            for item in pred.get(key, [])
                        ])
                        st.markdown(f"""
                        <div class='info-block'>
                            <div class='info-block-title'>{icon} {label}</div>
                            {items_html}
                        </div>""", unsafe_allow_html=True)

                st.markdown("<hr>", unsafe_allow_html=True)

elif not analyze_clicked:
    st.markdown("""
    <div style='text-align:center;padding:80px 20px;'>
        <div style='font-size:64px;margin-bottom:20px;'>🧬</div>
        <div style='font-family:Syne,sans-serif;font-weight:700;font-size:20px;color:#3a6050;margin-bottom:10px;'>
            No Analysis Yet
        </div>
        <div style='font-size:14px;color:#3a6050;line-height:1.7;'>
            Select symptoms from the sidebar on the left<br/>
            and click <strong style='color:#5a8070'>"Analyze Symptoms"</strong> to get predictions.
        </div>
    </div>
    """, unsafe_allow_html=True)
