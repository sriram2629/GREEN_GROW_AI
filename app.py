from flask import Flask, render_template, request, jsonify
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import torch
import torch.nn as nn
from PIL import Image
import io
import torchvision.transforms as transforms
from datetime import datetime
import requests

app = Flask(__name__)

# --- CONFIGURATION ---
# IMPORTANT: Replace "YOUR_API_KEY_HERE" with your actual OpenWeatherMap API key.
# The weather feature will not work without it.
OPENWEATHER_API_KEY = "795d0923724219bcfa69d64f03766bdd"

# --- DEEP LEARNING MODEL DEFINITION ---
# Load trained CNN model for waste classification
class WasteClassifierCNN(nn.Module):
    def __init__(self):
        super(WasteClassifierCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 25 * 25, 128)
        self.fc2 = nn.Linear(128, 2)  # Binary: Compostable (0), Non-Compostable (1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 25 * 25)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# --- MODEL LOADING ---
# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
waste_model = WasteClassifierCNN().to(device)
try:
    waste_model.load_state_dict(torch.load("waste_classifier.pth", map_location=device))
    waste_model.eval()
except FileNotFoundError:
    print("Warning: waste_classifier.pth not found. The waste classification module will not work.")

# --- TRANSLATIONS & STATIC DATA ---
# UI Translations Dictionary
ui_texts = {
    'en': {
        'title': 'Smart Farming Assistant',
        'welcome': 'GreenGrow AI 𓀚  𓃔𓃽',
        'bio_title': 'Biofertilizer Recommendation',
        'waste_title': 'Waste Image Classification',
        'compost_title': 'Compost Tracker',
        'soil_type': 'Soil Type',
        'crop_type': 'Crop Type',
        'get_rec': 'Get Recommendation',
        'upload_image': 'Choose File',
        'classify': 'Submit',
        'smell': 'Smell',
        'color': 'Color',
        'heat': 'Heat',
        'moisture': 'Moisture',
        'days': 'Days',
        'track_stage': 'Submit',
        'recommendation': 'Recommendation',
        'classification': 'Classification',
        'stage': 'Stage',
        'home': 'Home',
        'english': 'English',
        'telugu': 'Telugu',
        'hindi': 'Hindi',
        'fresh': 'Fresh', 'ammonia': 'Ammonia', 'earthy': 'Earthy',
        'mixed': 'Mixed', 'brown': 'Brown', 'dark': 'Dark',
        'low': 'Low', 'medium': 'Medium', 'high': 'High',
        'dry': 'Dry', 'moist': 'Moist', 'wet': 'Wet',
        'days_options': ['5', '10', '20', '40', '70'],
        'loading': 'Analyzing...',
        'try_again': 'Try Again',
        'not_suitable': '{crop} is not suitable for {soil}. Please select a different crop or soil.',
        'error_fetch': 'Error fetching result. Please try again.',
        'compost_process_title': 'The Composting Process',
        'initial_stage_title': 'Initial Stage',
        'initial_stage_desc': 'This is the preparatory phase where organic materials are gathered. Microorganisms consume readily available sugars and amino acids, generating a small amount of heat.',
        'mesophilic_stage_title': 'Mesophilic Stage',
        'mesophilic_stage_desc': 'Lasting a few days to a week, this stage involves mesophilic microorganisms thriving at 20°C to 45°C (68°F to 113°F). They break down soluble compounds, causing the pile to heat up.',
        'thermophilic_stage_title': 'Thermophilic Stage',
        'thermophilic_stage_desc': 'The most active phase, lasting weeks to months, with thermophilic microorganisms thriving at 45°C to 75°C (113°F to 167°F). Rapid decomposition occurs, and temperatures above 55°C kill pathogens and weed seeds. Regular aeration is essential.',
        'cooling_stage_title': 'Cooling Stage',
        'cooling_stage_desc': 'As complex compounds are broken down, microbial activity slows, and the pile cools below 45°C. Mesophilic microorganisms recolonize and continue decomposition at a slower pace.',
        'mature_stage_title': 'Mature Stage (Curing)',
        'mature_stage_desc': 'This months-long phase stabilizes the compost into humus, with a neutral pH and nutrients in plant-available forms. The compost becomes dark, crumbly, and earthy-smelling.',
        'weather_title': "Today's Weather & Farming Alerts",
        'fetching_location': 'Fetching your location for a local forecast...',
        'location_denied': 'Location access denied. Please enable it in your browser to see the weather forecast.',
        'humidity': 'Humidity',
        'wind': 'Wind',
        'forecast_title': '5-Day Forecast',
        'alert_rain': 'Rain is expected. Consider delaying spraying or irrigation.',
        'alert_heat': 'High temperatures expected. Ensure crops are well-hydrated.',
        'alert_wind': 'Strong winds forecasted. Young or tall plants may need support.',
        'alert_clear': 'Conditions look good for farming activities.',
    },
    'te': {
        'title': 'స్మార్ట్ ఫార్మింగ్ అసిస్టెంట్',
        'welcome': 'గ్రీన్ గ్రో ఏఐ 𓀚  𓃔𓃽',
        'bio_title': 'బయోఫెర్టిలైజర్ సిఫార్సు',
        'waste_title': 'వ్యర్థాల వర్గీకరణ',
        'compost_title': 'కంపోస్ట్ ట్రాకర్',
        'soil_type': 'మట్టి రకం',
        'crop_type': 'పంట రకం',
        'get_rec': 'సిఫార్సు పొందండి',
        'upload_image': 'ఫైల్ ఎంచుకోండి',
        'classify': 'సమర్పించండి',
        'smell': 'వాసన',
        'color': 'రంగు',
        'heat': 'వేడి',
        'moisture': 'తేమ',
        'days': 'రోజులు',
        'track_stage': 'సమర్పించండి',
        'recommendation': 'సిఫార్సు',
        'classification': 'వర్గీకరణ',
        'stage': 'దశ',
        'home': 'హోమ్',
        'english': 'ఇంగ్లీష్',
        'telugu': 'తెలుగు',
        'hindi': 'హిందీ',
        'fresh': 'తాజా', 'ammonia': 'అమ్మోనియా', 'earthy': 'మట్టి వాసన',
        'mixed': 'మిశ్రమం', 'brown': 'గోధుమ', 'dark': 'ముదురు',
        'low': 'తక్కువ', 'medium': 'మధ్యస్థం', 'high': 'అధికం',
        'dry': 'పొడి', 'moist': 'తేమ', 'wet': 'తడి',
        'days_options': ['5', '10', '20', '40', '70'],
        'loading': 'విశ్లేషిస్తోంది...',
        'try_again': 'మళ్లీ ప్రయత్నించండి',
        'not_suitable': '{crop} {soil} కి సరిపోదు. దయచేసి వేరే పంట లేదా మట్టిని ఎంచుకోండి.',
        'error_fetch': 'ఫలితం పొందడంలో లోపం. దయచేసి మళ్లీ ప్రయత్నించండి.',
        'compost_process_title': 'కంపోస్టింగ్ ప్రక్రియ',
        'initial_stage_title': 'ప్రారంభ దశ',
        'initial_stage_desc': 'ఇది సేంద్రీయ పదార్థాలను సేకరించిన వెంటనే ప్రారంభమయ్యే సిద్ధపరిచే దశ. సులభంగా అందుబాటులో ఉన్న షుగర్లు మరియు అమైనో ఆమ్లాలను సూక్ష్మజీవులు వేగంగా వినియోగిస్తాయి, తక్కువ మొత్తంలో వేడిని ఉత్పత్తి చేస్తాయి.',
        'mesophilic_stage_title': 'మెసోఫిలిక్ దశ',
        'mesophilic_stage_desc': 'కొన్ని రోజుల నుండి ఒక వారం వరకు ఉండే ఈ దశలో, 20°C నుండి 45°C (68°F నుండి 113°F) వద్ద వృద్ధి చెందే మెసోఫిలిక్ సూక్ష్మజీవులు ఆధిపత్యం వహిస్తాయి. అవి సులభంగా క్షీణించే సమ్మేళనాలను విచ్ఛిన్నం చేస్తాయి, దీనివల్ల కుప్ప వేడెక్కుతుంది.',
        'thermophilic_stage_title': 'థర్మోఫిలిక్ దశ',
        'thermophilic_stage_desc': 'కొన్ని వారాల నుండి నెలల వరకు ఉండే అత్యంత చురుకైన దశ, 45°C నుండి 75°C (113°F నుండి 167°F) వద్ద వృద్ధి చెందే థర్మోఫిలిక్ సూక్ష్మజీవులు ఆధిపత్యం వహిస్తాయి. వేగవంతమైన క్షీణనం జరుగుతుంది, మరియు 55°C పైన ఉష్ణోగ్రతలు రోగకారకాలను మరియు కలుపు విత్తనాలను నాశనం చేస్తాయి. సాధారణ గాలి ప్రసరణ అవసరం.',
        'cooling_stage_title': 'కూలింగ్ దశ',
        'cooling_stage_desc': 'సంక్లిష్ట సమ్మేళనాలు విచ్ఛిన్నమైన తర్వాత, సూక్ష్మజీవుల కార్యకలాపం మందగిస్తుంది, మరియు కుప్ప 45°C కంటే తక్కువకు చల్లబడుతుంది. మెసోఫిలిక్ సూక్ష్మజీవులు తిరిగి స్థిరపడతాయి మరియు నెమ్మదిగా క్షీణనం కొనసాగిస్తాయి.',
        'mature_stage_title': 'పరిపక్వ దశ (క్యూరింగ్)',
        'mature_stage_desc': 'ఈ నెలల పాటు కొనసాగే దశలో కంపోస్ట్ హ్యూమస్‌గా స్థిరపడుతుంది, తటస్థ pH మరియు మొక్కలకు అందుబాటులో ఉండే పోషకాలతో. కంపోస్ట్ చీకటిగా, మెత్తగా, మరియు మట్టి వాసనతో ఉంటుంది.',
        'weather_title': 'నేటి వాతావరణం & వ్యవసాయ హెచ్చరికలు',
        'fetching_location': 'మీ స్థానిక సూచన కోసం మీ స్థానాన్ని పొందుతోంది...',
        'location_denied': 'స్థాన యాక్సెస్ నిరాకరించబడింది. వాతావరణ సూచనను చూడటానికి దయచేసి మీ బ్రౌజర్‌లో దీన్ని ప్రారంభించండి.',
        'humidity': 'తేమ',
        'wind': 'గాలి',
        'forecast_title': '5-రోజుల సూచన',
        'alert_rain': 'వర్షం కురిసే అవకాశం ఉంది. స్ప్రేయింగ్ లేదా నీటిపారుదలని వాయిదా వేయండి.',
        'alert_heat': 'అధిక ఉష్ణోగ్రతలు ఉండే అవకాశం ఉంది. పంటలకు తగినంత నీరు అందేలా చూసుకోండి.',
        'alert_wind': 'బలమైన గాలులు వీచే అవకాశం ఉంది. లేత లేదా పొడవాటి మొక్కలకు మద్దతు అవసరం కావచ్చు.',
        'alert_clear': 'వ్యవసాయ పనులకు వాతావరణం అనుకూలంగా ఉంది.',
    },
    'hi': {
        'title': 'स्मार्ट फार्मिंग असिस्टेंट',
        'welcome': 'ग्रीन ग्रो एआई 𓀚  𓃔𓃽',
        'bio_title': 'बायोफर्टिलाइजर अनुशंसा',
        'waste_title': 'अपशिष्ट छवि वर्गीकरण',
        'compost_title': 'खाद ट्रैकर',
        'soil_type': 'मिट्टी का प्रकार',
        'crop_type': 'फसल का प्रकार',
        'get_rec': 'अनुशंसा प्राप्त करें',
        'upload_image': 'फ़ाइल चुनें',
        'classify': 'जमा करें',
        'smell': 'गंध',
        'color': 'रंग',
        'heat': 'गर्मी',
        'moisture': 'नमी',
        'days': 'दिन',
        'track_stage': 'जमा करें',
        'recommendation': 'अनुशंसा',
        'classification': 'वर्गीकरण',
        'stage': 'चरण',
        'home': 'होम',
        'english': 'अंग्रेजी',
        'telugu': 'తెలుగు',
        'hindi': 'हिंदी',
        'fresh': 'ताजा', 'ammonia': 'अमोनिया', 'earthy': 'मिट्टी जैसी',
        'mixed': 'मिश्रित', 'brown': 'भूरा', 'dark': 'गहरा',
        'low': 'कम', 'medium': 'मध्यम', 'high': 'उच्च',
        'dry': 'सूखा', 'moist': 'नम', 'wet': 'गीला',
        'days_options': ['5', '10', '20', '40', '70'],
        'loading': 'विश्लेषण हो रहा है...',
        'try_again': 'पुनः प्रयास करें',
        'not_suitable': '{crop} {soil} के लिए उपयुक्त नहीं है। कृपया एक अलग फसल या मिट्टी चुनें।',
        'error_fetch': 'परिणाम प्राप्त करने में त्रुटि। कृपया पुनः प्रयास करें।',
        'compost_process_title': 'खाद बनाने की प्रक्रिया',
        'initial_stage_title': 'प्रारंभिक चरण',
        'initial_stage_desc': 'यह वह प्रारंभिक चरण है जहां जैविक सामग्री एकत्र की जाती है। सूक्ष्मजीव आसानी से उपलब्ध शर्करा और अमीनो एसिड का उपभोग करते हैं, जिससे थोड़ी मात्रा में गर्मी उत्पन्न होती है।',
        'mesophilic_stage_title': 'मेसोफिलिक चरण',
        'mesophilic_stage_desc': 'कुछ दिनों से एक सप्ताह तक चलने वाला यह चरण 20°C से 45°C (68°F से 113°F) पर पनपने वाले मेसोफिलिक सूक्ष्मजीवों द्वारा प्रभुत्व रखता है। वे आसानी से विघटित होने वाले यौगिकों को तोड़ते हैं, जिससे ढेर गर्म होता है।',
        'thermophilic_stage_title': 'थर्मोफिलिक चरण',
        'thermophilic_stage_desc': 'सबसे सक्रिय चरण, जो हफ्तों से महीनों तक चलता है, जिसमें 45°C से 75°C (113°F से 167°F) पर पनपने वाले थर्मोफिलिक सूक्ष्मजीव प्रभुत्व रखते हैं। तीव्र विघटन होता है, और 55°C से ऊपर का तापमान रोगजनकों और खरपतवार के बीजों को नष्ट करता है। नियमित वातन आवश्यक है।',
        'cooling_stage_title': 'कूलिंग चरण',
        'cooling_stage_desc': 'जटिल यौगिकों के विघटित होने के बाद, सूक्ष्मजीवी गतिविधि धीमी हो जाती है, और ढेर 45°C से नीचे ठंडा हो जाता है। मेसोफिलिक सूक्ष्मजीव फिर से बस जाते हैं और धीमी गति से विघटन जारी रखते हैं।',
        'mature_stage_title': 'परिपक्व चरण (क्योरिंग)',
        'mature_stage_desc': 'यह महीनों तक चलने वाला चरण खाद को ह्यूमस में स्थिर करता है, जिसमें तटस्थ pH और पौधों के लिए उपलब्ध पोषक तत्व होते हैं। खाद गहरा, टुकड़े-टुकड़े, और मिट्टी की गंध वाला होता है।',
        'weather_title': 'आज का मौसम और कृषि अलर्ट',
        'fetching_location': 'स्थानीय पूर्वानुमान के लिए आपका स्थान प्राप्त किया जा रहा है...',
        'location_denied': 'स्थान पहुंच से इनकार कर दिया गया। मौसम पूर्वानुमान देखने के लिए कृपया इसे अपने ब्राउज़र में सक्षम करें।',
        'humidity': 'नमी',
        'wind': 'हवा',
        'forecast_title': '5-दिन का पूर्वानुमान',
        'alert_rain': 'बारिश की उम्मीद है। छिड़काव या सिंचाई में देरी करने पर विचार करें।',
        'alert_heat': 'उच्च तापमान की उम्मीद है। सुनिश्चित करें कि फसलें अच्छी तरह से हाइड्रेटेड हैं।',
        'alert_wind': 'तेज हवाओं का पूर्वानुमान। युवा या लंबे पौधों को सहारे की आवश्यकता हो सकती है।',
        'alert_clear': 'कृषि गतिविधियों के लिए स्थितियाँ अच्छी दिख रही हैं।',
    }
}

# English names are used as keys in the suitability_map
indian_soils_en = ['Alluvial', 'Black', 'Red', 'Laterite', 'Arid', 'Forest & Mountain', 'Desert', 'Saline & Alkaline', 'Peaty & Marshy', 'Sandy', 'Clay', 'Loamy', 'Calcareous', 'Acidic', 'Ferruginous', 'Regur', 'Podzol', 'Terra Rossa', 'Humus-rich', 'Khadar']
indian_crops_en = ['Rice', 'Wheat', 'Maize', 'Sugarcane', 'Cotton', 'Jute', 'Tea', 'Coffee', 'Pulses', 'Groundnut', 'Millets', 'Mustard', 'Potato', 'Onion', 'Tomato', 'Chilli', 'Turmeric', 'Ginger', 'Banana', 'Mango']

# Translation dictionary for soils and crops
soil_crop_trans = {
    'en': {'soils': indian_soils_en, 'crops': indian_crops_en},
    'te': {
        'soils': ['ఒండ్రు', 'నల్ల రేగడి', 'ఎర్ర', 'లేటరైట్', 'శుష్క', 'అటవీ & పర్వత', 'ఎడారి', 'క్షార & ఆల్కలీన్', 'పీటీ & చిత్తడి', 'ఇసుక', 'బంకమట్టి', 'లోమీ', 'సున్నపు', 'ఆమ్ల', 'ఫెర్రుగినస్', 'రేగడి', 'పోడ్జోల్', 'టెర్రా రోసా', 'హ్యూమస్-రిచ్', 'ఖదర్'],
        'crops': ['వరి', 'గోధుమ', 'మొక్కజొన్న', 'చెరకు', 'పత్తి', 'జనపనార', 'తేయాకు', 'కాఫీ', 'పప్పుధాన్యాలు', 'వేరుశనగ', 'చిరుధాన్యాలు', 'ఆవాలు', 'బంగాళదుంప', 'ఉల్లిపాయ', 'టమోటా', 'మిరప', 'పసుపు', 'అల్లం', 'అరటి', 'మామిడి']
    },
    'hi': {
        'soils': ['जलोढ़', 'काली', 'लाल', 'लैटेराइट', 'शुष्क', 'वन और पर्वतीय', 'मरुस्थलीय', 'लवणीय और क्षारीय', 'पीट और दलदली', 'रेतीली', 'चिकनी', 'दोमट', 'कैल्केरियस', 'अम्लीय', 'लौहमय', 'रेगुर', 'पॉडज़ोल', 'टेरा रोसा', 'ह्यूमस-युक्त', 'खादर'],
        'crops': ['चावल', 'गेहूं', 'मक्का', 'गन्ना', 'कपास', 'जूट', 'चाय', 'कॉफी', 'दालें', 'मूंगफली', 'बाजरा', 'सरसों', 'आलू', 'प्याज', 'टमाटर', 'मिर्च', 'हल्दी', 'अदरक', 'केला', 'आम']
    }
}

# Create reverse maps for translation lookup
te_soils_to_en = {te: en for en, te in zip(soil_crop_trans['en']['soils'], soil_crop_trans['te']['soils'])}
hi_soils_to_en = {hi: en for en, hi in zip(soil_crop_trans['en']['soils'], soil_crop_trans['hi']['soils'])}
te_crops_to_en = {te: en for en, te in zip(soil_crop_trans['en']['crops'], soil_crop_trans['te']['crops'])}
hi_crops_to_en = {hi: en for en, hi in zip(soil_crop_trans['en']['crops'], soil_crop_trans['hi']['crops'])}

# Biofertilizer recommendation lookup table
suitability_map = {
    'Alluvial': {
        'Rice': ('Suitable', ['Azospirillum', 'Phosphate Solubilizing Bacteria', 'Blue-Green Algae', 'Azolla']),
        'Wheat': ('Suitable', ['Azotobacter', 'Azospirillum', 'Phosphate Solubilizing Bacteria']),
        'Maize': ('Suitable', ['Azospirillum', 'Phosphate Solubilizing Bacteria']),
        'Sugarcane': ('Suitable', ['Acetobacter', 'Azospirillum', 'Phosphate Solubilizing Bacteria', 'Potassium Solubilizing Bacteria']),
        'Cotton': ('Suitable', ['Azotobacter', 'Phosphate Solubilizing Bacteria', 'Arbuscular Mycorrhizal Fungi']),
        'Jute': ('Suitable', ['Azotobacter', 'Phosphate Solubilizing Bacteria']),
        'Tea': ('Not Suitable', []), 'Coffee': ('Not Suitable', []),
        'Pulses': ('Suitable', ['Rhizobium', 'Phosphate Solubilizing Bacteria']),
        'Groundnut': ('Suitable', ['Rhizobium', 'Phosphate Solubilizing Bacteria']),
        'Millets': ('Suitable', ['Azospirillum', 'Phosphate Solubilizing Bacteria']),
        'Mustard': ('Suitable', ['Azotobacter', 'Phosphate Solubilizing Bacteria']),
        'Potato': ('Suitable', ['Azotobacter', 'Plant Growth Promoting Rhizobacteria', 'Potassium Solubilizing Bacteria']),
        'Onion': ('Suitable', ['Azotobacter', 'Phosphate Solubilizing Bacteria', 'Arbuscular Mycorrhizal Fungi']),
        'Tomato': ('Suitable', ['Azotobacter', 'Phosphate Solubilizing Bacteria', 'Potassium Solubilizing Bacteria']),
        'Chilli': ('Suitable', ['Azotobacter', 'Azospirillum', 'Phosphate Solubilizing Bacteria']),
        'Turmeric': ('Suitable', ['Azotobacter', 'Phosphate Solubilizing Bacteria']),
        'Ginger': ('Suitable', ['Azotobacter', 'Phosphate Solubilizing Bacteria', 'Potassium Solubilizing Bacteria']),
        'Banana': ('Suitable', ['Phosphate Solubilizing Bacteria', 'Potassium Solubilizing Bacteria', 'Arbuscular Mycorrhizal Fungi']),
        'Mango': ('Suitable', ['Azotobacter', 'Phosphate Solubilizing Bacteria', 'Arbuscular Mycorrhizal Fungi'])
    },
    'Black': {
        'Rice': ('Moderately suitable', ['Azospirillum', 'Phosphate Solubilizing Bacteria', 'Zinc Solubilizing Bacteria']),
        'Wheat': ('Suitable', ['Azotobacter', 'Phosphate Solubilizing Bacteria']),
        'Maize': ('Suitable', ['Azospirillum', 'Phosphate Solubilizing Bacteria']),
        'Sugarcane': ('Suitable', ['Acetobacter', 'Azospirillum', 'Phosphate Solubilizing Bacteria']),
        'Cotton': ('Suitable', ['Azotobacter', 'Phosphate Solubilizing Bacteria', 'Arbuscular Mycorrhizal Fungi']),
        'Jute': ('Not Suitable', []), 'Tea': ('Not Suitable', []), 'Coffee': ('Not Suitable', []),
        'Pulses': ('Suitable', ['Rhizobium', 'Phosphate Solubilizing Bacteria']),
        'Groundnut': ('Suitable', ['Rhizobium', 'Phosphate Solubilizing Bacteria']),
        'Millets': ('Suitable', ['Azospirillum', 'Phosphate Solubilizing Bacteria']),
        'Mustard': ('Less suitable', ['Azotobacter', 'Phosphate Solubilizing Bacteria']),
        'Potato': ('Less suitable', ['Azotobacter', 'Plant Growth Promoting Rhizobacteria', 'Potassium Solubilizing Bacteria']),
        'Onion': ('Less suitable', ['Azotobacter', 'Phosphate Solubilizing Bacteria', 'Arbuscular Mycorrhizal Fungi']),
        'Tomato': ('Suitable', ['Azotobacter', 'Phosphate Solubilizing Bacteria', 'Potassium Solubilizing Bacteria']),
        'Chilli': ('Suitable', ['Azotobacter', 'Azospirillum', 'Phosphate Solubilizing Bacteria']),
        'Turmeric': ('Less suitable', ['Azotobacter', 'Phosphate Solubilizing Bacteria']),
        'Ginger': ('Less suitable', ['Azotobacter', 'Phosphate Solubilizing Bacteria', 'Potassium Solubilizing Bacteria']),
        'Banana': ('Suitable', ['Phosphate Solubilizing Bacteria', 'Potassium Solubilizing Bacteria']),
        'Mango': ('Moderately suitable', ['Azotobacter', 'Phosphate Solubilizing Bacteria', 'Arbuscular Mycorrhizal Fungi'])
    },
    'Red': {
        'Rice': ('Suitable', ['Azospirillum', 'Phosphate Solubilizing Bacteria', 'Blue-Green Algae']),
        'Wheat': ('Less suitable', ['Azotobacter', 'Phosphate Solubilizing Bacteria']),
        'Maize': ('Suitable', ['Azospirillum', 'Phosphate Solubilizing Bacteria']),
        'Sugarcane': ('Less suitable', ['Azospirillum', 'Acetobacter', 'Phosphate Solubilizing Bacteria']),
        'Cotton': ('Less suitable', ['Azotobacter', 'Phosphate Solubilizing Bacteria', 'Arbuscular Mycorrhizal Fungi']),
        'Jute': ('Not Suitable', []),
        'Tea': ('Moderately suitable', ['Azotobacter', 'Phosphate Solubilizing Bacteria', 'Arbuscular Mycorrhizal Fungi']),
        'Coffee': ('Moderately suitable', ['Azotobacter', 'Phosphate Solubilizing Bacteria', 'Arbuscular Mycorrhizal Fungi']),
        'Pulses': ('Suitable', ['Rhizobium', 'Phosphate Solubilizing Bacteria']),
        'Groundnut': ('Suitable', ['Rhizobium', 'Phosphate Solubilizing Bacteria']),
        'Millets': ('Suitable', ['Azospirillum', 'Phosphate Solubilizing Bacteria']),
        'Mustard': ('Suitable', ['Azotobacter', 'Phosphate Solubilizing Bacteria']),
        'Potato': ('Suitable', ['Azotobacter', 'Plant Growth Promoting Rhizobacteria', 'Potassium Solubilizing Bacteria']),
        'Onion': ('Suitable', ['Azotobacter', 'Phosphate Solubilizing Bacteria', 'Arbuscular Mycorrhizal Fungi']),
        'Tomato': ('Suitable', ['Azotobacter', 'Phosphate Solubilizing Bacteria', 'Potassium Solubilizing Bacteria']),
        'Chilli': ('Suitable', ['Azotobacter', 'Azospirillum', 'Phosphate Solubilizing Bacteria']),
        'Turmeric': ('Suitable', ['Azotobacter', 'Pseudomonas', 'Phosphate Solubilizing Bacteria']),
        'Ginger': ('Suitable', ['Azotobacter', 'Phosphate Solubilizing Bacteria', 'Potassium Solubilizing Bacteria']),
        'Banana': ('Moderately suitable', ['Phosphate Solubilizing Bacteria', 'Potassium Solubilizing Bacteria', 'Arbuscular Mycorrhizal Fungi']),
        'Mango': ('Suitable', ['Azotobacter', 'Phosphate Solubilizing Bacteria', 'Arbuscular Mycorrhizal Fungi'])
    },
    'Laterite': {
        'Rice': ('Less suitable', ['Azospirillum', 'Phosphate Solubilizing Bacteria']),
        'Wheat': ('Not Suitable', []), 'Maize': ('Less suitable', ['Azospirillum', 'Phosphate Solubilizing Bacteria', 'Arbuscular Mycorrhizal Fungi']),
        'Sugarcane': ('Not Suitable', []), 'Cotton': ('Not Suitable', []), 'Jute': ('Not Suitable', []),
        'Tea': ('Suitable', ['Azotobacter', 'Phosphate Solubilizing Bacteria', 'Arbuscular Mycorrhizal Fungi']),
        'Coffee': ('Suitable', ['Azotobacter', 'Phosphate Solubilizing Bacteria', 'Arbuscular Mycorrhizal Fungi']),
        'Pulses': ('Less suitable', ['Rhizobium', 'Phosphate Solubilizing Bacteria']),
        'Groundnut': ('Less suitable', ['Rhizobium', 'Phosphate Solubilizing Bacteria']),
        'Millets': ('Suitable', ['Azospirillum', 'Phosphate Solubilizing Bacteria', 'Arbuscular Mycorrhizal Fungi']),
        'Mustard': ('Not Suitable', []),
        'Potato': ('Moderately suitable', ['Azotobacter', 'Plant Growth Promoting Rhizobacteria']),
        'Onion': ('Not Suitable', []),
        'Tomato': ('Less suitable', ['Azotobacter', 'Phosphate Solubilizing Bacteria', 'Arbuscular Mycorrhizal Fungi']),
        'Chilli': ('Less suitable', ['Azotobacter', 'Azospirillum', 'Phosphate Solubilizing Bacteria']),
        'Turmeric': ('Less suitable', ['Azotobacter', 'Phosphate Solubilizing Bacteria']),
        'Ginger': ('Less suitable', ['Azotobacter', 'Phosphate Solubilizing Bacteria']),
        'Banana': ('Less suitable', ['Phosphate Solubilizing Bacteria', 'Potassium Solubilizing Bacteria', 'Arbuscular Mycorrhizal Fungi']),
        'Mango': ('Suitable', ['Azotobacter', 'Phosphate Solubilizing Bacteria', 'Arbuscular Mycorrhizal Fungi'])
    },
    'Arid': {
        'Rice': ('Not Suitable', []),
        'Wheat': ('Less suitable', ['Azotobacter', 'Plant Growth Promoting Rhizobacteria', 'Phosphate Solubilizing Bacteria']),
        'Maize': ('Less suitable', ['Azospirillum', 'Plant Growth Promoting Rhizobacteria', 'Phosphate Solubilizing Bacteria']),
        'Sugarcane': ('Not Suitable', []),
        'Cotton': ('Less suitable', ['Azotobacter', 'Plant Growth Promoting Rhizobacteria', 'Phosphate Solubilizing Bacteria']),
        'Jute': ('Not Suitable', []), 'Tea': ('Not Suitable', []), 'Coffee': ('Not Suitable', []),
        'Pulses': ('Suitable', ['Rhizobium', 'Phosphate Solubilizing Bacteria', 'Plant Growth Promoting Rhizobacteria']),
        'Groundnut': ('Less suitable', ['Rhizobium', 'Phosphate Solubilizing Bacteria']),
        'Millets': ('Suitable', ['Azospirillum', 'Phosphate Solubilizing Bacteria', 'Plant Growth Promoting Rhizobacteria']),
        'Mustard': ('Suitable', ['Azotobacter', 'Phosphate Solubilizing Bacteria']),
        'Potato': ('Not Suitable', []),
        'Onion': ('Less suitable', ['Azotobacter', 'Phosphate Solubilizing Bacteria']),
        'Tomato': ('Less suitable', ['Azotobacter', 'Phosphate Solubilizing Bacteria', 'Plant Growth Promoting Rhizobacteria']),
        'Chilli': ('Less suitable', ['Azotobacter', 'Azospirillum', 'Phosphate Solubilizing Bacteria']),
        'Turmeric': ('Not Suitable', []), 'Ginger': ('Not Suitable', []), 'Banana': ('Not Suitable', []),
        'Mango': ('Less suitable', ['Azotobacter', 'Phosphate Solubilizing Bacteria', 'Arbuscular Mycorrhizal Fungi'])
    },
    'Forest & Mountain': {
        'Rice': ('Less suitable', ['Azotobacter', 'Phosphate Solubilizing Bacteria']),
        'Wheat': ('Less suitable', ['Azotobacter', 'Phosphate Solubilizing Bacteria']),
        'Maize': ('Suitable', ['Azospirillum', 'Phosphate Solubilizing Bacteria']),
        'Sugarcane': ('Not Suitable', []), 'Cotton': ('Not Suitable', []), 'Jute': ('Not Suitable', []),
        'Tea': ('Suitable', ['Azotobacter', 'Phosphate Solubilizing Bacteria', 'Arbuscular Mycorrhizal Fungi']),
        'Coffee': ('Suitable', ['Azotobacter', 'Phosphate Solubilizing Bacteria', 'Arbuscular Mycorrhizal Fungi']),
        'Pulses': ('Suitable', ['Rhizobium', 'Phosphate Solubilizing Bacteria']),
        'Groundnut': ('Not Suitable', []),
        'Millets': ('Suitable', ['Azospirillum', 'Phosphate Solubilizing Bacteria']),
        'Mustard': ('Less suitable', ['Azotobacter', 'Phosphate Solubilizing Bacteria']),
        'Potato': ('Suitable', ['Azotobacter', 'Plant Growth Promoting Rhizobacteria', 'Arbuscular Mycorrhizal Fungi']),
        'Onion': ('Less suitable', ['Azotobacter', 'Phosphate Solubilizing Bacteria', 'Arbuscular Mycorrhizal Fungi']),
        'Tomato': ('Suitable', ['Azotobacter', 'Phosphate Solubilizing Bacteria', 'Potassium Solubilizing Bacteria']),
        'Chilli': ('Suitable', ['Azotobacter', 'Azospirillum', 'Phosphate Solubilizing Bacteria']),
        'Turmeric': ('Suitable', ['Azotobacter', 'Pseudomonas', 'Phosphate Solubilizing Bacteria']),
        'Ginger': ('Suitable', ['Azotobacter', 'Phosphate Solubilizing Bacteria', 'Potassium Solubilizing Bacteria']),
        'Banana': ('Not Suitable', []),
        'Mango': ('Less suitable', ['Azotobacter', 'Phosphate Solubilizing Bacteria', 'Arbuscular Mycorrhizal Fungi'])
    },
    'Desert': {
        'Rice': ('Not Suitable', []), 'Wheat': ('Not Suitable', []), 'Maize': ('Not Suitable', []),
        'Sugarcane': ('Not Suitable', []), 'Cotton': ('Not Suitable', []), 'Jute': ('Not Suitable', []),
        'Tea': ('Not Suitable', []), 'Coffee': ('Not Suitable', []),
        'Pulses': ('Less suitable', ['Rhizobium', 'Plant Growth Promoting Rhizobacteria']),
        'Groundnut': ('Not Suitable', []),
        'Millets': ('Suitable', ['Azospirillum', 'Plant Growth Promoting Rhizobacteria', 'Phosphate Solubilizing Bacteria']),
        'Mustard': ('Less suitable', ['Azotobacter', 'Phosphate Solubilizing Bacteria', 'Plant Growth Promoting Rhizobacteria']),
        'Potato': ('Not Suitable', []), 'Onion': ('Not Suitable', []), 'Tomato': ('Not Suitable', []),
        'Chilli': ('Not Suitable', []), 'Turmeric': ('Not Suitable', []), 'Ginger': ('Not Suitable', []),
        'Banana': ('Not Suitable', []), 'Mango': ('Not Suitable', [])
    },
    'Saline & Alkaline': {
        'Rice': ('Less suitable', ['Blue-Green Algae', 'Azospirillum', 'Phosphate Solubilizing Bacteria']),
        'Wheat': ('Less suitable', ['Azotobacter', 'Plant Growth Promoting Rhizobacteria', 'Phosphate Solubilizing Bacteria']),
        'Maize': ('Not Suitable', []),
        'Sugarcane': ('Less suitable', ['Azospirillum', 'Phosphate Solubilizing Bacteria']),
        'Cotton': ('Less suitable', ['Azotobacter', 'Plant Growth Promoting Rhizobacteria', 'Phosphate Solubilizing Bacteria']),
        'Jute': ('Not Suitable', []), 'Tea': ('Not Suitable', []), 'Coffee': ('Not Suitable', []),
        'Pulses': ('Not Suitable', []), 'Groundnut': ('Not Suitable', []),
        'Millets': ('Less suitable', ['Azospirillum', 'Plant Growth Promoting Rhizobacteria', 'Phosphate Solubilizing Bacteria']),
        'Mustard': ('Less suitable', ['Azotobacter', 'Phosphate Solubilizing Bacteria']),
        'Potato': ('Not Suitable', []), 'Onion': ('Not Suitable', []), 'Tomato': ('Not Suitable', []),
        'Chilli': ('Not Suitable', []), 'Turmeric': ('Not Suitable', []), 'Ginger': ('Not Suitable', []),
        'Banana': ('Not Suitable', []), 'Mango': ('Not Suitable', [])
    },
    'Peaty & Marshy': {
        'Rice': ('Suitable', ['Blue-Green Algae', 'Azolla', 'Azospirillum', 'Phosphate Solubilizing Bacteria']),
        'Wheat': ('Not Suitable', []), 'Maize': ('Not Suitable', []), 'Sugarcane': ('Not Suitable', []),
        'Cotton': ('Not Suitable', []),
        'Jute': ('Suitable', ['Azotobacter', 'Phosphate Solubilizing Bacteria']),
        'Tea': ('Not Suitable', []), 'Coffee': ('Not Suitable', []), 'Pulses': ('Not Suitable', []),
        'Groundnut': ('Not Suitable', []), 'Millets': ('Not Suitable', []), 'Mustard': ('Not Suitable', []),
        'Potato': ('Not Suitable', []), 'Onion': ('Not Suitable', []), 'Tomato': ('Not Suitable', []),
        'Chilli': ('Not Suitable', []), 'Turmeric': ('Not Suitable', []), 'Ginger': ('Not Suitable', []),
        'Banana': ('Not Suitable', []), 'Mango': ('Not Suitable', [])
    },
    'Sandy': {
        'Rice': ('Not Suitable', []),
        'Wheat': ('Less suitable', ['Azotobacter', 'Phosphate Solubilizing Bacteria']),
        'Maize': ('Less suitable', ['Azospirillum', 'Phosphate Solubilizing Bacteria', 'Arbuscular Mycorrhizal Fungi']),
        'Sugarcane': ('Not Suitable', []),
        'Cotton': ('Less suitable', ['Azotobacter', 'Phosphate Solubilizing Bacteria', 'Arbuscular Mycorrhizal Fungi']),
        'Jute': ('Not Suitable', []), 'Tea': ('Not Suitable', []), 'Coffee': ('Not Suitable', []),
        'Pulses': ('Less suitable', ['Rhizobium', 'Phosphate Solubilizing Bacteria', 'Arbuscular Mycorrhizal Fungi']),
        'Groundnut': ('Suitable', ['Rhizobium', 'Phosphate Solubilizing Bacteria']),
        'Millets': ('Suitable', ['Azospirillum', 'Phosphate Solubilizing Bacteria']),
        'Mustard': ('Less suitable', ['Azotobacter', 'Phosphate Solubilizing Bacteria']),
        'Potato': ('Suitable', ['Azotobacter', 'Plant Growth Promoting Rhizobacteria']),
        'Onion': ('Less suitable', ['Azotobacter', 'Phosphate Solubilizing Bacteria', 'Arbuscular Mycorrhizal Fungi']),
        'Tomato': ('Less suitable', ['Azotobacter', 'Phosphate Solubilizing Bacteria', 'Arbuscular Mycorrhizal Fungi']),
        'Chilli': ('Less suitable', ['Azotobacter', 'Azospirillum', 'Phosphate Solubilizing Bacteria']),
        'Turmeric': ('Not Suitable', []), 'Ginger': ('Not Suitable', []), 'Banana': ('Not Suitable', []),
        'Mango': ('Less suitable', ['Azotobacter', 'Phosphate Solubilizing Bacteria', 'Arbuscular Mycorrhizal Fungi'])
    },
    'Clay': {
        'Rice': ('Suitable', ['Azospirillum', 'Phosphate Solubilizing Bacteria', 'Blue-Green Algae', 'Azolla', 'Zinc Solubilizing Bacteria']),
        'Wheat': ('Moderately suitable', ['Azotobacter', 'Phosphate Solubilizing Bacteria']),
        'Maize': ('Less suitable', ['Azospirillum', 'Phosphate Solubilizing Bacteria']),
        'Sugarcane': ('Suitable', ['Acetobacter', 'Azospirillum', 'Phosphate Solubilizing Bacteria']),
        'Cotton': ('Suitable', ['Azotobacter', 'Phosphate Solubilizing Bacteria']),
        'Jute': ('Suitable', ['Azotobacter', 'Phosphate Solubilizing Bacteria']),
        'Tea': ('Not Suitable', []), 'Coffee': ('Not Suitable', []),
        'Pulses': ('Less suitable', ['Rhizobium', 'Phosphate Solubilizing Bacteria']),
        'Groundnut': ('Not Suitable', []),
        'Millets': ('Less suitable', ['Azospirillum', 'Phosphate Solubilizing Bacteria']),
        'Mustard': ('Less suitable', ['Azotobacter', 'Phosphate Solubilizing Bacteria']),
        'Potato': ('Not Suitable', []), 'Onion': ('Not Suitable', []),
        'Tomato': ('Moderately suitable', ['Azotobacter', 'Phosphate Solubilizing Bacteria']),
        'Chilli': ('Moderately suitable', ['Azotobacter', 'Azospirillum', 'Phosphate Solubilizing Bacteria']),
        'Turmeric': ('Not Suitable', []), 'Ginger': ('Not Suitable', []),
        'Banana': ('Suitable', ['Phosphate Solubilizing Bacteria', 'Potassium Solubilizing Bacteria', 'Arbuscular Mycorrhizal Fungi']),
        'Mango': ('Moderately suitable', ['Azotobacter', 'Phosphate Solubilizing Bacteria', 'Arbuscular Mycorrhizal Fungi'])
    },
    'Loamy': {
        'Rice': ('Suitable', ['Azospirillum', 'Phosphate Solubilizing Bacteria', 'Blue-Green Algae', 'Azolla']),
        'Wheat': ('Suitable', ['Azotobacter', 'Azospirillum', 'Phosphate Solubilizing Bacteria']),
        'Maize': ('Suitable', ['Azospirillum', 'Phosphate Solubilizing Bacteria']),
        'Sugarcane': ('Suitable', ['Acetobacter', 'Azospirillum', 'Phosphate Solubilizing Bacteria']),
        'Cotton': ('Suitable', ['Azotobacter', 'Phosphate Solubilizing Bacteria', 'Arbuscular Mycorrhizal Fungi']),
        'Jute': ('Suitable', ['Azotobacter', 'Phosphate Solubilizing Bacteria']),
        'Tea': ('Not Suitable', []), 'Coffee': ('Not Suitable', []),
        'Pulses': ('Suitable', ['Rhizobium', 'Phosphate Solubilizing Bacteria']),
        'Groundnut': ('Suitable', ['Rhizobium', 'Phosphate Solubilizing Bacteria']),
        'Millets': ('Suitable', ['Azospirillum', 'Phosphate Solubilizing Bacteria']),
        'Mustard': ('Suitable', ['Azotobacter', 'Phosphate Solubilizing Bacteria']),
        'Potato': ('Suitable', ['Azotobacter', 'Plant Growth Promoting Rhizobacteria', 'Potassium Solubilizing Bacteria']),
        'Onion': ('Suitable', ['Azotobacter', 'Phosphate Solubilizing Bacteria', 'Arbuscular Mycorrhizal Fungi']),
        'Tomato': ('Suitable', ['Azotobacter', 'Phosphate Solubilizing Bacteria', 'Potassium Solubilizing Bacteria']),
        'Chilli': ('Suitable', ['Azotobacter', 'Azospirillum', 'Phosphate Solubilizing Bacteria']),
        'Turmeric': ('Suitable', ['Azotobacter', 'Phosphate Solubilizing Bacteria']),
        'Ginger': ('Suitable', ['Azotobacter', 'Phosphate Solubilizing Bacteria', 'Potassium Solubilizing Bacteria']),
        'Banana': ('Suitable', ['Phosphate Solubilizing Bacteria', 'Potassium Solubilizing Bacteria', 'Arbuscular Mycorrhizal Fungi']),
        'Mango': ('Suitable', ['Azotobacter', 'Phosphate Solubilizing Bacteria', 'Arbuscular Mycorrhizal Fungi'])
    },
    'Calcareous': {
        'Rice': ('Not Suitable', []),
        'Wheat': ('Suitable', ['Azotobacter', 'Phosphate Solubilizing Bacteria', 'Zinc Solubilizing Bacteria']),
        'Maize': ('Suitable', ['Azospirillum', 'Phosphate Solubilizing Bacteria', 'Zinc Solubilizing Bacteria']),
        'Sugarcane': ('Suitable', ['Acetobacter', 'Azospirillum', 'Phosphate Solubilizing Bacteria']),
        'Cotton': ('Less suitable', ['Azotobacter', 'Phosphate Solubilizing Bacteria']),
        'Jute': ('Not Suitable', []), 'Tea': ('Not Suitable', []), 'Coffee': ('Not Suitable', []),
        'Pulses': ('Suitable', ['Rhizobium', 'Phosphate Solubilizing Bacteria']),
        'Groundnut': ('Less suitable', ['Rhizobium', 'Phosphate Solubilizing Bacteria']),
        'Millets': ('Suitable', ['Azospirillum', 'Phosphate Solubilizing Bacteria']),
        'Mustard': ('Suitable', ['Azotobacter', 'Phosphate Solubilizing Bacteria']),
        'Potato': ('Not Suitable', []),
        'Onion': ('Moderately suitable', ['Azotobacter', 'Phosphate Solubilizing Bacteria']),
        'Tomato': ('Moderately suitable', ['Azotobacter', 'Phosphate Solubilizing Bacteria']),
        'Chilli': ('Moderately suitable', ['Azotobacter', 'Azospirillum', 'Phosphate Solubilizing Bacteria']),
        'Turmeric': ('Not Suitable', []), 'Ginger': ('Not Suitable', []),
        'Banana': ('Less suitable', ['Phosphate Solubilizing Bacteria', 'Potassium Solubilizing Bacteria']),
        'Mango': ('Suitable', ['Azotobacter', 'Phosphate Solubilizing Bacteria', 'Arbuscular Mycorrhizal Fungi'])
    },
    'Acidic': {
        'Rice': ('Suitable', ['Azospirillum', 'Phosphate Solubilizing Bacteria']),
        'Wheat': ('Less suitable', ['Azotobacter', 'Phosphate Solubilizing Bacteria']),
        'Maize': ('Less suitable', ['Azospirillum', 'Phosphate Solubilizing Bacteria']),
        'Sugarcane': ('Not Suitable', []), 'Cotton': ('Not Suitable', []), 'Jute': ('Not Suitable', []),
        'Tea': ('Suitable', ['Azotobacter', 'Phosphate Solubilizing Bacteria', 'Arbuscular Mycorrhizal Fungi']),
        'Coffee': ('Suitable', ['Azotobacter', 'Phosphate Solubilizing Bacteria', 'Arbuscular Mycorrhizal Fungi']),
        'Pulses': ('Less suitable', ['Rhizobium', 'Phosphate Solubilizing Bacteria']),
        'Groundnut': ('Less suitable', ['Rhizobium', 'Phosphate Solubilizing Bacteria']),
        'Millets': ('Suitable', ['Azospirillum', 'Phosphate Solubilizing Bacteria']),
        'Mustard': ('Not Suitable', []),
        'Potato': ('Suitable', ['Azotobacter', 'Plant Growth Promoting Rhizobacteria']),
        'Onion': ('Not Suitable', []),
        'Tomato': ('Less suitable', ['Azotobacter', 'Phosphate Solubilizing Bacteria']),
        'Chilli': ('Less suitable', ['Azotobacter', 'Phosphate Solubilizing Bacteria']),
        'Turmeric': ('Less suitable', ['Azotobacter', 'Phosphate Solubilizing Bacteria']),
        'Ginger': ('Less suitable', ['Azotobacter', 'Phosphate Solubilizing Bacteria']),
        'Banana': ('Less suitable', ['Phosphate Solubilizing Bacteria', 'Potassium Solubilizing Bacteria', 'Arbuscular Mycorrhizal Fungi']),
        'Mango': ('Less suitable', ['Azotobacter', 'Phosphate Solubilizing Bacteria', 'Arbuscular Mycorrhizal Fungi'])
    },
    'Ferruginous': {
        'Rice': ('Suitable', ['Azospirillum', 'Phosphate Solubilizing Bacteria']),
        'Wheat': ('Less suitable', ['Azotobacter', 'Phosphate Solubilizing Bacteria']),
        'Maize': ('Suitable', ['Azospirillum', 'Phosphate Solubilizing Bacteria']),
        'Sugarcane': ('Less suitable', ['Azospirillum', 'Acetobacter', 'Phosphate Solubilizing Bacteria']),
        'Cotton': ('Less suitable', ['Azotobacter', 'Phosphate Solubilizing Bacteria', 'Arbuscular Mycorrhizal Fungi']),
        'Jute': ('Not Suitable', []),
        'Tea': ('Moderately suitable', ['Azotobacter', 'Phosphate Solubilizing Bacteria', 'Arbuscular Mycorrhizal Fungi']),
        'Coffee': ('Moderately suitable', ['Azotobacter', 'Phosphate Solubilizing Bacteria', 'Arbuscular Mycorrhizal Fungi']),
        'Pulses': ('Suitable', ['Rhizobium', 'Phosphate Solubilizing Bacteria']),
        'Groundnut': ('Suitable', ['Rhizobium', 'Phosphate Solubilizing Bacteria']),
        'Millets': ('Suitable', ['Azospirillum', 'Phosphate Solubilizing Bacteria']),
        'Mustard': ('Suitable', ['Azotobacter', 'Phosphate Solubilizing Bacteria']),
        'Potato': ('Suitable', ['Azotobacter', 'Plant Growth Promoting Rhizobacteria', 'Potassium Solubilizing Bacteria']),
        'Onion': ('Suitable', ['Azotobacter', 'Phosphate Solubilizing Bacteria', 'Arbuscular Mycorrhizal Fungi']),
        'Tomato': ('Suitable', ['Azotobacter', 'Phosphate Solubilizing Bacteria', 'Potassium Solubilizing Bacteria']),
        'Chilli': ('Suitable', ['Azotobacter', 'Azospirillum', 'Phosphate Solubilizing Bacteria']),
        'Turmeric': ('Suitable', ['Azotobacter', 'Phosphate Solubilizing Bacteria']),
        'Ginger': ('Suitable', ['Azotobacter', 'Phosphate Solubilizing Bacteria', 'Potassium Solubilizing Bacteria']),
        'Banana': ('Moderately suitable', ['Phosphate Solubilizing Bacteria', 'Potassium Solubilizing Bacteria', 'Arbuscular Mycorrhizal Fungi']),
        'Mango': ('Suitable', ['Azotobacter', 'Phosphate Solubilizing Bacteria', 'Arbuscular Mycorrhizal Fungi'])
    },
    'Regur': { # Same as Black soil
        'Rice': ('Moderately suitable', ['Azospirillum', 'Phosphate Solubilizing Bacteria', 'Zinc Solubilizing Bacteria']),
        'Wheat': ('Suitable', ['Azotobacter', 'Phosphate Solubilizing Bacteria']),
        'Maize': ('Suitable', ['Azospirillum', 'Phosphate Solubilizing Bacteria']),
        'Sugarcane': ('Suitable', ['Acetobacter', 'Azospirillum', 'Phosphate Solubilizing Bacteria']),
        'Cotton': ('Suitable', ['Azotobacter', 'Phosphate Solubilizing Bacteria', 'Arbuscular Mycorrhizal Fungi']),
        'Jute': ('Not Suitable', []), 'Tea': ('Not Suitable', []), 'Coffee': ('Not Suitable', []),
        'Pulses': ('Suitable', ['Rhizobium', 'Phosphate Solubilizing Bacteria']),
        'Groundnut': ('Suitable', ['Rhizobium', 'Phosphate Solubilizing Bacteria']),
        'Millets': ('Suitable', ['Azospirillum', 'Phosphate Solubilizing Bacteria']),
        'Mustard': ('Less suitable', ['Azotobacter', 'Phosphate Solubilizing Bacteria']),
        'Potato': ('Less suitable', ['Azotobacter', 'Plant Growth Promoting Rhizobacteria', 'Potassium Solubilizing Bacteria']),
        'Onion': ('Less suitable', ['Azotobacter', 'Phosphate Solubilizing Bacteria', 'Arbuscular Mycorrhizal Fungi']),
        'Tomato': ('Suitable', ['Azotobacter', 'Phosphate Solubilizing Bacteria', 'Potassium Solubilizing Bacteria']),
        'Chilli': ('Suitable', ['Azotobacter', 'Azospirillum', 'Phosphate Solubilizing Bacteria']),
        'Turmeric': ('Less suitable', ['Azotobacter', 'Phosphate Solubilizing Bacteria']),
        'Ginger': ('Less suitable', ['Azotobacter', 'Phosphate Solubilizing Bacteria', 'Potassium Solubilizing Bacteria']),
        'Banana': ('Suitable', ['Phosphate Solubilizing Bacteria', 'Potassium Solubilizing Bacteria']),
        'Mango': ('Moderately suitable', ['Azotobacter', 'Phosphate Solubilizing Bacteria', 'Arbuscular Mycorrhizal Fungi'])
    },
    'Podzol': {
        'Rice': ('Not Suitable', []), 'Wheat': ('Not Suitable', []),
        'Maize': ('Less suitable', ['Azospirillum', 'Phosphate Solubilizing Bacteria', 'Arbuscular Mycorrhizal Fungi']),
        'Sugarcane': ('Not Suitable', []), 'Cotton': ('Not Suitable', []), 'Jute': ('Not Suitable', []),
        'Tea': ('Moderately suitable', ['Azotobacter', 'Phosphate Solubilizing Bacteria', 'Arbuscular Mycorrhizal Fungi']),
        'Coffee': ('Less suitable', ['Azotobacter', 'Phosphate Solubilizing Bacteria', 'Arbuscular Mycorrhizal Fungi']),
        'Pulses': ('Not Suitable', []), 'Groundnut': ('Not Suitable', []),
        'Millets': ('Less suitable', ['Azospirillum', 'Phosphate Solubilizing Bacteria']),
        'Mustard': ('Not Suitable', []),
        'Potato': ('Suitable', ['Azotobacter', 'Plant Growth Promoting Rhizobacteria']),
        'Onion': ('Not Suitable', []), 'Tomato': ('Not Suitable', []), 'Chilli': ('Not Suitable', []),
        'Turmeric': ('Not Suitable', []), 'Ginger': ('Not Suitable', []), 'Banana': ('Not Suitable', []),
        'Mango': ('Not Suitable', [])
    },
    'Terra Rossa': {
        'Rice': ('Not Suitable', []),
        'Wheat': ('Suitable', ['Azotobacter', 'Phosphate Solubilizing Bacteria']),
        'Maize': ('Suitable', ['Azospirillum', 'Phosphate Solubilizing Bacteria']),
        'Sugarcane': ('Less suitable', ['Azospirillum', 'Phosphate Solubilizing Bacteria']),
        'Cotton': ('Less suitable', ['Azotobacter', 'Phosphate Solubilizing Bacteria']),
        'Jute': ('Not Suitable', []), 'Tea': ('Not Suitable', []), 'Coffee': ('Not Suitable', []),
        'Pulses': ('Suitable', ['Rhizobium', 'Phosphate Solubilizing Bacteria']),
        'Groundnut': ('Less suitable', ['Rhizobium', 'Phosphate Solubilizing Bacteria']),
        'Millets': ('Suitable', ['Azospirillum', 'Phosphate Solubilizing Bacteria']),
        'Mustard': ('Suitable', ['Azotobacter', 'Phosphate Solubilizing Bacteria']),
        'Potato': ('Not Suitable', []),
        'Onion': ('Moderately suitable', ['Azotobacter', 'Phosphate Solubilizing Bacteria']),
        'Tomato': ('Suitable', ['Azotobacter', 'Phosphate Solubilizing Bacteria']),
        'Chilli': ('Suitable', ['Azotobacter', 'Azospirillum', 'Phosphate Solubilizing Bacteria']),
        'Turmeric': ('Less suitable', ['Azotobacter', 'Phosphate Solubilizing Bacteria']),
        'Ginger': ('Less suitable', ['Azotobacter', 'Phosphate Solubilizing Bacteria']),
        'Banana': ('Less suitable', ['Phosphate Solubilizing Bacteria', 'Potassium Solubilizing Bacteria']),
        'Mango': ('Suitable', ['Azotobacter', 'Phosphate Solubilizing Bacteria', 'Arbuscular Mycorrhizal Fungi'])
    },
    'Humus-rich': {
        'Rice': ('Suitable', ['Azospirillum', 'Phosphate Solubilizing Bacteria', 'Blue-Green Algae', 'Azolla']),
        'Wheat': ('Suitable', ['Azotobacter', 'Azospirillum', 'Phosphate Solubilizing Bacteria']),
        'Maize': ('Suitable', ['Azospirillum', 'Phosphate Solubilizing Bacteria']),
        'Sugarcane': ('Suitable', ['Acetobacter', 'Azospirillum', 'Phosphate Solubilizing Bacteria', 'Potassium Solubilizing Bacteria']),
        'Cotton': ('Suitable', ['Azotobacter', 'Phosphate Solubilizing Bacteria', 'Arbuscular Mycorrhizal Fungi']),
        'Jute': ('Suitable', ['Azotobacter', 'Phosphate Solubilizing Bacteria']),
        'Tea': ('Suitable', ['Azotobacter', 'Phosphate Solubilizing Bacteria', 'Arbuscular Mycorrhizal Fungi']),
        'Coffee': ('Suitable', ['Azotobacter', 'Phosphate Solubilizing Bacteria', 'Arbuscular Mycorrhizal Fungi']),
        'Pulses': ('Suitable', ['Rhizobium', 'Phosphate Solubilizing Bacteria']),
        'Groundnut': ('Suitable', ['Rhizobium', 'Phosphate Solubilizing Bacteria']),
        'Millets': ('Suitable', ['Azospirillum', 'Phosphate Solubilizing Bacteria']),
        'Mustard': ('Suitable', ['Azotobacter', 'Phosphate Solubilizing Bacteria']),
        'Potato': ('Suitable', ['Azotobacter', 'Plant Growth Promoting Rhizobacteria', 'Potassium Solubilizing Bacteria']),
        'Onion': ('Suitable', ['Azotobacter', 'Phosphate Solubilizing Bacteria', 'Arbuscular Mycorrhizal Fungi']),
        'Tomato': ('Suitable', ['Azotobacter', 'Phosphate Solubilizing Bacteria', 'Potassium Solubilizing Bacteria']),
        'Chilli': ('Suitable', ['Azotobacter', 'Azospirillum', 'Phosphate Solubilizing Bacteria']),
        'Turmeric': ('Suitable', ['Azotobacter', 'Phosphate Solubilizing Bacteria']),
        'Ginger': ('Suitable', ['Azotobacter', 'Phosphate Solubilizing Bacteria', 'Potassium Solubilizing Bacteria']),
        'Banana': ('Suitable', ['Phosphate Solubilizing Bacteria', 'Potassium Solubilizing Bacteria', 'Arbuscular Mycorrhizal Fungi']),
        'Mango': ('Suitable', ['Azotobacter', 'Phosphate Solubilizing Bacteria', 'Arbuscular Mycorrhizal Fungi'])
    },
    'Khadar': { # Same as Alluvial soil
        'Rice': ('Suitable', ['Azospirillum', 'Phosphate Solubilizing Bacteria', 'Blue-Green Algae', 'Azolla']),
        'Wheat': ('Suitable', ['Azotobacter', 'Azospirillum', 'Phosphate Solubilizing Bacteria']),
        'Maize': ('Suitable', ['Azospirillum', 'Phosphate Solubilizing Bacteria']),
        'Sugarcane': ('Suitable', ['Acetobacter', 'Azospirillum', 'Phosphate Solubilizing Bacteria', 'Potassium Solubilizing Bacteria']),
        'Cotton': ('Suitable', ['Azotobacter', 'Phosphate Solubilizing Bacteria', 'Arbuscular Mycorrhizal Fungi']),
        'Jute': ('Suitable', ['Azotobacter', 'Phosphate Solubilizing Bacteria']),
        'Tea': ('Not Suitable', []), 'Coffee': ('Not Suitable', []),
        'Pulses': ('Suitable', ['Rhizobium', 'Phosphate Solubilizing Bacteria']),
        'Groundnut': ('Suitable', ['Rhizobium', 'Phosphate Solubilizing Bacteria']),
        'Millets': ('Suitable', ['Azospirillum', 'Phosphate Solubilizing Bacteria']),
        'Mustard': ('Suitable', ['Azotobacter', 'Phosphate Solubilizing Bacteria']),
        'Potato': ('Suitable', ['Azotobacter', 'Plant Growth Promoting Rhizobacteria', 'Potassium Solubilizing Bacteria']),
        'Onion': ('Suitable', ['Azotobacter', 'Phosphate Solubilizing Bacteria', 'Arbuscular Mycorrhizal Fungi']),
        'Tomato': ('Suitable', ['Azotobacter', 'Phosphate Solubilizing Bacteria', 'Potassium Solubilizing Bacteria']),
        'Chilli': ('Suitable', ['Azotobacter', 'Azospirillum', 'Phosphate Solubilizing Bacteria']),
        'Turmeric': ('Suitable', ['Azotobacter', 'Phosphate Solubilizing Bacteria']),
        'Ginger': ('Suitable', ['Azotobacter', 'Phosphate Solubilizing Bacteria', 'Potassium Solubilizing Bacteria']),
        'Banana': ('Suitable', ['Phosphate Solubilizing Bacteria', 'Potassium Solubilizing Bacteria', 'Arbuscular Mycorrhizal Fungi']),
        'Mango': ('Suitable', ['Azotobacter', 'Phosphate Solubilizing Bacteria', 'Arbuscular Mycorrhizal Fungi'])
    }
}

# Biofertilizer translations
bio_trans = {
    'en': {
        'Azotobacter': 'Azotobacter', 'Azospirillum': 'Azospirillum',
        'Arbuscular Mycorrhizal Fungi': 'Arbuscular Mycorrhizal Fungi',
        'Phosphate Solubilizing Bacteria': 'Phosphate Solubilizing Bacteria',
        'Rhizobium': 'Rhizobium', 'Acetobacter': 'Acetobacter',
        'Blue-Green Algae': 'Blue-Green Algae', 'Azolla': 'Azolla',
        'Potassium Solubilizing Bacteria': 'Potassium Solubilizing Bacteria',
        'Plant Growth Promoting Rhizobacteria': 'Plant Growth Promoting Rhizobacteria',
        'Zinc Solubilizing Bacteria': 'Zinc Solubilizing Bacteria', 'Pseudomonas': 'Pseudomonas'
    },
    'te': {
        'Azotobacter': 'అజోటోబాక్టర్', 'Azospirillum': 'అజోస్పిరిలమ్',
        'Arbuscular Mycorrhizal Fungi': 'ఆర్బస్కులర్ మైకోరైజల్ ఫంగై',
        'Phosphate Solubilizing Bacteria': 'ఫాస్ఫేట్ సోల్యూబిలైజింగ్ బ్యాక్టీరియా',
        'Rhizobium': 'రైజోబియం', 'Acetobacter': 'ఆసిటోబాక్టర్',
        'Blue-Green Algae': 'నీలి-ఆకుపచ్చ శైవలాలు', 'Azolla': 'అజోల్లా',
        'Potassium Solubilizing Bacteria': 'పొటాషియం సోల్యూబిలైజింగ్ బ్యాక్టీరియా',
        'Plant Growth Promoting Rhizobacteria': 'మొక్కల పెరుగుదలను ప్రోత్సహించే రైజోబాక్టీరియా',
        'Zinc Solubilizing Bacteria': 'జింక్ సోల్యూబిలైజింగ్ బ్యాక్టీరియా', 'Pseudomonas': 'సూడోమోనస్'
    },
    'hi': {
        'Azotobacter': 'एजोटोबैक्टर', 'Azospirillum': 'एजोस्पिरिलम',
        'Arbuscular Mycorrhizal Fungi': 'आर्बस्कुलर माइकोराइजल फंगी',
        'Phosphate Solubilizing Bacteria': 'फॉस्फेट सॉल्यूबिलाइजिंग बैक्टीरिया',
        'Rhizobium': 'राइजोबियम', 'Acetobacter': 'एसिटोबैक्टर',
        'Blue-Green Algae': 'नील-हरित शैवाल', 'Azolla': 'अजोला',
        'Potassium Solubilizing Bacteria': 'पोटैशियम सॉल्यूबिलाइजिंग बैक्टीरिया',
        'Plant Growth Promoting Rhizobacteria': 'प्लांट ग्रोथ प्रोमोटिंग राइजोबैक्टीरिया',
        'Zinc Solubilizing Bacteria': 'जिंक सॉल्यूबिलाइजिंग बैक्टीरिया', 'Pseudomonas': 'स्यूडोमोनास'
    }
}

# --- HELPER FUNCTIONS ---

def get_farming_alerts(forecast_data, lang):
    """Analyzes forecast and returns farming alerts."""
    alerts = []
    # Check for rain in the next 24 hours (8 * 3-hour intervals)
    will_rain = any('rain' in item for item in forecast_data['list'][:8])
    if will_rain:
        alerts.append(ui_texts[lang]['alert_rain'])

    # Check for high temperatures > 35°C
    high_temp = any(item['main']['temp'] > 35 for item in forecast_data['list'][:8])
    if high_temp:
        alerts.append(ui_texts[lang]['alert_heat'])

    # Check for strong winds > 10 m/s
    strong_wind = any(item['wind']['speed'] > 10 for item in forecast_data['list'][:8])
    if strong_wind:
        alerts.append(ui_texts[lang]['alert_wind'])

    # If no specific alerts, give a positive message
    if not alerts:
        alerts.append(ui_texts[lang]['alert_clear'])
        
    return alerts

def recommend_biofertilizer(soil_type_en, crop_type_en, lang):
    """Recommends biofertilizer based on soil and crop type."""
    try:
        suitability, biofertilizers = suitability_map[soil_type_en][crop_type_en]
        if suitability == 'Not Suitable':
            soil_display, crop_display = soil_type_en, crop_type_en
            if lang == 'te':
                en_soils_to_te = {v: k for k, v in te_soils_to_en.items()}
                en_crops_to_te = {v: k for k, v in te_crops_to_en.items()}
                soil_display = en_soils_to_te.get(soil_type_en, soil_type_en)
                crop_display = en_crops_to_te.get(crop_type_en, crop_type_en)
            elif lang == 'hi':
                en_soils_to_hi = {v: k for k, v in hi_soils_to_en.items()}
                en_crops_to_hi = {v: k for k, v in hi_crops_to_en.items()}
                soil_display = en_soils_to_hi.get(soil_type_en, soil_type_en)
                crop_display = en_crops_to_hi.get(crop_type_en, crop_type_en)
            return {'rec': ui_texts[lang]['not_suitable'].format(crop=crop_display, soil=soil_display), 'suitability': suitability}

        if not biofertilizers:
            return {'rec': 'No specific recommendation available.', 'suitability': suitability}
            
        translated_bio = [bio_trans[lang].get(bio, bio) for bio in biofertilizers]
        return {'rec': ', '.join(translated_bio), 'suitability': suitability}
    except KeyError:
        return {'rec': ui_texts[lang]['error_fetch'], 'suitability': 'Error'}

def train_compost_model():
    """Trains a simple decision tree model for compost stage prediction."""
    X = np.array([[0, 0, 0, 1, 5], [1, 1, 2, 1, 10], [1, 1, 2, 1, 20], [2, 2, 1, 1, 40], [2, 2, 0, 1, 70]])
    y = np.array([0, 1, 2, 3, 4])
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X, y)
    return clf

compost_model = train_compost_model()
stages = {
    'en': ['Initial', 'Mesophilic', 'Thermophilic', 'Cooling', 'Mature'],
    'te': ['ప్రారంభ', 'మెసోఫిలిక్', 'థర్మోఫిలిక్', 'కూలింగ్', 'పరిపక్వం'],
    'hi': ['प्रारंभिक', 'मेसोफिलिक', 'थर्मोफिलिक', 'कूलिंग', 'परिपक्व']
}

def classify_waste(image_bytes, lang):
    """Classifies waste image as compostable or non-compostable."""
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((100, 100)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        img_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = waste_model(img_tensor)
            _, pred_class = torch.max(outputs, 1)
        return classes[lang][pred_class.item()]
    except Exception as e:
        print(f"Error classifying waste: {e}")
        return classes[lang]['invalid_image']

classes = {
    'en': {0: 'Compostable', 1: 'Non-Compostable', 'no_image': 'No image uploaded', 'invalid_image': 'Not a valid waste image'},
    'te': {0: 'కంపోస్ట్ చేయదగిన', 1: 'కంపోస్ట్ చేయనిది', 'no_image': 'ఎటువంటి చిత్రం అప్‌లోడ్ చేయబడలేదు', 'invalid_image': 'చెల్లుబాటు అయ్యే వ్యర్థ చిత్రం కాదు'},
    'hi': {0: 'कंपोस्ट योग्य', 1: 'गैर-कंपोस्ट', 'no_image': 'कोई छवि अपलोड नहीं की गई', 'invalid_image': 'एक मान्य अपशिष्ट छवि नहीं है'}
}

# --- FLASK ROUTES ---
@app.route('/')
def home():
    lang = request.args.get('lang', 'en')
    ui = ui_texts[lang]
    return render_template('index.html', lang=lang, ui=ui)

@app.route('/weather')
def weather():
    lat = request.args.get('lat')
    lon = request.args.get('lon')
    lang = request.args.get('lang', 'en')

    if not lat or not lon:
        return jsonify({"error": "Latitude and longitude are required"}), 400
    if not OPENWEATHER_API_KEY or OPENWEATHER_API_KEY == "YOUR_API_KEY_HERE":
         return jsonify({"error": "Weather API key not configured on the server"}), 500

    try:
        current_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
        current_res = requests.get(current_url)
        current_res.raise_for_status()
        current_data = current_res.json()

        forecast_url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
        forecast_res = requests.get(forecast_url)
        forecast_res.raise_for_status()
        forecast_data = forecast_res.json()

        daily_forecast = []
        seen_days = set()
        for item in forecast_data['list']:
            day = datetime.fromtimestamp(item['dt']).strftime('%A')
            if day not in seen_days:
                daily_forecast.append({
                    "day": day,
                    "temp": item['main']['temp'],
                    "icon": item['weather'][0]['icon'],
                    "description": item['weather'][0]['description']
                })
                seen_days.add(day)
            if len(daily_forecast) == 5:
                break
        
        alerts = get_farming_alerts(forecast_data, lang)

        weather_info = {
            "current": {
                "temp": current_data['main']['temp'],
                "humidity": current_data['main']['humidity'],
                "wind": current_data['wind']['speed'],
                "description": current_data['weather'][0]['description'],
                "icon": current_data['weather'][0]['icon'],
                "city": current_data.get('name', 'Your Location')
            },
            "forecast": daily_forecast,
            "alerts": alerts
        }
        return jsonify(weather_info)

    except requests.exceptions.RequestException as e:
        print(f"Error fetching weather data: {e}")
        return jsonify({"error": "Failed to fetch weather data"}), 500

@app.route('/biofertilizer', methods=['GET', 'POST'])
def biofertilizer():
    lang = request.form.get('lang', request.args.get('lang', 'en'))
    ui = ui_texts[lang]
    if request.method == 'POST':
        soil_from_form = request.form.get('soil')
        crop_from_form = request.form.get('crop')
        if not soil_from_form or not crop_from_form:
            return jsonify({'rec': ui_texts[lang]['error_fetch'], 'suitability': 'Error'}), 400
        
        soil_en = soil_from_form
        crop_en = crop_from_form
        if lang == 'te':
            soil_en = te_soils_to_en.get(soil_from_form, soil_from_form)
            crop_en = te_crops_to_en.get(crop_from_form, crop_from_form)
        elif lang == 'hi':
            soil_en = hi_soils_to_en.get(soil_from_form, soil_from_form)
            crop_en = hi_crops_to_en.get(crop_from_form, crop_from_form)
            
        rec = recommend_biofertilizer(soil_en, crop_en, lang)
        return jsonify(rec)
    
    soils_display = soil_crop_trans[lang]['soils']
    crops_display = soil_crop_trans[lang]['crops']
    return render_template('biofertilizer.html', lang=lang, ui=ui, soils=soils_display, crops=crops_display)

@app.route('/waste', methods=['GET', 'POST'])
def waste():
    lang = request.form.get('lang', request.args.get('lang', 'en'))
    ui = ui_texts[lang]
    if request.method == 'POST':
        file = request.files.get('image')
        if file and file.filename != '':
            img_bytes = file.read()
            clas = classify_waste(img_bytes, lang)
            return jsonify({'clas': clas})
        return jsonify({'clas': classes[lang]['no_image']})
    return render_template('waste.html', lang=lang, ui=ui)

@app.route('/compost', methods=['GET', 'POST'])
def compost():
    lang = request.form.get('lang', request.args.get('lang', 'en'))
    ui = ui_texts[lang]
    if request.method == 'POST':
        smell = int(request.form.get('smell', 0))
        color = int(request.form.get('color', 0))
        heat = int(request.form.get('heat', 0))
        moisture = int(request.form.get('moisture', 0))
        days = int(request.form.get('days', 5))
        features = np.array([[smell, color, heat, moisture, days]])
        pred = compost_model.predict(features)[0]
        
        stage_key = stages['en'][pred]
        stage_display = stages[lang][pred]
        
        return jsonify({'stage': stage_display, 'stage_key': stage_key})
        
    return render_template('compost.html', lang=lang, ui=ui)

if __name__ == '__main__':
    app.run(debug=True)
