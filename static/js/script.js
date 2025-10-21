// --- BASE URL CONFIG ---
const BASE_URL = window.location.hostname.includes('localhost') 
                 ? '' 
                 : 'https://green-grow-ai-a6q3.onrender.com';

// --- STATE & CONFIG ---
let voiceEnabled = localStorage.getItem('voiceEnabled') === 'true';
let isListening = false;
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
let recognition;

// --- INITIALIZATION ---
document.addEventListener('DOMContentLoaded', () => {
    initializeExitAnimations();
    updateVoiceButtonState();
    if (SpeechRecognition) {
        initializeRecognition();
    } else {
        console.warn("Speech Recognition API is not supported in this browser.");
        const voiceBtn = document.getElementById('voiceBtn');
        if (voiceBtn) voiceBtn.style.display = 'none';
    }

    // Override forms to use BASE_URL
    const compostForm = document.getElementById('compostForm');
    const wasteForm = document.getElementById('wasteForm');
    if (compostForm) overrideFormFetch(compostForm, '/compost');
    if (wasteForm) overrideFormFetch(wasteForm, '/waste');
});

// --- SPEECH RECOGNITION ---
function initializeRecognition() {
    recognition = new SpeechRecognition();
    const pageLang = document.documentElement.lang || 'en';
    const targetLangCode = pageLang === 'te' ? 'te-IN' : pageLang === 'hi' ? 'hi-IN' : 'en-US';
    
    recognition.lang = targetLangCode;
    recognition.continuous = false;
    recognition.interimResults = false;

    recognition.onstart = () => { isListening = true; updateVoiceButtonState(); };
    recognition.onend = () => { isListening = false; updateVoiceButtonState(); };
    recognition.onerror = (event) => { console.error("Speech recognition error:", event.error); isListening = false; updateVoiceButtonState(); };
    recognition.onresult = (event) => { processVoiceCommand(event.results[0][0].transcript.trim().toLowerCase()); };
}

function toggleVoice() {
    if (!SpeechRecognition) return;
    voiceEnabled = !voiceEnabled;
    localStorage.setItem('voiceEnabled', voiceEnabled);
    if (voiceEnabled && !isListening) {
        try { recognition.start(); } catch (e) { console.error(e); }
    } else {
        recognition.stop();
    }
    updateVoiceButtonState();
}

// --- COMMAND PROCESSING ---
function processVoiceCommand(transcript) {
    const page = document.body.dataset.page;
    if (!page) return;
    switch(page) {
        case 'home': handleHomePageCommands(transcript); break;
        case 'biofertilizer': handleBiofertilizerCommands(transcript); break;
        case 'compost': handleCompostCommands(transcript); break;
        case 'waste': handleWasteCommands(transcript); break;
    }
}

function handleHomePageCommands(transcript) {
    const lang = document.documentElement.lang || 'en';
    const navKeywords = {
        'en': { bio: 'biofertilizer', waste: 'waste', compost: 'compost' },
        'te': { bio: 'బయో', waste: 'వ్యర్థాల', compost: 'కంపోస్ట్' },
        'hi': { bio: 'बायो', waste: 'अपशिष्ट', compost: 'खाद' }
    };
    if (transcript.includes(navKeywords[lang].bio)) speakAndNavigate("Navigating to Biofertilizer.", `/biofertilizer?lang=${lang}`);
    else if (transcript.includes(navKeywords[lang].waste)) speakAndNavigate("Navigating to Waste Classification.", `/waste?lang=${lang}`);
    else if (transcript.includes(navKeywords[lang].compost)) speakAndNavigate("Navigating to Compost Tracker.", `/compost?lang=${lang}`);
}

function handleBiofertilizerCommands(transcript) {
    const soilSelect = document.querySelector('select[name="soil"]');
    const cropSelect = document.querySelector('select[name="crop"]');
    const lang = document.documentElement.lang || 'en';
    const keywords = { 'en': { soil: 'soil', crop: 'crop' }, 'te': { soil: 'మట్టి', crop: 'పంట' }, 'hi': { soil: 'मिट्टी', crop: 'फसल' } };

    const parts = transcript.split(keywords[lang].crop);
    if (parts.length < 2) return;
    const soilValue = parts[0].replace(keywords[lang].soil, '').trim();
    const cropValue = parts[1].trim();
    const soilSet = setSelectBySpokenText(soilSelect, soilValue);
    const cropSet = setSelectBySpokenText(cropSelect, cropValue);
    if (soilSet && cropSet) document.getElementById('bioForm').querySelector('button[type="submit"]').click();
    else { if(!soilSet) speak(`Could not find soil ${soilValue}`); if(!cropSet) speak(`Could not find crop ${cropValue}`); }
}

function handleCompostCommands(transcript) {
    const form = document.getElementById('compostForm');
    let commandsFound = 0;
    const lang = document.documentElement.lang || 'en';
    const keywords = {
        'en': { smell: 'smell', color: 'color', heat: 'heat', moisture: 'moisture', days: 'days' },
        'te': { smell: 'వాసన', color: 'రంగు', heat: 'వేడి', moisture: 'తేమ', days: 'రోజులు' },
        'hi': { smell: 'गंध', color: 'रंग', heat: 'गर्मी', moisture: 'नमी', days: 'दिन' }
    };

    for(const key in keywords[lang]){
        const regex = new RegExp(`${keywords[lang][key]}\\s+([^,]+)`);
        const match = transcript.match(regex);
        if(match && match[1]){
            const value = match[1].trim();
            const select = form.querySelector(`select[name="${key}"]`);
            if(setSelectBySpokenText(select, value)) commandsFound++;
            else speak(`Could not find value ${value} for ${key}`);
        }
    }
    if(commandsFound > 0) form.querySelector('button[type="submit"]').click();
}

function handleWasteCommands(transcript) {
    const lang = document.documentElement.lang || 'en';
    const keywords = { 'en': { upload:'upload', choose:'choose', submit:'submit', classify:'classify' },
                       'te': { upload:'అప్లోడ్', choose:'ఎంచుకోండి', submit:'సమర్పించండి', classify:'వర్గీకరించండి' },
                       'hi': { upload:'अपलोड', choose:'चुनें', submit:'जमा', classify:'वर्गीकृत' } };
    if(transcript.includes(keywords[lang].upload) || transcript.includes(keywords[lang].choose)) document.querySelector('input[type="file"]').click();
    else if(transcript.includes(keywords[lang].submit) || transcript.includes(keywords[lang].classify)) document.getElementById('wasteForm').querySelector('button[type="submit"]').click();
}

// --- HELPERS ---
function setSelectBySpokenText(selectEl, spokenText) {
    if(!selectEl || !spokenText) return false;
    const option = Array.from(selectEl.options).find(opt => opt.textContent.trim().toLowerCase() === spokenText.toLowerCase());
    if(option){ selectEl.value = option.value; selectEl.dispatchEvent(new Event('change')); return true; }
    return false;
}

function speakAndNavigate(text, url){
    speak(text);
    setTimeout(()=>{ window.location.href = `${BASE_URL}${url}`; }, 1500);
}

function updateVoiceButtonState(){
    const btn = document.getElementById('voiceBtn');
    if(!btn) return;
    btn.classList.remove('btn-success','btn-secondary','active','is-listening');
    if(voiceEnabled){ btn.classList.add('btn-success','active'); if(isListening) btn.classList.add('is-listening'); }
    else btn.classList.add('btn-secondary');
}

function speak(text){
    if(!voiceEnabled || !('speechSynthesis' in window)) return;
    speechSynthesis.cancel();
    const pageLang = document.documentElement.lang || 'en';
    const targetLangCode = pageLang==='te'?'te-IN':pageLang==='hi'?'hi-IN':'en-US';
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = targetLangCode;
    const setVoice = () => {
        const voices = speechSynthesis.getVoices();
        const selectedVoice = voices.find(v=>v.lang===targetLangCode) || voices.find(v=>v.lang.startsWith(pageLang));
        if(selectedVoice) utterance.voice = selectedVoice;
        speechSynthesis.speak(utterance);
    };
    if(speechSynthesis.getVoices().length>0) setVoice();
    else speechSynthesis.onvoiceschanged=setVoice;
}

function initializeExitAnimations(){
    const moduleLinks = document.querySelectorAll('.module-link');
    if(!moduleLinks.length) return;
    moduleLinks.forEach(link => {
        link.addEventListener('click', e => {
            e.preventDefault();
            link.querySelector('.card').classList.add('is-exiting');
            setTimeout(()=>{ window.location.href = link.href; },500);
        });
    });
}

function changeLang(){
    const lang = document.getElementById('langSelect').value;
    const currentUrl = new URL(window.location.href);
    currentUrl.searchParams.set('lang', lang);
    window.location.href = currentUrl.toString();
}

// --- OVERRIDE FORMS ---
function overrideFormFetch(form, endpoint){
    form.addEventListener('submit', e=>{
        e.preventDefault();
        const button = form.querySelector('.btn-morph');
        const resultDiv = document.getElementById('result');
        if(button.classList.contains('is-loading')) return;
        button.disabled = true; button.classList.add('is-loading');
        resultDiv.style.display='none'; resultDiv.classList.remove('show');
        const revertButton = ()=>{ button.classList.remove('is-success','is-loading'); button.disabled=false; };
        const formData = new FormData(form);
        setTimeout(()=>{
            fetch(`${BASE_URL}${endpoint}`,{method:'POST', body:formData})
            .then(res=>res.json())
            .then(data=>{
                button.classList.remove('is-loading'); button.classList.add('is-success');
                if(endpoint==='/compost'){
                    resultDiv.innerHTML=`<h5>Stage: ${data.stage}</h5><p>${data.description||''}</p>`;
                } else if(endpoint==='/waste'){
                    resultDiv.innerHTML=`<h5>Classification: ${data.clas}</h5>`;
                }
                speak(data.stage||data.clas||'Done');
                setTimeout(()=>{ revertButton(); resultDiv.style.display='block'; resultDiv.classList.add('show'); },1500);
            })
            .catch(err=>{ console.error(err); resultDiv.innerHTML='<p class="text-danger">Error fetching data</p>'; revertButton(); resultDiv.style.display='block'; resultDiv.classList.add('show'); });
        },500);
    });
}
