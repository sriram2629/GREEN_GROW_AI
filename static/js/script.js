// --- STATE & CONFIG ---
let voiceEnabled = localStorage.getItem('voiceEnabled') === 'true';
let isListening = false;
// Check for browser compatibility for the Speech Recognition API
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
let recognition;

// --- INITIALIZATION ---
// Initialize all functions when the page content is fully loaded.
document.addEventListener('DOMContentLoaded', () => {
    initializeExitAnimations();
    updateVoiceButtonState();
    if (SpeechRecognition) {
        initializeRecognition();
    } else {
        console.warn("Speech Recognition API is not supported in this browser.");
        const voiceBtn = document.getElementById('voiceBtn');
        if (voiceBtn) voiceBtn.style.display = 'none'; // Hide button if not supported
    }
});


// --- CORE SPEECH RECOGNITION LOGIC ---

/**
 * Sets up the SpeechRecognition object with correct language and event handlers.
 */
function initializeRecognition() {
    recognition = new SpeechRecognition();
    const pageLang = document.documentElement.lang || 'en';
    const targetLangCode = pageLang === 'te' ? 'te-IN' : pageLang === 'hi' ? 'hi-IN' : 'en-US';
    
    recognition.lang = targetLangCode;
    recognition.continuous = false; // Process single commands at a time
    recognition.interimResults = false;

    recognition.onstart = () => {
        isListening = true;
        updateVoiceButtonState(); // Visually indicate that the app is listening
    };

    recognition.onend = () => {
        isListening = false;
        updateVoiceButtonState();
    };

    recognition.onerror = (event) => {
        console.error("Speech recognition error:", event.error);
        isListening = false;
        updateVoiceButtonState();
    };

    recognition.onresult = (event) => {
        const transcript = event.results[0][0].transcript.trim().toLowerCase();
        console.log("Heard:", transcript); // For debugging purposes
        processVoiceCommand(transcript);
    };
}

/**
 * Toggles voice recognition on or off when the user clicks the voice button.
 */
function toggleVoice() {
    if (!SpeechRecognition) return;

    voiceEnabled = !voiceEnabled;
    localStorage.setItem('voiceEnabled', voiceEnabled); // Save user preference

    if (voiceEnabled && !isListening) {
        try {
            recognition.start();
        } catch (e) {
            console.error("Could not start recognition:", e);
        }
    } else {
        recognition.stop();
    }
    updateVoiceButtonState();
}


// --- COMMAND PROCESSING ---

/**
 * Routes the transcribed text to the correct handler based on the current page.
 * @param {string} transcript - The lowercase, trimmed text from speech recognition.
 */
function processVoiceCommand(transcript) {
    const page = document.body.dataset.page;
    if (!page) return;

    switch (page) {
        case 'home':
            handleHomePageCommands(transcript);
            break;
        case 'biofertilizer':
            handleBiofertilizerCommands(transcript);
            break;
        case 'compost':
            handleCompostCommands(transcript);
            break;
        case 'waste':
            handleWasteCommands(transcript);
            break;
    }
}

/**
 * Handles voice commands on the home page for navigation.
 */
function handleHomePageCommands(transcript) {
    const lang = document.documentElement.lang || 'en';
    // Keywords for navigation in each language
    const navKeywords = {
        'en': { bio: 'biofertilizer', waste: 'waste', compost: 'compost' },
        'te': { bio: 'బయో', waste: 'వ్యర్థాల', compost: 'కంపోస్ట్' },
        'hi': { bio: 'बायो', waste: 'अपशिष्ट', compost: 'खाद' }
    };
    
    if (transcript.includes(navKeywords[lang].bio)) {
        speakAndNavigate("Navigating to Biofertilizer Recommendation.", `/biofertilizer?lang=${lang}`);
    } else if (transcript.includes(navKeywords[lang].waste)) {
        speakAndNavigate("Navigating to Waste Classification.", `/waste?lang=${lang}`);
    } else if (transcript.includes(navKeywords[lang].compost)) {
        speakAndNavigate("Navigating to Compost Tracker.", `/compost?lang=${lang}`);
    }
}

/**
 * Handles voice commands on the Biofertilizer page to select soil and crop.
 * Example command: "soil black crop rice"
 */
function handleBiofertilizerCommands(transcript) {
    const soilSelect = document.querySelector('select[name="soil"]');
    const cropSelect = document.querySelector('select[name="crop"]');
    
    const keywords = {
        'en': { soil: 'soil', crop: 'crop' },
        'te': { soil: 'మట్టి', crop: 'పంట' },
        'hi': { soil: 'मिट्टी', crop: 'फसल' }
    };
    const lang = document.documentElement.lang || 'en';

    const parts = transcript.split(keywords[lang].crop);
    if (parts.length < 2) return; // Command incomplete

    const soilValue = parts[0].replace(keywords[lang].soil, '').trim();
    const cropValue = parts[1].trim();

    const soilSet = setSelectBySpokenText(soilSelect, soilValue);
    const cropSet = setSelectBySpokenText(cropSelect, cropValue);

    if (soilSet && cropSet) {
        speak("Submitting your selection.");
        document.getElementById('bioForm').querySelector('button[type="submit"]').click();
    } else {
        if (!soilSet) speak(`Could not find soil type ${soilValue}.`);
        if (!cropSet) speak(`Could not find crop type ${cropValue}.`);
    }
}

/**
 * Handles voice commands on the Compost Tracker page.
 * Example: "smell earthy color dark heat low moisture moist days 70"
 */
function handleCompostCommands(transcript) {
    const form = document.getElementById('compostForm');
    let commandsFound = 0;

    const keywords = {
        'en': { smell: 'smell', color: 'color', heat: 'heat', moisture: 'moisture', days: 'days' },
        'te': { smell: 'వాసన', color: 'రంగు', heat: 'వేడి', moisture: 'తేమ', days: 'రోజులు' },
        'hi': { smell: 'गंध', color: 'रंग', heat: 'गर्मी', moisture: 'नमी', days: 'दिन' }
    };
    const lang = document.documentElement.lang || 'en';

    for (const key in keywords[lang]) {
        const keyword = keywords[lang][key];
        const regex = new RegExp(`${keyword}\\s+([^,]+)`);
        const match = transcript.match(regex);
        
        if (match && match[1]) {
            const value = match[1].trim();
            const select = form.querySelector(`select[name="${key}"]`);
            if (setSelectBySpokenText(select, value)) {
                commandsFound++;
            } else {
                speak(`Could not find value ${value} for ${keyword}.`);
            }
        }
    }

    if (commandsFound > 0) {
        speak("Submitting compost data.");
        form.querySelector('button[type="submit"]').click();
    }
}

/**
 * Handles voice commands on the Waste Classification page.
 * Example: "upload image" or "classify"
 */
function handleWasteCommands(transcript) {
    const keywords = {
        'en': { upload: 'upload', choose: 'choose', submit: 'submit', classify: 'classify' },
        'te': { upload: 'అప్లోడ్', choose: 'ఎంచుకోండి', submit: 'సమర్పించండి', classify: 'వర్గీకరించండి' },
        'hi': { upload: 'अपलोड', choose: 'चुनें', submit: 'जमा', classify: 'वर्गीकृत' }
    };
    const lang = document.documentElement.lang || 'en';
    
    if (transcript.includes(keywords[lang].upload) || transcript.includes(keywords[lang].choose)) {
        speak("Opening file selection.");
        document.querySelector('input[type="file"]').click();
    } else if (transcript.includes(keywords[lang].submit) || transcript.includes(keywords[lang].classify)) {
        speak("Classifying image.");
        document.getElementById('wasteForm').querySelector('button[type="submit"]').click();
    }
}


// --- HELPER & UI FUNCTIONS ---

/**
 * Finds an option in a dropdown by matching its visible text content to spoken text.
 * @param {HTMLSelectElement} selectEl The select element to search within.
 * @param {string} spokenText The text to match against the options.
 * @returns {boolean} True if a match was found and set, false otherwise.
 */
function setSelectBySpokenText(selectEl, spokenText) {
    if (!selectEl || !spokenText) return false;
    
    const option = Array.from(selectEl.options).find(opt => opt.textContent.trim().toLowerCase() === spokenText.toLowerCase());

    if (option) {
        selectEl.value = option.value;
        selectEl.dispatchEvent(new Event('change')); // Trigger animations
        return true;
    }
    console.warn(`Could not find option for spoken text: "${spokenText}"`);
    return false;
}

/**
 * Speaks a confirmation message and then navigates to a new page.
 */
function speakAndNavigate(text, url) {
    speak(text);
    setTimeout(() => {
        const link = document.querySelector(`a[href^="${url.split('?')[0]}"]`);
        if (link && link.querySelector('.card')) {
            link.querySelector('.card').classList.add('is-exiting');
        }
        setTimeout(() => {
            window.location.href = url;
        }, 500);
    }, 1500);
}

/**
 * Updates the visual state of the voice button (color and animation).
 */
function updateVoiceButtonState() {
    const btn = document.getElementById('voiceBtn');
    if (!btn) return;

    btn.classList.remove('btn-success', 'btn-secondary', 'active', 'is-listening');

    if (voiceEnabled) {
        btn.classList.add('btn-success', 'active');
        if (isListening) {
            btn.classList.add('is-listening'); // Add class for listening animation
        }
    } else {
        btn.classList.add('btn-secondary');
    }
}

/**
 * Speaks text using the browser's TTS engine (Text-to-Speech).
 * @param {string} text - The text to be spoken.
 */
function speak(text) {
    if (!voiceEnabled || !('speechSynthesis' in window)) return;
    speechSynthesis.cancel(); // Stop any previous speech
    const pageLang = document.documentElement.lang || 'en';
    const targetLangCode = pageLang === 'te' ? 'te-IN' : pageLang === 'hi' ? 'hi-IN' : 'en-US';
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = targetLangCode;
    const setVoice = () => {
        const voices = speechSynthesis.getVoices();
        const selectedVoice = voices.find(voice => voice.lang === targetLangCode) || voices.find(voice => voice.lang.startsWith(pageLang));
        if (selectedVoice) utterance.voice = selectedVoice;
        speechSynthesis.speak(utterance);
    };
    if (speechSynthesis.getVoices().length > 0) setVoice();
    else speechSynthesis.onvoiceschanged = setVoice;
}

/**
 * Applies a page-exit animation when a module card is clicked.
 */
function initializeExitAnimations() {
    const moduleLinks = document.querySelectorAll('.module-link');
    if (!moduleLinks.length) return;

    moduleLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const destination = this.href;
            this.querySelector('.card').classList.add('is-exiting');
            setTimeout(() => { window.location.href = destination; }, 500);
        });
    });
}

/**
 * Updates the language of the page by reloading with a new query parameter.
 * NOTE: This function was moved from the inline script in index.html to here.
 */
function changeLang() {
    const lang = document.getElementById('langSelect').value;
    const currentUrl = new URL(window.location.href);
    currentUrl.searchParams.set('lang', lang);
    window.location.href = currentUrl.toString();
}